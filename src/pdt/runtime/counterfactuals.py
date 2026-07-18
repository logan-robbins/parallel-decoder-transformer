"""Counterfactual interventions for the pre-registered ablations.

Implements the paper-level interventions (paper \u00a76 /
evolution log \u00a76):

- **A -- SNC gate ablation.** Force \u03bb_l \u2190 0 at every instrumented layer.
  Implemented by passing ``snc_force_gate=False`` into the per-layer
  ``LayerRuntimeContext``. The SNC delta collapses to the zero tensor (proof
  in ``tests/smoke/pdt_tests/test_diag_build_step2.py``).

- **B -- Norm-matched sibling-write scramble.** Replace sibling notes with
  Gaussian vectors rescaled to the empirical per-note norm. Keep gates as
  trained so this isolates *informational content* from attention-softmax
  numerics.

- **C -- Complete-plan intervention.** Swap, zero, or norm-match-randomize
  the persistent plan before any lane emits tokens.

- **D -- Source swap.** Deterministically replace sibling dynamic-note
  payloads from another example or explicit donor window while preserving
  producer slots.

- **E -- Bus mutation.** Increment one transmitted product-code index for one
  producer at one published block as a direct causal-path probe. The mutation
  is guaranteed to select a different in-alphabet message.

These are applied at the bus / window layer before the orchestrator's
forward call, so the trained checkpoint is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

__all__ = [
    "CounterfactualConfig",
    "apply_gate_ablation",
    "apply_norm_scramble",
    "apply_plan_intervention",
    "apply_source_swap",
    "apply_bus_mutation",
]


@dataclass(slots=True)
class CounterfactualConfig:
    mode: Optional[
        Literal[
            "gate_zero",
            "norm_scramble",
            "plan_swap",
            "lane_swap",
            "plan_zero",
            "random_plan",
            "source_swap",
            "bus_mutation",
            "none",
        ]
    ] = None
    # Plan-swap needs a full plan tensor from a different prompt.
    alt_prompt_plan_nodes: Optional[torch.Tensor] = None  # (K, N, planner_width)
    # Physical-lane plan swap; defaults to D1 <-> D2.
    plan_swap_lanes: tuple[int, int] = (0, 1)
    # Source-swap donor window. If omitted, batching must provide B >= 2 and
    # donor notes are obtained by a deterministic roll across examples.
    source_swap_donor: Optional[torch.Tensor] = None  # (B, addressed_slots, d_notes)
    source_swap_donor_mask: Optional[torch.Tensor] = None  # (B, addressed_slots)
    # Seed for reproducibility.
    seed: Optional[int] = None
    # Bus-mutation target. ``None`` selects the first configured producer.
    mutation_producer: Optional[str] = None
    mutation_block: int = 0
    mutation_code_offset: int = 1


def apply_gate_ablation() -> object:
    """Return the ``snc_force_gate`` value that closes SNC at every layer.

    Intended use: set ``LayerRuntimeContext.snc_force_gate = apply_gate_ablation()``
    on every instrumented layer prior to a forward pass.
    """
    return False  # The SNC module interprets False as "force closed".


def apply_norm_scramble(
    notes: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    slot_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replace ``notes`` with Gaussian noise rescaled to per-note norm.

    Args:
        notes: ``(B, S, d_notes)`` visible window tensor.
        generator: Optional RNG for reproducibility.

    Returns:
        Same-shape tensor with each note vector replaced by a Gaussian
        sample rescaled to match the original note's L2 norm.
    """
    if notes.dim() != 3:
        raise ValueError(f"notes must be rank 3 (B, S, d), got rank {notes.dim()}.")
    if notes.size(1) == 0:
        return notes
    if slot_mask is not None and slot_mask.shape != (notes.size(1),):
        raise ValueError("slot_mask must have shape (S,).")
    if generator is None:
        scramble = torch.randn_like(notes)
    else:
        # A CPU generator is reproducible on CPU, MPS, and CUDA. Generate in
        # float32 and then transfer so the intervention is device-independent.
        scramble = torch.randn(
            notes.size(),
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        ).to(device=notes.device, dtype=notes.dtype)
    norms = torch.linalg.vector_norm(notes, dim=-1, keepdim=True)
    scramble_norms = torch.linalg.vector_norm(scramble, dim=-1, keepdim=True).clamp(min=1e-12)
    scrambled = scramble / scramble_norms * norms
    if slot_mask is None:
        return scrambled
    return torch.where(slot_mask[None, :, None], scrambled, notes)


def apply_plan_intervention(
    plan_nodes: torch.Tensor,
    *,
    mode: Literal["plan_swap", "lane_swap", "plan_zero", "random_plan"],
    alternate: Optional[torch.Tensor] = None,
    lane_pair: tuple[int, int] = (0, 1),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply one registered intervention to `[B, K, N, P]` static plans."""

    if mode == "lane_swap":
        if plan_nodes.dim() < 3:
            raise ValueError("lane_swap requires a tensor shaped [B, K, ...].")
        if alternate is not None:
            raise ValueError("lane_swap does not accept alt_prompt_plan_nodes.")
        if (
            len(lane_pair) != 2
            or any(type(index) is not int for index in lane_pair)
            or lane_pair[0] == lane_pair[1]
            or any(not 0 <= index < plan_nodes.size(1) for index in lane_pair)
        ):
            raise ValueError(
                f"lane_pair must name two distinct lanes in [0, {plan_nodes.size(1)})."
            )
        order = torch.arange(
            plan_nodes.size(1),
            device=plan_nodes.device,
            dtype=torch.long,
        )
        left, right = lane_pair
        order[left], order[right] = order[right].clone(), order[left].clone()
        return plan_nodes.index_select(1, order)
    if plan_nodes.dim() != 4:
        raise ValueError("Plan interventions require shape [B, K, N, planner_width].")
    if mode == "plan_swap":
        if alternate is None:
            raise ValueError("plan_swap requires alt_prompt_plan_nodes.")
        candidate = alternate
        if candidate.dim() == 3:
            candidate = candidate.unsqueeze(0)
        if candidate.shape != plan_nodes.shape:
            raise ValueError(
                "alt_prompt_plan_nodes must exactly match plan_nodes shape; "
                f"got {tuple(candidate.shape)} and {tuple(plan_nodes.shape)}."
            )
        return candidate.to(device=plan_nodes.device, dtype=plan_nodes.dtype).clone()
    if alternate is not None:
        raise ValueError(f"{mode} does not accept alt_prompt_plan_nodes.")
    if mode == "plan_zero":
        return torch.zeros_like(plan_nodes)
    if mode == "random_plan":
        if generator is None:
            noise = torch.randn_like(plan_nodes)
        else:
            noise = torch.randn(
                plan_nodes.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=plan_nodes.device, dtype=plan_nodes.dtype)
        source_norm = torch.linalg.vector_norm(plan_nodes, dim=-1, keepdim=True)
        noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
        return noise * (source_norm / noise_norm)
    raise ValueError(f"Unsupported plan intervention {mode!r}.")


def apply_source_swap(
    notes: torch.Tensor,
    mask: torch.Tensor,
    *,
    producer_indices: torch.Tensor,
    consumer_index: int,
    donor_notes: Optional[torch.Tensor] = None,
    donor_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap sibling dynamic payloads across examples or from a donor window.

    The receiver's own dynamic note remains unchanged.
    Without an explicit donor, the batch is rolled by one example. A
    within-window producer permutation is intentionally not substituted for a
    donor intervention because it changes multiple addressed sources at once.
    """
    if notes.dim() != 3 or mask.shape != notes.shape[:2]:
        raise ValueError("notes/mask must have shapes (B, S, d) and (B, S).")
    slots = notes.size(1)
    if producer_indices.shape != (slots,):
        raise ValueError("producer_indices must have shape (S,).")
    producer_count = int(producer_indices.max().item()) + 1
    if consumer_index < 0 or consumer_index >= producer_count:
        raise ValueError(f"consumer_index must be in [0, {producer_count}), got {consumer_index}.")
    sibling_slots = torch.nonzero(
        producer_indices != consumer_index, as_tuple=False
    ).flatten()
    if sibling_slots.numel() < 2:
        raise ValueError("source_swap requires at least two sibling dynamic slots.")

    if (donor_notes is None) != (donor_mask is None):
        raise ValueError("donor_notes and donor_mask must be provided together.")
    if donor_notes is None:
        if notes.size(0) < 2:
            raise ValueError(
                "source_swap requires B >= 2 or an explicit donor window; "
                "a within-window permutation is not the registered donor intervention."
            )
        source_notes = torch.roll(notes, shifts=1, dims=0)
        source_mask = torch.roll(mask, shifts=1, dims=0)
    else:
        assert donor_mask is not None
        if donor_notes.shape != notes.shape or donor_mask.shape != mask.shape:
            raise ValueError("donor_notes/donor_mask must exactly match notes/mask shapes.")
        source_notes = donor_notes.to(device=notes.device, dtype=notes.dtype)
        source_mask = donor_mask.to(device=mask.device, dtype=torch.bool)

    swapped_notes = notes.clone()
    swapped_mask = mask.clone()
    swapped_notes[:, sibling_slots] = source_notes[:, sibling_slots]
    swapped_mask[:, sibling_slots] = source_mask[:, sibling_slots]
    return swapped_notes, swapped_mask


def apply_bus_mutation(
    code_indices: torch.Tensor,
    *,
    codes_per_codebook: int,
    code_offset: int = 1,
) -> torch.Tensor:
    """Guarantee a different finite message by cycling its first sub-code."""

    if code_indices.dim() < 1 or code_indices.size(-1) == 0:
        raise ValueError("code_indices must have a non-empty product-code axis.")
    if (
        code_indices.dtype == torch.bool
        or code_indices.is_floating_point()
        or code_indices.is_complex()
    ):
        raise TypeError("code_indices must use a non-bool integer dtype.")
    if type(codes_per_codebook) is not int or codes_per_codebook <= 1:
        raise ValueError("codes_per_codebook must be an integer greater than one.")
    if type(code_offset) is not int or not 0 < code_offset < codes_per_codebook:
        raise ValueError(
            f"code_offset must be an integer in [1, {codes_per_codebook}); got {code_offset!r}."
        )
    if bool(((code_indices < 0) | (code_indices >= codes_per_codebook)).any()):
        raise ValueError(f"code_indices must lie in [0, {codes_per_codebook}).")
    mutated = code_indices.clone()
    mutated[..., 0] = torch.remainder(mutated[..., 0] + code_offset, codes_per_codebook)
    if torch.equal(mutated, code_indices):
        raise RuntimeError("Finite bus mutation failed to change the transmitted message.")
    return mutated
