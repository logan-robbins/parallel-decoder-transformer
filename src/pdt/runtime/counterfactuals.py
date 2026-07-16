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

- **C -- Anchor swap.** Replace stream ``k``'s snapshot-0 anchor with one
  drawn from a *different prompt entirely*. Tests whether SNC interprets
  sibling position prompt-conditionally rather than as generic noise.

- **D -- Source swap.** Deterministically replace sibling dynamic-note
  payloads from another example or explicit donor window while preserving
  producer slots.

- **E -- Bus mutation.** Add a deterministic perturbation to one producer's
  dynamic note at one published block as a direct causal-path probe.

These are applied at the bus / window layer before the orchestrator's
forward call, so the trained checkpoint is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch

from pdt.runtime.dnb_bus import DynamicNotesBus


__all__ = [
    "CounterfactualConfig",
    "apply_gate_ablation",
    "apply_norm_scramble",
    "apply_anchor_swap",
    "apply_source_swap",
    "apply_bus_mutation",
]


@dataclass(slots=True)
class CounterfactualConfig:
    mode: Optional[
        Literal[
            "gate_zero",
            "norm_scramble",
            "anchor_swap",
            "source_swap",
            "bus_mutation",
            "none",
        ]
    ] = None
    # Anchor-swap needs a reference snapshot-0 tensor from a different prompt.
    alt_prompt_anchors: Optional[torch.Tensor] = None  # (K, d_notes)
    # Source-swap donor window. If omitted, batching must provide B >= 2 and
    # donor notes are obtained by a deterministic roll across examples.
    source_swap_donor: Optional[torch.Tensor] = None  # (B, 2K, d_notes)
    source_swap_donor_mask: Optional[torch.Tensor] = None  # (B, 2K)
    # Seed for reproducibility.
    seed: Optional[int] = None
    # Bus-mutation target. ``None`` selects the first configured producer.
    mutation_producer: Optional[str] = None
    mutation_block: int = 0
    mutation_magnitude: float = 1.0


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


def apply_anchor_swap(
    bus: DynamicNotesBus,
    alt_prompt_anchors: torch.Tensor,
    stream_order: tuple[str, ...],
) -> None:
    """Overwrite each stream's snapshot-0 with the alt-prompt anchor.

    Mutates the bus in place. Must be called before any stream begins
    emitting tokens (i.e. after ``plan_notes_proj`` published snapshot 0 but
    before the first block).

    Args:
        bus: shared addressed DNB containing every stream's anchor.
        alt_prompt_anchors: ``(K, d_notes)`` tensor -- snapshot-0 anchors
            computed on a *different prompt entirely*.
        stream_order: iteration order matching ``alt_prompt_anchors[k]``.
    """
    if alt_prompt_anchors.dim() != 2:
        raise ValueError(
            f"alt_prompt_anchors must be rank 2 (K, d_notes), got rank {alt_prompt_anchors.dim()}"
        )
    if alt_prompt_anchors.size(0) != len(stream_order):
        raise ValueError(
            f"alt_prompt_anchors.size(0)={alt_prompt_anchors.size(0)} != "
            f"len(stream_order)={len(stream_order)}"
        )
    for idx, stream in enumerate(stream_order):
        bus.replace_anchor(stream, alt_prompt_anchors[idx])


def apply_source_swap(
    notes: torch.Tensor,
    mask: torch.Tensor,
    *,
    anchor_mask: torch.Tensor,
    producer_indices: torch.Tensor,
    consumer_index: int,
    donor_notes: Optional[torch.Tensor] = None,
    donor_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap sibling dynamic payloads across examples or from a donor window.

    The receiver's own dynamic note and all prompt anchors remain unchanged.
    Without an explicit donor, the batch is rolled by one example. A
    within-window producer permutation is intentionally unsupported because
    unaddressed cross-attention is exactly permutation-invariant to that move.
    """
    if notes.dim() != 3 or mask.shape != notes.shape[:2]:
        raise ValueError("notes/mask must have shapes (B, S, d) and (B, S).")
    slots = notes.size(1)
    if anchor_mask.shape != (slots,) or producer_indices.shape != (slots,):
        raise ValueError("anchor_mask and producer_indices must each have shape (S,).")
    producer_count = int(producer_indices.max().item()) + 1
    if consumer_index < 0 or consumer_index >= producer_count:
        raise ValueError(f"consumer_index must be in [0, {producer_count}), got {consumer_index}.")
    sibling_slots = torch.nonzero(
        (~anchor_mask) & (producer_indices != consumer_index), as_tuple=False
    ).flatten()
    if sibling_slots.numel() < 2:
        raise ValueError("source_swap requires at least two sibling dynamic slots.")

    if (donor_notes is None) != (donor_mask is None):
        raise ValueError("donor_notes and donor_mask must be provided together.")
    if donor_notes is None:
        if notes.size(0) < 2:
            raise ValueError(
                "source_swap requires B >= 2 or an explicit donor window; "
                "within-window producer permutation is an exact null."
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


def apply_bus_mutation(notes: torch.Tensor, *, magnitude: float = 1.0) -> torch.Tensor:
    """Add a deterministic first-coordinate perturbation to a dynamic note."""
    if notes.dim() not in (1, 2) or notes.size(-1) == 0:
        raise ValueError("notes must have shape (d,) or (B, d) with d > 0.")
    if not math.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("mutation magnitude must be finite and non-zero.")
    mutated = notes.clone()
    mutated[..., 0] = mutated[..., 0] + magnitude
    return mutated
