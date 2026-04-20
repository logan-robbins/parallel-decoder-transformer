"""Counterfactual interventions for the pre-registered ablations.

Implements the three paper-level interventions exactly (paper \u00a76 /
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

These are applied at the bus / window layer before the orchestrator's
forward call, so the trained checkpoint is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from pdt.runtime.dnb_bus import DynamicNotesBus


__all__ = [
    "CounterfactualConfig",
    "apply_gate_ablation",
    "apply_norm_scramble",
    "apply_anchor_swap",
]


@dataclass(slots=True)
class CounterfactualConfig:
    mode: Optional[Literal["gate_zero", "norm_scramble", "anchor_swap", "none"]] = None
    # Anchor-swap needs a reference snapshot-0 tensor from a different prompt.
    alt_prompt_anchors: Optional[torch.Tensor] = None  # (K, d_notes)
    # Seed for reproducibility.
    seed: Optional[int] = None


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
    scramble = torch.randn(
        notes.size(),
        dtype=notes.dtype,
        device=notes.device,
        generator=generator,
    )
    norms = torch.linalg.vector_norm(notes, dim=-1, keepdim=True)
    scramble_norms = torch.linalg.vector_norm(scramble, dim=-1, keepdim=True).clamp(min=1e-12)
    return scramble / scramble_norms * norms


def apply_anchor_swap(
    bus_by_stream: dict[str, DynamicNotesBus],
    alt_prompt_anchors: torch.Tensor,
    stream_order: tuple[str, ...],
) -> None:
    """Overwrite each stream's snapshot-0 with the alt-prompt anchor.

    Mutates the bus in place. Must be called before any stream begins
    emitting tokens (i.e. after ``plan_notes_proj`` published snapshot 0 but
    before the first block).

    Args:
        bus_by_stream: mapping from stream name to its DNB instance.
        alt_prompt_anchors: ``(K, d_notes)`` tensor -- snapshot-0 anchors
            computed on a *different prompt entirely*.
        stream_order: iteration order matching ``alt_prompt_anchors[k]``.
    """
    if alt_prompt_anchors.dim() != 2:
        raise ValueError(
            f"alt_prompt_anchors must be rank 2 (K, d_notes), got rank "
            f"{alt_prompt_anchors.dim()}"
        )
    if alt_prompt_anchors.size(0) != len(stream_order):
        raise ValueError(
            f"alt_prompt_anchors.size(0)={alt_prompt_anchors.size(0)} != "
            f"len(stream_order)={len(stream_order)}"
        )
    for idx, stream in enumerate(stream_order):
        bus = bus_by_stream.get(stream)
        if bus is None or len(bus) == 0:
            continue
        # Replace the earliest (snapshot-0) entry. We rebuild by popping and
        # re-pushing because our bus only supports push/pop via the deque.
        buffer = bus._buffer  # intentional access; counterfactuals are a
        # meta-operation on the bus that the public API does not expose.
        if len(buffer) == 0:
            continue
        earliest = buffer[0]
        replacement_notes = alt_prompt_anchors[idx].to(
            dtype=earliest.notes.dtype, device=earliest.notes.device
        )
        # Snapshot is immutable (frozen dataclass) -- replace the reference.
        from pdt.runtime.dnb_bus import Snapshot

        buffer[0] = Snapshot(
            version=earliest.version,
            stride=earliest.stride,
            notes=replacement_notes,
            metadata=earliest.metadata,
        )
