"""Learned retention scoring and eviction policy for the Dynamic Notes Bus.

Implements importance-weighted snapshot retention using EMA-accumulated
attention scores (H2O-style), recency pinning (StreamingLLM-style),
and optional compressive merge of low-scoring snapshots.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch

from .dnb_bus import Snapshot

LOGGER = logging.getLogger("parallel_decoder_transformer.retention")


@dataclass(slots=True)
class RetentionConfig:
    """Configuration for snapshot retention scoring and eviction policy.

    Attributes:
        mode: Retention strategy. ``"fifo"`` reproduces current behavior;
            ``"scored"`` uses attention-EMA eviction; ``"compressed"`` adds
            merge; ``"hybrid"`` combines scored eviction with recency pinning
            and optional compression.
        ema_decay: Exponential moving average decay for importance scores.
        recency_pinned_fraction: Fraction of the buffer that is unconditionally
            retained (most-recent snapshots).
        compression_enabled: When True, the two lowest-scoring candidates are
            merged rather than dropping one.
        detach_compressed_grads: Detach the merged notes tensor from the
            autograd graph (safe default for inference and early training).
        eviction_network_enabled: Reserved for Phase 5 differentiable eviction.
        eviction_network_hidden: Hidden size for the eviction MLP (Phase 5).
        score_floor: Initial importance score for newly pushed snapshots.
        min_score_spread: When the spread (max - min) of candidate scores is
            below this threshold, fall back to FIFO to avoid noisy eviction.
    """

    mode: Literal["fifo", "scored", "compressed", "hybrid"] = "fifo"
    ema_decay: float = 0.9
    recency_pinned_fraction: float = 0.5
    compression_enabled: bool = True
    detach_compressed_grads: bool = True
    eviction_network_enabled: bool = False
    eviction_network_hidden: int = 128
    score_floor: float = 1e-6
    min_score_spread: float = 0.05


class SnapshotScoreBank:
    """Running EMA importance scores keyed by snapshot version.

    Each call to :meth:`update` ingests a detached cross-attention weight
    tensor and updates per-slot EMA scores.  Scores are stored by snapshot
    version so they survive buffer reordering.
    """

    def __init__(self, config: RetentionConfig) -> None:
        self._config = config
        self._scores: Dict[int, float] = {}

    def update(
        self,
        attn_weights: torch.Tensor,
        slot_versions: List[int],
    ) -> None:
        """Ingest attention weights and update EMA importance scores.

        Args:
            attn_weights: Detached cross-attention weights with shape
                ``[B, H, T, K]`` where ``K`` matches ``len(slot_versions)``.
            slot_versions: Snapshot version for each of the ``K`` visible
                slots.  Versions not yet tracked are initialised at
                ``score_floor``.
        """
        if attn_weights.numel() == 0 or not slot_versions:
            return
        # Mean over batch, heads, and query positions -> [K]
        per_slot = attn_weights.detach().float().mean(dim=(0, 1, 2))
        alpha = self._config.ema_decay
        floor = self._config.score_floor
        for idx, version in enumerate(slot_versions):
            if idx >= per_slot.size(0):
                break
            raw = float(per_slot[idx].item())
            prev = self._scores.get(version, floor)
            self._scores[version] = alpha * prev + (1.0 - alpha) * raw

    def scores_for(self, snapshots: List[Snapshot]) -> torch.Tensor:
        """Return a float32 tensor of importance scores aligned to *snapshots*.

        Args:
            snapshots: Ordered buffer contents.

        Returns:
            Tensor of shape ``[K]`` with one score per snapshot.
        """
        floor = self._config.score_floor
        values = [self._scores.get(s.version, floor) for s in snapshots]
        return torch.tensor(values, dtype=torch.float32)

    def reset_slot(self, version: int) -> None:
        """Remove the tracked score for a specific snapshot version."""
        self._scores.pop(version, None)


class RetentionPolicy:
    """Selects which snapshot(s) to evict when the buffer is full.

    Supports FIFO, scored, and hybrid (scored + recency pinning + optional
    compression) modes.
    """

    def __init__(self, config: RetentionConfig, max_snapshots: int) -> None:
        self._config = config
        self._max_snapshots = max_snapshots

    def select_eviction(
        self,
        buffer: List[Snapshot],
        scores: torch.Tensor,
        lag: int,
    ) -> Tuple[int, Optional[int]]:
        """Identify the slot(s) to evict or merge.

        Args:
            buffer: Current buffer contents in temporal order (oldest first).
            scores: Importance scores aligned to *buffer*, shape ``[K]``.
            lag: Current lag parameter; the most-recent ``lag`` slots are
                protected from eviction.

        Returns:
            A tuple ``(evict_idx, merge_with_idx)``.  ``merge_with_idx`` is
            ``None`` when compression is disabled or only one candidate exists.
        """
        K = len(buffer)
        if K == 0:
            raise ValueError("Cannot evict from an empty buffer.")

        mode = self._config.mode
        if mode == "fifo":
            return (0, None)

        # --- Build candidate set -----------------------------------------------
        K_recent = max(1, math.ceil(K * self._config.recency_pinned_fraction))
        recency_start = K - K_recent
        lag_start = K - lag if lag > 0 else K

        candidates: List[int] = []
        for i in range(K):
            if i >= recency_start:
                continue
            if i >= lag_start:
                continue
            meta = buffer[i].metadata
            if isinstance(meta, dict) and meta.get("pin", False):
                continue
            candidates.append(i)

        # Safety valve: if all slots are protected, fall back to FIFO.
        if not candidates:
            LOGGER.debug(
                "retention_policy_fallback_fifo | all %d slots protected", K
            )
            return (0, None)

        # --- Score spread guard -------------------------------------------------
        candidate_scores = scores[candidates]
        spread = float((candidate_scores.max() - candidate_scores.min()).item())
        if spread < self._config.min_score_spread:
            LOGGER.debug(
                "retention_policy_fallback_fifo | score_spread=%.4f < min=%.4f",
                spread,
                self._config.min_score_spread,
            )
            return (candidates[0], None)

        # --- Select lowest-scoring candidate ------------------------------------
        sorted_indices = candidate_scores.argsort()
        evict_idx = candidates[int(sorted_indices[0].item())]

        merge_idx: Optional[int] = None
        use_compression = (
            self._config.compression_enabled
            and mode in ("compressed", "hybrid")
            and len(candidates) >= 2
        )
        if use_compression:
            merge_idx = candidates[int(sorted_indices[1].item())]

        return (evict_idx, merge_idx)


def compress_snapshots(
    a: Snapshot,
    b: Snapshot,
    score_a: float,
    score_b: float,
    *,
    detach: bool = True,
) -> Snapshot:
    """Merge two snapshots via importance-weighted averaging.

    The merge is fully differentiable when ``detach=False``.

    Args:
        a: First snapshot (typically lower-scoring).
        b: Second snapshot.
        score_a: Importance score for *a*.
        score_b: Importance score for *b*.
        detach: If True, detach the merged notes tensor from the autograd
            graph.

    Returns:
        A new :class:`Snapshot` whose notes are the weighted average of
        *a* and *b*, version is ``max(a.version, b.version)``, and stride
        is the sum of both strides.
    """
    total = score_a + score_b
    if total == 0.0:
        w_a = 0.5
        w_b = 0.5
    else:
        w_a = score_a / total
        w_b = score_b / total

    merged_notes = w_a * a.notes.float() + w_b * b.notes.float()
    merged_notes = merged_notes.to(dtype=a.notes.dtype)
    if detach:
        merged_notes = merged_notes.detach()

    return Snapshot(
        version=max(a.version, b.version),
        stride=a.stride + b.stride,
        notes=merged_notes,
        metadata={},
    )


__all__ = [
    "RetentionConfig",
    "RetentionPolicy",
    "SnapshotScoreBank",
    "compress_snapshots",
]
