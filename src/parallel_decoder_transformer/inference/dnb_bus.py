"""Dynamic Notes Bus implementation used by the GPT-OSS integration."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from .retention import RetentionConfig

LOGGER = logging.getLogger("parallel_decoder_transformer.dnb_bus")


@dataclass(slots=True)
class DynamicNotesBusConfig:
    snapshot_dim: int
    max_snapshots: int = 4
    lag: int = 1
    device: Optional[str] = None
    dtype: str = "bfloat16"
    retention: Optional["RetentionConfig"] = None


@dataclass(frozen=True, slots=True)
class Snapshot:
    version: int
    stride: int
    notes: torch.Tensor
    metadata: Dict[str, torch.Tensor]


class DynamicNotesBus:
    """Maintains lagged note snapshots with compaction.

    When ``config.retention`` is ``None`` or its mode is ``"fifo"``, compaction
    uses the original FIFO strategy (popleft).  When a scored/compressed/hybrid
    retention config is provided, compaction delegates to
    :class:`~.retention.RetentionPolicy`.
    """

    def __init__(self, config: DynamicNotesBusConfig) -> None:
        self.config = config
        self._buffer: Deque[Snapshot] = deque()
        self._version = 0

        # Retention scoring -- lazily imported to avoid circular deps.
        self._score_bank: Optional[object] = None
        self._retention_policy: Optional[object] = None
        if config.retention is not None and config.retention.mode != "fifo":
            from .retention import RetentionPolicy, SnapshotScoreBank

            self._score_bank = SnapshotScoreBank(config.retention)
            self._retention_policy = RetentionPolicy(
                config.retention, config.max_snapshots
            )

    def push(
        self,
        notes: torch.Tensor,
        *,
        stride: int,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Snapshot:
        if notes.size(-1) != self.config.snapshot_dim:
            raise ValueError(
                f"Snapshot dimension mismatch. Expected {self.config.snapshot_dim}, got {notes.size(-1)}."
            )
        device = self.config.device or notes.device
        dtype = _resolve_dtype(self.config.dtype)
        payload = Snapshot(
            version=self._version + 1,
            stride=stride,
            notes=notes.to(device=device, dtype=dtype),
            metadata=metadata or {},
        )
        self._buffer.append(payload)
        self._version = payload.version
        self._compact()
        return payload

    def snapshot(
        self,
        *,
        lag: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Snapshot]:
        if not self._buffer:
            return []
        effective_lag = self.config.lag if lag is None else lag
        effective_limit = self.config.max_snapshots if limit is None else limit
        snapshots = list(self._buffer)
        cut_index = max(0, len(snapshots) - effective_lag)
        window = snapshots[:cut_index]
        if effective_limit:
            window = window[-effective_limit:]
        return window

    def masked_snapshot(self, mask: torch.Tensor) -> List[Snapshot]:
        masked: List[Snapshot] = []
        for snapshot in self.snapshot():
            masked_notes = snapshot.notes * mask
            masked.append(
                Snapshot(
                    version=snapshot.version,
                    stride=snapshot.stride,
                    notes=masked_notes,
                    metadata=snapshot.metadata,
                )
            )
        return masked

    def latest_version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Retention scoring
    # ------------------------------------------------------------------

    def update_scores(
        self,
        attn_weights: torch.Tensor,
        slot_versions: List[int],
    ) -> None:
        """Ingest cross-attention weights into the score bank.

        This is a no-op when retention scoring is not configured.

        Args:
            attn_weights: Detached attention weights ``[B, H, T, K]``.
            slot_versions: Snapshot version for each of the ``K`` visible
                slots.
        """
        if self._score_bank is not None:
            from .retention import SnapshotScoreBank

            assert isinstance(self._score_bank, SnapshotScoreBank)
            self._score_bank.update(attn_weights, slot_versions)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        """Remove or merge excess snapshots to respect ``max_snapshots``."""
        while len(self._buffer) > self.config.max_snapshots:
            if self._retention_policy is None:
                # FIFO fallback -- bit-identical to original behavior.
                self._buffer.popleft()
            else:
                self._scored_compact()

    def _scored_compact(self) -> None:
        """Run one round of scored eviction or compressive merge."""
        from .retention import (
            RetentionPolicy,
            SnapshotScoreBank,
            compress_snapshots,
        )

        assert isinstance(self._score_bank, SnapshotScoreBank)
        assert isinstance(self._retention_policy, RetentionPolicy)
        assert self.config.retention is not None

        buffer_list = list(self._buffer)
        scores = self._score_bank.scores_for(buffer_list)
        evict_idx, merge_idx = self._retention_policy.select_eviction(
            buffer_list, scores, self.config.lag
        )

        if merge_idx is not None:
            a = buffer_list[evict_idx]
            b = buffer_list[merge_idx]
            score_a = float(scores[evict_idx].item())
            score_b = float(scores[merge_idx].item())
            merged = compress_snapshots(
                a,
                b,
                score_a,
                score_b,
                detach=self.config.retention.detach_compressed_grads,
            )
            # Remove both, insert merged at the earlier position.
            lo = min(evict_idx, merge_idx)
            hi = max(evict_idx, merge_idx)
            del buffer_list[hi]
            del buffer_list[lo]
            buffer_list.insert(lo, merged)
            # Clean up stale scores.
            self._score_bank.reset_slot(a.version)
            self._score_bank.reset_slot(b.version)
        else:
            evicted = buffer_list.pop(evict_idx)
            self._score_bank.reset_slot(evicted.version)

        self._buffer = deque(buffer_list)


def _resolve_dtype(alias: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[alias]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported dtype alias: {alias!r}") from exc


__all__ = ["DynamicNotesBus", "DynamicNotesBusConfig", "Snapshot"]
