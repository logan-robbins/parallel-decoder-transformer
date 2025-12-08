"""Dynamic Notes Bus implementation used by the GPT-OSS integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch


@dataclass(slots=True)
class DynamicNotesBusConfig:
    snapshot_dim: int
    max_snapshots: int = 4
    lag: int = 1
    device: Optional[str] = None
    dtype: str = "bfloat16"


@dataclass(frozen=True, slots=True)
class Snapshot:
    version: int
    stride: int
    notes: torch.Tensor
    metadata: Dict[str, torch.Tensor]


class DynamicNotesBus:
    """Maintains lagged note snapshots with compaction."""

    def __init__(self, config: DynamicNotesBusConfig) -> None:
        self.config = config
        self._buffer: Deque[Snapshot] = deque()
        self._version = 0

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

    def _compact(self) -> None:
        while len(self._buffer) > self.config.max_snapshots:
            self._buffer.popleft()


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
