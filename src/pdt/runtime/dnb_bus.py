"""Dynamic Notes Bus: versioned FIFO of per-stream note snapshots.

One bus per stream. Each push produces a versioned snapshot; consumers read
a lagged window (``lag`` snapshots back) bounded by ``max_snapshots``.

The implementation is tokenizer- and V_p-blind -- it stores raw tensors
and metadata. Shape conventions are enforced at push time via the
``snapshot_dim`` config value.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch

from pdt.config.schemas import NotesBusConfig


LOGGER = logging.getLogger("pdt.runtime.dnb_bus")

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


__all__ = ["DynamicNotesBus", "DynamicNotesBusConfig", "Snapshot"]


# Re-export under both names for backwards discoverability.
DynamicNotesBusConfig = NotesBusConfig


@dataclass(frozen=True, slots=True)
class Snapshot:
    version: int
    stride: int
    notes: torch.Tensor
    metadata: Dict[str, torch.Tensor]


class DynamicNotesBus:
    """Lagged FIFO of snapshots for a single stream."""

    def __init__(self, config: NotesBusConfig, device: Optional[str] = None) -> None:
        self.config = config
        self.device = device
        self.dtype = _DTYPE_MAP[config.dtype]
        self._buffer: Deque[Snapshot] = deque()
        self._version = 0

    def push(
        self,
        notes: torch.Tensor,
        *,
        stride: int,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Snapshot:
        """Append a new snapshot, return it, and FIFO-compact."""
        if notes.size(-1) != self.config.snapshot_dim:
            raise ValueError(
                f"Snapshot dim mismatch: expected {self.config.snapshot_dim}, "
                f"got {notes.size(-1)}."
            )
        target_device = self.device or notes.device
        payload = Snapshot(
            version=self._version + 1,
            stride=stride,
            notes=notes.to(device=target_device, dtype=self.dtype),
            metadata=metadata or {},
        )
        self._buffer.append(payload)
        self._version = payload.version
        while len(self._buffer) > self.config.max_snapshots:
            self._buffer.popleft()
        return payload

    def snapshot(
        self,
        *,
        lag: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Snapshot]:
        """Return the lag-aware visible window, ordered oldest -> newest."""
        if not self._buffer:
            return []
        effective_lag = self.config.lag if lag is None else lag
        effective_limit = self.config.max_snapshots if limit is None else limit
        snapshots = list(self._buffer)
        cut = max(0, len(snapshots) - effective_lag)
        window = snapshots[:cut]
        if effective_limit:
            window = window[-effective_limit:]
        return window

    def latest_version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return len(self._buffer)
