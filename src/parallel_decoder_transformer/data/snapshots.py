"""Shared snapshot structures for DNB-aware training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(slots=True)
class SnapshotFeatures:
    """Container describing a single set of notes for a bus snapshot."""

    notes: torch.Tensor
    stride: int = 0
    version: int = 0
    stream_id: Optional[str] = None
    coverage: Optional[torch.Tensor] = None
    source: str = "teacher"

    def to(self, dtype: torch.dtype) -> "SnapshotFeatures":
        """Return a copy converted to the requested dtype."""
        coverage = (
            self.coverage.to(dtype) if isinstance(self.coverage, torch.Tensor) else self.coverage
        )
        return SnapshotFeatures(
            notes=self.notes.to(dtype=dtype),
            stride=self.stride,
            version=self.version,
            stream_id=self.stream_id,
            coverage=coverage,
            source=self.source,
        )


__all__ = ["SnapshotFeatures"]
