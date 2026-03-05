"""Tests for retention-related training integration.

Tests cover LossWeights.retain, _select_bus_evict_idx, _retention_loss,
and the scored-eviction path in _update_student_bus.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stub teacher_runner before importing trainer internals
# ---------------------------------------------------------------------------

if "parallel_decoder_transformer.data.teacher_runner" not in sys.modules:
    teacher_runner_stub = types.ModuleType("parallel_decoder_transformer.data.teacher_runner")

    @dataclass(slots=True)
    class DatasetTeacherConfig:
        cache_dir: str | None = None
        max_snapshots: int = 3
        id_field: str = "example_id"
        refresh_cache: bool = False

    @dataclass(slots=True)
    class TeacherSnapshotText:
        version: int
        stride: int
        stream_notes: Mapping[str, Sequence[str]]
        coverage: Mapping[str, float] | None = None
        source: str = "teacher"

    @dataclass(slots=True)
    class TeacherRunResult:
        example_id: str
        stream_notes: Mapping[str, Sequence[str]]
        snapshots: Sequence[TeacherSnapshotText] = field(default_factory=tuple)
        teacher_plan: Mapping[str, Any] | None = None

    def normalize_stream_id(value: str) -> str:
        stream = str(value or "").strip().lower()
        if not stream:
            return "stream_unknown"
        if not stream.startswith("stream_"):
            stream = f"stream_{stream}"
        return stream

    def normalize_stream_notes(
        payload: Mapping[str, Sequence[str]] | None,
    ) -> Mapping[str, Sequence[str]]:
        return payload or {}

    teacher_runner_stub.DatasetTeacherConfig = DatasetTeacherConfig
    teacher_runner_stub.TeacherSnapshotText = TeacherSnapshotText
    teacher_runner_stub.TeacherRunResult = TeacherRunResult
    teacher_runner_stub.normalize_stream_id = normalize_stream_id
    teacher_runner_stub.normalize_stream_notes = normalize_stream_notes
    sys.modules["parallel_decoder_transformer.data.teacher_runner"] = teacher_runner_stub

from parallel_decoder_transformer.training.trainer import LossWeights


# ---------------------------------------------------------------------------
# LossWeights tests
# ---------------------------------------------------------------------------


def test_loss_weights_retain_default_zero() -> None:
    """The default retain weight should be 0.0 (no retention loss)."""
    weights = LossWeights()
    assert weights.retain == 0.0


def test_loss_weights_retain_configurable() -> None:
    """The retain weight should accept a positive value."""
    weights = LossWeights(retain=0.5)
    assert weights.retain == 0.5


def test_loss_weights_retain_in_dataclass_fields() -> None:
    """retain should be a proper dataclass field."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(LossWeights)}
    assert "retain" in field_names


# ---------------------------------------------------------------------------
# _select_bus_evict_idx tests (via a minimal trainer mock)
# ---------------------------------------------------------------------------


class _MinimalTrainer:
    """Minimal stand-in to test _select_bus_evict_idx without full Trainer init."""

    def __init__(self, retain_weight: float = 0.0) -> None:
        self.device = torch.device("cpu")

        # Mimic the nested config structure expected by _select_bus_evict_idx.
        self.config = types.SimpleNamespace(
            loss_weights=LossWeights(retain=retain_weight),
            curriculum=types.SimpleNamespace(B=4),
        )

    # Bind the method from Trainer.
    from parallel_decoder_transformer.training.trainer import Trainer

    _select_bus_evict_idx = Trainer._select_bus_evict_idx


def test_select_evict_fifo_when_retain_zero() -> None:
    """With retain=0, _select_bus_evict_idx should return shift_start (FIFO)."""
    trainer = _MinimalTrainer(retain_weight=0.0)
    coverage = torch.tensor([[0.1, 0.9, 0.5]])  # [1, 3]
    mask = torch.tensor([[True, True, True]])
    result = trainer._select_bus_evict_idx(0, 3, 0, coverage, mask)
    assert result == 0  # FIFO


def test_select_evict_scored_when_retain_positive() -> None:
    """With retain>0 and coverage, should evict the lowest-coverage slot."""
    trainer = _MinimalTrainer(retain_weight=1.0)
    # coverage_bus: [B, max_snapshots, S]
    # Slot 0 has high coverage, slot 1 low, slot 2 medium.
    coverage = torch.tensor([[[0.9, 0.8], [0.1, 0.1], [0.5, 0.6]]])
    mask = torch.tensor([[True, True, True]])
    result = trainer._select_bus_evict_idx(0, 3, 0, coverage, mask)
    assert result == 1  # lowest mean coverage


def test_select_evict_respects_shift_start() -> None:
    """Frozen slots below shift_start should be protected."""
    trainer = _MinimalTrainer(retain_weight=1.0)
    # Slot 0 is frozen (shift_start=1).  Slot 1 has lowest coverage.
    coverage = torch.tensor([[[0.01, 0.01], [0.1, 0.1], [0.9, 0.9]]])
    mask = torch.tensor([[True, True, True]])
    result = trainer._select_bus_evict_idx(0, 3, 1, coverage, mask)
    assert result == 1  # lowest among candidates [1, 2]


def test_select_evict_low_spread_falls_back() -> None:
    """When coverage spread is < 0.05, fall back to FIFO (shift_start)."""
    trainer = _MinimalTrainer(retain_weight=1.0)
    coverage = torch.tensor([[[0.50, 0.50], [0.51, 0.51], [0.50, 0.50]]])
    mask = torch.tensor([[True, True, True]])
    result = trainer._select_bus_evict_idx(0, 3, 0, coverage, mask)
    assert result == 0  # FIFO fallback due to low spread


# ---------------------------------------------------------------------------
# _retention_loss tests (via a minimal trainer mock)
# ---------------------------------------------------------------------------


class _RetentionLossTrainer:
    """Minimal stand-in to test _retention_loss."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    from parallel_decoder_transformer.training.trainer import Trainer

    _retention_loss = Trainer._retention_loss


def test_retention_loss_zero_without_coverage_bus() -> None:
    """Without a coverage bus, retention loss should be 0."""
    trainer = _RetentionLossTrainer()
    batch = {
        "student_notes_bus": torch.randn(2, 3, 2, 4),
        "student_bus_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    loss = trainer._retention_loss(batch)
    assert loss.item() == 0.0


def test_retention_loss_zero_single_snapshot() -> None:
    """With fewer than 2 active snapshots, retention loss should be 0."""
    trainer = _RetentionLossTrainer()
    batch = {
        "student_notes_bus": torch.randn(1, 1, 2, 4),
        "student_bus_mask": torch.ones(1, 1, dtype=torch.bool),
        "student_bus_coverage": torch.randn(1, 1, 3),
    }
    loss = trainer._retention_loss(batch)
    assert loss.item() == 0.0


def test_retention_loss_positive_with_similar_coverage() -> None:
    """When active snapshots have identical coverage, loss should be positive."""
    trainer = _RetentionLossTrainer()
    # Two active snapshots with identical coverage vectors -> high similarity.
    cov = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    batch = {
        "student_notes_bus": torch.randn(1, 2, 2, 4),
        "student_bus_mask": torch.ones(1, 2, dtype=torch.bool),
        "student_bus_coverage": cov,
    }
    loss = trainer._retention_loss(batch)
    assert loss.item() > 0.0
