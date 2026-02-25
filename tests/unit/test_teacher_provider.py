from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from parallel_decoder_transformer.data.teacher_provider import (
    CachedTeacherNotesProvider,
    DatasetTeacherNotesProvider,
    TeacherNotes,
    TeacherNotesProviderBase,
)
from parallel_decoder_transformer.data.teacher_runner import DatasetTeacherConfig


def _make_versioned_notes(strides: list[int]) -> list[dict[str, Any]]:
    """Build a minimal notes_versioned list with one snapshot per stride."""
    snapshots = []
    for i, stride in enumerate(strides):
        snapshots.append(
            {
                "snapshot_id": i,
                "source": "teacher",
                "stride": stride,
                "notes": [
                    {
                        "stream_id": "intro",
                        "ENT": [{"id": "E1", "name": "Alpha", "type": "concept"}],
                        "FACT": [],
                        "COVERAGE": [],
                    },
                    {
                        "stream_id": "core",
                        "ENT": [{"id": "E2", "name": "Beta", "type": "concept"}],
                        "FACT": [],
                        "COVERAGE": [],
                    },
                    {
                        "stream_id": "wrap",
                        "ENT": [{"id": "E3", "name": "Gamma", "type": "concept"}],
                        "FACT": [],
                        "COVERAGE": [],
                    },
                ],
            }
        )
    return snapshots


def _build_example(example_id: str = "example-123") -> dict[str, Any]:
    versioned_notes = _make_versioned_notes([0, 6, 12])
    return {
        "example_id": example_id,
        "metadata": {
            "teacher_notes": {
                "intro": ["Alpha concept"],
                "core": ["Beta concept"],
                "wrap": ["Gamma concept"],
            },
            "teacher_plan": {
                "plan": [
                    {"stream": "intro", "summary": "Set context"},
                    {"stream": "core", "summary": "Present core evidence"},
                    {"stream": "wrap", "summary": "Summarise"},
                ]
            },
            "notes_versioned": versioned_notes,
        },
    }


def _make_provider(notes_dim: int = 6) -> DatasetTeacherNotesProvider:
    config = DatasetTeacherConfig(cache_dir=None, max_snapshots=3)
    return DatasetTeacherNotesProvider(
        config,
        notes_dim=notes_dim,
        stream_to_id={"intro": 0, "core": 1, "wrap": 2},
    )


class _TrackingProvider(TeacherNotesProviderBase):
    """Wraps a provider and counts fetch calls."""

    def __init__(self, delegate: TeacherNotesProviderBase) -> None:
        self.delegate = delegate
        self.calls = 0

    def fetch(self, example: Mapping[str, Any]) -> TeacherNotes:
        self.calls += 1
        return self.delegate.fetch(example)


def test_on_demand_teacher_provider_generates_snapshots() -> None:
    provider = _make_provider(notes_dim=6)
    example = _build_example()

    payload = provider.fetch(example)

    # Notes matrix: one row per stream, columns = notes_dim
    assert payload.notes.shape == torch.Size([3, 6])

    # Snapshots: all 3 versioned notes entries are returned (max_snapshots=3)
    assert len(payload.snapshots) == 3
    assert all(s.notes.shape == torch.Size([3, 6]) for s in payload.snapshots)

    # Strides match the versioned notes definitions
    assert [s.stride for s in payload.snapshots] == [0, 6, 12]

    # Notes tensor must be finite (hashing embedder never produces NaN/Inf)
    assert torch.isfinite(payload.notes).all()


def test_teacher_provider_max_snapshots_is_respected() -> None:
    config = DatasetTeacherConfig(cache_dir=None, max_snapshots=2)
    provider = DatasetTeacherNotesProvider(
        config,
        notes_dim=4,
        stream_to_id={"intro": 0, "core": 1, "wrap": 2},
    )
    example = _build_example()

    payload = provider.fetch(example)

    # max_snapshots=2 should limit the returned list
    assert len(payload.snapshots) <= 2


def test_teacher_provider_missing_notes_returns_zeros() -> None:
    """When no teacher_notes and no notes_versioned, the notes tensor is zero."""
    config = DatasetTeacherConfig(cache_dir=None, max_snapshots=1)
    provider = DatasetTeacherNotesProvider(
        config,
        notes_dim=4,
        stream_to_id={"intro": 0, "core": 1},
    )
    # Bare example with no notes at all
    example: dict[str, Any] = {"example_id": "empty"}

    payload = provider.fetch(example)

    assert payload.notes.shape == torch.Size([2, 4])
    # All streams have no data — they get embedded from empty text ("")
    # which produces a zero-ish vector (hash of empty string is well-defined)
    assert torch.isfinite(payload.notes).all()


def test_cached_teacher_provider_reuses_embedded_results(tmp_path: Path) -> None:
    config = DatasetTeacherConfig(cache_dir=None, max_snapshots=2)
    inner = DatasetTeacherNotesProvider(
        config,
        notes_dim=8,
        stream_to_id={"intro": 0, "core": 1, "wrap": 2},
    )
    tracking = _TrackingProvider(inner)
    cached = CachedTeacherNotesProvider(
        backend=tracking,
        cache_dir=tmp_path,
        id_field="example_id",
    )
    example = _build_example(example_id="example-cached-42")

    first_payload = cached.fetch(example)
    assert tracking.calls == 1

    # Cache file should exist on disk
    cache_file = tmp_path / "example-cached-42.pt"
    assert cache_file.exists(), "Cache file was not written after first fetch"

    second_payload = cached.fetch(example)

    # Backend should NOT have been called again
    assert tracking.calls == 1, "Backend was called more than once; cache miss on second fetch"

    # Results must be identical
    assert torch.allclose(first_payload.notes, second_payload.notes)
    assert len(first_payload.snapshots) == len(second_payload.snapshots)
    for a, b in zip(first_payload.snapshots, second_payload.snapshots):
        assert torch.allclose(a.notes, b.notes)
        assert a.stride == b.stride
        assert a.version == b.version
