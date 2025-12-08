"""Lightweight dataclasses describing teacher runner outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class DatasetTeacherConfig:
    """Configuration for extracting pre-generated teacher notes from dataset.

    Teacher notes must be generated during dataset pipeline (Stage 3: Notes Generation).
    This config controls how those pre-generated notes are read and cached during training.
    """

    cache_dir: str | None = None
    max_snapshots: int = 3
    id_field: str = "example_id"
    refresh_cache: bool = False


@dataclass(slots=True)
class TeacherSnapshotText:
    """Textual snapshot emitted by a teacher runner invocation."""

    version: int
    stride: int
    stream_notes: Mapping[str, Sequence[str]]
    coverage: Mapping[str, float] | None = None
    source: str = "teacher"


@dataclass(slots=True)
class TeacherRunResult:
    """Structured payload returned by a teacher runner."""

    example_id: str
    stream_notes: Mapping[str, Sequence[str]]
    snapshots: Sequence[TeacherSnapshotText] = field(default_factory=tuple)
    teacher_plan: Mapping[str, Any] | None = None


def normalize_stream_notes(payload: Mapping[str, Sequence[str]] | None) -> dict[str, list[str]]:
    """Return a mutable copy with deterministic stream IDs."""

    if not payload:
        return {}
    normalized: MutableMapping[str, list[str]] = {}
    for stream, notes in payload.items():
        key = normalize_stream_id(stream)
        normalized[key] = [str(note) for note in notes]
    return dict(normalized)


def normalize_stream_id(value: str) -> str:
    stream = str(value or "").strip().lower()
    if not stream:
        return "stream_unknown"
    if not stream.startswith("stream_"):
        stream = f"stream_{stream}"
    return stream


__all__ = [
    "DatasetTeacherConfig",
    "TeacherRunResult",
    "TeacherSnapshotText",
    "normalize_stream_id",
    "normalize_stream_notes",
]
