"""Teacher notes providers used to seed KD collators."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from .snapshots import SnapshotFeatures
from .teacher_runner import (
    DatasetTeacherConfig,
    TeacherRunResult,
    TeacherSnapshotText,
    normalize_stream_notes,
    normalize_stream_id,
)


@dataclass(slots=True)
class TeacherNotes:
    """Tensor payload plus auxiliary metadata for teacher supervision."""

    notes: torch.Tensor
    snapshots: list[SnapshotFeatures]
    raw_notes: Mapping[str, Sequence[str]]


class TeacherNotesProviderBase:
    """Interface implemented by KD trainers to fetch teacher notes."""

    def fetch(self, example: Mapping[str, object]) -> TeacherNotes:  # pragma: no cover - interface
        raise NotImplementedError


class DatasetTeacherNotesProvider(TeacherNotesProviderBase):
    """Reads pre-generated teacher notes from dataset metadata.

    This provider extracts teacher notes that were already generated
    during the dataset pipeline (Stage 3: Notes Generation). It does
    NOT call any LLM APIs - all notes must be present in the dataset.
    """

    def __init__(
        self,
        config: DatasetTeacherConfig,
        *,
        notes_dim: int,
        stream_to_id: Mapping[str, int],
    ) -> None:
        self.config = config
        self.notes_dim = int(notes_dim)
        self.stream_to_id = dict(stream_to_id)
        self._embedder = _HashingEmbedder()
        self._runner = _DatasetMetadataRunner(config, self.stream_to_id)

    def fetch(self, example: Mapping[str, object]) -> TeacherNotes:
        result = self._runner.run(example)
        plan_snapshots = _snapshots_from_example(example)
        if any(snapshot.source == "plan_contract" for snapshot in result.snapshots):
            plan_snapshots = []
        snapshot_texts = list(plan_snapshots) + list(result.snapshots)
        if not snapshot_texts:
            snapshot_texts = [
                TeacherSnapshotText(
                    version=0,
                    stride=0,
                    stream_notes=result.stream_notes,
                    coverage=None,
                    source="teacher",
                )
            ]
        raw_notes = normalize_stream_notes(result.stream_notes)
        notes_tensor = self._embed_stream_map(raw_notes)
        snapshots = [
            self._snapshot_to_features(snapshot)
            for snapshot in snapshot_texts[
                : max(1, self.config.max_snapshots or len(snapshot_texts))
            ]
        ]
        return TeacherNotes(notes=notes_tensor, snapshots=snapshots, raw_notes=raw_notes)

    def _embed_stream_map(self, stream_map: Mapping[str, Sequence[str]]) -> torch.Tensor:
        stream_count = len(self.stream_to_id)
        notes = torch.zeros(stream_count, self.notes_dim, dtype=torch.float32)
        normalized_map: MutableMapping[str, Sequence[str]] = {}
        for key, value in stream_map.items():
            normalized_map[normalize_stream_id(key)] = value
        for stream, index in self.stream_to_id.items():
            key = normalize_stream_id(stream)
            texts = normalized_map.get(key)
            if texts is None and key.startswith("stream_"):
                texts = normalized_map.get(key.split("stream_", 1)[-1])
            encoded = self._embed_texts(texts)
            notes[index] = encoded
        return notes

    def _snapshot_to_features(self, snapshot: TeacherSnapshotText) -> SnapshotFeatures:
        stream_map = normalize_stream_notes(snapshot.stream_notes)
        notes = self._embed_stream_map(stream_map)
        coverage_tensor = None
        if snapshot.coverage:
            coverage_tensor = torch.zeros(len(self.stream_to_id), dtype=torch.float32)
            for stream, value in snapshot.coverage.items():
                idx = self.stream_to_id.get(stream, self.stream_to_id.get(f"stream_{stream}"))
                if idx is None:
                    continue
                coverage_tensor[idx] = float(value)
        return SnapshotFeatures(
            notes=notes,
            stride=snapshot.stride,
            version=snapshot.version,
            coverage=coverage_tensor,
            source=snapshot.source,
        )

    def _embed_texts(self, texts: Sequence[str] | None) -> torch.Tensor:
        payload = [str(text) for text in texts or [] if str(text).strip()]
        if not payload:
            payload = [""]
        vector = self._embedder.aggregate(payload, self.notes_dim)
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        return vector.to(dtype=torch.float32)


class CachedTeacherNotesProvider(TeacherNotesProviderBase):
    """Wraps a provider with simple on-disk caching."""

    def __init__(
        self,
        *,
        backend: TeacherNotesProviderBase,
        cache_dir: Path | None,
        id_field: str = "example_id",
        refresh: bool = False,
    ) -> None:
        self.backend = backend
        self.cache_dir = cache_dir
        self.id_field = id_field
        self.refresh = refresh

    def fetch(self, example: Mapping[str, object]) -> TeacherNotes:
        example_id = str(
            example.get(self.id_field)
            or example.get("sample_id")
            or example.get("example_id")
            or ""
        ).strip()
        cache_path: Path | None = None
        if self.cache_dir and example_id:
            cache_path = Path(self.cache_dir) / f"{example_id}.pt"

        # Check if cache exists before any expensive operations
        if cache_path and cache_path.exists() and not self.refresh:
            # Explicitly allow pickled TeacherNotes objects (PyTorch 2.6 defaults weights_only=True)
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        # Generate teacher payload (expensive operation)
        payload = self.backend.fetch(example)

        # DDP-safe cache writing: Skip caching in DataLoader workers only
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()

        if cache_path and worker_info is None:
            # Atomic write: Only write if file doesn't exist (prevents corruption)
            # Multiple ranks may race, but only first succeeds
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Try to create file exclusively (fails if exists)
            temp_path = cache_path.with_suffix(".tmp")
            try:
                # Write to temp file first
                torch.save(payload, temp_path)
                # Atomic move - only succeeds if target doesn't exist
                # On Unix, rename is atomic
                temp_path.rename(cache_path)
            except (OSError, FileExistsError):
                # Another rank won the race or file already exists - clean up temp
                temp_path.unlink(missing_ok=True)

        return payload


class _HashingEmbedder:
    """Deterministic embedder backed by SHA-256 hashes."""

    def aggregate(self, texts: Sequence[str], target_dim: int) -> torch.Tensor:
        vector = torch.zeros(target_dim, dtype=torch.float32)
        for offset, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index, byte in enumerate(digest):
                position = (index + offset) % target_dim
                vector[position] += 1.0 if byte % 2 == 0 else -1.0
        norm = torch.linalg.vector_norm(vector)
        if norm > 0:
            return vector / norm
        return vector


class _DatasetMetadataRunner:
    """Extracts teacher notes from pre-generated dataset metadata."""

    def __init__(self, config: DatasetTeacherConfig, stream_to_id: Mapping[str, int]) -> None:
        self.config = config
        self.stream_to_id = dict(stream_to_id)

    def run(self, example: Mapping[str, object]) -> TeacherRunResult:
        metadata = example.get("metadata") if isinstance(example, Mapping) else None
        metadata = metadata or {}
        teacher_notes = metadata.get("teacher_notes") or example.get("teacher_notes") or {}
        plan = metadata.get("teacher_plan") or example.get("teacher_plan")
        example_id = str(
            example.get(self.config.id_field)
            or metadata.get(self.config.id_field)
            or example.get("sample_id")
            or "unknown"
        )
        if not teacher_notes:
            plan_snapshots = _snapshots_from_example(example)
            if plan_snapshots:
                teacher_notes = plan_snapshots[0].stream_notes
        return TeacherRunResult(
            example_id=example_id,
            stream_notes=teacher_notes,
            snapshots=tuple(),
            teacher_plan=plan,
        )


def _snapshots_from_example(example: Mapping[str, object]) -> list[TeacherSnapshotText]:
    payload = example.get("notes_versioned") or example.get("versioned_notes")
    if payload is None:
        metadata = example.get("metadata") if isinstance(example, Mapping) else None
        if isinstance(metadata, Mapping):
            payload = metadata.get("notes_versioned")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, Sequence):
        return []
    snapshots: list[TeacherSnapshotText] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        notes_block = entry.get("notes")
        if not isinstance(notes_block, Sequence):
            continue
        stream_map: MutableMapping[str, list[str]] = {}
        for stream in notes_block:
            if not isinstance(stream, Mapping):
                continue
            stream_id = normalize_stream_id(stream.get("stream_id") or stream.get("stream") or "")
            if not stream_id:
                continue
            stream_map[stream_id] = _stringify_stream_notes(stream)
        if not stream_map:
            continue
        snapshot_id = int(entry.get("snapshot_id", len(snapshots)))
        stride = int(entry.get("stride", 0))
        source = str(entry.get("source", "plan_contract"))
        snapshots.append(
            TeacherSnapshotText(
                version=snapshot_id,
                stride=stride,
                stream_notes=stream_map,
                coverage=None,
                source=source,
            )
        )
    return snapshots


def _stringify_stream_notes(stream: Mapping[str, Any]) -> list[str]:
    parts: list[str] = []
    for entity in stream.get("ENT", []) or []:
        if not isinstance(entity, Mapping):
            continue
        name = entity.get("name") or entity.get("id")
        entity_type = entity.get("type") or "entity"
        parts.append(f"ENT::{name}::{entity_type}")
    for fact in stream.get("FACT", []) or []:
        if not isinstance(fact, Mapping):
            continue
        subj = fact.get("subj_id") or fact.get("subject")
        predicate = fact.get("predicate") or "relates_to"
        obj = fact.get("object") or fact.get("object_id")
        parts.append(f"FACT::{subj}::{predicate}::{obj}")
        span = fact.get("evidence_span")
        if isinstance(span, Mapping) and span.get("text"):
            parts.append(str(span.get("text")))
    for coverage in stream.get("COVERAGE", []) or []:
        if not isinstance(coverage, Mapping):
            continue
        plan_item = coverage.get("plan_item_id")
        status = coverage.get("status")
        parts.append(f"COVER::{plan_item}::{status}")
    summary = stream.get("summary")
    if summary:
        parts.append(str(summary))
    return [part for part in parts if part]


__all__ = [
    "CachedTeacherNotesProvider",
    "DatasetTeacherNotesProvider",
    "TeacherNotes",
    "TeacherNotesProviderBase",
]
