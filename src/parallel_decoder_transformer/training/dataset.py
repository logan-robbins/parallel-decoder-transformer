"""JSONL dataset feeding the knowledge distillation collator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..data.extraction import CoverageStatus, StreamNotes, load_stream_notes
from ..data.snapshots import SnapshotFeatures

COVERAGE_STATUS_TO_SCORE = {
    CoverageStatus.COVERED: 1.0,
    CoverageStatus.PARTIAL: 0.5,
    CoverageStatus.MISSING: 0.0,
}


@dataclass(slots=True)
class KDRecord:
    """Decoded JSONL record ready for collation."""

    student_ids: torch.Tensor
    student_labels: torch.Tensor
    planner_ids: torch.Tensor
    notes_student: torch.Tensor
    notes_teacher: Optional[torch.Tensor]
    stream_id: str
    teacher_snapshots: List[SnapshotFeatures] = field(default_factory=list)
    student_snapshots: List[SnapshotFeatures] = field(default_factory=list)
    stride_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    commit_points: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    agreement_labels: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    metadata: Dict[str, Any] = field(default_factory=dict)
    example_id: Optional[str] = None
    plan_items: List[str] = field(default_factory=list)
    plan_catalog: List[str] = field(default_factory=list)
    plan_catalog_streams: List[str] = field(default_factory=list)
    coverage_targets: List[float] = field(default_factory=list)
    coverage_supervision_mask: List[bool] = field(default_factory=list)
    notes_text: str = ""
    raw_teacher_notes: Dict[str, List[str]] = field(default_factory=dict)


class KDJsonlDataset(Dataset[Dict[str, object]]):
    """Loads KD training examples from a JSONL manifest.

    Each line must encode keys: ``student_ids``, ``student_labels``, ``planner_ids``,
    ``notes_student``, ``notes_teacher``, and ``stream_id``. Additional optional keys are
    supported to express per-stride snapshots, agreement labels, and metadata.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.exists():
            raise FileNotFoundError(f"KDJsonlDataset expects dataset at {path} to exist.")

        # Lazy loading: Build an index of line offsets
        self._offsets: List[int] = [0]
        with path.open("rb") as f:
            while f.readline():
                self._offsets.append(f.tell())
        # The last offset points to EOF, remove it
        self._offsets.pop()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._offsets)

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        offset = self._offsets[index]
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
        record = self._decode(line)
        return {
            "student_ids": record.student_ids,
            "student_labels": record.student_labels,
            "planner_ids": record.planner_ids,
            "notes_student": record.notes_student,
            "notes_teacher": record.notes_teacher,
            "stream_id": record.stream_id,
            "teacher_snapshots": record.teacher_snapshots,
            "student_snapshots": record.student_snapshots,
            "stride_ids": record.stride_ids,
            "commit_points": record.commit_points,
            "agreement_labels": record.agreement_labels,
            "metadata": record.metadata,
            "example_id": record.example_id,
            "plan_items": record.plan_items,
            "plan_catalog": record.plan_catalog,
            "plan_catalog_streams": record.plan_catalog_streams,
            "coverage_targets": record.coverage_targets,
            "coverage_supervision_mask": record.coverage_supervision_mask,
            "notes_text": record.notes_text,
            "raw_teacher_notes": record.raw_teacher_notes,
            "sectional_independence": record.metadata.get("sectional_independence", False),
        }

    def _decode(self, payload: str) -> KDRecord:
        data = json.loads(payload)
        student_ids = self._tensor(data["student_ids"], torch.long)
        student_labels = self._tensor(data.get("student_labels", data["student_ids"]), torch.long)
        planner_ids = self._tensor(data["planner_ids"], torch.long)
        notes_student = self._tensor(data["notes_student"], torch.float32)
        teacher_notes_raw = data.get("notes_teacher")
        notes_teacher = (
            self._tensor(teacher_notes_raw, torch.float32)
            if teacher_notes_raw is not None
            else None
        )
        metadata = dict(data.get("metadata", {}))
        if "document_text" not in metadata or "document_paragraphs" not in metadata:
            raise KeyError(
                "KDJsonlDataset metadata must include 'document_text' and 'document_paragraphs'."
            )
        metadata["document_text"] = str(metadata["document_text"])
        metadata["document_paragraphs"] = [
            str(paragraph) for paragraph in metadata.get("document_paragraphs", [])
        ]
        if "sectional_independence" not in metadata and "sectional_independence" in data:
            metadata["sectional_independence"] = bool(data.get("sectional_independence"))
        if "notes_versioned" not in metadata and data.get("notes_versioned") is not None:
            metadata["notes_versioned"] = data.get("notes_versioned")
        teacher_notes_text = {
            self._normalize_stream_id(stream_id).lower(): [str(note) for note in notes]
            for stream_id, notes in (metadata.get("teacher_notes") or {}).items()
        }
        stream_notes_map = self._load_stream_notes_map(data.get("true_notes"))
        teacher_snapshots = self._build_snapshots(
            self._snapshot_payload(data, "teacher"), notes_teacher, source="teacher"
        )
        student_snapshots = self._build_snapshots(
            self._snapshot_payload(data, "student"), notes_student, source="student"
        )
        stride_ids = self._tensor(data.get("stride_ids", [0]), torch.long)
        commit_points = self._tensor(data.get("commit_points", []), torch.long)
        agreement_labels = self._tensor(data.get("agreement_labels", []), torch.long)
        example_id = self._resolve_example_id(data, metadata)
        stream_id_value = data.get("stream_id") or data.get("stream") or "stream_unknown"
        stream_id = self._normalize_stream_id(stream_id_value)
        (
            plan_items,
            coverage_targets,
            notes_text,
            plan_catalog,
            plan_catalog_streams,
        ) = self._extract_plan_items(
            stream_id=stream_id,
            plan_tokens=data.get("plan_tokens", []),
            notes_tokens=data.get("notes_tokens", []),
            metadata=metadata,
            stream_notes=stream_notes_map,
        )
        coverage_supervision_mask = self._coverage_supervision_mask(metadata, plan_catalog)
        return KDRecord(
            student_ids=student_ids,
            student_labels=student_labels,
            planner_ids=planner_ids,
            notes_student=notes_student,
            notes_teacher=notes_teacher,
            stream_id=stream_id,
            teacher_snapshots=teacher_snapshots,
            student_snapshots=student_snapshots,
            stride_ids=stride_ids,
            commit_points=commit_points,
            agreement_labels=agreement_labels,
            metadata=metadata,
            example_id=example_id,
            plan_items=plan_items,
            plan_catalog=plan_catalog,
            plan_catalog_streams=plan_catalog_streams,
            coverage_targets=coverage_targets,
            coverage_supervision_mask=coverage_supervision_mask,
            notes_text=notes_text,
            raw_teacher_notes=teacher_notes_text,
        )

    def _tensor(self, values: Sequence, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=dtype)
        tensor = torch.tensor(values, dtype=dtype)
        return tensor

    def _build_snapshots(
        self,
        payload: Sequence[Mapping[str, Any]],
        default_notes: Optional[torch.Tensor],
        *,
        source: str,
    ) -> List[SnapshotFeatures]:
        if not payload:
            if default_notes is None:
                return []
            return [SnapshotFeatures(notes=default_notes.clone(), source=source)]
        snapshots: List[SnapshotFeatures] = []
        for index, item in enumerate(payload):
            if "notes" in item:
                notes = self._tensor(item["notes"], torch.float32)
            elif default_notes is not None:
                notes = default_notes.clone()
            else:
                raise KeyError("Snapshot payload missing notes and no default provided.")
            stride = int(item.get("stride", item.get("stride_id", 0)))
            version = int(item.get("version", item.get("snapshot_id", index)))
            stream_label = item.get("stream_id") or item.get("stream")
            coverage = item.get("coverage_flags")
            coverage_tensor = None
            if coverage is not None:
                coverage_tensor = self._tensor(coverage, torch.float32)
            snapshots.append(
                SnapshotFeatures(
                    notes=notes,
                    stride=stride,
                    version=version,
                    stream_id=str(stream_label).strip().lower() if stream_label else None,
                    coverage=coverage_tensor,
                    source=source,
                )
            )
        return snapshots

    def _snapshot_payload(
        self, data: Mapping[str, Any], prefix: str
    ) -> Sequence[Mapping[str, Any]]:
        if prefix == "teacher":
            return data.get("teacher_snapshots") or data.get("snapshots") or []
        if prefix == "student":
            return data.get("student_snapshots") or data.get("speculative_snapshots") or []
        return []

    def _resolve_example_id(
        self, data: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> Optional[str]:
        candidate = data.get("example_id") or data.get("id") or metadata.get("id")
        if candidate is None:
            return None
        return str(candidate)

    def _extract_plan_items(
        self,
        *,
        stream_id: str,
        plan_tokens: Sequence[str],
        notes_tokens: Sequence[str],
        metadata: Mapping[str, Any],
        stream_notes: Mapping[str, StreamNotes],
    ) -> tuple[List[str], List[float], str, List[str], List[str]]:
        stream_lower = stream_id.lower()
        teacher_plan = metadata.get("teacher_plan", {}) or {}
        teacher_notes = metadata.get("teacher_notes", {}) or {}
        notes_text = " ".join(str(token) for token in notes_tokens)

        catalog: List[tuple[str, str]] = []

        def _append_item(entry_stream: str, text: Any) -> None:
            item_text = str(text).strip()
            if not item_text:
                return
            catalog.append((self._normalize_stream_id(entry_stream).lower(), item_text))

        for entry in teacher_plan.get("plan", []):
            entry_stream = str(entry.get("stream_id") or entry.get("stream") or "").strip().lower()
            if not entry_stream:
                continue
            for note in entry.get("notes_contract") or entry.get("notes") or []:
                _append_item(entry_stream, note)
            summary = entry.get("summary")
            if summary is not None:
                _append_item(entry_stream, summary)

        if not catalog and plan_tokens:
            for token in plan_tokens:
                if not isinstance(token, str):
                    continue
                token = token.strip()
                if not token:
                    continue
                if "::" in token:
                    prefix, body = token.split("::", 1)
                    entry_stream = prefix.strip().lower() or stream_lower
                    item_text = body
                else:
                    entry_stream = stream_lower
                    item_text = token
                _append_item(entry_stream, item_text)

        if teacher_notes:
            stream_notes_text = teacher_notes.get(stream_lower) or teacher_notes.get(stream_id)
            if stream_notes_text:
                notes_text = " ".join(str(note) for note in stream_notes_text)
        if not notes_text:
            normalized_notes = stream_notes.get(stream_lower)
            if normalized_notes:
                notes_text = self._notes_to_text(normalized_notes)

        plan_catalog_streams = [stream_name for stream_name, _ in catalog]
        plan_catalog = [text for _, text in catalog]
        plan_items_for_stream = [
            text for stream_name, text in catalog if stream_name == stream_lower
        ]
        coverage_targets = self._coverage_targets_from_catalog(catalog, stream_notes)

        if not plan_catalog and plan_tokens:
            plan_items_for_stream = []
            coverage_targets = []

        return (
            plan_items_for_stream,
            coverage_targets,
            notes_text,
            plan_catalog,
            plan_catalog_streams,
        )

    def _coverage_supervision_mask(
        self,
        metadata: Mapping[str, Any],
        plan_catalog: Sequence[str],
    ) -> List[bool]:
        if not plan_catalog:
            return []
        provenance = metadata.get("coverage_provenance") if isinstance(metadata, Mapping) else None
        confirmed_items: set[str] = set()
        default_confirmed = True
        if isinstance(provenance, Mapping):
            method = str(provenance.get("method", "")).strip().lower()
            strength = str(provenance.get("strength", "confirmed")).strip().lower()

            # Relaxed check for production training on distilled notes
            if method == "notes_json" and strength == "hint":
                default_confirmed = True
            else:
                confirmed_flag = provenance.get("confirmed")
                if isinstance(confirmed_flag, bool):
                    default_confirmed = confirmed_flag
                else:
                    default_confirmed = strength not in {"hint", "weak"} and method not in {
                        "text_hint",
                        "notes_text",
                    }
            raw_confirmed = provenance.get("confirmed_plan_items")
            if isinstance(raw_confirmed, Mapping):
                for values in raw_confirmed.values():
                    if not isinstance(values, Sequence):
                        continue
                    for item in values:
                        normalized = str(item).strip().lower()
                        if normalized:
                            confirmed_items.add(normalized)
            elif isinstance(raw_confirmed, Sequence):
                for item in raw_confirmed:
                    normalized = str(item).strip().lower()
                    if normalized:
                        confirmed_items.add(normalized)
        mask: List[bool] = []
        for item in plan_catalog:
            normalized = str(item).strip().lower()
            if confirmed_items:
                mask.append(normalized in confirmed_items)
            else:
                mask.append(default_confirmed)
        return mask

    def _load_stream_notes_map(self, payload: Any) -> Dict[str, StreamNotes]:
        notes_map: Dict[str, StreamNotes] = {}
        if not isinstance(payload, Sequence):
            return notes_map
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            try:
                notes = load_stream_notes(item)
            except (ValueError, TypeError):
                continue
            stream_id = (notes.stream_id or str(item.get("stream_id", ""))).strip()
            if not stream_id:
                continue
            notes.stream_id = stream_id
            notes_map[stream_id.lower()] = notes
        return notes_map

    def _coverage_targets_from_catalog(
        self,
        catalog: Sequence[tuple[str, str]],
        stream_notes: Mapping[str, StreamNotes],
    ) -> List[float]:
        if not catalog:
            return []
        lookup: Dict[str, Dict[str, float]] = {
            stream_id: self._coverage_scores(notes) for stream_id, notes in stream_notes.items()
        }
        scores: List[float] = []
        for stream_name, item_text in catalog:
            stream_key = stream_name.lower()
            stream_scores = lookup.get(stream_key)
            scores.append(self._score_for_plan_item(stream_scores, item_text))
        return scores

    def _coverage_scores(self, notes: StreamNotes) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for signal in notes.coverage:
            plan_item = signal.plan_item_id.strip()
            # Remove "COVER::" prefix
            if plan_item.startswith("COVER::"):
                plan_item = plan_item[len("COVER::") :].strip()
            # Remove status suffix (e.g., "::covered") if present
            if "::" in plan_item:
                base_plan_item, _ = plan_item.rsplit("::", 1)
                plan_item = base_plan_item.strip()
            if not plan_item:
                continue
            scores[plan_item] = COVERAGE_STATUS_TO_SCORE.get(signal.status, 0.0)
        return scores

    def _score_for_plan_item(
        self,
        score_map: Optional[Mapping[str, float]],
        plan_item: str,
    ) -> float:
        if not score_map:
            return 0.0
        if plan_item in score_map:
            return float(score_map[plan_item])
        normalized = plan_item.strip().lower()
        if not normalized:
            return 0.0
        for key, value in score_map.items():
            if key.strip().lower() == normalized:
                return float(value)
        return 0.0

    def _normalize_stream_id(self, value: Any) -> str:
        stream_id = str(value or "").strip()
        if not stream_id:
            stream_id = "stream_unknown"
        if not stream_id.startswith("stream_"):
            stream_id = f"stream_{stream_id}"
        return stream_id

    def _notes_to_text(self, notes: StreamNotes) -> str:
        parts: List[str] = []
        for entity in notes.entities:
            if entity.name:
                parts.append(entity.name)
            parts.extend(alias for alias in entity.aliases if alias)
        for fact in notes.facts:
            snippet = " ".join(part for part in (fact.subj_id, fact.predicate, fact.object) if part)
            if snippet:
                parts.append(snippet)
            if fact.evidence_span.text:
                parts.append(fact.evidence_span.text)
        for coverage in notes.coverage:
            parts.append(f"{coverage.plan_item_id}:{coverage.status.value}")
        return " ".join(parts)

    def _iter_lines(self, path: Path) -> Iterator[str]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped


__all__ = ["KDJsonlDataset", "KDRecord"]
