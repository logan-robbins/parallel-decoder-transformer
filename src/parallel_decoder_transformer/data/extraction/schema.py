"""Normalized schema for ENT/FACT/COVERAGE teacher notes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping

NOTES_SCHEMA_VERSION = "2.0"


class CoverageStatus(str, Enum):
    """Categorical coverage supervision values."""

    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(slots=True)
class EvidenceSpan:
    """Grounded character span used to justify a fact."""

    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span offsets: start={self.start}, end={self.end}")
        self.text = self.text.strip()


@dataclass(slots=True)
class EntityCard:
    """Normalized entity payload shared across streams."""

    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    type: str = "entity"
    canonical: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": list(dict.fromkeys(self.aliases)),
            "type": self.type,
            "canonical": bool(self.canonical),
        }


@dataclass(slots=True)
class FactStatement:
    """FACT tuples link back to canonical entities via subj_id."""

    subj_id: str
    predicate: str
    object: str
    evidence_span: EvidenceSpan
    certainty: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.certainty <= 1.0:
            raise ValueError(f"certainty must be within [0, 1], got {self.certainty}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subj_id": self.subj_id,
            "predicate": self.predicate,
            "object": self.object,
            "evidence_span": {
                "start": self.evidence_span.start,
                "end": self.evidence_span.end,
                "text": self.evidence_span.text,
            },
            "certainty": self.certainty,
        }


@dataclass(slots=True)
class CoverageSignal:
    """Coverage supervision for a plan item."""

    plan_item_id: str
    status: CoverageStatus

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_item_id": self.plan_item_id,
            "status": self.status.value,
        }


@dataclass(slots=True)
class StreamNotes:
    """Normalized note payload scoped to a stream."""

    stream_id: str
    entities: list[EntityCard] = field(default_factory=list)
    facts: list[FactStatement] = field(default_factory=list)
    coverage: list[CoverageSignal] = field(default_factory=list)

    def __post_init__(self) -> None:
        stream_id = self.stream_id.strip()
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        if not stream_id.startswith("stream_"):
            raise ValueError("stream_id must start with 'stream_'")
        self.stream_id = stream_id

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "ENT": [entity.as_dict() for entity in self.entities],
            "FACT": [fact.as_dict() for fact in self.facts],
            "COVERAGE": [cov.as_dict() for cov in self.coverage],
        }


@dataclass(slots=True)
class ExtractedNoteSet:
    """Complete teacher note payload for a document."""

    document_id: str
    notes: Dict[str, StreamNotes]
    schema_version: str = NOTES_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def stream(self, id: str) -> StreamNotes:
        return self.notes.get(id) or StreamNotes(stream_id=id)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "schema_version": self.schema_version,
            "notes": {stream_id: payload.as_dict() for stream_id, payload in self.notes.items()},
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ExtractionResult:
    """Result wrapper for teacher note extraction."""

    success: bool
    note_set: ExtractedNoteSet | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)


def load_stream_notes(raw: Mapping[str, Any]) -> StreamNotes:
    """Utility used by ingest paths to hydrate StreamNotes from dictionaries."""

    def _coerce_entities(items: Any) -> list[EntityCard]:
        entities: list[EntityCard] = []
        if not isinstance(items, list):
            return entities
        for item in items:
            if not isinstance(item, Mapping):
                continue
            entity_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            if not entity_id or not name:
                continue
            aliases = [
                str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()
            ]
            entities.append(
                EntityCard(
                    id=entity_id,
                    name=name,
                    aliases=aliases,
                    type=str(item.get("type", "entity")).strip() or "entity",
                    canonical=bool(item.get("canonical", True)),
                )
            )
        return entities

    def _coerce_facts(items: Any) -> list[FactStatement]:
        facts: list[FactStatement] = []
        if not isinstance(items, list):
            return facts
        for item in items:
            if not isinstance(item, Mapping):
                continue
            subj_id = str(item.get("subj_id", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not subj_id or not predicate or not obj:
                continue
            evidence_payload = item.get("evidence_span") or {}
            try:
                start = int(evidence_payload.get("start", 0))
                end = int(evidence_payload.get("end", start))
            except (TypeError, ValueError):
                start = 0
                end = 0
            end = max(end, start)
            span_text = str(evidence_payload.get("text", "")).strip()
            try:
                span = EvidenceSpan(start=start, end=end, text=span_text)
            except ValueError:
                span = EvidenceSpan(start=0, end=0, text=span_text)
            certainty_value = float(item.get("certainty", 1.0))
            certainty = (
                0.0 if certainty_value < 0.0 else 1.0 if certainty_value > 1.0 else certainty_value
            )
            facts.append(
                FactStatement(
                    subj_id=subj_id,
                    predicate=predicate,
                    object=obj,
                    evidence_span=span,
                    certainty=certainty,
                )
            )
        return facts

    def _coerce_coverage(items: Any) -> list[CoverageSignal]:
        coverage: list[CoverageSignal] = []
        if not isinstance(items, list):
            return coverage
        for item in items:
            if not isinstance(item, Mapping):
                continue
            plan_item_id = str(item.get("plan_item_id", "")).strip()
            if not plan_item_id:
                continue
            status_value = str(item.get("status", CoverageStatus.MISSING.value)).lower()
            try:
                status = CoverageStatus(status_value)
            except ValueError:
                status = CoverageStatus.MISSING
            coverage.append(
                CoverageSignal(
                    plan_item_id=plan_item_id,
                    status=status,
                )
            )
        return coverage

    stream_id_value = str(raw.get("stream_id") or raw.get("stream") or "").strip() or "unknown"
    stream_id = (
        stream_id_value if stream_id_value.startswith("stream_") else f"stream_{stream_id_value}"
    )
    return StreamNotes(
        stream_id=stream_id,
        entities=_coerce_entities(raw.get("ENT")),
        facts=_coerce_facts(raw.get("FACT")),
        coverage=_coerce_coverage(raw.get("COVERAGE")),
    )


__all__ = [
    "CoverageSignal",
    "CoverageStatus",
    "EntityCard",
    "EvidenceSpan",
    "ExtractedNoteSet",
    "ExtractionResult",
    "FactStatement",
    "StreamNotes",
    "NOTES_SCHEMA_VERSION",
    "load_stream_notes",
]
