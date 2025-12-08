"""Attribute consistency metric utilities."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from ..data.extraction import StreamNotes

_TOKEN_NORMALISER = re.compile(r"[^a-z0-9]+")
_SEGMENT_SPLIT = re.compile(r"[;\n•·\*\u2022\u2023\u25AA\u25CF]+")
_THREE_FIELD_PATTERN = re.compile(
    r"^\s*(?P<subject>[^:;,\-]+?)\s*[-:]\s*(?P<attribute>[^:;,\-]+?)\s*[-:=]\s*(?P<value>.+?)\s*$"
)
_POSSESSIVE_PATTERN = re.compile(
    r"^\s*(?P<subject>[^:;,\-]+?)'s\s+(?P<attribute>[A-Za-z0-9 _/]+?)\s*(?:is|are|=)\s*(?P<value>.+?)\s*$"
)
_VERB_PATTERN = re.compile(
    r"^\s*(?P<subject>[^:;,\-]+?)\s+(?P<attribute>has|have|includes|contains|features)\s+(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the",
    "and",
    "plan",
    "section",
    "stream",
    "part",
    "focus",
    "topic",
    "notes",
}


@dataclass(frozen=True)
class AttributeTuple:
    """Single mined attribute tuple."""

    stream_id: str
    subject: str
    attribute: str
    value: str
    time_index: int
    source: str = "text"

    def subject_norm(self) -> str:
        return _normalize_token(self.subject)

    def attribute_norm(self) -> str:
        return _normalize_token(self.attribute)

    def value_norm(self) -> str:
        return _normalize_token(self.value)


@dataclass(frozen=True)
class AttributeConsistencyResult:
    """Structured payload describing attribute consistency statistics."""

    cross_stream_total: int
    cross_stream_consistent: int
    per_stream_totals: Dict[str, int]
    per_stream_violations: Dict[str, int]
    time_totals: Dict[str, int]
    time_violations: Dict[str, int]
    total_tuples: int
    source_counts: Dict[str, int]

    def to_payload(self) -> Dict[str, Any]:
        """Render the result into a serialisable dictionary."""

        def _rates(
            totals: Mapping[str, int],
            violations: Mapping[str, int],
        ) -> Dict[str, Dict[str, float | int | None]]:
            payload: Dict[str, Dict[str, float | int | None]] = {}
            for stream_id, total in totals.items():
                violation = violations.get(stream_id, 0)
                rate = None if total == 0 else (total - violation) / float(total)
                payload[stream_id] = {
                    "total": total,
                    "violations": violation,
                    "consistency_rate": rate,
                }
            return payload

        cross_rate = (
            None
            if self.cross_stream_total == 0
            else self.cross_stream_consistent / float(self.cross_stream_total)
        )
        return {
            "cross_stream": {
                "total": self.cross_stream_total,
                "consistent": self.cross_stream_consistent,
                "violations": self.cross_stream_total - self.cross_stream_consistent,
                "consistency_rate": cross_rate,
            },
            "per_stream": _rates(self.per_stream_totals, self.per_stream_violations),
            "time": _rates(self.time_totals, self.time_violations),
            "total_tuples": self.total_tuples,
            "source_counts": dict(self.source_counts),
        }


def compute_attribute_consistency(
    notes_by_stream: Mapping[str, Sequence[str]],
    plan_by_stream: Mapping[str, Sequence[str]] | None = None,
    *,
    structured_notes: Mapping[str, StreamNotes] | None = None,
) -> AttributeConsistencyResult:
    """Compute attribute consistency metrics from notes text and optional structured payloads."""

    normalized_plan = _normalize_plan(plan_by_stream or {})
    tuples_by_stream: Dict[str, List[AttributeTuple]] = defaultdict(list)
    source_counts: Dict[str, int] = defaultdict(int)

    for stream_id, notes in notes_by_stream.items():
        stream_key = _normalize_stream_id(stream_id)
        if isinstance(notes, str):
            note_sequence: Sequence[str] = [notes]
        elif isinstance(notes, Sequence):
            note_sequence = notes
        else:
            continue
        for time_index, text in enumerate(note_sequence):
            if not isinstance(text, str):
                continue
            for subject, attribute, value in _extract_text_triples(text):
                candidate = AttributeTuple(
                    stream_id=stream_key,
                    subject=subject,
                    attribute=attribute,
                    value=value,
                    time_index=time_index,
                    source="text",
                )
                if _accept_tuple(candidate, normalized_plan):
                    tuples_by_stream[stream_key].append(candidate)
                    source_counts["text"] += 1

    if structured_notes:
        for stream_id, payload in structured_notes.items():
            stream_key = _normalize_stream_id(stream_id)
            for tuple_ in _tuples_from_stream_notes(stream_key, payload):
                if _accept_tuple(tuple_, normalized_plan):
                    tuples_by_stream[stream_key].append(tuple_)
                    source_counts["structured"] += 1

    return _evaluate_consistency(tuples_by_stream, source_counts)


def _normalize_stream_id(stream_id: str) -> str:
    return str(stream_id or "").strip().lower()


def _normalize_token(text: str) -> str:
    lowered = _TOKEN_NORMALISER.sub(" ", str(text).strip().lower())
    return " ".join(lowered.split())


def _is_informative(token: str) -> bool:
    return bool(token) and len(token) >= 3 and token not in _STOPWORDS


def _normalize_plan(plan_by_stream: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for stream_id, entries in plan_by_stream.items():
        stream_key = _normalize_stream_id(stream_id)
        normalized_entries = [
            entry for entry in (_normalize_token(item) for item in entries) if entry
        ]
        if normalized_entries:
            normalized[stream_key] = normalized_entries
    return normalized


def _extract_text_triples(note_text: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for raw_segment in _SEGMENT_SPLIT.split(note_text):
        segment = raw_segment.strip()
        if not segment:
            continue
        segment = segment.strip("-–—•* ")
        if not segment:
            continue
        segment = segment.replace("—", "-")
        match = _THREE_FIELD_PATTERN.match(segment)
        if match:
            triples.append(
                (
                    match.group("subject").strip(),
                    match.group("attribute").strip(),
                    match.group("value").strip(),
                )
            )
            continue
        match = _POSSESSIVE_PATTERN.match(segment)
        if match:
            triples.append(
                (
                    match.group("subject").strip(),
                    match.group("attribute").strip(),
                    match.group("value").strip(),
                )
            )
            continue
        match = _VERB_PATTERN.match(segment)
        if match:
            triples.append(
                (
                    match.group("subject").strip(),
                    match.group("attribute").strip(),
                    match.group("value").strip(),
                )
            )
            continue
        parts = re.split(r"\s*[-:]\s*", segment, maxsplit=2)
        if len(parts) == 3:
            triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return triples


def _tuples_from_stream_notes(stream_id: str, notes: StreamNotes) -> Iterable[AttributeTuple]:
    entities = {entity.id: (entity.name or entity.id) for entity in notes.entities}
    for fact in notes.facts:
        subject = entities.get(fact.subj_id, fact.subj_id)
        if not subject or not fact.predicate or not fact.object:
            continue
        yield AttributeTuple(
            stream_id=_normalize_stream_id(stream_id),
            subject=subject,
            attribute=fact.predicate,
            value=fact.object,
            time_index=0,
            source="structured",
        )


def _accept_tuple(tuple_: AttributeTuple, plan_lookup: Mapping[str, Sequence[str]]) -> bool:
    subject_norm = tuple_.subject_norm()
    attribute_norm = tuple_.attribute_norm()
    value_norm = tuple_.value_norm()
    if not (_is_informative(subject_norm) and _is_informative(attribute_norm) and value_norm):
        return False
    plan_entries = plan_lookup.get(tuple_.stream_id)
    if not plan_entries:
        return True
    for entry in plan_entries:
        if subject_norm and subject_norm in entry:
            return True
        if attribute_norm and attribute_norm in entry:
            return True
    return False


def _evaluate_consistency(
    tuples_by_stream: Mapping[str, Sequence[AttributeTuple]],
    source_counts: Mapping[str, int],
) -> AttributeConsistencyResult:
    tuples_by_key: Dict[Tuple[str, str], Dict[str, List[AttributeTuple]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total_tuples = 0
    for stream_id, tuples in tuples_by_stream.items():
        total_tuples += len(tuples)
        for tuple_ in tuples:
            subj = tuple_.subject_norm()
            attr = tuple_.attribute_norm()
            if not subj or not attr:
                continue
            tuples_by_key[(subj, attr)][stream_id].append(tuple_)

    cross_total = 0
    cross_consistent = 0
    per_stream_totals: Dict[str, int] = defaultdict(int)
    per_stream_violations: Dict[str, int] = defaultdict(int)
    time_totals: Dict[str, int] = defaultdict(int)
    time_violations: Dict[str, int] = defaultdict(int)

    for stream_map in tuples_by_key.values():
        for stream_id, entries in stream_map.items():
            if len(entries) < 2:
                continue
            time_totals[stream_id] += 1
            unique_values = {
                _normalize_token(entry.value) for entry in entries if entry.value_norm()
            }
            if len(unique_values) > 1:
                time_violations[stream_id] += 1

    for stream_map in tuples_by_key.values():
        if len(stream_map) < 2:
            continue
        cross_total += 1
        majority_by_stream: Dict[str, str] = {}
        for stream_id, entries in stream_map.items():
            per_stream_totals[stream_id] += 1
            majority_by_stream[stream_id] = _majority_value(entries)
        unique_values = {value for value in majority_by_stream.values() if value}
        if len(unique_values) <= 1:
            cross_consistent += 1
            continue
        dominant_value = _dominant_value(majority_by_stream.values())
        for stream_id, value in majority_by_stream.items():
            if value != dominant_value:
                per_stream_violations[stream_id] += 1

    return AttributeConsistencyResult(
        cross_stream_total=cross_total,
        cross_stream_consistent=cross_consistent,
        per_stream_totals=dict(per_stream_totals),
        per_stream_violations=dict(per_stream_violations),
        time_totals=dict(time_totals),
        time_violations=dict(time_violations),
        total_tuples=total_tuples,
        source_counts=dict(source_counts),
    )


def _majority_value(entries: Sequence[AttributeTuple]) -> str:
    counts: MutableMapping[str, int] = defaultdict(int)
    latest_index: MutableMapping[str, int] = {}
    for entry in entries:
        value_norm = entry.value_norm()
        if not value_norm:
            continue
        counts[value_norm] += 1
        latest_index[value_norm] = max(latest_index.get(value_norm, -1), entry.time_index)
    if not counts:
        return ""
    best_value = ""
    best_signature = (-1, -1)
    for value, count in counts.items():
        signature = (count, latest_index.get(value, -1))
        if signature > best_signature:
            best_signature = signature
            best_value = value
    return best_value


def _dominant_value(values: Iterable[str]) -> str:
    counter = Counter(value for value in values if value)
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


__all__ = [
    "AttributeConsistencyResult",
    "AttributeTuple",
    "compute_attribute_consistency",
]
