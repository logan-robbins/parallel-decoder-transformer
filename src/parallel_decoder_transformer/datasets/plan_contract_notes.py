"""Utilities for deriving deterministic notes snapshots from plan contracts."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from parallel_decoder_transformer.data.extraction import (
    CoverageSignal,
    CoverageStatus,
    EntityCard,
    EvidenceSpan,
    FactStatement,
    StreamNotes,
)

_KEY_VALUE_PATTERN = re.compile(r"(\w+)=((?:'[^']*')|(?:\"[^\"]*\")|[^,]+)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ANSWER_KEYWORD_PATTERN = re.compile(r"\bfinal[\s_-]*answers?\b|\banswers?\b", re.IGNORECASE)
_ANSWER_FACT_PREDICATES = {
    "answers",
    "provides_answer",
    "delivers_answer",
    "yields",
    "states_answer",
}
_CERTAINTY_MAP = {
    "low": 0.35,
    "medium": 0.6,
    "mid": 0.6,
    "moderate": 0.6,
    "high": 0.85,
    "certain": 1.0,
    "sure": 0.9,
}


def derive_initial_notes_from_plan(
    plan_payload: Mapping[str, Any],
    *,
    input_text: str = "",
) -> list[StreamNotes]:
    """Convert planner payloads into machine-usable per-stream notes."""

    streams = plan_payload.get("streams", [])
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes)):
        return []
    derived: list[StreamNotes] = []
    for idx, raw_stream in enumerate(streams):
        if not isinstance(raw_stream, Mapping):
            continue
        derived.append(_derive_stream_notes(raw_stream, idx, input_text))
    return derived


def merge_seed_notes(
    plan_streams: Sequence[Mapping[str, Any]] | Sequence[Any],
    true_notes: Sequence[StreamNotes],
    seed_notes: Sequence[StreamNotes],
    *,
    input_text: str = "",
) -> list[StreamNotes]:
    """Align true notes to plan order and ensure each stream has plan-specified seed data."""

    plan_meta: list[tuple[str, str, str]] = []
    for idx, raw in enumerate(plan_streams or []):
        if not isinstance(raw, Mapping):
            continue
        stream_id = str(raw.get("stream_id") or f"stream_{idx + 1}")
        header = str(raw.get("header") or stream_id)
        summary = str(raw.get("summary") or header)
        plan_meta.append((stream_id, header, summary))
    if not plan_meta:
        plan_meta = [
            (note.stream_id, note.stream_id, note.stream_id) for note in seed_notes or true_notes
        ]
    true_map = {note.stream_id: note for note in true_notes}
    seed_map = {note.stream_id: note for note in seed_notes}
    merged: list[StreamNotes] = []
    for stream_id, header, summary in plan_meta:
        base = true_map.get(stream_id)
        if base is None:
            base = StreamNotes(stream_id=stream_id)
        seed = seed_map.get(stream_id)
        if seed is not None:
            _merge_stream_notes(base, seed)
        _ensure_minimum_payload(base, header, summary, input_text)
        merged.append(base)
    return merged


# REMOVED: build_versioned_note_snapshots - replaced by procedural_snapshots.py
# which generates snapshots deterministically based on evidence_span positions


def _derive_stream_notes(stream: Mapping[str, Any], index: int, input_text: str) -> StreamNotes:
    stream_id = str(stream.get("stream_id") or f"stream_{index + 1}")
    header = str(stream.get("header") or f"Stream {index + 1}")
    summary = str(stream.get("summary") or header)
    entity_templates = stream.get("entities") or []
    constraint_templates = stream.get("constraints") or []
    notes_contracts = stream.get("notes_contract") or []
    section_contract = stream.get("section_contract") or {}

    entities = _dedupe_entities(
        _parse_entity_templates(entity_templates, stream_id, header)
        + _parse_entity_templates(notes_contracts, stream_id, header)
    )
    facts, coverage = _parse_constraint_templates(
        constraint_templates, stream_id, summary, input_text
    )
    contract_facts, contract_cov = _parse_notes_contract_entries(
        notes_contracts, stream_id, summary, input_text
    )
    facts = _dedupe_facts(facts + contract_facts)
    coverage = _dedupe_coverage(
        coverage + contract_cov + _coverage_from_section_contract(section_contract, stream_id)
    )

    entities, facts = _ensure_answer_semantics(
        entities,
        facts,
        coverage,
        stream_id=stream_id,
        header=header,
        summary=summary,
        input_text=input_text,
        notes_contracts=notes_contracts,
    )

    if not entities:
        entities = [_default_entity(stream_id, header)]
    if not facts:
        facts = [_default_fact(stream_id, entities[0], summary, input_text)]
    if not coverage:
        coverage = [_default_coverage(stream_id)]

    return StreamNotes(stream_id=stream_id, entities=entities, facts=facts, coverage=coverage)


def _parse_entity_templates(
    templates: Sequence[Any],
    stream_id: str,
    header: str,
) -> list[EntityCard]:
    entities: list[EntityCard] = []
    for idx, template in enumerate(templates or []):
        prefix, fields = _extract_prefix_and_fields(template)
        if prefix != "ENT":
            continue
        entity_id = _normalize_identifier(fields.get("id") or f"{stream_id}_ent_{idx + 1}")
        name = fields.get("name") or header
        entity_type = fields.get("type") or "entity"
        aliases = _split_aliases(fields.get("aliases"))
        canonical = _to_bool(fields.get("canonical"), default=True)
        entities.append(
            EntityCard(
                id=entity_id, name=name, aliases=aliases, type=entity_type, canonical=canonical
            )
        )
    return entities


def _parse_constraint_templates(
    templates: Sequence[Any],
    stream_id: str,
    summary: str,
    input_text: str,
) -> tuple[list[FactStatement], list[CoverageSignal]]:
    facts: list[FactStatement] = []
    coverage: list[CoverageSignal] = []
    for idx, template in enumerate(templates or []):
        prefix, fields = _extract_prefix_and_fields(template)
        if prefix == "FACT":
            facts.append(_fact_from_fields(fields, stream_id, summary, input_text))
        elif prefix == "COVERAGE":
            coverage.append(_coverage_from_fields(fields, stream_id, suffix=f"plan_{idx + 1}"))
    return facts, coverage


def _parse_notes_contract_entries(
    entries: Sequence[Any],
    stream_id: str,
    summary: str,
    input_text: str,
) -> tuple[list[FactStatement], list[CoverageSignal]]:
    facts: list[FactStatement] = []
    coverage: list[CoverageSignal] = []
    for idx, entry in enumerate(entries or []):
        prefix, fields = _extract_prefix_and_fields(entry)
        if prefix == "FACT":
            facts.append(_fact_from_fields(fields, stream_id, summary, input_text))
        elif prefix == "COVERAGE":
            coverage.append(_coverage_from_fields(fields, stream_id, suffix=f"contract_{idx + 1}"))
        elif prefix == "ENT":
            # Entities parsed elsewhere; ignore duplicates.
            continue
        else:
            text = str(fields.get("text") or entry).strip()
            if not text:
                continue
            plan_item = f"{stream_id}_{_slugify(text)}"
            coverage.append(CoverageSignal(plan_item_id=plan_item, status=CoverageStatus.PARTIAL))
    return facts, coverage


def _coverage_from_fields(
    fields: Mapping[str, Any],
    stream_id: str,
    *,
    suffix: str,
) -> CoverageSignal:
    plan_item_id = str(fields.get("plan_item_id") or f"{stream_id}_{suffix}")
    status = _coverage_status(fields.get("status"))
    return CoverageSignal(plan_item_id=plan_item_id, status=status)


def _coverage_from_section_contract(contract: Any, stream_id: str) -> list[CoverageSignal]:
    if not isinstance(contract, Mapping) or not contract:
        return []
    contract_type = str(contract.get("type") or "section")
    details = [f"{key}={value}" for key, value in contract.items() if key != "type"]
    descriptor = f"{contract_type}:{'|'.join(details)}" if details else contract_type
    plan_item_id = f"{stream_id}_{_slugify(descriptor)}"
    return [CoverageSignal(plan_item_id=plan_item_id, status=CoverageStatus.COVERED)]


def _merge_stream_notes(target: StreamNotes, seed: StreamNotes) -> None:
    target.entities = _dedupe_entities(list(target.entities) + list(seed.entities))
    target.facts = _dedupe_facts(list(target.facts) + list(seed.facts))
    target.coverage = _dedupe_coverage(list(target.coverage) + list(seed.coverage))


def _ensure_minimum_payload(
    note: StreamNotes,
    header: str,
    summary: str,
    input_text: str,
) -> None:
    if not note.entities:
        note.entities = [_default_entity(note.stream_id, header)]
    if not note.facts:
        note.facts = [_default_fact(note.stream_id, note.entities[0], summary, input_text)]
    if not note.coverage:
        note.coverage = [_default_coverage(note.stream_id)]


def _default_entity(stream_id: str, header: str) -> EntityCard:
    return EntityCard(
        id=f"{stream_id}_entity",
        name=header or stream_id,
        aliases=[],
        type="section",
        canonical=True,
    )


def _ensure_answer_semantics(
    entities: Sequence[EntityCard],
    facts: Sequence[FactStatement],
    coverage: Sequence[CoverageSignal],
    *,
    stream_id: str,
    header: str,
    summary: str,
    input_text: str,
    notes_contracts: Sequence[Any],
) -> tuple[list[EntityCard], list[FactStatement]]:
    if not _requires_answer_semantics(notes_contracts, coverage):
        return list(entities), list(facts)

    updated_entities = list(entities)
    answer_entity = _find_answer_entity(updated_entities)
    if answer_entity is None:
        answer_name = summary.strip() or header or stream_id
        aliases = [summary.strip()] if summary.strip() and summary.strip() != answer_name else []
        answer_entity = EntityCard(
            id=f"{stream_id}_answer",
            name=answer_name,
            aliases=aliases,
            type="final_answer",
            canonical=True,
        )
        updated_entities.append(answer_entity)

    updated_facts = list(facts)
    if not _has_answer_fact(updated_facts, answer_entity.id):
        snippet = summary.strip() or input_text.strip() or answer_entity.name
        fact_object = snippet or answer_entity.name
        evidence = _make_evidence_span(fact_object, input_text)
        updated_facts.append(
            FactStatement(
                subj_id=answer_entity.id,
                predicate="answers",
                object=fact_object,
                evidence_span=evidence,
                certainty=0.9,
            )
        )

    return _dedupe_entities(updated_entities), _dedupe_facts(updated_facts)


def _requires_answer_semantics(
    notes_contracts: Sequence[Any],
    coverage: Sequence[CoverageSignal],
) -> bool:
    for entry in notes_contracts or []:
        if _entry_mentions_answer(entry):
            return True
    for signal in coverage or []:
        if _contains_answer_keyword(signal.plan_item_id):
            return True
    return False


def _entry_mentions_answer(entry: Any) -> bool:
    if isinstance(entry, Mapping):
        for value in entry.values():
            if isinstance(value, str) and _contains_answer_keyword(value):
                return True
        return False
    if isinstance(entry, str):
        return _contains_answer_keyword(entry)
    return False


def _contains_answer_keyword(text: str) -> bool:
    return bool(_ANSWER_KEYWORD_PATTERN.search(str(text)))


def _find_answer_entity(entities: Sequence[EntityCard]) -> EntityCard | None:
    for entity in entities:
        name = entity.name.lower()
        if entity.type == "final_answer":
            return entity
        if entity.id.endswith("_answer"):
            return entity
        if "answer" in name:
            return entity
    return None


def _has_answer_fact(facts: Sequence[FactStatement], answer_entity_id: str) -> bool:
    for fact in facts:
        if fact.subj_id == answer_entity_id and fact.predicate in _ANSWER_FACT_PREDICATES:
            return True
        if fact.predicate in _ANSWER_FACT_PREDICATES and _contains_answer_keyword(fact.object):
            return True
    return False


def _default_fact(
    stream_id: str,
    entity: EntityCard,
    summary: str,
    input_text: str,
) -> FactStatement:
    snippet = (summary or input_text[:160]).strip() or stream_id
    evidence = _make_evidence_span(snippet, input_text)
    return FactStatement(
        subj_id=entity.id,
        predicate="describes_section",
        object=snippet,
        evidence_span=evidence,
        certainty=0.8,
    )


def _default_coverage(stream_id: str) -> CoverageSignal:
    return CoverageSignal(plan_item_id=f"{stream_id}_section", status=CoverageStatus.PARTIAL)


def _fact_from_fields(
    fields: Mapping[str, Any],
    stream_id: str,
    summary: str,
    input_text: str,
) -> FactStatement:
    subj_id = _normalize_identifier(
        fields.get("subj_id") or fields.get("subject") or f"{stream_id}_ent_1"
    )
    predicate = str(fields.get("predicate") or "relates_to")
    obj = str(fields.get("object") or fields.get("object_id") or summary or stream_id)
    if obj.startswith("ENT:"):
        obj = obj.split(":", 1)[-1]
    certainty = _certainty_value(fields.get("certainty"))
    start = _int_from_value(fields.get("start"))
    end = _int_from_value(fields.get("end"), default=start + len(obj))
    text = str(fields.get("text") or summary or obj)
    evidence = _make_evidence_span(text, input_text, start_override=start, end_override=end)
    return FactStatement(
        subj_id=subj_id,
        predicate=predicate,
        object=obj,
        evidence_span=evidence,
        certainty=certainty,
    )


def _make_evidence_span(
    hint: str,
    input_text: str,
    *,
    start_override: int | None = None,
    end_override: int | None = None,
) -> EvidenceSpan:
    snippet = (hint or input_text[:160]).strip()
    if not snippet and input_text:
        snippet = input_text[:160]
    snippet = snippet[:160]
    if snippet and snippet in input_text:
        start = input_text.index(snippet)
        end = start + len(snippet)
    else:
        start = start_override if start_override is not None else 0
        end = end_override if end_override is not None else start + len(snippet)
    if end < start:
        end = start
    return EvidenceSpan(start=start, end=end, text=snippet)


def _extract_prefix_and_fields(entry: Any) -> tuple[str, dict[str, str]]:
    if isinstance(entry, Mapping):
        prefix = (
            str(
                entry.get("note_type")
                or entry.get("kind")
                or entry.get("schema")
                or entry.get("prefix")
                or ""
            )
            .strip()
            .upper()
        )
        body = {
            str(k): str(v)
            for k, v in entry.items()
            if k not in {"note_type", "kind", "schema", "prefix"}
        }
        if prefix:
            return prefix, body
        return "TEXT", {"text": json.dumps(entry, ensure_ascii=False)}
    if isinstance(entry, str):
        value = entry.strip()
        if not value:
            return "", {}
        if ":" not in value:
            return "TEXT", {"text": value}
        prefix, rest = value.split(":", 1)
        return prefix.strip().upper(), _parse_key_values(rest)
    return "TEXT", {"text": str(entry)}


def _parse_key_values(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _KEY_VALUE_PATTERN.finditer(body):
        key = match.group(1)
        raw_value = match.group(2).strip()
        if raw_value.startswith("'") and raw_value.endswith("'"):
            value = raw_value[1:-1]
        elif raw_value.startswith('"') and raw_value.endswith('"'):
            value = raw_value[1:-1]
        else:
            value = raw_value
        fields[key] = value.strip()
    return fields


def _split_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(alias).strip() for alias in value if str(alias).strip()]
    raw = str(value)
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(alias).strip() for alias in parsed if str(alias).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[|;/]", raw) if part.strip()]


def _normalize_identifier(value: Any) -> str:
    text = str(value or "").strip().replace(" ", "_")
    return text or "ent"


def _dedupe_entities(entities: Sequence[EntityCard]) -> list[EntityCard]:
    seen: set[str] = set()
    deduped: list[EntityCard] = []
    for entity in entities:
        key = entity.id.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _dedupe_facts(facts: Sequence[FactStatement]) -> list[FactStatement]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[FactStatement] = []
    for fact in facts:
        key = (fact.subj_id.lower(), fact.predicate, str(fact.object))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _dedupe_coverage(coverage: Sequence[CoverageSignal]) -> list[CoverageSignal]:
    seen: set[str] = set()
    deduped: list[CoverageSignal] = []
    for signal in coverage:
        if signal.plan_item_id in seen:
            continue
        seen.add(signal.plan_item_id)
        deduped.append(signal)
    return deduped


def _coverage_status(value: Any) -> CoverageStatus:
    if isinstance(value, CoverageStatus):
        return value
    lowered = str(value or "").strip().lower()
    if lowered in {"complete", "covered", "done"}:
        return CoverageStatus.COVERED
    if lowered in {"partial", "progress", "ongoing", "in_progress"}:
        return CoverageStatus.PARTIAL
    if lowered:
        return CoverageStatus.MISSING
    return CoverageStatus.PARTIAL


def _certainty_value(value: Any) -> float:
    if value is None:
        return 0.75
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    lowered = str(value).strip().lower()
    if lowered in _CERTAINTY_MAP:
        return _CERTAINTY_MAP[lowered]
    try:
        numeric = float(lowered)
        if 0.0 <= numeric <= 1.0:
            return numeric
    except ValueError:
        pass
    return 0.75


def _int_from_value(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slugify(value: str, *, fallback: str | None = None) -> str:
    slug = _NON_ALNUM.sub("_", value.lower()).strip("_")
    if len(slug) > 48:
        slug = slug[:48]
    if not slug:
        slug = fallback or "item"
    return slug


# REMOVED: _snapshot_entry helper - no longer needed with procedural snapshots


def _to_bool(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value or "").strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return default


__all__ = [
    "derive_initial_notes_from_plan",
    "merge_seed_notes",
]
