"""Convert source-grounded manual audits into canonical real-plan records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from model_intrinsic_parallel.training_example import TrainingExample
from pdt.config.schemas import TrunkProfile
from pdt.datasets.historical_source import (
    AssessmentQuality,
    HISTORICAL_RENDERER_ID,
    HISTORICAL_SOURCE_SCHEMA,
    HistoricalCategory,
    HistoricalParagraph,
    HistoricalSection,
    HistoricalSource,
    PageAssessment,
    ReferenceKind,
    ReferenceRecord,
    render_historical_sections,
    split_for_family,
)
from pdt.datasets.real_plan_schema import (
    REAL_PLAN_SCHEMA,
    AtomicFact,
    CrossPlanDependency,
    FactExtractionOutput,
    FactRole,
    JointPlanOutput,
    LaneFactLabel,
    OutlineNode,
    ProvenanceSpan,
    RealPlanExample,
    StreamPlan,
    TargetParagraph,
    TargetSection,
    validate_real_plan_example,
)


AUDITED_TRAINER_CATALOG_SCHEMA = "pdt-audited-trainer-catalog-v1"

__all__ = [
    "AUDITED_TRAINER_CATALOG_SCHEMA",
    "AuditedTrainerCatalogEntry",
    "build_audited_real_plan_example",
    "load_audited_catalog",
    "load_catalog_raw_revision",
    "load_training_example",
]


class AuditedTrainerCatalogEntry(BaseModel):
    """Human-reviewed metadata required to bridge an audited source revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=rf"^{AUDITED_TRAINER_CATALOG_SCHEMA}$")
    example_file: str = Field(pattern=r"^[a-z0-9_]+\.json$")
    raw_revision_file: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    revision_id: int = Field(gt=0)
    historical_category: HistoricalCategory
    event_end_year: int = Field(ge=1, le=2200)
    dump_date: date

    @field_validator("raw_revision_file")
    @classmethod
    def validate_raw_revision_file(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("raw_revision_file must be a repository-relative path.")
        return value


def load_audited_catalog(path: Path) -> tuple[AuditedTrainerCatalogEntry, ...]:
    """Load an immutable catalog and reject duplicate source or example identity."""

    if not path.is_file():
        raise FileNotFoundError(f"Audited trainer catalog does not exist: {path}")
    entries: list[AuditedTrainerCatalogEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(AuditedTrainerCatalogEntry.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} violates the catalog schema.") from exc
    if not entries:
        raise ValueError(f"Audited trainer catalog is empty: {path}")
    _require_unique(
        [entry.example_file for entry in entries],
        label="example_file",
        context=path,
    )
    _require_unique(
        [entry.source_id for entry in entries],
        label="source_id",
        context=path,
    )
    return tuple(entries)


def load_training_example(path: Path) -> TrainingExample:
    """Load one already-audited teacher example without changing it."""

    if not path.is_file():
        raise FileNotFoundError(f"Audited training example does not exist: {path}")
    try:
        return TrainingExample.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Audited training example violates its schema: {path}") from exc


def build_audited_real_plan_example(
    *,
    example: TrainingExample,
    catalog: AuditedTrainerCatalogEntry,
    raw_revision: Mapping[str, object],
    tokenizer: Any,
    trunk_profile: TrunkProfile,
) -> RealPlanExample:
    """Build one canonical v2 record while preserving audited semantic labels."""

    _validate_identity(example, catalog, raw_revision)
    sections = _historical_sections(example)
    rendered_source = render_historical_sections(
        example.source_document.source.title,
        sections,
    )
    source_digest = hashlib.sha256(rendered_source.encode("utf-8")).hexdigest()
    source_token_ids = tokenizer.encode(rendered_source, add_special_tokens=False)
    if not source_token_ids:
        raise ValueError(f"{catalog.source_id} produced an empty source token sequence.")
    family_id = "family-" + hashlib.sha256(
        catalog.source_id.encode("utf-8")
    ).hexdigest()[:24]
    source = HistoricalSource(
        schema_version=HISTORICAL_SOURCE_SCHEMA,
        renderer_id=HISTORICAL_RENDERER_ID,
        source_id=catalog.source_id,
        page_id=example.source_document.source.page_id,
        revision_id=catalog.revision_id,
        revision_timestamp=datetime.fromisoformat(
            example.source_document.source.revision_timestamp.replace("Z", "+00:00")
        ),
        dump_date=catalog.dump_date,
        title=example.source_document.source.title,
        source_url=example.source_document.source.source_url,
        license=example.source_document.source.license,
        revision_sha1=_required_text(raw_revision, "revision_sha1"),
        historical_category=catalog.historical_category,
        event_end_year=catalog.event_end_year,
        assessments=_page_assessments(raw_revision),
        family_id=family_id,
        split=split_for_family(family_id),
        tokenizer=trunk_profile.base_model,
        tokenizer_revision=trunk_profile.revision,
        qwen_token_count=len(source_token_ids),
        sections=sections,
        references=_reference_records(example),
        raw_wikitext_sha256=example.source_document.source.raw_wikitext_sha256,
        semantic_html_sha256=example.source_document.source.semantic_html_sha256,
        model_visible_sha256=source_digest,
    )

    fact_id = {
        fact.fact_id: f"fact_{index:03d}"
        for index, fact in enumerate(example.facts)
    }
    facts = FactExtractionOutput(
        source_id=source.source_id,
        source_revision_id=source.revision_id,
        source_model_visible_sha256=source.model_visible_sha256,
        facts=[
            AtomicFact(
                fact_id=fact_id[fact.fact_id],
                statement=fact.statement,
                importance=fact.importance,
                provenance=[
                    ProvenanceSpan(
                        paragraph_id=fact.evidence.paragraph_id,
                        start_char=fact.evidence.char_start,
                        end_char=fact.evidence.char_end,
                        exact_quote=fact.evidence.exact_quote,
                        reference_ids=list(fact.evidence.citation_ids),
                    )
                ],
                hard_negative=fact.hard_negative,
            )
            for fact in example.facts
        ],
    )

    plan_id = {
        lane.lane_id: f"plan_{index}" for index, lane in enumerate(example.lanes)
    }
    node_id = {
        lane.lane_id: {
            node.node_id: f"node_{index}"
            for index, node in enumerate(lane.plan_nodes)
        }
        for lane in example.lanes
    }
    paragraph_to_node = {
        lane.lane_id: {
            node.target_paragraph_id: node_id[lane.lane_id][node.node_id]
            for node in lane.plan_nodes
        }
        for lane in example.lanes
    }
    dependencies_by_target: dict[tuple[str, str], list[CrossPlanDependency]] = {}
    for dependency in example.dependencies:
        source_node = paragraph_to_node[dependency.source_lane_id].get(
            dependency.source_target_paragraph_id
        )
        if source_node is None:
            raise ValueError(
                f"{catalog.source_id} dependency source paragraph is not assigned "
                f"to a node: {dependency.source_target_paragraph_id}."
            )
        converted = CrossPlanDependency(
            source_plan_id=plan_id[dependency.source_lane_id],
            source_node_id=source_node,
            source_target_paragraph_id=dependency.source_target_paragraph_id,
            source_evidence_quote=dependency.source_exact_quote,
            required_by_target_paragraph_id=dependency.target_paragraph_id,
            required_target_evidence_quote=dependency.target_exact_quote,
            fact_ids=[fact_id[dependency.fact_id]],
        )
        dependencies_by_target.setdefault(
            (dependency.target_lane_id, dependency.target_paragraph_id),
            [],
        ).append(converted)

    plans: list[StreamPlan] = []
    targets: list[TargetSection] = []
    for lane in example.lanes:
        labels = {
            label.fact_id: label
            for label in lane.fact_labels
        }
        references_by_paragraph: dict[str, list[str]] = {}
        for label in lane.fact_labels:
            if label.role == "REFERENCE":
                if label.target_paragraph_id is None:
                    raise ValueError(
                        f"{catalog.source_id}/{lane.lane_id}/{label.fact_id} "
                        "REFERENCE label has no target paragraph."
                    )
                references_by_paragraph.setdefault(label.target_paragraph_id, []).append(
                    fact_id[label.fact_id]
                )
        plans.append(
            StreamPlan(
                plan_id=plan_id[lane.lane_id],
                heading=lane.focus,
                role_summary=lane.selection_rationale,
                nodes=[
                    OutlineNode(
                        node_id=node_id[lane.lane_id][node.node_id],
                        objective=node.purpose,
                        owned_fact_ids=[
                            fact_id[value] for value in node.owner_fact_ids
                        ],
                        reference_fact_ids=references_by_paragraph.get(
                            node.target_paragraph_id,
                            [],
                        ),
                        dependencies=dependencies_by_target.get(
                            (lane.lane_id, node.target_paragraph_id),
                            [],
                        ),
                        target_paragraph_ids=[node.target_paragraph_id],
                    )
                    for node in lane.plan_nodes
                ],
            )
        )
        targets.append(
            TargetSection(
                plan_id=plan_id[lane.lane_id],
                paragraphs=[
                    TargetParagraph(
                        paragraph_id=paragraph.paragraph_id,
                        outline_node_id=paragraph_to_node[lane.lane_id][
                            paragraph.paragraph_id
                        ],
                        text=paragraph.text,
                    )
                    for paragraph in lane.target_paragraphs
                ],
                fact_labels=[
                    LaneFactLabel(
                        fact_id=fact_id[value],
                        role=FactRole(labels[value].role),
                        target_evidence_quote=labels[value].target_exact_quote,
                    )
                    for value in fact_id
                ],
            )
        )

    converted_example = RealPlanExample(
        schema_version=REAL_PLAN_SCHEMA,
        source=source,
        facts=facts,
        teacher=JointPlanOutput(
            source_id=source.source_id,
            source_revision_id=source.revision_id,
            source_model_visible_sha256=source.model_visible_sha256,
            expository_prompt=example.question,
            plans=plans,
            target_sections=targets,
            presentation_order=[plan_id[value] for value in example.presentation_order],
        ),
    )
    validate_real_plan_example(converted_example)
    return converted_example


def _historical_sections(
    example: TrainingExample,
) -> tuple[HistoricalSection, ...]:
    sections: list[HistoricalSection] = []
    current_path: tuple[str, ...] | None = None
    current_paragraphs: list[HistoricalParagraph] = []
    for paragraph in example.source_document.paragraphs:
        if current_path is not None and paragraph.section_path != current_path:
            sections.append(
                HistoricalSection(
                    section_id=f"section_{len(sections):03d}",
                    heading_path=current_path,
                    paragraphs=tuple(current_paragraphs),
                )
            )
            current_paragraphs = []
        current_path = paragraph.section_path
        current_paragraphs.append(
            HistoricalParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                reference_ids=paragraph.citation_ids,
            )
        )
    if current_path is None:
        raise ValueError(f"{example.source_document.source.source_id} has no source paragraphs.")
    sections.append(
        HistoricalSection(
            section_id=f"section_{len(sections):03d}",
            heading_path=current_path,
            paragraphs=tuple(current_paragraphs),
        )
    )
    return tuple(sections)


def _reference_records(example: TrainingExample) -> tuple[ReferenceRecord, ...]:
    records: list[ReferenceRecord] = []
    valid_kinds = {kind.value for kind in ReferenceKind}
    for citation in example.source_document.citations:
        year = citation.year if type(citation.year) is int else None
        source_type = (
            ReferenceKind(citation.source_type)
            if citation.source_type in valid_kinds
            else ReferenceKind.OTHER
        )
        records.append(
            ReferenceRecord(
                reference_id=citation.reference_id,
                source_type=source_type,
                citation_text=citation.citation_text,
                title=citation.title,
                authors=citation.authors,
                publication=citation.publication,
                year=year,
                identifiers=citation.identifiers,
            )
        )
    return tuple(records)


def _page_assessments(
    raw_revision: Mapping[str, object],
) -> tuple[PageAssessment, ...]:
    raw = raw_revision.get("assessments")
    assessments: list[PageAssessment] = []
    if isinstance(raw, list):
        for value in raw:
            if not isinstance(value, Mapping):
                raise ValueError("Raw assessment list entries must be objects.")
            assessments.append(
                PageAssessment(
                    project=_mapping_text(value, "project"),
                    quality=AssessmentQuality(_mapping_text(value, "quality")),
                )
            )
    elif isinstance(raw, Mapping):
        for project, value in raw.items():
            if not isinstance(project, str) or not isinstance(value, Mapping):
                raise ValueError("Raw assessment map entries must be project objects.")
            assessments.append(
                PageAssessment(
                    project=project,
                    quality=AssessmentQuality(_mapping_text(value, "class")),
                )
            )
    else:
        raise ValueError("Raw revision must contain list or mapping assessments.")
    if not assessments or not any(
        assessment.quality is AssessmentQuality.FA for assessment in assessments
    ):
        raise ValueError("Audited trainer source must retain at least one FA assessment.")
    return tuple(assessments)


def _validate_identity(
    example: TrainingExample,
    catalog: AuditedTrainerCatalogEntry,
    raw_revision: Mapping[str, object],
) -> None:
    source = example.source_document.source
    exact = {
        "source_id": source.source_id,
        "page_id": source.page_id,
        "revision_id": source.revision_id,
        "revision_timestamp": source.revision_timestamp,
        "title": source.title,
        "source_url": source.source_url,
        "license": source.license,
        "raw_wikitext_sha256": source.raw_wikitext_sha256,
        "semantic_html_sha256": source.semantic_html_sha256,
    }
    if catalog.source_id != source.source_id or catalog.revision_id != source.revision_id:
        raise ValueError(
            f"Catalog identity does not match audited example {catalog.example_file}."
        )
    mismatches = {
        key: (expected, raw_revision.get(key))
        for key, expected in exact.items()
        if raw_revision.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Raw revision identity does not match {catalog.example_file}: {mismatches}."
        )


def _load_raw_revision(
    path: Path,
    *,
    source_id: str,
    revision_id: int,
) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned raw revision JSONL does not exist: {path}")
    matches: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be an object.")
            if (
                value.get("source_id") == source_id
                and value.get("revision_id") == revision_id
            ):
                matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one pinned raw revision for {source_id}@{revision_id} "
            f"in {path}; found {len(matches)}."
        )
    return matches[0]


def load_catalog_raw_revision(
    repository_root: Path,
    entry: AuditedTrainerCatalogEntry,
) -> Mapping[str, object]:
    """Load the catalog-pinned raw row used to verify immutable source identity."""

    return _load_raw_revision(
        repository_root / entry.raw_revision_file,
        source_id=entry.source_id,
        revision_id=entry.revision_id,
    )


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Raw revision {key!r} must be non-empty text.")
    return value


def _mapping_text(record: Mapping[object, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Assessment {key!r} must be non-empty text.")
    return value


def _require_unique(values: Sequence[str], *, label: str, context: Path) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{context} contains duplicate {label} values.")
