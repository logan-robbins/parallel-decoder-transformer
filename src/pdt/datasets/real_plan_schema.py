"""Strict source-grounded schema for three-lane long-form training data."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pdt.datasets.historical_source import HistoricalSource


__all__ = [
    "AtomicFact",
    "FactExtractionOutput",
    "FactRole",
    "JointPlanOutput",
    "REAL_PLAN_SCHEMA",
    "RealPlanExample",
    "validate_fact_extraction",
    "validate_real_plan_example",
]

REAL_PLAN_SCHEMA = "pdt-real-plan-v2"
MIN_EXTRACTED_FACTS = 18
MAX_EXTRACTED_FACTS = 48
MIN_FACT_SOURCE_PARAGRAPHS = 12
MIN_FACT_SOURCE_SECTIONS = 4


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProvenanceSpan(StrictModel):
    paragraph_id: str = Field(min_length=1, max_length=80)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    exact_quote: str = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_offsets(self) -> ProvenanceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("Provenance end_char must be greater than start_char.")
        if len(set(self.reference_ids)) != len(self.reference_ids):
            raise ValueError("Provenance reference_ids must not contain duplicates.")
        return self


class AtomicFact(StrictModel):
    fact_id: str = Field(pattern=r"^fact_[0-9]{3}$")
    statement: str = Field(min_length=10, max_length=800)
    importance: int = Field(ge=1, le=5)
    provenance: list[ProvenanceSpan] = Field(min_length=1, max_length=4)
    hard_negative: str = Field(min_length=10, max_length=800)


class FactExtractionOutput(StrictModel):
    source_id: str = Field(min_length=1, max_length=160)
    source_revision_id: int = Field(gt=0)
    source_model_visible_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    facts: list[AtomicFact] = Field(
        min_length=MIN_EXTRACTED_FACTS,
        max_length=MAX_EXTRACTED_FACTS,
    )

    @model_validator(mode="after")
    def validate_fact_ids(self) -> FactExtractionOutput:
        actual = [fact.fact_id for fact in self.facts]
        expected = [f"fact_{index:03d}" for index in range(len(self.facts))]
        if actual != expected:
            raise ValueError(
                f"Atomic fact IDs must be consecutive and ordered as {expected}; got {actual}."
            )
        return self


class FactRole(StrEnum):
    OWNER = "OWNER"
    REFERENCE = "REFERENCE"
    ABSENT = "ABSENT"


class CrossPlanDependency(StrictModel):
    source_plan_id: str = Field(pattern=r"^plan_[0-2]$")
    source_node_id: str = Field(pattern=r"^node_[0-7]$")
    source_target_paragraph_id: str = Field(min_length=1, max_length=80)
    source_evidence_quote: str = Field(min_length=20, max_length=1000)
    required_by_target_paragraph_id: str = Field(min_length=1, max_length=80)
    required_target_evidence_quote: str = Field(min_length=20, max_length=1000)
    fact_ids: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_fact_ids(self) -> CrossPlanDependency:
        if len(set(self.fact_ids)) != len(self.fact_ids):
            raise ValueError("Dependency fact_ids must not contain duplicates.")
        return self


class OutlineNode(StrictModel):
    node_id: str = Field(pattern=r"^node_[0-7]$")
    objective: str = Field(min_length=20, max_length=800)
    owned_fact_ids: list[str] = Field(max_length=32)
    reference_fact_ids: list[str] = Field(max_length=32)
    dependencies: list[CrossPlanDependency] = Field(max_length=16)
    target_paragraph_ids: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_fact_sets(self) -> OutlineNode:
        owned = set(self.owned_fact_ids)
        referenced = set(self.reference_fact_ids)
        if len(owned) != len(self.owned_fact_ids):
            raise ValueError("owned_fact_ids must not contain duplicates.")
        if len(referenced) != len(self.reference_fact_ids):
            raise ValueError("reference_fact_ids must not contain duplicates.")
        if owned & referenced:
            raise ValueError("A node cannot both own and reference the same fact.")
        return self


class StreamPlan(StrictModel):
    plan_id: str = Field(pattern=r"^plan_[0-2]$")
    heading: str = Field(min_length=3, max_length=300)
    role_summary: str = Field(min_length=20, max_length=800)
    nodes: list[OutlineNode] = Field(min_length=4, max_length=8)

    @model_validator(mode="after")
    def validate_node_ids(self) -> StreamPlan:
        expected = [f"node_{index}" for index in range(len(self.nodes))]
        actual = [node.node_id for node in self.nodes]
        if actual != expected:
            raise ValueError(
                f"Plan nodes must be consecutively ordered as {expected}, got {actual}."
            )
        return self


class TargetParagraph(StrictModel):
    paragraph_id: str = Field(min_length=1, max_length=80)
    outline_node_id: str = Field(pattern=r"^node_[0-7]$")
    text: str = Field(min_length=200)


class LaneFactLabel(StrictModel):
    fact_id: str = Field(pattern=r"^fact_[0-9]{3}$")
    role: FactRole
    target_evidence_quote: str

    @model_validator(mode="after")
    def validate_evidence(self) -> LaneFactLabel:
        if self.role is FactRole.ABSENT and self.target_evidence_quote:
            raise ValueError("ABSENT facts must use an empty target_evidence_quote.")
        if self.role is not FactRole.ABSENT and not self.target_evidence_quote:
            raise ValueError("OWNER and REFERENCE facts require a target evidence quote.")
        return self


class TargetSection(StrictModel):
    plan_id: str = Field(pattern=r"^plan_[0-2]$")
    paragraphs: list[TargetParagraph] = Field(min_length=4, max_length=12)
    fact_labels: list[LaneFactLabel] = Field(
        min_length=MIN_EXTRACTED_FACTS,
        max_length=MAX_EXTRACTED_FACTS,
    )

    @model_validator(mode="after")
    def validate_target_ids(self) -> TargetSection:
        paragraph_ids = [paragraph.paragraph_id for paragraph in self.paragraphs]
        if len(set(paragraph_ids)) != len(paragraph_ids):
            raise ValueError("Target paragraph IDs must be unique within a section.")
        fact_ids = [label.fact_id for label in self.fact_labels]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("Every target section must label each fact exactly once.")
        return self

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)


class JointPlanOutput(StrictModel):
    source_id: str = Field(min_length=1, max_length=160)
    source_revision_id: int = Field(gt=0)
    source_model_visible_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expository_prompt: str = Field(min_length=20, max_length=1000)
    plans: list[StreamPlan] = Field(min_length=3, max_length=3)
    target_sections: list[TargetSection] = Field(min_length=3, max_length=3)
    presentation_order: list[str] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_plan_axes(self) -> JointPlanOutput:
        expected = {"plan_0", "plan_1", "plan_2"}
        plan_ids = [plan.plan_id for plan in self.plans]
        target_ids = [target.plan_id for target in self.target_sections]
        if set(plan_ids) != expected or len(set(plan_ids)) != 3:
            raise ValueError("plans must contain plan_0, plan_1, and plan_2 exactly once.")
        if set(target_ids) != expected or len(set(target_ids)) != 3:
            raise ValueError(
                "target_sections must contain plan_0, plan_1, and plan_2 exactly once."
            )
        if set(self.presentation_order) != expected or len(set(self.presentation_order)) != 3:
            raise ValueError(
                "presentation_order must be a permutation of plan_0, plan_1, plan_2."
            )
        return self


class RealPlanExample(StrictModel):
    schema_version: str = Field(pattern=rf"^{REAL_PLAN_SCHEMA}$")
    source: HistoricalSource
    facts: FactExtractionOutput
    teacher: JointPlanOutput


def validate_real_plan_example(example: RealPlanExample) -> None:
    """Validate provenance, exact lane ownership, and plan/target alignment."""

    source = example.source
    facts = example.facts
    teacher = example.teacher
    validate_fact_extraction(source, facts)
    expected_identity = (
        source.source_id,
        source.revision_id,
        source.model_visible_sha256,
    )
    fact_identity = (
        facts.source_id,
        facts.source_revision_id,
        facts.source_model_visible_sha256,
    )
    teacher_identity = (
        teacher.source_id,
        teacher.source_revision_id,
        teacher.source_model_visible_sha256,
    )
    if fact_identity != expected_identity or teacher_identity != expected_identity:
        raise ValueError(
            "Source, fact extraction, and teacher output must name the exact same "
            "source ID, revision, and model-visible SHA-256."
        )

    fact_ids = {fact.fact_id for fact in facts.facts}
    plan_by_id = {plan.plan_id: plan for plan in teacher.plans}
    target_by_id = {target.plan_id: target for target in teacher.target_sections}
    owner_count = {fact_id: 0 for fact_id in fact_ids}
    for plan_id, plan in plan_by_id.items():
        target = target_by_id[plan_id]
        labels = {label.fact_id: label for label in target.fact_labels}
        if set(labels) != fact_ids:
            missing = sorted(fact_ids - set(labels))
            extra = sorted(set(labels) - fact_ids)
            raise ValueError(
                f"{plan_id} must label every extracted fact; missing={missing}, extra={extra}."
            )
        target_paragraphs = {
            paragraph.paragraph_id: paragraph for paragraph in target.paragraphs
        }
        node_by_id = {node.node_id: node for node in plan.nodes}
        owned_occurrences: dict[str, int] = {}
        reference_occurrences: dict[str, int] = {}
        dependency_occurrences: dict[str, int] = {}
        for paragraph in target.paragraphs:
            if paragraph.outline_node_id not in node_by_id:
                raise ValueError(
                    f"{plan_id} target paragraph references unknown outline node "
                    f"{paragraph.outline_node_id!r}."
                )
        for node in plan.nodes:
            if not (
                set(node.owned_fact_ids) | set(node.reference_fact_ids)
            ) <= fact_ids:
                raise ValueError(f"{plan_id}/{node.node_id} references unknown facts.")
            if not set(node.target_paragraph_ids) <= set(target_paragraphs):
                raise ValueError(f"{plan_id}/{node.node_id} references unknown target paragraphs.")
            assigned = {
                paragraph_id
                for paragraph_id, paragraph in target_paragraphs.items()
                if paragraph.outline_node_id == node.node_id
            }
            if set(node.target_paragraph_ids) != assigned:
                raise ValueError(
                    f"{plan_id}/{node.node_id} target paragraph mapping is not exact."
                )
            for fact_id in node.owned_fact_ids:
                owned_occurrences[fact_id] = owned_occurrences.get(fact_id, 0) + 1
                if labels[fact_id].role is not FactRole.OWNER:
                    raise ValueError(f"{plan_id}/{node.node_id} owned fact label is inconsistent.")
            for fact_id in node.reference_fact_ids:
                reference_occurrences[fact_id] = (
                    reference_occurrences.get(fact_id, 0) + 1
                )
                if labels[fact_id].role is not FactRole.REFERENCE:
                    raise ValueError(
                        f"{plan_id}/{node.node_id} reference fact label is inconsistent."
                    )
            for dependency in node.dependencies:
                _validate_dependency(
                    dependency,
                    target_plan_id=plan_id,
                    target_paragraphs=target_paragraphs,
                    plan_by_id=plan_by_id,
                    target_by_id=target_by_id,
                    fact_ids=fact_ids,
                    target_node=node,
                    labels=labels,
                )
                for fact_id in dependency.fact_ids:
                    dependency_occurrences[fact_id] = (
                        dependency_occurrences.get(fact_id, 0) + 1
                    )
        owner_labels = {
            fact_id
            for fact_id, label in labels.items()
            if label.role is FactRole.OWNER
        }
        reference_labels = {
            fact_id
            for fact_id, label in labels.items()
            if label.role is FactRole.REFERENCE
        }
        absent_labels = {
            fact_id
            for fact_id, label in labels.items()
            if label.role is FactRole.ABSENT
        }
        if set(owned_occurrences) != owner_labels or any(
            count != 1 for count in owned_occurrences.values()
        ):
            raise ValueError(
                f"{plan_id} must route every OWNER fact to exactly one outline node."
            )
        if set(reference_occurrences) != reference_labels or any(
            count != 1 for count in reference_occurrences.values()
        ):
            raise ValueError(
                f"{plan_id} must route every REFERENCE fact to exactly one outline node."
            )
        if len(absent_labels) < 2:
            raise ValueError(
                f"{plan_id} requires at least two positive facts that must remain absent."
            )
        if set(dependency_occurrences) != reference_labels or any(
            count != 1 for count in dependency_occurrences.values()
        ):
            raise ValueError(
                f"{plan_id} dependencies must cover every REFERENCE fact exactly once."
            )
        for label in target.fact_labels:
            if label.role is FactRole.OWNER:
                owner_count[label.fact_id] += 1
            if label.target_evidence_quote and label.target_evidence_quote not in target.text:
                raise ValueError(
                    f"{plan_id}/{label.fact_id} target evidence quote is not in its section."
                )
    incorrectly_owned = {
        fact_id: count for fact_id, count in owner_count.items() if count != 1
    }
    if incorrectly_owned:
        raise ValueError(
            "Every extracted fact must have exactly one physical-lane owner; "
            f"violations={incorrectly_owned}."
        )


def validate_fact_extraction(
    source: HistoricalSource,
    facts: FactExtractionOutput,
) -> None:
    """Validate exact source identity, citation lineage, and provenance coverage."""

    if (
        facts.source_id != source.source_id
        or facts.source_revision_id != source.revision_id
        or facts.source_model_visible_sha256 != source.model_visible_sha256
    ):
        raise ValueError(
            "Historical source and fact extraction must match by source ID, revision, "
            "and model-visible SHA-256."
        )
    paragraph_by_id = {
        paragraph.paragraph_id: paragraph
        for section in source.sections
        for paragraph in section.paragraphs
    }
    section_by_paragraph = {
        paragraph.paragraph_id: section.section_id
        for section in source.sections
        for paragraph in section.paragraphs
    }
    covered_paragraphs: set[str] = set()
    covered_sections: set[str] = set()
    for fact in facts.facts:
        if fact.hard_negative == fact.statement:
            raise ValueError(f"{fact.fact_id} hard negative must differ from the true fact.")
        for span in fact.provenance:
            source_paragraph = paragraph_by_id.get(span.paragraph_id)
            if source_paragraph is None:
                raise ValueError(
                    f"{fact.fact_id} references unknown source paragraph {span.paragraph_id!r}."
                )
            if not set(span.reference_ids) <= set(source_paragraph.reference_ids):
                raise ValueError(
                    f"{fact.fact_id} cites references that are not attached to source "
                    f"paragraph {span.paragraph_id!r}."
                )
            if span.end_char > len(source_paragraph.text):
                raise ValueError(f"{fact.fact_id} provenance extends beyond its paragraph.")
            if (
                source_paragraph.text[span.start_char : span.end_char]
                != span.exact_quote
            ):
                raise ValueError(f"{fact.fact_id} provenance quote does not match source text.")
            covered_paragraphs.add(span.paragraph_id)
            covered_sections.add(section_by_paragraph[span.paragraph_id])
    if len(covered_paragraphs) < MIN_FACT_SOURCE_PARAGRAPHS:
        raise ValueError(
            f"Fact inventory must span at least {MIN_FACT_SOURCE_PARAGRAPHS} source "
            f"paragraphs; got {len(covered_paragraphs)}."
        )
    if len(covered_sections) < MIN_FACT_SOURCE_SECTIONS:
        raise ValueError(
            f"Fact inventory must span at least {MIN_FACT_SOURCE_SECTIONS} source "
            f"sections; got {len(covered_sections)}."
        )


def _validate_dependency(
    dependency: CrossPlanDependency,
    *,
    target_plan_id: str,
    target_paragraphs: dict[str, TargetParagraph],
    plan_by_id: dict[str, StreamPlan],
    target_by_id: dict[str, TargetSection],
    fact_ids: set[str],
    target_node: OutlineNode,
    labels: dict[str, LaneFactLabel],
) -> None:
    if dependency.source_plan_id == target_plan_id:
        raise ValueError("Cross-plan dependencies must name a sibling plan.")
    source_plan = plan_by_id.get(dependency.source_plan_id)
    if source_plan is None:
        raise ValueError("Cross-plan dependency names an unknown source plan.")
    source_node_by_id = {node.node_id: node for node in source_plan.nodes}
    if dependency.source_node_id not in source_node_by_id:
        raise ValueError("Cross-plan dependency names an unknown source node.")
    source_node = source_node_by_id[dependency.source_node_id]
    source_target = target_by_id[dependency.source_plan_id]
    source_target_by_id = {
        paragraph.paragraph_id: paragraph for paragraph in source_target.paragraphs
    }
    source_indices = {
        paragraph.paragraph_id: index
        for index, paragraph in enumerate(source_target.paragraphs)
    }
    target_indices = {
        paragraph.paragraph_id: index
        for index, paragraph in enumerate(target_paragraphs.values())
    }
    if dependency.source_target_paragraph_id not in source_indices:
        raise ValueError("Cross-plan dependency names an unknown source paragraph.")
    if dependency.required_by_target_paragraph_id not in target_indices:
        raise ValueError("Cross-plan dependency names an unknown target paragraph.")
    if not set(dependency.fact_ids) <= fact_ids:
        raise ValueError("Cross-plan dependency names unknown facts.")
    if (
        source_indices[dependency.source_target_paragraph_id]
        >= target_indices[dependency.required_by_target_paragraph_id]
    ):
        raise ValueError(
            "Cross-plan dependencies must point from an earlier sibling paragraph "
            "to a later receiver paragraph."
        )
    source_paragraph = source_target_by_id[dependency.source_target_paragraph_id]
    target_paragraph = target_paragraphs[dependency.required_by_target_paragraph_id]
    if source_paragraph.outline_node_id != dependency.source_node_id:
        raise ValueError("Dependency source paragraph does not belong to source_node_id.")
    if target_paragraph.outline_node_id != target_node.node_id:
        raise ValueError("Dependency receiver paragraph does not belong to its target node.")
    if dependency.source_evidence_quote not in source_paragraph.text:
        raise ValueError("Dependency source evidence quote is not in its source paragraph.")
    if dependency.required_target_evidence_quote not in target_paragraph.text:
        raise ValueError(
            "Dependency receiver evidence quote is not in its required target paragraph."
        )
    source_labels = {
        label.fact_id: label for label in source_target.fact_labels
    }
    for fact_id in dependency.fact_ids:
        if fact_id not in source_node.owned_fact_ids:
            raise ValueError("Dependency facts must be owned by the named source node.")
        if fact_id not in target_node.reference_fact_ids:
            raise ValueError("Dependency facts must be referenced by the receiver node.")
        if source_labels[fact_id].role is not FactRole.OWNER:
            raise ValueError("Dependency source fact must have OWNER role.")
        if labels[fact_id].role is not FactRole.REFERENCE:
            raise ValueError("Dependency receiver fact must have REFERENCE role.")
