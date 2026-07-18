from __future__ import annotations

import hashlib
import itertools
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from model_intrinsic_parallel.wikipedia_source import (
    SourceParagraph,
    StrictModel,
    WikipediaSourceDocument,
    render_model_visible_text,
)


MIN_FACTS = 18
MAX_FACTS = 48
LANE_COUNT = 3
MIN_OWNER_FACTS_PER_LANE = 5
MIN_TARGET_PARAGRAPHS = 4
MAX_TARGET_PARAGRAPHS = 6
MIN_TARGET_PARAGRAPH_WORDS = 110
MIN_TARGET_TOKENS = 700
MAX_TARGET_TOKENS = 1_000
PhysicalDecoder = Literal["D1", "D2", "D3"]
PHYSICAL_DECODERS: tuple[PhysicalDecoder, PhysicalDecoder, PhysicalDecoder] = (
    "D1",
    "D2",
    "D3",
)


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class CurationFact(StrictModel):
    fact_id: str = Field(pattern=r"^f[0-9]{3}$")
    statement: str = Field(min_length=20, max_length=300)
    hard_negative: str = Field(min_length=20, max_length=300)
    importance: int = Field(ge=1, le=5)
    source_exact_quote: str = Field(min_length=20, max_length=1_600)


class CurationPlanNode(StrictModel):
    node_id: str = Field(pattern=r"^n[0-9]{2}$")
    target_paragraph_id: str = Field(pattern=r"^t[0-9]{2}$")
    purpose: str = Field(min_length=20, max_length=500)
    owner_fact_ids: tuple[str, ...] = Field(min_length=1)


class CurationTargetParagraph(StrictModel):
    paragraph_id: str = Field(pattern=r"^t[0-9]{2}$")
    text: str = Field(min_length=500)
    support_fact_ids: tuple[str, ...] = Field(min_length=1)


class CurationFactRealization(StrictModel):
    fact_id: str
    role: Literal["OWNER", "REFERENCE"]
    target_paragraph_id: str
    target_exact_quote: str = Field(min_length=20, max_length=800)


class CurationLane(StrictModel):
    lane_id: str = Field(pattern=r"^s[0-9]{2}$")
    focus: str = Field(min_length=20, max_length=300)
    selection_rationale: str = Field(min_length=80, max_length=1_000)
    plan_nodes: tuple[CurationPlanNode, ...] = Field(
        min_length=MIN_TARGET_PARAGRAPHS,
        max_length=MAX_TARGET_PARAGRAPHS,
    )
    target_paragraphs: tuple[CurationTargetParagraph, ...] = Field(
        min_length=MIN_TARGET_PARAGRAPHS,
        max_length=MAX_TARGET_PARAGRAPHS,
    )
    fact_realizations: tuple[CurationFactRealization, ...] = Field(min_length=1)


class CurationDependency(StrictModel):
    fact_id: str
    source_lane_id: str
    source_target_paragraph_id: str
    source_exact_quote: str = Field(min_length=20, max_length=800)
    target_lane_id: str
    target_paragraph_id: str
    target_exact_quote: str = Field(min_length=20, max_length=800)
    reason: str = Field(min_length=40, max_length=800)


class CompetingPartition(StrictModel):
    label: str = Field(min_length=5, max_length=100)
    proposed_lane_focuses: tuple[str, str, str]
    rejection_reason: str = Field(min_length=80, max_length=1_000)


class TeacherAudit(StrictModel):
    article_quality: Literal["featured-article"]
    quality_basis: str = Field(min_length=80, max_length=1_000)
    representative_reference_ids: tuple[str, ...] = Field(min_length=6)
    reference_quality_conclusion: str = Field(min_length=150, max_length=2_000)
    claim_audit_conclusion: str = Field(min_length=150, max_length=2_000)
    decomposition_audit_conclusion: str = Field(min_length=150, max_length=2_000)


class InspectionCuration(StrictModel):
    schema_version: Literal["model-intrinsic-parallel-inspection-curation-v1"]
    source_id: str
    question: str = Field(min_length=40, max_length=500)
    decomposition_rationale: str = Field(min_length=150, max_length=2_000)
    competing_partitions: tuple[CompetingPartition, ...] = Field(min_length=2)
    teacher_audit: TeacherAudit
    facts: tuple[CurationFact, ...] = Field(min_length=MIN_FACTS, max_length=MAX_FACTS)
    lanes: tuple[CurationLane, ...] = Field(
        min_length=LANE_COUNT, max_length=LANE_COUNT
    )
    presentation_order: tuple[str, str, str]
    dependencies: tuple[CurationDependency, ...]

    @model_validator(mode="after")
    def validate_identifiers(self) -> InspectionCuration:
        fact_ids = [fact.fact_id for fact in self.facts]
        expected_fact_ids = [f"f{index:03d}" for index in range(1, len(self.facts) + 1)]
        if fact_ids != expected_fact_ids:
            raise ValueError(
                "facts must use consecutive IDs f001..fNNN in source order"
            )
        if any(fact.statement == fact.hard_negative for fact in self.facts):
            raise ValueError("a fact hard negative cannot equal its true statement")

        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("semantic lane IDs must be unique")
        if set(lane_ids) & set(PHYSICAL_DECODERS):
            raise ValueError("semantic lane IDs cannot be physical decoder IDs")
        if len({lane.focus.casefold() for lane in self.lanes}) != LANE_COUNT:
            raise ValueError("all three semantic lane focuses must be distinct")
        if set(self.presentation_order) != set(lane_ids):
            raise ValueError(
                "presentation_order must contain each semantic lane exactly once"
            )
        return self


class FactEvidence(StrictModel):
    paragraph_id: str
    section_path: tuple[str, ...]
    exact_quote: str
    char_start: int
    char_end: int
    citation_ids: tuple[str, ...]


class TrainingFact(StrictModel):
    fact_id: str
    statement: str
    hard_negative: str
    importance: int
    owner_lane_id: str
    evidence: FactEvidence


class FactLabel(StrictModel):
    fact_id: str
    role: Literal["OWNER", "REFERENCE", "ABSENT"]
    target_paragraph_id: str | None
    target_exact_quote: str


class TrainingPlanNode(StrictModel):
    node_id: str
    target_paragraph_id: str
    purpose: str
    owner_fact_ids: tuple[str, ...]


class TrainingLane(StrictModel):
    lane_id: str
    focus: str
    selection_rationale: str
    plan_nodes: tuple[TrainingPlanNode, ...]
    target_paragraphs: tuple[CurationTargetParagraph, ...]
    target_text: str
    fact_labels: tuple[FactLabel, ...]


class TrainingDependency(StrictModel):
    fact_id: str
    source_lane_id: str
    source_target_paragraph_id: str
    source_exact_quote: str
    target_lane_id: str
    target_paragraph_id: str
    target_exact_quote: str
    reason: str


class PhysicalBinding(StrictModel):
    semantic_to_physical: dict[str, PhysicalDecoder]


class TokenizedSequence(StrictModel):
    token_count: int
    input_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_ids: tuple[int, ...]

    @model_validator(mode="after")
    def validate_tokens(self) -> TokenizedSequence:
        if self.token_count != len(self.input_ids):
            raise ValueError(
                f"token_count={self.token_count} does not match "
                f"{len(self.input_ids)} stored token IDs"
            )
        if any(input_id < 0 or input_id > 2**32 - 1 for input_id in self.input_ids):
            raise ValueError("stored token IDs must be unsigned 32-bit integers")
        packed_ids = b"".join(
            struct.pack("<I", input_id) for input_id in self.input_ids
        )
        actual_hash = hashlib.sha256(packed_ids).hexdigest()
        if actual_hash != self.input_ids_sha256:
            raise ValueError("stored token IDs do not match input_ids_sha256")
        return self


class Tokenization(StrictModel):
    tokenizer_model: str
    tokenizer_revision: str
    add_special_tokens: Literal[False]
    eos_token_id: int
    planner_input: TokenizedSequence
    decoder_targets: dict[str, TokenizedSequence]


class TrainingExample(StrictModel):
    schema_version: Literal["model-intrinsic-parallel-training-example-v1"]
    creation_method: Literal["manual-source-grounded-inspection"]
    empirical_status: Literal["data-contract-inspection-only"]
    source_document: WikipediaSourceDocument
    question: str
    decomposition_rationale: str
    competing_partitions: tuple[CompetingPartition, ...]
    teacher_audit: TeacherAudit
    presentation_order: tuple[str, str, str]
    physical_decoder_assignment_is_unordered: Literal[True]
    physical_binding_policy: Literal["uniform-random-per-example-per-epoch"]
    allowed_physical_bindings: tuple[PhysicalBinding, ...]
    facts: tuple[TrainingFact, ...]
    lanes: tuple[TrainingLane, ...]
    dependencies: tuple[TrainingDependency, ...]
    tokenization: Tokenization

    @model_validator(mode="after")
    def validate_compiled_contract(self) -> TrainingExample:
        fact_ids = tuple(fact.fact_id for fact in self.facts)
        lane_ids = tuple(lane.lane_id for lane in self.lanes)
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("compiled fact IDs must be unique")
        if len(lane_ids) != LANE_COUNT or len(lane_ids) != len(set(lane_ids)):
            raise ValueError(
                "compiled example must contain three unique semantic lanes"
            )
        if set(self.presentation_order) != set(lane_ids):
            raise ValueError(
                "presentation_order must contain each semantic lane exactly once"
            )

        expected_bindings = {
            tuple(zip(lane_ids, physical_order, strict=True))
            for physical_order in itertools.permutations(PHYSICAL_DECODERS)
        }
        actual_bindings = {
            tuple(
                (lane_id, binding.semantic_to_physical.get(lane_id))
                for lane_id in lane_ids
            )
            for binding in self.allowed_physical_bindings
        }
        if actual_bindings != expected_bindings:
            raise ValueError(
                "allowed physical bindings must contain all six permutations"
            )

        owner_by_fact_id = {fact.fact_id: fact.owner_lane_id for fact in self.facts}
        lane_by_id = {lane.lane_id: lane for lane in self.lanes}
        for lane in self.lanes:
            label_fact_ids = tuple(label.fact_id for label in lane.fact_labels)
            if label_fact_ids != fact_ids:
                raise ValueError(
                    f"{lane.lane_id} must label every fact exactly once in canonical order"
                )
            for label in lane.fact_labels:
                expected_owner = owner_by_fact_id[label.fact_id]
                if label.role == "OWNER" and expected_owner != lane.lane_id:
                    raise ValueError(
                        f"{lane.lane_id} cannot own {label.fact_id}; "
                        f"compiled owner is {expected_owner}"
                    )
                if label.role == "REFERENCE" and expected_owner == lane.lane_id:
                    raise ValueError(
                        f"{lane.lane_id} cannot reference its own fact {label.fact_id}"
                    )
                if label.role == "ABSENT" and (
                    label.target_paragraph_id is not None or label.target_exact_quote
                ):
                    raise ValueError("ABSENT fact labels cannot carry target evidence")
            supported_by_paragraph = {
                paragraph.paragraph_id: set(paragraph.support_fact_ids)
                for paragraph in lane.target_paragraphs
            }
            realized_by_paragraph: dict[str, set[str]] = {
                paragraph.paragraph_id: set() for paragraph in lane.target_paragraphs
            }
            for label in lane.fact_labels:
                if label.target_paragraph_id is not None:
                    realized_by_paragraph[label.target_paragraph_id].add(label.fact_id)
            if supported_by_paragraph != realized_by_paragraph:
                raise ValueError(
                    f"{lane.lane_id} paragraph support sets do not match compiled fact labels"
                )

        expected_references = {
            (lane.lane_id, label.fact_id)
            for lane in self.lanes
            for label in lane.fact_labels
            if label.role == "REFERENCE"
        }
        presentation_index = {
            lane_id: index for index, lane_id in enumerate(self.presentation_order)
        }
        seen_references: set[tuple[str, str]] = set()
        for dependency in self.dependencies:
            source_lane = lane_by_id.get(dependency.source_lane_id)
            target_lane = lane_by_id.get(dependency.target_lane_id)
            if source_lane is None or target_lane is None:
                raise ValueError(
                    "compiled dependency references an unknown semantic lane"
                )
            if (
                presentation_index[dependency.source_lane_id]
                >= presentation_index[dependency.target_lane_id]
            ):
                raise ValueError(
                    f"compiled dependency {dependency.fact_id} violates presentation order"
                )
            if owner_by_fact_id.get(dependency.fact_id) != dependency.source_lane_id:
                raise ValueError(
                    f"compiled dependency source does not own {dependency.fact_id}"
                )
            source_label = next(
                label
                for label in source_lane.fact_labels
                if label.fact_id == dependency.fact_id
            )
            target_label = next(
                label
                for label in target_lane.fact_labels
                if label.fact_id == dependency.fact_id
            )
            if (
                source_label.role != "OWNER"
                or source_label.target_paragraph_id
                != dependency.source_target_paragraph_id
                or source_label.target_exact_quote != dependency.source_exact_quote
            ):
                raise ValueError(
                    f"compiled dependency owner evidence is invalid for {dependency.fact_id}"
                )
            if (
                target_label.role != "REFERENCE"
                or target_label.target_paragraph_id != dependency.target_paragraph_id
                or target_label.target_exact_quote != dependency.target_exact_quote
            ):
                raise ValueError(
                    f"compiled dependency reference evidence is invalid for "
                    f"{dependency.fact_id}"
                )
            source_index = _training_target_paragraph_index(
                source_lane, dependency.source_target_paragraph_id
            )
            target_index = _training_target_paragraph_index(
                target_lane, dependency.target_paragraph_id
            )
            if source_index >= target_index:
                raise ValueError(
                    f"compiled dependency {dependency.fact_id} is not delayed"
                )
            reference_key = (dependency.target_lane_id, dependency.fact_id)
            if reference_key in seen_references:
                raise ValueError(
                    f"compiled dependencies repeat target reference {reference_key}"
                )
            seen_references.add(reference_key)
        if seen_references != expected_references:
            raise ValueError(
                "compiled dependencies must cover every cross-lane reference exactly once"
            )

        if set(self.tokenization.decoder_targets) != set(lane_ids):
            raise ValueError("tokenized decoder targets must match semantic lane IDs")
        return self


def load_source_document(path: Path) -> WikipediaSourceDocument:
    return WikipediaSourceDocument.model_validate_json(path.read_text(encoding="utf-8"))


def load_curation(path: Path) -> InspectionCuration:
    return InspectionCuration.model_validate_json(path.read_text(encoding="utf-8"))


def compile_training_example(
    source: WikipediaSourceDocument,
    curation: InspectionCuration,
    *,
    tokenizer: Tokenizer,
    tokenizer_model: str,
    tokenizer_revision: str,
) -> TrainingExample:
    if curation.source_id != source.source.source_id:
        raise ValueError(
            f"curation source_id={curation.source_id!r} does not match "
            f"source document {source.source.source_id!r}"
        )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token ID")

    source_paragraph_by_fact_id = _resolve_fact_source_evidence(
        curation.facts, source.paragraphs
    )
    _validate_teacher_audit(
        curation.teacher_audit,
        source,
        tuple(source_paragraph_by_fact_id.values()),
    )

    lane_by_id = {lane.lane_id: lane for lane in curation.lanes}
    fact_ids = tuple(fact.fact_id for fact in curation.facts)
    owner_by_fact_id = _owner_by_fact_id(curation.lanes, fact_ids)
    realizations_by_lane = _validate_lanes(curation, owner_by_fact_id, tokenizer)
    _validate_dependencies(
        curation.dependencies,
        lane_by_id,
        owner_by_fact_id,
        realizations_by_lane,
        curation.presentation_order,
    )

    facts = tuple(
        _compile_fact(fact, source_paragraph_by_fact_id[fact.fact_id], owner_by_fact_id)
        for fact in curation.facts
    )
    lanes = tuple(
        _compile_lane(lane, fact_ids, realizations_by_lane[lane.lane_id])
        for lane in curation.lanes
    )
    dependencies = tuple(
        TrainingDependency.model_validate(dependency.model_dump())
        for dependency in curation.dependencies
    )
    allowed_bindings = _all_physical_bindings(tuple(lane_by_id))

    visible_source = render_model_visible_text(source.headings, source.paragraphs)
    planner_input_text = f"Question: {curation.question}\n\nSource:\n{visible_source}"
    planner_ids = tokenizer.encode(planner_input_text, add_special_tokens=False)
    target_sequences: dict[str, TokenizedSequence] = {}
    for lane in lanes:
        target_ids = tokenizer.encode(lane.target_text, add_special_tokens=False)
        target_ids.append(tokenizer.eos_token_id)
        target_sequences[lane.lane_id] = _tokenized_sequence(target_ids)

    return TrainingExample(
        schema_version="model-intrinsic-parallel-training-example-v1",
        creation_method="manual-source-grounded-inspection",
        empirical_status="data-contract-inspection-only",
        source_document=source,
        question=curation.question,
        decomposition_rationale=curation.decomposition_rationale,
        competing_partitions=curation.competing_partitions,
        teacher_audit=curation.teacher_audit,
        presentation_order=curation.presentation_order,
        physical_decoder_assignment_is_unordered=True,
        physical_binding_policy="uniform-random-per-example-per-epoch",
        allowed_physical_bindings=allowed_bindings,
        facts=facts,
        lanes=lanes,
        dependencies=dependencies,
        tokenization=Tokenization(
            tokenizer_model=tokenizer_model,
            tokenizer_revision=tokenizer_revision,
            add_special_tokens=False,
            eos_token_id=tokenizer.eos_token_id,
            planner_input=_tokenized_sequence(planner_ids),
            decoder_targets=target_sequences,
        ),
    )


def write_training_example(example: TrainingExample, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        example.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def _resolve_fact_source_evidence(
    facts: tuple[CurationFact, ...],
    paragraphs: tuple[SourceParagraph, ...],
) -> dict[str, SourceParagraph]:
    used_paragraphs: set[str] = set()
    used_top_level_sections: set[str] = set()
    paragraph_by_fact_id: dict[str, SourceParagraph] = {}
    for fact in facts:
        matching_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if fact.source_exact_quote in paragraph.text
        ]
        if len(matching_paragraphs) != 1:
            raise ValueError(
                f"{fact.fact_id} source quote must resolve to exactly one source paragraph; "
                f"found {[paragraph.paragraph_id for paragraph in matching_paragraphs]}"
            )
        paragraph = matching_paragraphs[0]
        if not paragraph.citation_ids:
            raise ValueError(f"{fact.fact_id} source paragraph has no local citations")
        occurrence_count = paragraph.text.count(fact.source_exact_quote)
        if occurrence_count != 1:
            raise ValueError(
                f"{fact.fact_id} source quote must occur exactly once in "
                f"{paragraph.paragraph_id}; found {occurrence_count}"
            )
        used_paragraphs.add(paragraph.paragraph_id)
        if paragraph.section_path:
            used_top_level_sections.add(paragraph.section_path[0])
        paragraph_by_fact_id[fact.fact_id] = paragraph

    if len(used_paragraphs) < 12:
        raise ValueError(
            f"facts must span at least 12 source paragraphs; found {len(used_paragraphs)}"
        )
    if len(used_top_level_sections) < 4:
        raise ValueError(
            f"facts must span at least four top-level sections; found "
            f"{len(used_top_level_sections)}"
        )
    return paragraph_by_fact_id


def _validate_teacher_audit(
    audit: TeacherAudit,
    source: WikipediaSourceDocument,
    evidence_paragraphs: tuple[SourceParagraph, ...],
) -> None:
    reference_ids = audit.representative_reference_ids
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("teacher audit repeats a representative reference ID")
    known_reference_ids = {citation.reference_id for citation in source.citations}
    unknown_reference_ids = set(reference_ids) - known_reference_ids
    if unknown_reference_ids:
        raise ValueError(
            "teacher audit references unknown source citations: "
            f"{sorted(unknown_reference_ids)}"
        )
    evidence_reference_ids = {
        reference_id
        for paragraph in evidence_paragraphs
        for reference_id in paragraph.citation_ids
    }
    unused_reference_ids = set(reference_ids) - evidence_reference_ids
    if unused_reference_ids:
        raise ValueError(
            "teacher audit representative citations must support selected fact "
            f"paragraphs: {sorted(unused_reference_ids)}"
        )


def _owner_by_fact_id(
    lanes: tuple[CurationLane, ...],
    fact_ids: tuple[str, ...],
) -> dict[str, str]:
    owner_candidates: dict[str, list[str]] = {fact_id: [] for fact_id in fact_ids}
    for lane in lanes:
        node_owner_ids = [
            fact_id for node in lane.plan_nodes for fact_id in node.owner_fact_ids
        ]
        if len(node_owner_ids) != len(set(node_owner_ids)):
            raise ValueError(
                f"{lane.lane_id} assigns an owner fact to multiple plan nodes"
            )
        if len(node_owner_ids) < MIN_OWNER_FACTS_PER_LANE:
            raise ValueError(
                f"{lane.lane_id} owns {len(node_owner_ids)} facts; "
                f"minimum is {MIN_OWNER_FACTS_PER_LANE}"
            )
        if len(node_owner_ids) > math.ceil(len(fact_ids) / 2):
            raise ValueError(
                f"{lane.lane_id} is a dominant lane with too many owned facts"
            )
        for fact_id in node_owner_ids:
            if fact_id not in owner_candidates:
                raise ValueError(f"{lane.lane_id} owns unknown fact {fact_id}")
            owner_candidates[fact_id].append(lane.lane_id)

    invalid_owners = {
        fact_id: owners
        for fact_id, owners in owner_candidates.items()
        if len(owners) != 1
    }
    if invalid_owners:
        raise ValueError(
            f"every fact must have exactly one owning lane: {invalid_owners}"
        )
    return {fact_id: owners[0] for fact_id, owners in owner_candidates.items()}


def _validate_lanes(
    curation: InspectionCuration,
    owner_by_fact_id: dict[str, str],
    tokenizer: Tokenizer,
) -> dict[str, dict[str, CurationFactRealization]]:
    known_fact_ids = set(owner_by_fact_id)
    result: dict[str, dict[str, CurationFactRealization]] = {}

    for lane in curation.lanes:
        paragraph_ids = [paragraph.paragraph_id for paragraph in lane.target_paragraphs]
        expected_paragraph_ids = [
            f"t{index:02d}" for index in range(len(lane.target_paragraphs))
        ]
        if paragraph_ids != expected_paragraph_ids:
            raise ValueError(
                f"{lane.lane_id} target paragraph IDs must be consecutive t00..tNN"
            )
        node_ids = [node.node_id for node in lane.plan_nodes]
        expected_node_ids = [f"n{index:02d}" for index in range(len(lane.plan_nodes))]
        if node_ids != expected_node_ids:
            raise ValueError(
                f"{lane.lane_id} plan node IDs must be consecutive n00..nNN"
            )
        node_paragraph_ids = [node.target_paragraph_id for node in lane.plan_nodes]
        if node_paragraph_ids != paragraph_ids:
            raise ValueError(
                f"{lane.lane_id} plan nodes must map one-to-one to target paragraphs"
            )

        paragraph_by_id = {
            paragraph.paragraph_id: paragraph for paragraph in lane.target_paragraphs
        }
        for paragraph in lane.target_paragraphs:
            word_count = len(paragraph.text.split())
            if word_count < MIN_TARGET_PARAGRAPH_WORDS:
                raise ValueError(
                    f"{lane.lane_id}/{paragraph.paragraph_id} has {word_count} words; "
                    f"minimum is {MIN_TARGET_PARAGRAPH_WORDS}"
                )

        target_text = "\n\n".join(
            paragraph.text for paragraph in lane.target_paragraphs
        )
        token_count = len(tokenizer.encode(target_text, add_special_tokens=False)) + 1
        if not MIN_TARGET_TOKENS <= token_count <= MAX_TARGET_TOKENS:
            raise ValueError(
                f"{lane.lane_id} target has {token_count} tokens including EOS; "
                f"required range is {MIN_TARGET_TOKENS}-{MAX_TARGET_TOKENS}"
            )

        realization_by_fact_id: dict[str, CurationFactRealization] = {}
        for realization in lane.fact_realizations:
            if realization.fact_id not in known_fact_ids:
                raise ValueError(
                    f"{lane.lane_id} realizes unknown fact {realization.fact_id}"
                )
            if realization.fact_id in realization_by_fact_id:
                raise ValueError(
                    f"{lane.lane_id} realizes {realization.fact_id} more than once"
                )
            target_paragraph = paragraph_by_id.get(realization.target_paragraph_id)
            if target_paragraph is None:
                raise ValueError(
                    f"{lane.lane_id}/{realization.fact_id} references missing target paragraph "
                    f"{realization.target_paragraph_id}"
                )
            occurrence_count = target_paragraph.text.count(
                realization.target_exact_quote
            )
            if occurrence_count != 1:
                raise ValueError(
                    f"{lane.lane_id}/{realization.fact_id} target quote must occur exactly "
                    f"once in {target_paragraph.paragraph_id}; found {occurrence_count}"
                )
            expected_role = (
                "OWNER"
                if owner_by_fact_id[realization.fact_id] == lane.lane_id
                else "REFERENCE"
            )
            if realization.role != expected_role:
                raise ValueError(
                    f"{lane.lane_id}/{realization.fact_id} role={realization.role}, "
                    f"expected {expected_role}"
                )
            realization_by_fact_id[realization.fact_id] = realization

        support_by_paragraph_id = {
            paragraph.paragraph_id: set(paragraph.support_fact_ids)
            for paragraph in lane.target_paragraphs
        }
        for paragraph in lane.target_paragraphs:
            if len(paragraph.support_fact_ids) != len(set(paragraph.support_fact_ids)):
                raise ValueError(
                    f"{lane.lane_id}/{paragraph.paragraph_id} repeats a support fact ID"
                )
            unknown_support = set(paragraph.support_fact_ids) - known_fact_ids
            if unknown_support:
                raise ValueError(
                    f"{lane.lane_id}/{paragraph.paragraph_id} has unknown support facts: "
                    f"{sorted(unknown_support)}"
                )
        realized_by_paragraph_id: dict[str, set[str]] = {
            paragraph_id: set() for paragraph_id in paragraph_by_id
        }
        for realization in realization_by_fact_id.values():
            realized_by_paragraph_id[realization.target_paragraph_id].add(
                realization.fact_id
            )
        if support_by_paragraph_id != realized_by_paragraph_id:
            raise ValueError(
                f"{lane.lane_id} paragraph support sets must exactly equal its "
                "OWNER/REFERENCE realizations"
            )

        owner_fact_ids = {
            fact_id
            for fact_id, owner_lane_id in owner_by_fact_id.items()
            if owner_lane_id == lane.lane_id
        }
        realized_owner_fact_ids = {
            fact_id
            for fact_id, realization in realization_by_fact_id.items()
            if realization.role == "OWNER"
        }
        if realized_owner_fact_ids != owner_fact_ids:
            raise ValueError(
                f"{lane.lane_id} owner realizations do not match its plan ownership: "
                f"missing={sorted(owner_fact_ids - realized_owner_fact_ids)}, "
                f"extra={sorted(realized_owner_fact_ids - owner_fact_ids)}"
            )

        absent_count = len(known_fact_ids - realization_by_fact_id.keys())
        if absent_count < 2:
            raise ValueError(f"{lane.lane_id} must leave at least two facts absent")

        result[lane.lane_id] = realization_by_fact_id
    return result


def _validate_dependencies(
    dependencies: tuple[CurationDependency, ...],
    lane_by_id: dict[str, CurationLane],
    owner_by_fact_id: dict[str, str],
    realizations_by_lane: dict[str, dict[str, CurationFactRealization]],
    presentation_order: tuple[str, str, str],
) -> None:
    presentation_index = {
        lane_id: index for index, lane_id in enumerate(presentation_order)
    }
    expected_references = {
        (lane_id, fact_id)
        for lane_id, realizations in realizations_by_lane.items()
        for fact_id, realization in realizations.items()
        if realization.role == "REFERENCE"
    }
    seen_references: set[tuple[str, str]] = set()
    for dependency in dependencies:
        source_lane = lane_by_id.get(dependency.source_lane_id)
        target_lane = lane_by_id.get(dependency.target_lane_id)
        if source_lane is None or target_lane is None:
            raise ValueError(
                f"dependency references an unknown semantic lane: {dependency}"
            )
        if dependency.source_lane_id == dependency.target_lane_id:
            raise ValueError(
                "cross-lane dependency cannot begin and end in the same lane"
            )
        if (
            presentation_index[dependency.source_lane_id]
            >= presentation_index[dependency.target_lane_id]
        ):
            raise ValueError(
                f"dependency {dependency.fact_id} violates presentation order: "
                f"{dependency.source_lane_id} must precede {dependency.target_lane_id}"
            )
        if owner_by_fact_id.get(dependency.fact_id) != dependency.source_lane_id:
            raise ValueError(
                f"dependency source lane does not own {dependency.fact_id}: {dependency}"
            )
        source_realization = realizations_by_lane[dependency.source_lane_id].get(
            dependency.fact_id
        )
        target_realization = realizations_by_lane[dependency.target_lane_id].get(
            dependency.fact_id
        )
        if source_realization is None or source_realization.role != "OWNER":
            raise ValueError(f"dependency lacks owner realization: {dependency}")
        if target_realization is None or target_realization.role != "REFERENCE":
            raise ValueError(f"dependency lacks reference realization: {dependency}")
        if (
            source_realization.target_paragraph_id
            != dependency.source_target_paragraph_id
            or source_realization.target_exact_quote != dependency.source_exact_quote
        ):
            raise ValueError(
                f"dependency source evidence does not match realization: {dependency}"
            )
        if (
            target_realization.target_paragraph_id != dependency.target_paragraph_id
            or target_realization.target_exact_quote != dependency.target_exact_quote
        ):
            raise ValueError(
                f"dependency target evidence does not match realization: {dependency}"
            )

        source_index = _target_paragraph_index(
            source_lane, dependency.source_target_paragraph_id
        )
        target_index = _target_paragraph_index(
            target_lane, dependency.target_paragraph_id
        )
        if source_index >= target_index:
            raise ValueError(
                f"dependency {dependency.fact_id} is not delayed: source paragraph "
                f"{source_index}, target paragraph {target_index}"
            )
        reference_key = (dependency.target_lane_id, dependency.fact_id)
        if reference_key in seen_references:
            raise ValueError(
                f"duplicate dependency for target reference {reference_key}"
            )
        seen_references.add(reference_key)

    if seen_references != expected_references:
        raise ValueError(
            "dependencies must cover every cross-lane reference exactly once: "
            f"missing={sorted(expected_references - seen_references)}, "
            f"extra={sorted(seen_references - expected_references)}"
        )


def _compile_fact(
    fact: CurationFact,
    paragraph: SourceParagraph,
    owner_by_fact_id: dict[str, str],
) -> TrainingFact:
    char_start = paragraph.text.index(fact.source_exact_quote)
    return TrainingFact(
        fact_id=fact.fact_id,
        statement=fact.statement,
        hard_negative=fact.hard_negative,
        importance=fact.importance,
        owner_lane_id=owner_by_fact_id[fact.fact_id],
        evidence=FactEvidence(
            paragraph_id=paragraph.paragraph_id,
            section_path=paragraph.section_path,
            exact_quote=fact.source_exact_quote,
            char_start=char_start,
            char_end=char_start + len(fact.source_exact_quote),
            citation_ids=paragraph.citation_ids,
        ),
    )


def _compile_lane(
    lane: CurationLane,
    fact_ids: tuple[str, ...],
    realization_by_fact_id: dict[str, CurationFactRealization],
) -> TrainingLane:
    labels: list[FactLabel] = []
    for fact_id in fact_ids:
        realization = realization_by_fact_id.get(fact_id)
        if realization is None:
            labels.append(
                FactLabel(
                    fact_id=fact_id,
                    role="ABSENT",
                    target_paragraph_id=None,
                    target_exact_quote="",
                )
            )
        else:
            labels.append(
                FactLabel(
                    fact_id=fact_id,
                    role=realization.role,
                    target_paragraph_id=realization.target_paragraph_id,
                    target_exact_quote=realization.target_exact_quote,
                )
            )
    return TrainingLane(
        lane_id=lane.lane_id,
        focus=lane.focus,
        selection_rationale=lane.selection_rationale,
        plan_nodes=tuple(
            TrainingPlanNode.model_validate(node.model_dump())
            for node in lane.plan_nodes
        ),
        target_paragraphs=lane.target_paragraphs,
        target_text="\n\n".join(paragraph.text for paragraph in lane.target_paragraphs),
        fact_labels=tuple(labels),
    )


def _all_physical_bindings(lane_ids: tuple[str, ...]) -> tuple[PhysicalBinding, ...]:
    if len(lane_ids) != LANE_COUNT or len(set(lane_ids)) != LANE_COUNT:
        raise ValueError(
            "physical bindings require exactly three unique semantic lanes"
        )
    return tuple(
        PhysicalBinding(
            semantic_to_physical=dict(zip(lane_ids, physical_order, strict=True))
        )
        for physical_order in itertools.permutations(PHYSICAL_DECODERS)
    )


def _tokenized_sequence(input_ids: list[int]) -> TokenizedSequence:
    if any(input_id < 0 for input_id in input_ids):
        raise ValueError("token IDs must be non-negative")
    packed_ids = b"".join(struct.pack("<I", input_id) for input_id in input_ids)
    return TokenizedSequence(
        token_count=len(input_ids),
        input_ids_sha256=hashlib.sha256(packed_ids).hexdigest(),
        input_ids=tuple(input_ids),
    )


def _target_paragraph_index(lane: CurationLane, paragraph_id: str) -> int:
    paragraph_ids = [paragraph.paragraph_id for paragraph in lane.target_paragraphs]
    try:
        return paragraph_ids.index(paragraph_id)
    except ValueError as exc:
        raise ValueError(
            f"{lane.lane_id} has no target paragraph {paragraph_id}"
        ) from exc


def _training_target_paragraph_index(
    lane: TrainingLane,
    paragraph_id: str,
) -> int:
    paragraph_ids = [paragraph.paragraph_id for paragraph in lane.target_paragraphs]
    try:
        return paragraph_ids.index(paragraph_id)
    except ValueError as exc:
        raise ValueError(
            f"{lane.lane_id} has no target paragraph {paragraph_id}"
        ) from exc


def summarize_example(example: TrainingExample) -> str:
    owner_counts = Counter(fact.owner_lane_id for fact in example.facts)
    reference_counts = {
        lane.lane_id: sum(label.role == "REFERENCE" for label in lane.fact_labels)
        for lane in example.lanes
    }
    target_tokens = {
        lane_id: sequence.token_count
        for lane_id, sequence in example.tokenization.decoder_targets.items()
    }
    fact_paragraph_count = len({fact.evidence.paragraph_id for fact in example.facts})
    fact_section_count = len(
        {
            fact.evidence.section_path[0]
            for fact in example.facts
            if fact.evidence.section_path
        }
    )
    return (
        f"source={example.source_document.source.title!r} "
        f"facts={len(example.facts)} "
        f"fact_paragraphs={fact_paragraph_count} "
        f"fact_sections={fact_section_count} "
        f"owner_counts={dict(owner_counts)} "
        f"reference_counts={reference_counts} "
        f"dependencies={len(example.dependencies)} "
        f"target_tokens={target_tokens} "
        f"physical_bindings={len(example.allowed_physical_bindings)}"
    )


def load_training_example(path: Path) -> TrainingExample:
    return TrainingExample.model_validate_json(path.read_text(encoding="utf-8"))
