"""Blinded human adjudication for lane-exact generated-fact evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pdt.datasets.historical_source import canonical_json_bytes, sha256_file
from pdt.datasets.immutable_io import write_bytes_new, write_jsonl_new
from pdt.datasets.real_plan_schema import FactRole, RealPlanExample, validate_real_plan_example
from pdt.evaluation.fact_ownership import FactOwnershipCounts
from pdt.evaluation.paired_causal import bootstrap_mean


MANUAL_AUDIT_ITEM_SCHEMA = "pdt-manual-fact-audit-item-v1"
MANUAL_AUDIT_KEY_SCHEMA = "pdt-manual-fact-audit-key-v2"
MANUAL_ANNOTATION_SCHEMA = "pdt-manual-fact-annotation-v1"
MANUAL_AUDIT_RESULT_SCHEMA = "pdt-manual-fact-audit-result-v2"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryKind(StrEnum):
    POSITIVE_FACT = "positive_fact"
    HARD_NEGATIVE = "hard_negative"


class ManualJudgment(StrEnum):
    ENTAILED = "ENTAILED"
    NOT_ENTAILED = "NOT_ENTAILED"
    UNCERTAIN = "UNCERTAIN"


class ManualAuditItem(_StrictModel):
    schema_version: str = Field(pattern=rf"^{MANUAL_AUDIT_ITEM_SCHEMA}$")
    item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    premise_text: str = Field(min_length=20)
    hypothesis: str = Field(min_length=10, max_length=800)


class ManualAuditKeyRecord(_StrictModel):
    item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    example_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    physical_lane: int = Field(ge=0, le=2)
    fact_id: str = Field(pattern=r"^fact_[0-9]{3}$")
    query_kind: QueryKind
    expected_role: FactRole
    unmoved_expected_role: FactRole | None = None


class ManualAuditKey(_StrictModel):
    schema_version: str = Field(pattern=rf"^{MANUAL_AUDIT_KEY_SCHEMA}$")
    generation_evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_examples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    queue_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    randomization_seed: int
    physical_lane_order: tuple[str, str, str]
    active_conditions: tuple[str, ...] = Field(min_length=1)
    records: tuple[ManualAuditKeyRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_key(self) -> ManualAuditKey:
        item_ids = [record.item_id for record in self.records]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Manual audit key item IDs must be unique.")
        return self


class ManualAnnotation(_StrictModel):
    schema_version: str = Field(pattern=rf"^{MANUAL_ANNOTATION_SCHEMA}$")
    item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotator_id: str = Field(min_length=1, max_length=200)
    judgment: ManualJudgment
    evidence_quote: str = Field(default="", max_length=2_000)
    notes: str = Field(default="", max_length=2_000)

    @model_validator(mode="after")
    def validate_evidence(self) -> ManualAnnotation:
        if self.judgment is ManualJudgment.ENTAILED and len(self.evidence_quote) < 5:
            raise ValueError("ENTAILED annotations require an evidence quote.")
        if self.judgment is not ManualJudgment.ENTAILED and self.evidence_quote:
            raise ValueError("Only ENTAILED annotations may contain an evidence quote.")
        return self


def export_blinded_fact_audit(
    *,
    generation_evaluation_path: Path,
    raw_examples_path: Path,
    queue_path: Path,
    key_path: Path,
    randomization_seed: int,
) -> int:
    """Export every condition/lane/fact decision without exposing its expected role."""

    if queue_path.exists() or key_path.exists():
        raise FileExistsError("Refusing to replace an existing manual audit queue or key.")
    evaluation = _load_json_object(generation_evaluation_path)
    if evaluation.get("schema_version") != "pdt-real-plan-generation-eval-v2":
        raise ValueError("Manual audit export requires generation evaluation schema v2.")
    if evaluation.get("human_fact_audit_required") is not True:
        raise ValueError("Generation evaluation does not declare a required human audit.")
    raw_by_id = _load_raw_examples(raw_examples_path)
    lane_order = _three_texts(
        evaluation.get("physical_lane_order"),
        context="physical_lane_order",
    )
    active_conditions = _text_tuple(
        evaluation.get("active_conditions"),
        context="active_conditions",
    )
    documents = evaluation.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Generation evaluation contains no documents.")

    queue_by_id: dict[str, ManualAuditItem] = {}
    key_records: list[ManualAuditKeyRecord] = []
    observed_examples: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Generation evaluation document entries must be objects.")
        example_id = _required_text(document, "example_id", context="document")
        example = raw_by_id.get(example_id)
        if example is None:
            raise ValueError(f"Generation evaluation names unknown example {example_id!r}.")
        if example_id in observed_examples:
            raise ValueError(f"Generation evaluation repeats example {example_id!r}.")
        observed_examples.add(example_id)
        condition_rows = _required_mapping(document, "conditions", context=example_id)
        if set(condition_rows) != set(active_conditions):
            raise ValueError(
                f"{example_id} conditions do not exactly match active_conditions."
            )
        owner_teacher, references_teacher, absences_teacher = _teacher_roles(example)
        facts = example.facts.facts
        lane_swap_unmoved_mapping = (
            _lane_mapping(
                _mapping(
                    condition_rows["learned_plan"],
                    context=f"{example_id}/learned_plan",
                ),
                context=f"{example_id}/learned_plan",
            )
            if "lane_swap" in active_conditions
            else None
        )
        for condition in active_conditions:
            condition_row = _mapping(
                condition_rows[condition],
                context=f"{example_id}/{condition}",
            )
            mapping = _lane_mapping(condition_row, context=f"{example_id}/{condition}")
            owner, references, absences = _remap_roles(
                owner_teacher,
                references_teacher,
                absences_teacher,
                mapping,
            )
            unmoved_roles: tuple[list[int], list[list[bool]], list[list[bool]]] | None = None
            if condition == "lane_swap":
                if lane_swap_unmoved_mapping is None:
                    raise RuntimeError("Lane-swap audit lost the learned-plan address mapping.")
                unmoved_roles = _remap_roles(
                    owner_teacher,
                    references_teacher,
                    absences_teacher,
                    lane_swap_unmoved_mapping,
                )
            texts = _required_mapping(
                condition_row,
                "text_by_physical_lane",
                context=f"{example_id}/{condition}",
            )
            if set(texts) != set(lane_order):
                raise ValueError(
                    f"{example_id}/{condition} lane texts do not match physical_lane_order."
                )
            for fact_index, fact in enumerate(facts):
                for physical_lane, stream in enumerate(lane_order):
                    premise = texts[stream]
                    if not isinstance(premise, str) or len(premise.strip()) < 20:
                        raise ValueError(
                            f"{example_id}/{condition}/{stream} has no auditable prose."
                        )
                    if owner[fact_index] == physical_lane:
                        role = FactRole.OWNER
                    elif references[physical_lane][fact_index]:
                        role = FactRole.REFERENCE
                    elif absences[physical_lane][fact_index]:
                        role = FactRole.ABSENT
                    else:
                        raise RuntimeError("Physical fact-role remapping is incomplete.")
                    unmoved_role = (
                        _physical_role(
                            fact_index=fact_index,
                            physical_lane=physical_lane,
                            owner=unmoved_roles[0],
                            references=unmoved_roles[1],
                            absences=unmoved_roles[2],
                        )
                        if unmoved_roles is not None
                        else None
                    )
                    _append_audit_item(
                        queue_by_id,
                        key_records,
                        example_id=example_id,
                        condition=condition,
                        physical_lane=physical_lane,
                        fact_id=fact.fact_id,
                        query_kind=QueryKind.POSITIVE_FACT,
                        expected_role=role,
                        unmoved_expected_role=unmoved_role,
                        premise_text=premise,
                        hypothesis=fact.statement,
                    )
                    _append_audit_item(
                        queue_by_id,
                        key_records,
                        example_id=example_id,
                        condition=condition,
                        physical_lane=physical_lane,
                        fact_id=fact.fact_id,
                        query_kind=QueryKind.HARD_NEGATIVE,
                        expected_role=FactRole.ABSENT,
                        unmoved_expected_role=(
                            FactRole.ABSENT if unmoved_roles is not None else None
                        ),
                        premise_text=premise,
                        hypothesis=fact.hard_negative,
                    )
    if observed_examples != set(raw_by_id):
        raise ValueError(
            "Manual audit requires raw examples to exactly match evaluated documents; "
            f"not_evaluated={sorted(set(raw_by_id) - observed_examples)}."
        )
    _validate_key_coverage(key_records, raw_by_id, active_conditions)
    ordered_items = sorted(
        queue_by_id.values(),
        key=lambda item: hashlib.sha256(
            f"{randomization_seed}:{item.item_id}".encode("utf-8")
        ).hexdigest(),
    )
    write_jsonl_new(
        queue_path,
        (item.model_dump(mode="json") for item in ordered_items),
    )
    key = ManualAuditKey(
        schema_version=MANUAL_AUDIT_KEY_SCHEMA,
        generation_evaluation_sha256=sha256_file(generation_evaluation_path),
        raw_examples_sha256=sha256_file(raw_examples_path),
        queue_sha256=sha256_file(queue_path),
        randomization_seed=randomization_seed,
        physical_lane_order=lane_order,
        active_conditions=active_conditions,
        records=tuple(sorted(key_records, key=lambda record: record.item_id)),
    )
    write_bytes_new(key_path, canonical_json_bytes(key))
    return len(ordered_items)


def adjudicate_fact_audit(
    *,
    queue_path: Path,
    key_path: Path,
    annotator_a_path: Path,
    annotator_b_path: Path,
    adjudicator_path: Path,
    output_path: Path,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    minimum_documents: int = 32,
) -> dict[str, object]:
    """Resolve two blinded reads plus third-party disagreements into final evidence."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to replace manual audit result: {output_path}")
    queue = _load_queue(queue_path)
    key = _load_key(key_path)
    if key.queue_sha256 != sha256_file(queue_path):
        raise ValueError("Manual audit queue differs from the immutable key.")
    key_by_id = {record.item_id: record for record in key.records}
    if set(queue) != set(key_by_id):
        raise ValueError("Manual audit queue and key do not exactly cover one another.")
    annotations_a, annotator_a = _load_annotations(
        annotator_a_path,
        expected_ids=set(queue),
        allow_empty=False,
    )
    annotations_b, annotator_b = _load_annotations(
        annotator_b_path,
        expected_ids=set(queue),
        allow_empty=False,
    )
    if annotator_a is None or annotator_b is None:
        raise RuntimeError("Non-empty primary annotation files lost annotator identity.")
    if annotator_a == annotator_b:
        raise ValueError("The two primary manual annotations must use distinct annotators.")
    _validate_evidence_quotes(queue, annotations_a)
    _validate_evidence_quotes(queue, annotations_b)
    disputed = {
        item_id
        for item_id in queue
        if (
            annotations_a[item_id].judgment
            != annotations_b[item_id].judgment
            or annotations_a[item_id].judgment is ManualJudgment.UNCERTAIN
            or annotations_b[item_id].judgment is ManualJudgment.UNCERTAIN
        )
    }
    adjudications, adjudicator = _load_annotations(
        adjudicator_path,
        expected_ids=disputed,
        allow_empty=not disputed,
    )
    if disputed and adjudicator in {annotator_a, annotator_b}:
        raise ValueError("Disputed items require a distinct third adjudicator.")
    _validate_evidence_quotes(queue, adjudications)
    if any(
        annotation.judgment is ManualJudgment.UNCERTAIN
        for annotation in adjudications.values()
    ):
        raise ValueError("Third-party adjudication cannot remain UNCERTAIN.")

    resolved: dict[str, ManualJudgment] = {}
    for item_id in queue:
        if item_id in disputed:
            resolved[item_id] = adjudications[item_id].judgment
        else:
            resolved[item_id] = annotations_a[item_id].judgment
    condition_documents, lane_swap_unmoved_documents = _manual_document_counts(
        key.records,
        resolved,
    )
    inference = {
        condition: _summarize_manual_condition(
            documents,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=key.randomization_seed + index * 10,
        )
        for index, (condition, documents) in enumerate(
            sorted(condition_documents.items())
        )
    }
    if lane_swap_unmoved_documents:
        inference["lane_swap_unmoved_address_control"] = _summarize_manual_condition(
            lane_swap_unmoved_documents,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=key.randomization_seed + 500,
        )
    gate = _manual_evidence_gate(
        condition_documents,
        lane_swap_unmoved=lane_swap_unmoved_documents,
        minimum_documents=minimum_documents,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=key.randomization_seed + 1_000,
    )
    annotation_hashes = {
        annotator_a: sha256_file(annotator_a_path),
        annotator_b: sha256_file(annotator_b_path),
    }
    if disputed and adjudicator is not None:
        annotation_hashes[adjudicator] = sha256_file(adjudicator_path)
    result: dict[str, object] = {
        "schema_version": MANUAL_AUDIT_RESULT_SCHEMA,
        "queue_sha256": sha256_file(queue_path),
        "key_sha256": sha256_file(key_path),
        "annotation_sha256": annotation_hashes,
        "items": len(queue),
        "primary_disagreements": len(disputed),
        "primary_disagreement_rate": len(disputed) / len(queue),
        "primary_cohen_kappa": _cohen_kappa(annotations_a, annotations_b),
        "resolved_by_adjudicator": len(adjudications),
        "condition_inference": inference,
        "manual_evidence_gate": gate,
    }
    write_bytes_new(
        output_path,
        (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    return result


def _append_audit_item(
    queue_by_id: dict[str, ManualAuditItem],
    key_records: list[ManualAuditKeyRecord],
    *,
    example_id: str,
    condition: str,
    physical_lane: int,
    fact_id: str,
    query_kind: QueryKind,
    expected_role: FactRole,
    unmoved_expected_role: FactRole | None,
    premise_text: str,
    hypothesis: str,
) -> None:
    identity = {
        "example_id": example_id,
        "condition": condition,
        "physical_lane": physical_lane,
        "fact_id": fact_id,
        "query_kind": query_kind.value,
        "premise_text": premise_text,
        "hypothesis": hypothesis,
    }
    item_id = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if item_id in queue_by_id:
        raise ValueError(f"Manual audit item identity collided for {item_id}.")
    queue_by_id[item_id] = ManualAuditItem(
        schema_version=MANUAL_AUDIT_ITEM_SCHEMA,
        item_id=item_id,
        premise_text=premise_text,
        hypothesis=hypothesis,
    )
    key_records.append(
        ManualAuditKeyRecord(
            item_id=item_id,
            example_id=example_id,
            condition=condition,
            physical_lane=physical_lane,
            fact_id=fact_id,
            query_kind=query_kind,
            expected_role=expected_role,
            unmoved_expected_role=unmoved_expected_role,
        )
    )


def _teacher_roles(
    example: RealPlanExample,
) -> tuple[list[int], list[list[bool]], list[list[bool]]]:
    plan_ids = [plan.plan_id for plan in example.teacher.plans]
    target_by_id = {
        target.plan_id: target for target in example.teacher.target_sections
    }
    fact_ids = [fact.fact_id for fact in example.facts.facts]
    owners = [-1] * len(fact_ids)
    references = [[False] * len(fact_ids) for _ in range(3)]
    absences = [[False] * len(fact_ids) for _ in range(3)]
    for fact_index, fact_id in enumerate(fact_ids):
        owner_rows: list[int] = []
        for teacher_lane, plan_id in enumerate(plan_ids):
            labels = {
                label.fact_id: label for label in target_by_id[plan_id].fact_labels
            }
            role = labels[fact_id].role
            if role is FactRole.OWNER:
                owner_rows.append(teacher_lane)
            elif role is FactRole.REFERENCE:
                references[teacher_lane][fact_index] = True
            else:
                absences[teacher_lane][fact_index] = True
        if len(owner_rows) != 1:
            raise ValueError(f"{fact_id} must have one teacher-plan owner.")
        owners[fact_index] = owner_rows[0]
    return owners, references, absences


def _remap_roles(
    owner_teacher: Sequence[int],
    reference_teacher: Sequence[Sequence[bool]],
    absence_teacher: Sequence[Sequence[bool]],
    teacher_to_physical: tuple[int, int, int],
) -> tuple[list[int], list[list[bool]], list[list[bool]]]:
    facts = len(owner_teacher)
    owner = [teacher_to_physical[value] for value in owner_teacher]
    references = [[False] * facts for _ in range(3)]
    absences = [[False] * facts for _ in range(3)]
    for teacher_lane, physical_lane in enumerate(teacher_to_physical):
        references[physical_lane] = list(reference_teacher[teacher_lane])
        absences[physical_lane] = list(absence_teacher[teacher_lane])
    return owner, references, absences


def _physical_role(
    *,
    fact_index: int,
    physical_lane: int,
    owner: Sequence[int],
    references: Sequence[Sequence[bool]],
    absences: Sequence[Sequence[bool]],
) -> FactRole:
    if owner[fact_index] == physical_lane:
        return FactRole.OWNER
    if references[physical_lane][fact_index]:
        return FactRole.REFERENCE
    if absences[physical_lane][fact_index]:
        return FactRole.ABSENT
    raise RuntimeError("Physical fact-role remapping is incomplete.")


def _validate_key_coverage(
    records: Sequence[ManualAuditKeyRecord],
    examples: Mapping[str, RealPlanExample],
    conditions: Sequence[str],
) -> None:
    grouped: dict[tuple[str, str, str, QueryKind], list[ManualAuditKeyRecord]] = defaultdict(
        list
    )
    for record in records:
        grouped[
            (
                record.example_id,
                record.condition,
                record.fact_id,
                record.query_kind,
            )
        ].append(record)
    for example_id, example in examples.items():
        for condition in conditions:
            for fact in example.facts.facts:
                for kind in QueryKind:
                    rows = grouped.get((example_id, condition, fact.fact_id, kind), [])
                    if sorted(record.physical_lane for record in rows) != [0, 1, 2]:
                        raise ValueError(
                            "Manual audit must contain every fact query at all three "
                            f"physical lanes: {example_id}/{condition}/{fact.fact_id}/{kind}."
                        )
                    roles = Counter(record.expected_role for record in rows)
                    if kind is QueryKind.POSITIVE_FACT:
                        if roles[FactRole.OWNER] != 1 or sum(roles.values()) != 3:
                            raise ValueError("Positive audit queries require exactly one owner.")
                    elif roles != Counter({FactRole.ABSENT: 3}):
                        raise ValueError("Hard-negative audit queries must all be ABSENT.")
                    unmoved_roles = [
                        record.unmoved_expected_role for record in rows
                    ]
                    if condition == "lane_swap":
                        if any(role is None for role in unmoved_roles):
                            raise ValueError(
                                "Lane-swap audit records require an unmoved expected role."
                            )
                        unmoved_counts = Counter(unmoved_roles)
                        if kind is QueryKind.POSITIVE_FACT:
                            if (
                                unmoved_counts[FactRole.OWNER] != 1
                                or sum(unmoved_counts.values()) != 3
                            ):
                                raise ValueError(
                                    "Lane-swap unmoved positive queries require one owner."
                                )
                        elif unmoved_counts != Counter({FactRole.ABSENT: 3}):
                            raise ValueError(
                                "Lane-swap unmoved hard negatives must all be ABSENT."
                            )
                    elif any(role is not None for role in unmoved_roles):
                        raise ValueError(
                            "Only lane-swap audit records may have unmoved roles."
                        )


def _manual_document_counts(
    records: Sequence[ManualAuditKeyRecord],
    resolved: Mapping[str, ManualJudgment],
) -> tuple[dict[str, list[FactOwnershipCounts]], list[FactOwnershipCounts]]:
    grouped: dict[tuple[str, str], list[ManualAuditKeyRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.condition, record.example_id)].append(record)
    result: dict[str, list[FactOwnershipCounts]] = defaultdict(list)
    lane_swap_unmoved: list[FactOwnershipCounts] = []
    for (condition, _), rows in sorted(grouped.items()):
        result[condition].append(_counts_for_roles(rows, resolved, use_unmoved=False))
        if condition == "lane_swap":
            lane_swap_unmoved.append(
                _counts_for_roles(rows, resolved, use_unmoved=True)
            )
    return dict(result), lane_swap_unmoved


def _counts_for_roles(
    rows: Sequence[ManualAuditKeyRecord],
    resolved: Mapping[str, ManualJudgment],
    *,
    use_unmoved: bool,
) -> FactOwnershipCounts:
    def role(record: ManualAuditKeyRecord) -> FactRole:
        selected = (
            record.unmoved_expected_role if use_unmoved else record.expected_role
        )
        if selected is None:
            raise ValueError("Manual audit record is missing its requested expected role.")
        return selected

    owner = [row for row in rows if role(row) is FactRole.OWNER]
    reference = [
        row
        for row in rows
        if row.query_kind is QueryKind.POSITIVE_FACT
        and role(row) is FactRole.REFERENCE
    ]
    absent = [
        row
        for row in rows
        if row.query_kind is QueryKind.POSITIVE_FACT
        and role(row) is FactRole.ABSENT
    ]
    hard_negative = [
        row for row in rows if row.query_kind is QueryKind.HARD_NEGATIVE
    ]
    return FactOwnershipCounts(
        owner_hits=sum(
            resolved[row.item_id] is ManualJudgment.ENTAILED for row in owner
        ),
        owner_total=len(owner),
        reference_hits=sum(
            resolved[row.item_id] is ManualJudgment.ENTAILED for row in reference
        ),
        reference_total=len(reference),
        unauthorized_hits=sum(
            resolved[row.item_id] is ManualJudgment.ENTAILED for row in absent
        ),
        unauthorized_total=len(absent),
        hard_negative_hits=sum(
            resolved[row.item_id] is ManualJudgment.ENTAILED
            for row in hard_negative
        ),
        hard_negative_total=len(hard_negative),
    )


def _summarize_manual_condition(
    documents: Sequence[FactOwnershipCounts],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    aggregate = documents[0]
    for document in documents[1:]:
        aggregate = aggregate + document
    return {
        "documents": len(documents),
        "aggregate": aggregate.to_dict(),
        "owner_recall_document_bootstrap": bootstrap_mean(
            [document.owner_recall for document in documents],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        ).to_dict(),
        "reference_recall_document_bootstrap": bootstrap_mean(
            [document.reference_recall for document in documents],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 1,
        ).to_dict(),
        "unauthorized_leakage_document_bootstrap": bootstrap_mean(
            [document.unauthorized_leakage for document in documents],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 2,
        ).to_dict(),
        "hard_negative_rate_document_bootstrap": bootstrap_mean(
            [document.hard_negative_rate for document in documents],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 3,
        ).to_dict(),
    }


def _manual_evidence_gate(
    conditions: Mapping[str, Sequence[FactOwnershipCounts]],
    *,
    lane_swap_unmoved: Sequence[FactOwnershipCounts],
    minimum_documents: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    oracle = conditions.get("oracle_plan")
    if not oracle:
        raise ValueError("Manual evidence requires the oracle_plan condition.")
    owner = bootstrap_mean(
        [row.owner_recall for row in oracle],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    leakage = bootstrap_mean(
        [row.unauthorized_leakage for row in oracle],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 1,
    )
    enough = len(oracle) >= minimum_documents
    checks: dict[str, Any] = {
        "minimum_documents": minimum_documents,
        "documents": len(oracle),
        "enough_documents": enough,
        "oracle_owner_recall_ci_lower_at_least_0_90": owner.lower >= 0.90,
        "oracle_unauthorized_leakage_ci_upper_at_most_0_10": leakage.upper <= 0.10,
        "paired_document_intervals": {
            "oracle_owner_recall": owner.to_dict(),
            "oracle_unauthorized_leakage": leakage.to_dict(),
        },
    }
    if {"learned_plan", "plan_zero", "lane_swap"} <= set(conditions):
        learned = conditions["learned_plan"]
        zero = conditions["plan_zero"]
        swapped = conditions["lane_swap"]
        if not (
            len(learned)
            == len(zero)
            == len(swapped)
            == len(lane_swap_unmoved)
            == len(oracle)
        ):
            raise ValueError("Manual evidence conditions must cover identical documents.")
        retention = bootstrap_mean(
            [
                learned_row.owner_recall - 0.8 * oracle_row.owner_recall
                for learned_row, oracle_row in zip(learned, oracle, strict=True)
            ],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 2,
        )
        zero_drop = bootstrap_mean(
            [
                learned_row.owner_recall - zero_row.owner_recall
                for learned_row, zero_row in zip(learned, zero, strict=True)
            ],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 3,
        )
        swap_retention = bootstrap_mean(
            [
                swapped_row.owner_recall - 0.8 * learned_row.owner_recall
                for swapped_row, learned_row in zip(swapped, learned, strict=True)
            ],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 4,
        )
        expected_address_advantage = bootstrap_mean(
            [
                swapped_row.owner_recall - unmoved_row.owner_recall
                for swapped_row, unmoved_row in zip(
                    swapped,
                    lane_swap_unmoved,
                    strict=True,
                )
            ],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 5,
        )
        checks.update(
            {
                "learned_owner_recall_at_least_80_percent_of_oracle_ci": (
                    retention.lower >= 0.0
                ),
                "plan_zero_owner_recall_drop_ci_lower_at_least_0_30": (
                    zero_drop.lower >= 0.30
                ),
                "lane_swap_retains_80_percent_of_learned_owner_recall_ci": (
                    swap_retention.lower >= 0.0
                ),
                "lane_swap_expected_address_advantage_ci_lower_at_least_0_30": (
                    expected_address_advantage.lower >= 0.30
                ),
            }
        )
        checks["paired_document_intervals"].update(
            {
                "learned_minus_80_percent_oracle_owner_recall": retention.to_dict(),
                "learned_minus_plan_zero_owner_recall": zero_drop.to_dict(),
                "lane_swap_minus_80_percent_learned_owner_recall": (
                    swap_retention.to_dict()
                ),
                "lane_swap_expected_minus_unmoved_owner_recall": (
                    expected_address_advantage.to_dict()
                ),
            }
        )
    boolean_checks = [
        value
        for key, value in checks.items()
        if key not in {"minimum_documents", "documents", "paired_document_intervals"}
    ]
    checks["passes"] = all(value is True for value in boolean_checks)
    return checks


def _cohen_kappa(
    left: Mapping[str, ManualAnnotation],
    right: Mapping[str, ManualAnnotation],
) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("Cohen kappa requires identical non-empty annotation keys.")
    categories = tuple(ManualJudgment)
    observed = sum(
        left[item_id].judgment is right[item_id].judgment for item_id in left
    ) / len(left)
    left_counts = Counter(annotation.judgment for annotation in left.values())
    right_counts = Counter(annotation.judgment for annotation in right.values())
    expected = sum(
        (left_counts[category] / len(left)) * (right_counts[category] / len(right))
        for category in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _load_queue(path: Path) -> dict[str, ManualAuditItem]:
    rows = _load_jsonl_models(path, ManualAuditItem)
    result = {row.item_id: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Manual audit queue repeats item IDs.")
    return result


def _load_key(path: Path) -> ManualAuditKey:
    if not path.is_file():
        raise FileNotFoundError(f"Manual audit key does not exist: {path}")
    try:
        return ManualAuditKey.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"Manual audit key is invalid: {path}") from exc


def _load_annotations(
    path: Path,
    *,
    expected_ids: set[str],
    allow_empty: bool,
) -> tuple[dict[str, ManualAnnotation], str | None]:
    rows = _load_jsonl_models(path, ManualAnnotation, allow_empty=allow_empty)
    result = {row.item_id: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Annotation file repeats item IDs: {path}")
    if set(result) != expected_ids:
        raise ValueError(
            f"Annotation file does not exactly cover required items: {path}; "
            f"missing={sorted(expected_ids - set(result))[:10]}, "
            f"extra={sorted(set(result) - expected_ids)[:10]}."
        )
    annotators = {row.annotator_id for row in rows}
    if len(annotators) > 1:
        raise ValueError(f"One annotation file must contain one annotator ID: {path}")
    return result, next(iter(annotators), None)


def _validate_evidence_quotes(
    queue: Mapping[str, ManualAuditItem],
    annotations: Mapping[str, ManualAnnotation],
) -> None:
    for item_id, annotation in annotations.items():
        if (
            annotation.judgment is ManualJudgment.ENTAILED
            and annotation.evidence_quote not in queue[item_id].premise_text
        ):
            raise ValueError(
                f"Annotation {item_id} evidence quote is not exact generated text."
            )


def _load_raw_examples(path: Path) -> dict[str, RealPlanExample]:
    rows = _load_jsonl_models(path, RealPlanExample)
    result: dict[str, RealPlanExample] = {}
    for row in rows:
        validate_real_plan_example(row)
        if row.source.source_id in result:
            raise ValueError(f"Raw examples repeat {row.source.source_id!r}.")
        result[row.source.source_id] = row
    return result


def _load_jsonl_models[
    ModelT: BaseModel
](
    path: Path,
    model_type: type[ModelT],
    *,
    allow_empty: bool = False,
) -> list[ModelT]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL does not exist: {path}")
    rows: list[ModelT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model_type.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} violates its schema.") from exc
    if not rows and not allow_empty:
        raise ValueError(f"{path} contains no records.")
    return rows


def _load_json_object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON input is invalid: {path}") from exc
    return _mapping(value, context=str(path))


def _lane_mapping(
    condition: Mapping[str, object],
    *,
    context: str,
) -> tuple[int, int, int]:
    value = condition.get("expected_teacher_to_physical_lane")
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(item) is not int for item in value)
        or sorted(value) != [0, 1, 2]
    ):
        raise ValueError(f"{context} has an invalid teacher-to-physical mapping.")
    return (value[0], value[1], value[2])


def _three_texts(value: object, *, context: str) -> tuple[str, str, str]:
    rows = _text_tuple(value, context=context)
    if len(rows) != 3 or len(set(rows)) != 3:
        raise ValueError(f"{context} must contain three unique strings.")
    return (rows[0], rows[1], rows[2])


def _text_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{context} must contain non-empty strings.")
    return tuple(value)


def _required_mapping(
    row: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> Mapping[str, object]:
    return _mapping(row.get(key), context=f"{context}.{key}")


def _required_text(row: Mapping[str, object], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be non-empty text.")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object.")
    return value


__all__ = [
    "MANUAL_ANNOTATION_SCHEMA",
    "MANUAL_AUDIT_ITEM_SCHEMA",
    "MANUAL_AUDIT_KEY_SCHEMA",
    "MANUAL_AUDIT_RESULT_SCHEMA",
    "ManualAnnotation",
    "ManualAuditItem",
    "ManualAuditKey",
    "ManualAuditKeyRecord",
    "ManualJudgment",
    "QueryKind",
    "adjudicate_fact_audit",
    "export_blinded_fact_audit",
]
