"""Source-grounded schema, Batch request, retokenization, and lane binding."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from pdt.datasets.real_plan_batch import (
    CANONICAL_BATCH_MODEL,
    RealPlanSplitManifest,
    build_fact_requests,
    build_joint_requests,
    split_real_plan_examples,
)
from pdt.datasets.historical_source import (
    HISTORICAL_RENDERER_ID,
    HISTORICAL_SOURCE_SCHEMA,
    AssessmentQuality,
    DatasetSplit,
    HistoricalCategory,
    HistoricalParagraph,
    HistoricalSection,
    HistoricalSource,
    PageAssessment,
    ReferenceKind,
    ReferenceRecord,
    canonical_json_bytes,
    manifest_for_source_file,
    sha256_file,
)
from pdt.datasets.immutable_io import write_jsonl_new
from pdt.datasets.real_plan_retokenize import retokenize_real_plan_example
from pdt.datasets.real_plan_schema import RealPlanExample, validate_real_plan_example
from pdt.evaluation.manual_fact_audit import (
    MANUAL_ANNOTATION_SCHEMA,
    ManualAnnotation,
    ManualAuditKey,
    ManualAuditItem,
    ManualJudgment,
    QueryKind,
    adjudicate_fact_audit,
    export_blinded_fact_audit,
)
from pdt.training.dataset import RealPlanCollator, RealPlanDataset


def _example() -> RealPlanExample:
    source_paragraphs: list[HistoricalParagraph] = []
    facts: list[dict[str, object]] = []
    next_fact = 0
    for paragraph_index in range(12):
        facts_in_paragraph = 2 if paragraph_index < 6 else 1
        fact_indices = tuple(range(next_fact, next_fact + facts_in_paragraph))
        next_fact += facts_in_paragraph
        quotes = [
            f"Source fact {fact_index:03d} states a precise grounded relationship."
            for fact_index in fact_indices
        ]
        text = (
            " ".join(quotes)
            + " "
            + "contextual material " * 450
        ).strip()
        reference_ids = tuple(f"r{fact_index:02d}" for fact_index in fact_indices)
        source_paragraphs.append(
            HistoricalParagraph(
                paragraph_id=f"p{paragraph_index:03d}",
                text=text,
                reference_ids=reference_ids,
            )
        )
        for fact_index, quote in zip(fact_indices, quotes, strict=True):
            start = text.index(quote)
            facts.append(
                {
                    "fact_id": f"fact_{fact_index:03d}",
                    "statement": (
                        f"Fact {fact_index:03d} records the true grounded relationship."
                    ),
                    "subject": f"subject {fact_index:03d}",
                    "relation": "records",
                    "object": f"object {fact_index:03d}",
                    "importance": 3,
                    "provenance": [
                        {
                            "paragraph_id": f"p{paragraph_index:03d}",
                            "start_char": start,
                            "end_char": start + len(quote),
                            "exact_quote": quote,
                            "reference_ids": [f"r{fact_index:02d}"],
                        }
                    ],
                    "hard_negative": (
                        f"Fact {fact_index:03d} records a deliberately contradicted relationship."
                    ),
                }
            )

    assert next_fact == 18
    sections = tuple(
        HistoricalSection(
            section_id=f"section_{section_index:03d}",
            heading_path=(f"Historical phase {section_index + 1}",),
            paragraphs=tuple(
                source_paragraphs[2 * section_index : 2 * section_index + 2]
            ),
        )
        for section_index in range(6)
    )
    title = "A Grounded Test Article"
    rendered_parts = [title]
    for section in sections:
        rendered_parts.extend(section.heading_path)
        rendered_parts.extend(paragraph.text for paragraph in section.paragraphs)
    model_visible = "\n\n".join(rendered_parts)
    source = HistoricalSource(
        schema_version=HISTORICAL_SOURCE_SCHEMA,
        renderer_id=HISTORICAL_RENDERER_ID,
        source_id="wikipedia-test-article",
        page_id=101,
        revision_id=202,
        revision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        dump_date=date(2026, 1, 1),
        title=title,
        source_url="https://en.wikipedia.org/?curid=101",
        license="CC BY-SA 4.0",
        revision_sha1="1" * 40,
        historical_category=HistoricalCategory.REVOLUTIONS_TRANSITIONS,
        event_end_year=1900,
        assessments=(
            PageAssessment(
                project="WikiProject History",
                quality=AssessmentQuality.GA,
            ),
        ),
        family_id="family-" + "2" * 24,
        split=DatasetSplit.TRAIN,
        tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="pinned-qwen-revision",
        qwen_token_count=4_000,
        sections=sections,
        references=tuple(
            ReferenceRecord(
                reference_id=f"r{index:02d}",
                source_type=ReferenceKind.BOOK,
                citation_text=f"Author {index}. Substantial historical source.",
            )
            for index in range(18)
        ),
        raw_wikitext_sha256="3" * 64,
        semantic_html_sha256="4" * 64,
        model_visible_sha256=hashlib.sha256(
            model_visible.encode("utf-8")
        ).hexdigest(),
    )

    owner_sets = {
        "plan_0": [f"fact_{index:03d}" for index in range(0, 6)],
        "plan_1": [f"fact_{index:03d}" for index in range(6, 12)],
        "plan_2": [f"fact_{index:03d}" for index in range(12, 18)],
    }
    reference_sets = {
        "plan_0": [("plan_1", "fact_006"), ("plan_2", "fact_012")],
        "plan_1": [("plan_0", "fact_000"), ("plan_2", "fact_012")],
        "plan_2": [("plan_0", "fact_000"), ("plan_1", "fact_006")],
    }
    plans: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    owner_quotes: dict[tuple[str, str], str] = {}
    reference_quotes: dict[tuple[str, str], str] = {}
    paragraph_text: dict[tuple[str, int], str] = {}
    for plan_id, owned in owner_sets.items():
        plan_number = int(plan_id[-1])
        for node_index, fact_id in enumerate(owned):
            owner_quotes[(plan_id, fact_id)] = (
                f"Plan {plan_number} explicitly owns {fact_id} in this carefully grounded section."
            )
        for _, fact_id in reference_sets[plan_id]:
            reference_quotes[(plan_id, fact_id)] = (
                f"Plan {plan_number} deliberately references sibling {fact_id} "
                "after its earlier realization."
            )
        paragraphs: list[dict[str, object]] = []
        for node_index, fact_id in enumerate(owned):
            evidence = [owner_quotes[(plan_id, fact_id)]]
            if node_index == 1:
                evidence.extend(
                    reference_quotes[(plan_id, reference_fact)]
                    for _, reference_fact in reference_sets[plan_id]
                )
            text = (
                " ".join(evidence)
                + " "
                + (
                    "This developed contextual sentence connects the evidence to the "
                    "broader historical explanation without reducing the prose to fragments. "
                )
                    * 8
            ).strip()
            paragraph_text[(plan_id, node_index)] = text
            paragraphs.append(
                {
                    "paragraph_id": f"{plan_id}_paragraph_{node_index}",
                    "outline_node_id": f"node_{node_index}",
                    "text": text,
                }
            )
        targets.append(
            {
                "plan_id": plan_id,
                "paragraphs": paragraphs,
                "fact_labels": [
                    {
                        "fact_id": f"fact_{fact_index:03d}",
                        "role": (
                            "OWNER"
                            if f"fact_{fact_index:03d}" in owned
                            else "REFERENCE"
                            if f"fact_{fact_index:03d}"
                            in {fact_id for _, fact_id in reference_sets[plan_id]}
                            else "ABSENT"
                        ),
                        "target_evidence_quote": (
                            owner_quotes[(plan_id, f"fact_{fact_index:03d}")]
                            if f"fact_{fact_index:03d}" in owned
                            else reference_quotes[
                                (plan_id, f"fact_{fact_index:03d}")
                            ]
                            if f"fact_{fact_index:03d}"
                            in {fact_id for _, fact_id in reference_sets[plan_id]}
                            else ""
                        ),
                    }
                    for fact_index in range(18)
                ],
            }
        )

    for plan_id, owned in owner_sets.items():
        nodes: list[dict[str, object]] = []
        for node_index, fact_id in enumerate(owned):
            references = (
                [reference_fact for _, reference_fact in reference_sets[plan_id]]
                if node_index == 1
                else []
            )
            dependencies = []
            if node_index == 1:
                for source_plan, reference_fact in reference_sets[plan_id]:
                    dependencies.append(
                        {
                            "source_plan_id": source_plan,
                            "source_node_id": "node_0",
                            "source_target_paragraph_id": (
                                f"{source_plan}_paragraph_0"
                            ),
                            "source_evidence_quote": owner_quotes[
                                (source_plan, reference_fact)
                            ],
                            "required_by_target_paragraph_id": (
                                f"{plan_id}_paragraph_1"
                            ),
                            "required_target_evidence_quote": reference_quotes[
                                (plan_id, reference_fact)
                            ],
                            "fact_ids": [reference_fact],
                        }
                    )
            nodes.append(
                {
                    "node_id": f"node_{node_index}",
                    "objective": (
                        f"Develop objective {node_index} for {plan_id} with grounded detail."
                    ),
                    "owned_fact_ids": [fact_id],
                    "reference_fact_ids": references,
                    "dependencies": dependencies,
                    "target_paragraph_ids": [
                        f"{plan_id}_paragraph_{node_index}"
                    ],
                }
            )
        plans.append(
            {
                "plan_id": plan_id,
                "heading": f"Grounded section {plan_id}",
                "role_summary": (
                    f"This prompt-specific role for {plan_id} develops a complementary section."
                ),
                "nodes": nodes,
            }
        )

    return RealPlanExample.model_validate(
        {
            "schema_version": "pdt-real-plan-v2",
            "source": source.model_dump(mode="json"),
            "facts": {
                "source_id": "wikipedia-test-article",
                "source_revision_id": source.revision_id,
                "source_model_visible_sha256": source.model_visible_sha256,
                "facts": facts,
            },
            "teacher": {
                "source_id": "wikipedia-test-article",
                "source_revision_id": source.revision_id,
                "source_model_visible_sha256": source.model_visible_sha256,
                "expository_prompt": (
                    "Explain the grounded subject through three complementary historical sections."
                ),
                "plans": plans,
                "target_sections": targets,
                "presentation_order": ["plan_1", "plan_0", "plan_2"],
            },
        }
    )


def _example_for_split(index: int, split: DatasetSplit) -> RealPlanExample:
    payload = _example().model_dump(mode="json")
    source_id = f"source-split-{index}"
    revision_id = 12_345 + index
    payload["source"]["source_id"] = source_id
    payload["source"]["page_id"] = 54_321 + index
    payload["source"]["revision_id"] = revision_id
    payload["source"]["family_id"] = f"family-{index:024x}"
    payload["source"]["split"] = split.value
    payload["facts"]["source_id"] = source_id
    payload["facts"]["source_revision_id"] = revision_id
    payload["teacher"]["source_id"] = source_id
    payload["teacher"]["source_revision_id"] = revision_id
    example = RealPlanExample.model_validate(payload)
    validate_real_plan_example(example)
    return example


class TokenizerContractDouble:
    eos_token_id = 3

    @staticmethod
    def _tokens(text: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0), match.start(), match.end())
            for match in re.finditer(r"\S+", text)
        ]

    @classmethod
    def _ids(cls, text: str) -> list[int]:
        return [(len(token) % 97) + 3 for token, _, _ in cls._tokens(text)]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
        tokenize: bool,
    ) -> list[int]:
        assert enable_thinking is False and tokenize is True
        prompt = [1] + self._ids(messages[0]["content"]) + [2]
        if add_generation_prompt:
            return prompt
        return prompt + self._ids(messages[1]["content"]) + [3]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False and return_offsets_mapping is True
        tokens = self._tokens(text)
        return {
            "input_ids": [(len(token) % 97) + 3 for token, _, _ in tokens],
            "offset_mapping": [(start, end) for _, start, end in tokens],
        }


class EmbeddingContractDouble:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % 1024] = 1.0
        return vectors


def test_schema_enforces_exact_delayed_reference_and_owner_contracts() -> None:
    example = _example()
    validate_real_plan_example(example)

    payload = example.model_dump(mode="json")
    payload["teacher"]["plans"][0]["nodes"][0]["owned_fact_ids"].append(
        "fact_001"
    )
    invalid = RealPlanExample.model_validate(payload)
    with pytest.raises(ValueError, match="exactly one outline node"):
        validate_real_plan_example(invalid)

    payload = example.model_dump(mode="json")
    dependency = payload["teacher"]["plans"][0]["nodes"][1]["dependencies"][0]
    dependency["required_target_evidence_quote"] = "missing dependency evidence quote"
    invalid = RealPlanExample.model_validate(payload)
    with pytest.raises(ValueError, match="receiver evidence quote"):
        validate_real_plan_example(invalid)


def test_fact_provenance_must_name_a_reference_on_its_exact_source_paragraph() -> None:
    payload = _example().model_dump(mode="json")
    payload["facts"]["facts"][0]["provenance"][0]["reference_ids"] = ["r17"]
    invalid = RealPlanExample.model_validate(payload)

    with pytest.raises(ValueError, match="not attached"):
        validate_real_plan_example(invalid)


def test_batch_requests_use_one_pinned_responses_schema_path(tmp_path: Path) -> None:
    example = _example()
    source_path = tmp_path / "sources.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    manifest_path = tmp_path / "accepted_manifest.json"
    facts_path = tmp_path / "facts.jsonl"
    fact_requests = tmp_path / "fact_requests.jsonl"
    joint_requests = tmp_path / "joint_requests.jsonl"
    raw_path.write_text('{"immutable":"test raw identity"}\n', encoding="utf-8")
    write_jsonl_new(
        source_path,
        (example.source.model_dump(mode="json"),),
    )
    manifest = manifest_for_source_file(
        source_path,
        (example.source,),
        input_path=raw_path,
        tokenizer_name=example.source.tokenizer,
        tokenizer_revision=example.source.tokenizer_revision,
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_sha256 = sha256_file(manifest_path)
    facts_path.write_text(
        example.facts.model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert (
        build_fact_requests(
            source_path,
            manifest_path,
            manifest_sha256,
            fact_requests,
        )
        == 1
    )
    assert (
        build_joint_requests(
            source_path,
            manifest_path,
            manifest_sha256,
            facts_path,
            joint_requests,
        )
        == 1
    )
    joint = json.loads(joint_requests.read_text(encoding="utf-8"))
    assert joint["url"] == "/v1/responses"
    assert joint["body"]["model"] == CANONICAL_BATCH_MODEL
    assert joint["body"]["text"]["format"]["type"] == "json_schema"
    assert joint["body"]["text"]["format"]["strict"] is True
    assert joint["body"]["metadata"] == {
        "pipeline": "pdt-real-plan-v2",
        "accepted_manifest_sha256": manifest_sha256,
    }
    prompt = joint["body"]["input"][0]["content"][0]["text"]
    assert "exactly three complementary" in prompt
    assert "short or choppy sentences" in prompt
    assert "receive at least one dependency from each sibling" in prompt
    with pytest.raises(ValueError, match="approved SHA-256"):
        build_fact_requests(
            source_path,
            manifest_path,
            "0" * 64,
            tmp_path / "unapproved_requests.jsonl",
        )


def test_retokenization_uses_true_and_hard_negative_fact_queries() -> None:
    record = retokenize_real_plan_example(
        _example(),
        tokenizer=TokenizerContractDouble(),
        embedder=EmbeddingContractDouble(),
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="pinned-qwen-revision",
    )
    assert record["schema_version"] == "pdt-real-plan-tokenized-v3"
    assert record["positive_fact_count"] == 18
    assert len(record["fact_query_ids"]) == 36
    assert len(record["fact_embeddings"]) == 36
    assert len(record["fact_route_targets"][0][0]) == 36
    for lane in record["lanes"]:
        for block_labels in lane["fact_write_targets"]:
            assert block_labels[18:] == [2] * 18


def test_validated_examples_publish_atomic_family_disjoint_splits(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "all_examples.jsonl"
    output_dir = tmp_path / "examples"
    examples = tuple(
        _example_for_split(index, split)
        for index, split in enumerate(DatasetSplit)
    )
    write_jsonl_new(
        input_path,
        (example.model_dump(mode="json") for example in examples),
    )

    assert split_real_plan_examples(input_path, output_dir) == 3

    manifest = RealPlanSplitManifest.model_validate_json(
        (output_dir / "manifest.json").read_bytes()
    )
    assert manifest.total_examples == 3
    assert [record.split for record in manifest.splits] == list(DatasetSplit)
    assert all(record.examples == 1 for record in manifest.splits)
    assert {
        path.name for path in output_dir.iterdir()
    } == {"train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"}

    crossing_input = tmp_path / "crossing.jsonl"
    crossing_output = tmp_path / "crossing"
    crossing = [
        _example_for_split(10, DatasetSplit.TRAIN),
        _example_for_split(11, DatasetSplit.VALIDATION),
        _example_for_split(12, DatasetSplit.TEST),
    ]
    crossing_payload = crossing[1].model_dump(mode="json")
    crossing_payload["source"]["family_id"] = crossing[0].source.family_id
    crossing[1] = RealPlanExample.model_validate(crossing_payload)
    write_jsonl_new(
        crossing_input,
        (example.model_dump(mode="json") for example in crossing),
    )
    with pytest.raises(ValueError, match="crosses"):
        split_real_plan_examples(crossing_input, crossing_output)
    assert not crossing_output.exists()


def test_retokenization_rejects_paragraph_order_without_block_visibility() -> None:
    payload = _example().model_dump(mode="json")
    dependency = payload["teacher"]["plans"][0]["nodes"][1]["dependencies"][0]
    source_plan_id = dependency["source_plan_id"]
    source_target = next(
        target
        for target in payload["teacher"]["target_sections"]
        if target["plan_id"] == source_plan_id
    )
    source_paragraph = source_target["paragraphs"][0]
    quote = dependency["source_evidence_quote"]
    source_paragraph["text"] = (
        source_paragraph["text"].removeprefix(quote).strip()
        + " "
        + (
            "Additional developed historical context deliberately delays the source "
            "evidence beyond the receiver's first usable communication round. "
        )
        * 6
        + quote
    )
    example = RealPlanExample.model_validate(payload)
    validate_real_plan_example(example)

    with pytest.raises(ValueError, match="not causally visible"):
        retokenize_real_plan_example(
            example,
            tokenizer=TokenizerContractDouble(),
            embedder=EmbeddingContractDouble(),
            tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="pinned-qwen-revision",
        )


def test_collator_randomly_rebinds_every_teacher_lane_axis(tmp_path: Path) -> None:
    record = retokenize_real_plan_example(
        _example(),
        tokenizer=TokenizerContractDouble(),
        embedder=EmbeddingContractDouble(),
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="pinned-qwen-revision",
    )
    for teacher_lane in range(3):
        record["plan_semantic_targets"][teacher_lane][0][0] = float(
            teacher_lane + 1
        )
        record["fact_route_targets"][teacher_lane][0][teacher_lane] = 1.0
        record["lanes"][teacher_lane]["target_block_ids"][0][0] = (
            100 + teacher_lane
        )
    path = tmp_path / "tokenized.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = RealPlanDataset(
        path,
        expected_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        expected_tokenizer_revision="pinned-qwen-revision",
    )
    collator = RealPlanCollator(
        pad_token_id=0,
        max_planner_prompt_length=len(record["planner_prompt_ids"]) + 8,
        seed=3,
    )
    batch = collator([dataset[0]])
    for physical_lane in range(3):
        teacher_lane = (
            int(batch.plan_semantic_targets[0, physical_lane, 0, 0].item()) - 1
        )
        assert batch.fact_route_targets[
            0,
            physical_lane,
            0,
            teacher_lane,
        ] == 1
        assert (
            batch.target_block_ids[0, physical_lane, 0, 0].item()
            == 100 + teacher_lane
        )
    assert batch.fact_mask[0, :36].all()
    assert batch.positive_fact_mask[0, :18].all()
    assert not batch.positive_fact_mask[0, 18:].any()


def test_immutable_jsonl_is_not_published_when_streaming_validation_fails(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "raw.jsonl"

    def invalid_rows():
        yield {"valid": True}
        raise ValueError("late validation failed")

    with pytest.raises(ValueError, match="late validation"):
        write_jsonl_new(destination, invalid_rows())
    assert not destination.exists()


def test_blinded_manual_audit_requires_exact_double_annotation_and_adjudication(
    tmp_path: Path,
) -> None:
    example = _example()
    raw_path = tmp_path / "evaluation_raw.jsonl"
    generation_path = tmp_path / "generation.json"
    queue_path = tmp_path / "manual_queue.jsonl"
    key_path = tmp_path / "manual_key.json"
    annotation_a_path = tmp_path / "annotation_a.jsonl"
    annotation_b_path = tmp_path / "annotation_b.jsonl"
    adjudication_path = tmp_path / "adjudication.jsonl"
    result_path = tmp_path / "manual_result.json"
    write_jsonl_new(raw_path, (example.model_dump(mode="json"),))
    lane_text = (
        "This developed generated paragraph provides auditable evidence in connected "
        "historical prose for the annotation contract."
    )
    generation_path.write_text(
        json.dumps(
            {
                "schema_version": "pdt-real-plan-generation-eval-v2",
                "human_fact_audit_required": True,
                "physical_lane_order": ["stream_0", "stream_1", "stream_2"],
                "active_conditions": ["oracle_plan"],
                "documents": [
                    {
                        "example_id": example.source.source_id,
                        "conditions": {
                            "oracle_plan": {
                                "expected_teacher_to_physical_lane": [0, 1, 2],
                                "text_by_physical_lane": {
                                    "stream_0": lane_text,
                                    "stream_1": lane_text,
                                    "stream_2": lane_text,
                                },
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    count = export_blinded_fact_audit(
        generation_evaluation_path=generation_path,
        raw_examples_path=raw_path,
        queue_path=queue_path,
        key_path=key_path,
        randomization_seed=17,
    )

    assert count == 18 * 3 * 2
    queue = [
        ManualAuditItem.model_validate_json(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
    ]
    key = ManualAuditKey.model_validate_json(key_path.read_text(encoding="utf-8"))
    assert set(queue[0].model_dump()) == {
        "schema_version",
        "item_id",
        "premise_text",
        "hypothesis",
    }
    assert len(key.records) == count

    # These synthetic judgments validate adjudication plumbing only; they are
    # not model outputs and establish no empirical fact-recall result.
    annotations_a: list[ManualAnnotation] = []
    annotations_b: list[ManualAnnotation] = []
    disputed_id = key.records[0].item_id
    for record in key.records:
        entailed = (
            record.query_kind is QueryKind.POSITIVE_FACT
            and record.expected_role.value in {"OWNER", "REFERENCE"}
        )
        judgment = (
            ManualJudgment.ENTAILED
            if entailed
            else ManualJudgment.NOT_ENTAILED
        )
        evidence = "This developed generated paragraph" if entailed else ""
        annotations_a.append(
            ManualAnnotation(
                schema_version=MANUAL_ANNOTATION_SCHEMA,
                item_id=record.item_id,
                annotator_id="annotator-a",
                judgment=judgment,
                evidence_quote=evidence,
            )
        )
        annotations_b.append(
            ManualAnnotation(
                schema_version=MANUAL_ANNOTATION_SCHEMA,
                item_id=record.item_id,
                annotator_id="annotator-b",
                judgment=(
                    ManualJudgment.UNCERTAIN
                    if record.item_id == disputed_id
                    else judgment
                ),
                evidence_quote=(
                    ""
                    if record.item_id == disputed_id
                    else evidence
                ),
            )
        )
    write_jsonl_new(
        annotation_a_path,
        (row.model_dump(mode="json") for row in annotations_a),
    )
    write_jsonl_new(
        annotation_b_path,
        (row.model_dump(mode="json") for row in annotations_b),
    )
    disputed_record = next(
        record for record in key.records if record.item_id == disputed_id
    )
    disputed_entails = (
        disputed_record.query_kind is QueryKind.POSITIVE_FACT
        and disputed_record.expected_role.value in {"OWNER", "REFERENCE"}
    )
    write_jsonl_new(
        adjudication_path,
        (
            ManualAnnotation(
                schema_version=MANUAL_ANNOTATION_SCHEMA,
                item_id=disputed_id,
                annotator_id="adjudicator-c",
                judgment=(
                    ManualJudgment.ENTAILED
                    if disputed_entails
                    else ManualJudgment.NOT_ENTAILED
                ),
                evidence_quote=(
                    "This developed generated paragraph"
                    if disputed_entails
                    else ""
                ),
            ).model_dump(mode="json"),
        ),
    )

    result = adjudicate_fact_audit(
        queue_path=queue_path,
        key_path=key_path,
        annotator_a_path=annotation_a_path,
        annotator_b_path=annotation_b_path,
        adjudicator_path=adjudication_path,
        output_path=result_path,
        bootstrap_samples=1_000,
        confidence_level=0.95,
        minimum_documents=1,
    )

    assert result["primary_disagreements"] == 1
    assert result["resolved_by_adjudicator"] == 1
    assert result["manual_evidence_gate"]["passes"] is True
