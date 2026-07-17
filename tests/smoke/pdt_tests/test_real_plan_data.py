"""Source-grounded schema, Batch request, retokenization, and lane binding."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from pdt.datasets.real_plan_batch import (
    CANONICAL_BATCH_MODEL,
    build_fact_requests,
    build_joint_requests,
)
from pdt.datasets.immutable_io import write_jsonl_new
from pdt.datasets.real_plan_retokenize import retokenize_real_plan_example
from pdt.datasets.real_plan_schema import RealPlanExample, validate_real_plan_example
from pdt.training.dataset import RealPlanCollator, RealPlanDataset


def _example() -> RealPlanExample:
    source_paragraphs: list[dict[str, object]] = []
    facts: list[dict[str, object]] = []
    for paragraph_index in range(6):
        fact_indices = (2 * paragraph_index, 2 * paragraph_index + 1)
        quotes = [
            f"Source fact {fact_index:03d} states a precise grounded relationship."
            for fact_index in fact_indices
        ]
        text = (
            " ".join(quotes)
            + " "
            + "contextual material " * 450
        ).strip()
        source_paragraphs.append(
            {
                "paragraph_id": f"p{paragraph_index:03d}",
                "text": text,
            }
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
                        }
                    ],
                    "hard_negative": (
                        f"Fact {fact_index:03d} records a deliberately contradicted relationship."
                    ),
                }
            )

    owner_sets = {
        "plan_0": [f"fact_{index:03d}" for index in range(0, 4)],
        "plan_1": [f"fact_{index:03d}" for index in range(4, 8)],
        "plan_2": [f"fact_{index:03d}" for index in range(8, 12)],
    }
    reference_sets = {
        "plan_0": [("plan_1", "fact_004"), ("plan_2", "fact_008")],
        "plan_1": [("plan_0", "fact_000"), ("plan_2", "fact_008")],
        "plan_2": [("plan_0", "fact_000"), ("plan_1", "fact_004")],
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
                * 12
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
                    for fact_index in range(12)
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
            "schema_version": "pdt-real-plan-v1",
            "source": {
                "source_id": "wikipedia-test-article",
                "title": "A Grounded Test Article",
                "source_url": "https://example.test/article",
                "license": "CC BY-SA 3.0",
                "paragraphs": source_paragraphs,
            },
            "facts": {
                "source_id": "wikipedia-test-article",
                "facts": facts,
            },
            "teacher": {
                "source_id": "wikipedia-test-article",
                "expository_prompt": (
                    "Explain the grounded subject through three complementary historical sections."
                ),
                "plans": plans,
                "target_sections": targets,
                "presentation_order": ["plan_1", "plan_0", "plan_2"],
            },
        }
    )


class _FakeTokenizer:
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


class _FakeEmbedder:
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


def test_batch_requests_use_one_pinned_responses_schema_path(tmp_path: Path) -> None:
    example = _example()
    source_path = tmp_path / "sources.jsonl"
    facts_path = tmp_path / "facts.jsonl"
    fact_requests = tmp_path / "fact_requests.jsonl"
    joint_requests = tmp_path / "joint_requests.jsonl"
    source_path.write_text(
        example.source.model_dump_json() + "\n",
        encoding="utf-8",
    )
    facts_path.write_text(
        example.facts.model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert build_fact_requests(source_path, fact_requests) == 1
    assert build_joint_requests(source_path, facts_path, joint_requests) == 1
    joint = json.loads(joint_requests.read_text(encoding="utf-8"))
    assert joint["url"] == "/v1/responses"
    assert joint["body"]["model"] == CANONICAL_BATCH_MODEL
    assert joint["body"]["text"]["format"]["type"] == "json_schema"
    assert joint["body"]["text"]["format"]["strict"] is True
    prompt = joint["body"]["input"][0]["content"][0]["text"]
    assert "exactly three complementary" in prompt
    assert "short or choppy sentences" in prompt
    assert "receive at least one dependency from each sibling" in prompt


def test_retokenization_uses_true_and_hard_negative_fact_queries() -> None:
    record = retokenize_real_plan_example(
        _example(),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        tokenizer_name="fake-qwen",
        tokenizer_revision="fake-revision",
    )
    assert record["positive_fact_count"] == 12
    assert len(record["fact_query_ids"]) == 24
    assert len(record["fact_embeddings"]) == 24
    assert len(record["fact_route_targets"][0][0]) == 24
    for lane in record["lanes"]:
        for block_labels in lane["fact_write_targets"]:
            assert block_labels[12:] == [2] * 12


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
        source_paragraph["text"].removeprefix(quote).strip() + " " + quote
    )
    example = RealPlanExample.model_validate(payload)
    validate_real_plan_example(example)

    with pytest.raises(ValueError, match="not causally visible"):
        retokenize_real_plan_example(
            example,
            tokenizer=_FakeTokenizer(),
            embedder=_FakeEmbedder(),
            tokenizer_name="fake-qwen",
            tokenizer_revision="fake-revision",
        )


def test_collator_randomly_rebinds_every_teacher_lane_axis(tmp_path: Path) -> None:
    record = retokenize_real_plan_example(
        _example(),
        tokenizer=_FakeTokenizer(),
        embedder=_FakeEmbedder(),
        tokenizer_name="fake-qwen",
        tokenizer_revision="fake-revision",
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
        expected_tokenizer="fake-qwen",
        expected_tokenizer_revision="fake-revision",
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
    assert batch.fact_mask[0, :24].all()
    assert batch.positive_fact_mask[0, :12].all()
    assert not batch.positive_fact_mask[0, 12:].any()


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
