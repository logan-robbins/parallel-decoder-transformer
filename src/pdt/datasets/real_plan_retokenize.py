"""Retokenize and embed validated real-plan records for packed PDT training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES
from pdt.datasets.immutable_io import write_jsonl_new
from pdt.datasets.real_plan_schema import (
    FactRole,
    RealPlanExample,
    StreamPlan,
    TargetSection,
    validate_real_plan_example,
)


SEMANTIC_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
SEMANTIC_EMBEDDING_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
SEMANTIC_EMBEDDING_DIM = 1024
PLAN_SEMANTIC_DIM = 512
PLAN_PROJECTION_SEED = 1729
PLAN_PROJECTION_ID = "bge-large-en-v1.5-gaussian-jl-512-seed-1729"
TARGET_BLOCK_TOKENS = 32
MIN_TARGET_TOKENS = 700
MAX_TARGET_TOKENS = 1000
MAX_TARGET_BLOCKS = 32
MAX_PLAN_NODES = 8
FACT_ROLE_INDEX = {
    FactRole.OWNER: 0,
    FactRole.REFERENCE: 1,
    FactRole.ABSENT: 2,
}
TOKENIZED_REAL_PLAN_SCHEMA = "pdt-real-plan-tokenized-v3"

__all__ = [
    "RealPlanRetokenizeConfig",
    "TOKENIZED_REAL_PLAN_SCHEMA",
    "render_planner_prompt",
    "retokenize_real_plan_example",
    "run_real_plan_retokenize",
]


@dataclass(frozen=True, slots=True)
class RealPlanRetokenizeConfig:
    input_path: Path
    output_path: Path
    trunk_profile: str = DEFAULT_TRUNK_PROFILE
    embedding_device: str = "cpu"


def run_real_plan_retokenize(config: RealPlanRetokenizeConfig) -> int:
    if config.input_path.resolve() == config.output_path.resolve():
        raise ValueError("Raw real-plan input is immutable; input and output must differ.")
    if not config.input_path.is_file():
        raise FileNotFoundError(f"Real-plan JSONL does not exist: {config.input_path}")
    if config.output_path.exists():
        raise FileExistsError(f"Refusing to replace processed output: {config.output_path}")
    profile = TRUNK_PROFILES.get(config.trunk_profile)
    if profile is None:
        raise ValueError(
            f"Unknown trunk profile {config.trunk_profile!r}; expected one of "
            f"{tuple(TRUNK_PROFILES)}."
        )

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        profile.base_model,
        revision=profile.revision,
        use_fast=True,
        local_files_only=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Real-plan retokenization requires a fast tokenizer.")
    embedder = SentenceTransformer(
        SEMANTIC_EMBEDDING_MODEL,
        revision=SEMANTIC_EMBEDDING_REVISION,
        device=config.embedding_device,
        local_files_only=True,
    )
    dimension = embedder.get_sentence_embedding_dimension()
    if dimension != SEMANTIC_EMBEDDING_DIM:
        raise RuntimeError(
            f"Semantic embedder dimension must be {SEMANTIC_EMBEDDING_DIM}, got {dimension}."
        )

    def records() -> Any:
        with config.input_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    example = RealPlanExample.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(
                        f"{config.input_path}:{line_number} violates the real-plan schema."
                    ) from exc
                validate_real_plan_example(example)
                yield retokenize_real_plan_example(
                    example,
                    tokenizer=tokenizer,
                    embedder=embedder,
                    tokenizer_name=profile.base_model,
                    tokenizer_revision=profile.revision,
                )

    return write_jsonl_new(config.output_path, records())


def retokenize_real_plan_example(
    example: RealPlanExample,
    *,
    tokenizer: Any,
    embedder: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> dict[str, object]:
    """Materialize one exact training row without mutating the raw example."""

    validate_real_plan_example(example)
    prompt_text = render_planner_prompt(example)
    prompt_ids = _chat_prompt_ids(tokenizer, prompt_text)
    fact_ids = [fact.fact_id for fact in example.facts.facts]
    fact_index = {fact_id: index for index, fact_id in enumerate(fact_ids)}
    fact_query_ids = fact_ids + [
        f"hard_negative:{fact_id}" for fact_id in fact_ids
    ]
    fact_query_texts = [
        fact.statement for fact in example.facts.facts
    ] + [
        fact.hard_negative for fact in example.facts.facts
    ]
    fact_embeddings = _encode(embedder, fact_query_texts)

    target_by_plan = {
        target.plan_id: target for target in example.teacher.target_sections
    }
    plans = list(example.teacher.plans)
    plan_by_id = {plan.plan_id: plan for plan in plans}
    plan_semantic_targets: list[list[list[float]]] = []
    plan_node_mask: list[list[bool]] = []
    fact_route_targets: list[list[list[float]]] = []
    lane_records: list[dict[str, object]] = []
    plan_index = {plan.plan_id: index for index, plan in enumerate(plans)}
    presentation_rank = [0] * 3
    for rank, plan_id in enumerate(example.teacher.presentation_order):
        presentation_rank[plan_index[plan_id]] = rank

    fact_statement = {fact.fact_id: fact.statement for fact in example.facts.facts}
    for plan in plans:
        target = target_by_plan[plan.plan_id]
        node_texts = [_node_semantic_text(plan, node_index, fact_statement) for node_index in range(len(plan.nodes))]
        embedded_nodes = _project_plan_embeddings(_encode(embedder, node_texts))
        padded_nodes = embedded_nodes + [
            [0.0] * PLAN_SEMANTIC_DIM
            for _ in range(MAX_PLAN_NODES - len(embedded_nodes))
        ]
        node_mask = [True] * len(embedded_nodes) + [False] * (
            MAX_PLAN_NODES - len(embedded_nodes)
        )
        route = [
            [0.0] * len(fact_query_ids)
            for _ in range(MAX_PLAN_NODES)
        ]
        for node_index, node in enumerate(plan.nodes):
            for fact_id in node.owned_fact_ids:
                route[node_index][fact_index[fact_id]] = 1.0
        plan_semantic_targets.append(padded_nodes)
        plan_node_mask.append(node_mask)
        fact_route_targets.append(route)
        lane_records.append(
            _retokenize_target(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                plan=plan,
                target=target,
                fact_ids=fact_ids,
                fact_query_count=len(fact_query_ids),
                plan_by_id=plan_by_id,
                target_by_plan=target_by_plan,
            )
        )

    return {
        "schema_version": TOKENIZED_REAL_PLAN_SCHEMA,
        "example_id": example.source.source_id,
        "source_revision_id": example.source.revision_id,
        "source_model_visible_sha256": example.source.model_visible_sha256,
        "source_family_id": example.source.family_id,
        "source_split": example.source.split.value,
        "source_historical_category": example.source.historical_category.value,
        "source_title": example.source.title,
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "semantic_embedding_model": SEMANTIC_EMBEDDING_MODEL,
        "semantic_embedding_revision": SEMANTIC_EMBEDDING_REVISION,
        "semantic_embedding_dim": SEMANTIC_EMBEDDING_DIM,
        "plan_semantic_dim": PLAN_SEMANTIC_DIM,
        "plan_projection_id": PLAN_PROJECTION_ID,
        "block_size_tokens": TARGET_BLOCK_TOKENS,
        "planner_prompt_ids": prompt_ids,
        "fact_ids": fact_ids,
        "fact_query_ids": fact_query_ids,
        "positive_fact_count": len(fact_ids),
        "fact_embeddings": fact_embeddings,
        "plan_semantic_targets": plan_semantic_targets,
        "plan_node_mask": plan_node_mask,
        "fact_route_targets": fact_route_targets,
        "presentation_rank_targets": presentation_rank,
        "lanes": lane_records,
    }


def _retokenize_target(
    *,
    tokenizer: Any,
    prompt_text: str,
    plan: StreamPlan,
    target: TargetSection,
    fact_ids: list[str],
    fact_query_count: int,
    plan_by_id: Mapping[str, StreamPlan],
    target_by_plan: Mapping[str, TargetSection],
) -> dict[str, object]:
    rendered, paragraph_spans = _render_target(plan, target)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets: list[tuple[int, int]] = []
    for pair in encoded["offset_mapping"]:
        if len(pair) != 2:
            raise RuntimeError(f"Tokenizer returned malformed offset pair {pair!r}.")
        offsets.append((int(pair[0]), int(pair[1])))
    if not MIN_TARGET_TOKENS <= len(token_ids) <= MAX_TARGET_TOKENS:
        raise ValueError(
            f"{plan.plan_id} target has {len(token_ids)} Qwen tokens; expected "
            f"{MIN_TARGET_TOKENS}-{MAX_TARGET_TOKENS}."
        )
    prompt_ids = _chat_prompt_ids(tokenizer, prompt_text)
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": rendered},
        ],
        add_generation_prompt=False,
        enable_thinking=False,
        tokenize=True,
    )
    continuation = [int(token_id) for token_id in full_ids[len(prompt_ids) :]]
    if continuation[: len(token_ids)] != token_ids:
        raise ValueError(
            f"{plan.plan_id} target does not compose exactly after the Qwen chat prompt."
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("The pinned Qwen tokenizer must define one EOS token ID.")
    if len(continuation) <= len(token_ids) or continuation[len(token_ids)] != eos_token_id:
        raise ValueError(
            f"{plan.plan_id} chat continuation does not terminate with the pinned EOS token."
        )
    training_token_ids = token_ids + [eos_token_id]
    block_count = math.ceil(len(training_token_ids) / TARGET_BLOCK_TOKENS)
    if block_count > MAX_TARGET_BLOCKS:
        raise ValueError(
            f"{plan.plan_id} requires {block_count} blocks; maximum is {MAX_TARGET_BLOCKS}."
        )
    target_blocks = [
        training_token_ids[start : start + TARGET_BLOCK_TOKENS]
        for start in range(0, len(training_token_ids), TARGET_BLOCK_TOKENS)
    ]
    block_masks = [[True] * len(block) for block in target_blocks]
    outline_progress = _outline_progress_targets(
        offsets,
        paragraph_spans=paragraph_spans,
        paragraph_to_node={
            paragraph.paragraph_id: int(paragraph.outline_node_id.removeprefix("node_"))
            for paragraph in target.paragraphs
        },
    )
    while len(outline_progress) < block_count:
        outline_progress.append(len(plan.nodes) - 1)
    fact_write = [
        [FACT_ROLE_INDEX[FactRole.ABSENT] for _ in range(fact_query_count)]
        for _ in range(block_count)
    ]
    dependency_token_mask = [
        [False] * len(block)
        for block in target_blocks
    ]
    dependency_edges: list[dict[str, object]] = []
    label_by_fact = {label.fact_id: label for label in target.fact_labels}
    for fact_column, fact_id in enumerate(fact_ids):
        label = label_by_fact[fact_id]
        if label.role is FactRole.ABSENT:
            continue
        quote_start = rendered.find(label.target_evidence_quote)
        if quote_start < 0:
            raise ValueError(f"{plan.plan_id}/{fact_id} evidence quote is missing.")
        if rendered.find(label.target_evidence_quote, quote_start + 1) >= 0:
            raise ValueError(
                f"{plan.plan_id}/{fact_id} evidence quote must occur exactly once."
            )
        quote_end = quote_start + len(label.target_evidence_quote)
        touched_blocks = {
            token_index // TARGET_BLOCK_TOKENS
            for token_index, (start, end) in enumerate(offsets)
            if start < quote_end and end > quote_start
        }
        if not touched_blocks:
            raise ValueError(f"{plan.plan_id}/{fact_id} evidence aligns to no tokens.")
        for block_index in touched_blocks:
            fact_write[block_index][fact_column] = FACT_ROLE_INDEX[label.role]
    for node in plan.nodes:
        for dependency in node.dependencies:
            quote = dependency.required_target_evidence_quote
            quote_start = rendered.find(quote)
            if quote_start < 0 or rendered.find(quote, quote_start + 1) >= 0:
                raise ValueError(
                    f"{plan.plan_id}/{node.node_id} dependency evidence must occur "
                    "exactly once."
                )
            quote_end = quote_start + len(quote)
            touched_tokens = {
                token_index
                for token_index, (start, end) in enumerate(offsets)
                if start < quote_end and end > quote_start
            }
            if not touched_tokens:
                raise ValueError(
                    f"{plan.plan_id}/{node.node_id} dependency evidence aligns to no tokens."
                )
            for token_index in touched_tokens:
                block_index, offset_in_block = divmod(
                    token_index,
                    TARGET_BLOCK_TOKENS,
                )
                dependency_token_mask[block_index][offset_in_block] = True
            source_plan = plan_by_id[dependency.source_plan_id]
            source_target = target_by_plan[dependency.source_plan_id]
            source_rendered, _ = _render_target(source_plan, source_target)
            source_quote = dependency.source_evidence_quote
            source_quote_start = source_rendered.find(source_quote)
            if (
                source_quote_start < 0
                or source_rendered.find(source_quote, source_quote_start + 1) >= 0
            ):
                raise ValueError(
                    f"{plan.plan_id}/{node.node_id} source dependency evidence must "
                    "occur exactly once."
                )
            source_encoded = tokenizer(
                source_rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            source_quote_end = source_quote_start + len(source_quote)
            source_touched_tokens = {
                token_index
                for token_index, pair in enumerate(source_encoded["offset_mapping"])
                if int(pair[0]) < source_quote_end and int(pair[1]) > source_quote_start
            }
            if not source_touched_tokens:
                raise ValueError(
                    f"{plan.plan_id}/{node.node_id} source dependency evidence "
                    "aligns to no tokens."
                )
            source_block = max(source_touched_tokens) // TARGET_BLOCK_TOKENS
            first_target_block = min(touched_tokens) // TARGET_BLOCK_TOKENS
            if source_block + 1 > first_target_block:
                raise ValueError(
                    f"{plan.plan_id}/{node.node_id} dependency is not causally visible: "
                    f"source block {source_block} cannot reach target block "
                    f"{first_target_block} with one-block delay."
                )
            dependency_edges.append(
                {
                    "source_plan_id": dependency.source_plan_id,
                    "source_block_index": source_block,
                    "target_token_coordinates": [
                        list(divmod(token_index, TARGET_BLOCK_TOKENS))
                        for token_index in sorted(touched_tokens)
                    ],
                    "fact_ids": list(dependency.fact_ids),
                }
            )
    return {
        "plan_id": plan.plan_id,
        "target_token_count": len(token_ids),
        "training_token_count": len(training_token_ids),
        "target_block_ids": target_blocks,
        "target_block_attention_mask": block_masks,
        "outline_progress_targets": outline_progress,
        "fact_write_targets": fact_write,
        "dependency_token_mask": dependency_token_mask,
        "dependency_edges": dependency_edges,
    }


def _outline_progress_targets(
    offsets: list[tuple[int, int]],
    *,
    paragraph_spans: Mapping[str, tuple[int, int]],
    paragraph_to_node: Mapping[str, int],
) -> list[int]:
    block_count = math.ceil(len(offsets) / TARGET_BLOCK_TOKENS)
    targets: list[int] = []
    ordered_paragraphs = tuple(paragraph_spans)
    for block_index in range(block_count):
        block_offsets = offsets[
            block_index * TARGET_BLOCK_TOKENS : (block_index + 1) * TARGET_BLOCK_TOKENS
        ]
        counts = {node_index: 0 for node_index in paragraph_to_node.values()}
        for start, end in block_offsets:
            for paragraph_id in ordered_paragraphs:
                paragraph_start, paragraph_end = paragraph_spans[paragraph_id]
                if start < paragraph_end and end > paragraph_start:
                    counts[paragraph_to_node[paragraph_id]] += 1
                    break
        if not counts or max(counts.values()) == 0:
            targets.append(0)
        else:
            targets.append(max(counts, key=lambda node_index: (counts[node_index], -node_index)))
    return targets


def _render_target(
    plan: StreamPlan,
    target: TargetSection,
) -> tuple[str, dict[str, tuple[int, int]]]:
    parts = [plan.heading]
    paragraph_spans: dict[str, tuple[int, int]] = {}
    cursor = len(plan.heading)
    for paragraph in target.paragraphs:
        separator = "\n\n"
        cursor += len(separator)
        start = cursor
        parts.append(paragraph.text)
        cursor += len(paragraph.text)
        paragraph_spans[paragraph.paragraph_id] = (start, cursor)
    return "\n\n".join(parts), paragraph_spans


def _node_semantic_text(
    plan: StreamPlan,
    node_index: int,
    fact_statement: Mapping[str, str],
) -> str:
    node = plan.nodes[node_index]
    owned = " ".join(fact_statement[fact_id] for fact_id in node.owned_fact_ids)
    referenced = " ".join(
        fact_statement[fact_id] for fact_id in node.reference_fact_ids
    )
    return (
        f"Section: {plan.heading}. Role: {plan.role_summary}. "
        f"Outline objective: {node.objective}. Owned facts: {owned}. "
        f"Permitted references: {referenced}."
    )


def render_planner_prompt(example: RealPlanExample) -> str:
    parts: list[str] = []
    previous_path: tuple[str, ...] = ()
    for section in example.source.sections:
        common = 0
        for left, right in zip(previous_path, section.heading_path):
            if left != right:
                break
            common += 1
        parts.extend(section.heading_path[common:])
        parts.extend(paragraph.text for paragraph in section.paragraphs)
        previous_path = section.heading_path
    source_text = "\n\n".join(parts)
    return (
        f"Source title: {example.source.title}\n\n{source_text}\n\n"
        f"Task: {example.teacher.expository_prompt}\n\n"
        "Write three complementary long-form sections that jointly answer the task. "
        "Each section must be several connected paragraphs. Do not write a short answer, "
        "a question-answer list, or a separate final synthesis."
    )


def _chat_prompt_ids(tokenizer: Any, user_text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    values = [int(token_id) for token_id in encoded]
    if not values:
        raise ValueError("Planner prompt tokenized empty.")
    return values


def _encode(embedder: Any, texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("Semantic embedding input cannot be empty.")
    vectors = embedder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vectors.shape != (len(texts), SEMANTIC_EMBEDDING_DIM):
        raise RuntimeError(
            "Semantic embedder returned the wrong shape: "
            f"expected {(len(texts), SEMANTIC_EMBEDDING_DIM)}, got {vectors.shape}."
        )
    return [[float(value) for value in row] for row in vectors]


def _project_plan_embeddings(vectors: list[list[float]]) -> list[list[float]]:
    """Apply one frozen Johnson-Lindenstrauss projection into planner width."""

    source = np.asarray(vectors, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != SEMANTIC_EMBEDDING_DIM:
        raise ValueError(
            f"Plan embeddings must have width {SEMANTIC_EMBEDDING_DIM}, got {source.shape}."
        )
    rng = np.random.default_rng(PLAN_PROJECTION_SEED)
    matrix = rng.standard_normal(
        (SEMANTIC_EMBEDDING_DIM, PLAN_SEMANTIC_DIM),
        dtype=np.float32,
    ) / math.sqrt(PLAN_SEMANTIC_DIM)
    projected = source @ matrix
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if bool((norms <= 0).any()):
        raise RuntimeError("Frozen plan projection produced a zero vector.")
    projected /= norms
    return [[float(value) for value in row] for row in projected]
