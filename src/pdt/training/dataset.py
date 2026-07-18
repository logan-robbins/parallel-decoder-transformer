"""Dataset and collator for source-grounded three-lane real-plan training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, cast

import torch
from torch.utils.data import Dataset

from pdt.datasets.real_plan_retokenize import (
    MAX_PLAN_NODES,
    MAX_TARGET_BLOCKS,
    PLAN_PROJECTION_ID,
    PLAN_SEMANTIC_DIM,
    SEMANTIC_EMBEDDING_DIM,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_EMBEDDING_REVISION,
    TARGET_BLOCK_TOKENS,
    TOKENIZED_REAL_PLAN_SCHEMA,
)
from pdt.datasets.real_plan_schema import MAX_EXTRACTED_FACTS, MIN_EXTRACTED_FACTS


LOGGER = logging.getLogger("pdt.training.dataset")

__all__ = ["RealPlanCollator", "RealPlanDataset", "SampleBatch"]

NUM_LANES = 3
MAX_SOURCE_FACTS = MAX_EXTRACTED_FACTS
MAX_FACT_QUERIES = 2 * MAX_SOURCE_FACTS
ABSENT_ROLE_INDEX = 2


@dataclass(slots=True)
class SampleBatch:
    """One batch preserves physical lane, outline-node, block, and fact axes."""

    example_ids: List[str]
    planner_prompt_ids: torch.Tensor
    planner_prompt_attention_mask: torch.Tensor
    target_block_ids: torch.Tensor
    target_block_labels: torch.Tensor
    target_block_attention_mask: torch.Tensor
    fact_embeddings: torch.Tensor
    fact_mask: torch.Tensor
    positive_fact_mask: torch.Tensor
    plan_semantic_targets: torch.Tensor
    plan_node_mask: torch.Tensor
    fact_route_targets: torch.Tensor
    outline_progress_targets: torch.Tensor
    fact_write_targets: torch.Tensor
    dependency_token_mask: torch.Tensor
    presentation_rank_targets: torch.Tensor
    raw: List[Mapping[str, object]]


class RealPlanDataset(Dataset):
    """Load one validated, retokenized real-plan document per JSONL row."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_tokenizer: str,
        expected_tokenizer_revision: str,
    ) -> None:
        self.path = Path(path)
        self.expected_tokenizer = expected_tokenizer
        self.expected_tokenizer_revision = expected_tokenizer_revision
        self._samples: list[Mapping[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Real-plan dataset does not exist: {self.path}")
        seen: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{line_number} is not valid JSON."
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"{self.path}:{line_number} must be a JSON object.")
                self._validate_record(record, line_number=line_number)
                example_id = str(record["example_id"])
                if example_id in seen:
                    raise ValueError(
                        f"{self.path}:{line_number} repeats example_id={example_id!r}."
                    )
                seen.add(example_id)
                self._samples.append(record)
        if not self._samples:
            raise ValueError(f"{self.path} contains no real-plan records.")
        LOGGER.info("Loaded %d real-plan examples from %s.", len(self), self.path)

    def _validate_record(
        self,
        record: Mapping[str, object],
        *,
        line_number: int,
    ) -> None:
        context = f"{self.path}:{line_number}"
        if record.get("schema_version") != TOKENIZED_REAL_PLAN_SCHEMA:
            raise ValueError(f"{context} has the wrong schema_version.")
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"{context} example_id must be non-empty text.")
        source_revision_id = record.get("source_revision_id")
        if type(source_revision_id) is not int or source_revision_id <= 0:
            raise ValueError(f"{context} source_revision_id must be a positive integer.")
        source_digest = record.get("source_model_visible_sha256")
        if (
            not isinstance(source_digest, str)
            or len(source_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_digest)
        ):
            raise ValueError(
                f"{context} source_model_visible_sha256 must be lowercase SHA-256."
            )
        source_family_id = record.get("source_family_id")
        if (
            not isinstance(source_family_id, str)
            or not source_family_id.startswith("family-")
        ):
            raise ValueError(f"{context} source_family_id must be present.")
        if record.get("source_split") not in {"train", "validation", "test"}:
            raise ValueError(f"{context} source_split is invalid.")
        if not isinstance(record.get("source_historical_category"), str):
            raise ValueError(f"{context} source_historical_category must be present.")
        if record.get("tokenizer") != self.expected_tokenizer:
            raise ValueError(f"{context} tokenizer does not match the selected trunk.")
        if record.get("tokenizer_revision") != self.expected_tokenizer_revision:
            raise ValueError(f"{context} tokenizer revision does not match the selected trunk.")
        if record.get("semantic_embedding_model") != SEMANTIC_EMBEDDING_MODEL:
            raise ValueError(f"{context} semantic embedding model is not canonical.")
        if record.get("semantic_embedding_revision") != SEMANTIC_EMBEDDING_REVISION:
            raise ValueError(f"{context} semantic embedding revision is not canonical.")
        if record.get("semantic_embedding_dim") != SEMANTIC_EMBEDDING_DIM:
            raise ValueError(f"{context} semantic embedding dimension is not canonical.")
        if record.get("plan_semantic_dim") != PLAN_SEMANTIC_DIM:
            raise ValueError(f"{context} plan semantic dimension is not canonical.")
        if record.get("plan_projection_id") != PLAN_PROJECTION_ID:
            raise ValueError(f"{context} plan projection identity is not canonical.")
        if record.get("block_size_tokens") != TARGET_BLOCK_TOKENS:
            raise ValueError(f"{context} target block size must equal {TARGET_BLOCK_TOKENS}.")
        _required_int_list(record, "planner_prompt_ids", context=context)
        fact_ids = _required_list(record, "fact_ids", context=context)
        if (
            not MIN_EXTRACTED_FACTS <= len(fact_ids) <= MAX_SOURCE_FACTS
            or len(set(map(str, fact_ids))) != len(fact_ids)
        ):
            raise ValueError(
                f"{context} fact_ids must contain {MIN_EXTRACTED_FACTS}-"
                f"{MAX_SOURCE_FACTS} unique IDs."
            )
        fact_query_ids = _required_list(record, "fact_query_ids", context=context)
        expected_query_ids = list(map(str, fact_ids)) + [
            f"hard_negative:{fact_id}" for fact_id in fact_ids
        ]
        if list(map(str, fact_query_ids)) != expected_query_ids:
            raise ValueError(
                f"{context} fact_query_ids must pair every source fact with its hard negative."
            )
        if record.get("positive_fact_count") != len(fact_ids):
            raise ValueError(f"{context} positive_fact_count must equal len(fact_ids).")
        fact_embeddings = _required_list(record, "fact_embeddings", context=context)
        if len(fact_embeddings) != len(fact_query_ids):
            raise ValueError(f"{context} fact_embeddings must align with fact_query_ids.")
        for vector in fact_embeddings:
            _validate_vector(vector, width=SEMANTIC_EMBEDDING_DIM, context=context)
        _validate_shape(
            record.get("plan_semantic_targets"),
            (NUM_LANES, MAX_PLAN_NODES, PLAN_SEMANTIC_DIM),
            context=f"{context} plan_semantic_targets",
        )
        _validate_shape(
            record.get("plan_node_mask"),
            (NUM_LANES, MAX_PLAN_NODES),
            context=f"{context} plan_node_mask",
        )
        _validate_shape(
            record.get("fact_route_targets"),
            (NUM_LANES, MAX_PLAN_NODES, len(fact_query_ids)),
            context=f"{context} fact_route_targets",
        )
        ranks = _required_list(record, "presentation_rank_targets", context=context)
        if any(type(rank) is not int for rank in ranks):
            raise ValueError(f"{context} presentation ranks must be integers.")
        if sorted(cast(list[int], ranks)) != [0, 1, 2]:
            raise ValueError(f"{context} presentation ranks must be a permutation of 0,1,2.")
        lanes = _required_list(record, "lanes", context=context)
        if len(lanes) != NUM_LANES:
            raise ValueError(f"{context} must contain exactly {NUM_LANES} lanes.")
        for lane_index, lane in enumerate(lanes):
            if not isinstance(lane, Mapping):
                raise ValueError(f"{context} lane {lane_index} must be an object.")
            blocks = _required_list(lane, "target_block_ids", context=context)
            block_masks = _required_list(
                lane,
                "target_block_attention_mask",
                context=context,
            )
            progress = _required_list(
                lane,
                "outline_progress_targets",
                context=context,
            )
            writes = _required_list(lane, "fact_write_targets", context=context)
            dependencies = _required_list(
                lane,
                "dependency_token_mask",
                context=context,
            )
            dependency_edges = _required_list(
                lane,
                "dependency_edges",
                context=context,
            )
            target_token_count = lane.get("target_token_count")
            training_token_count = lane.get("training_token_count")
            if (
                type(target_token_count) is not int
                or not 700 <= target_token_count <= 1000
                or type(training_token_count) is not int
                or training_token_count != target_token_count + 1
            ):
                raise ValueError(
                    f"{context} lane {lane_index} requires 700-1000 prose tokens "
                    "followed by exactly one training EOS token."
                )
            if not 1 <= len(blocks) <= MAX_TARGET_BLOCKS:
                raise ValueError(f"{context} lane {lane_index} has invalid block count.")
            if sum(len(block) for block in blocks if isinstance(block, list)) != (
                training_token_count
            ):
                raise ValueError(
                    f"{context} lane {lane_index} training token count is inconsistent."
                )
            if not (
                len(blocks)
                == len(block_masks)
                == len(progress)
                == len(writes)
                == len(dependencies)
            ):
                raise ValueError(f"{context} lane {lane_index} block labels are misaligned.")
            dependency_observations = 0
            for block, mask, write, dependency in zip(
                blocks,
                block_masks,
                writes,
                dependencies,
                strict=True,
            ):
                if (
                    not isinstance(block, list)
                    or not isinstance(mask, list)
                    or not isinstance(dependency, list)
                ):
                    raise ValueError(f"{context} target block IDs and masks must be lists.")
                if (
                    not 1 <= len(block) <= TARGET_BLOCK_TOKENS
                    or len(block) != len(mask)
                    or len(block) != len(dependency)
                ):
                    raise ValueError(f"{context} target block shape is invalid.")
                if any(type(value) is not bool for value in dependency):
                    raise ValueError(
                        f"{context} lane {lane_index} dependency mask must be boolean."
                    )
                dependency_observations += sum(dependency)
                if not isinstance(write, list) or len(write) != len(fact_query_ids):
                    raise ValueError(
                        f"{context} fact-write labels must cover every fact query."
                    )
                if any(type(role) is not int or role not in (0, 1, 2) for role in write):
                    raise ValueError(f"{context} fact-write roles must be 0, 1, or 2.")
                if any(
                    role != ABSENT_ROLE_INDEX
                    for role in write[len(fact_ids) :]
                ):
                    raise ValueError(
                        f"{context} hard-negative fact queries must always be ABSENT."
                    )
            if bool(dependency_observations) != bool(dependency_edges):
                raise ValueError(
                    f"{context} lane {lane_index} dependency masks and causal edges "
                    "must either both be empty or both be present."
                )
            for edge in dependency_edges:
                if not isinstance(edge, Mapping):
                    raise ValueError(f"{context} dependency edges must be objects.")
                source_plan_id = edge.get("source_plan_id")
                source_block = edge.get("source_block_index")
                coordinates = edge.get("target_token_coordinates")
                edge_fact_ids = edge.get("fact_ids")
                if (
                    not isinstance(source_plan_id, str)
                    or source_plan_id == lane.get("plan_id")
                    or type(source_block) is not int
                    or source_block < 0
                    or not isinstance(coordinates, list)
                    or not coordinates
                    or not isinstance(edge_fact_ids, list)
                    or not edge_fact_ids
                    or not set(map(str, edge_fact_ids)) <= set(map(str, fact_ids))
                ):
                    raise ValueError(f"{context} dependency edge timing is malformed.")
                for coordinate in coordinates:
                    if (
                        not isinstance(coordinate, list)
                        or len(coordinate) != 2
                        or any(type(index) is not int for index in coordinate)
                    ):
                        raise ValueError(
                            f"{context} dependency token coordinates are malformed."
                        )
                    block_index = cast(int, coordinate[0])
                    token_offset = cast(int, coordinate[1])
                    if not 0 <= block_index < len(dependencies):
                        raise ValueError(
                            f"{context} dependency edge violates delayed token alignment."
                        )
                    dependency_row = dependencies[block_index]
                    if not isinstance(dependency_row, list) or (
                        not 0 <= token_offset < len(dependency_row)
                        or dependency_row[token_offset] is not True
                        or source_block + 1 > block_index
                    ):
                        raise ValueError(
                            f"{context} dependency edge violates delayed token alignment."
                        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self._samples[index]


class RealPlanCollator:
    """Pad real-plan rows and randomly bind teacher plans to physical lanes."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        max_planner_prompt_length: int,
        seed: int = 1729,
    ) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative.")
        if max_planner_prompt_length <= 0:
            raise ValueError("max_planner_prompt_length must be positive.")
        self.pad_token_id = pad_token_id
        self.max_planner_prompt_length = max_planner_prompt_length
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def __call__(self, batch: Sequence[Mapping[str, object]]) -> SampleBatch:
        if not batch:
            raise ValueError("RealPlanCollator cannot collate an empty batch.")
        batch_size = len(batch)
        prompts = [
            _required_int_list(
                record,
                "planner_prompt_ids",
                context=str(record.get("example_id", f"batch_{index}")),
            )
            for index, record in enumerate(batch)
        ]
        too_long = [
            (str(record.get("example_id", f"batch_{index}")), len(prompt))
            for index, (record, prompt) in enumerate(zip(batch, prompts, strict=True))
            if len(prompt) > self.max_planner_prompt_length
        ]
        if too_long:
            raise ValueError(
                "Planner prompts exceed the configured fail-fast maximum "
                f"{self.max_planner_prompt_length} and cannot be truncated: {too_long}."
            )
        prompt_width = max(len(prompt) for prompt in prompts)
        prompt_ids = torch.full(
            (batch_size, prompt_width),
            self.pad_token_id,
            dtype=torch.long,
        )
        prompt_mask = torch.zeros_like(prompt_ids, dtype=torch.bool)
        target_ids = torch.full(
            (
                batch_size,
                NUM_LANES,
                MAX_TARGET_BLOCKS,
                TARGET_BLOCK_TOKENS,
            ),
            self.pad_token_id,
            dtype=torch.long,
        )
        target_labels = torch.full_like(target_ids, -100)
        target_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        fact_embeddings = torch.zeros(
            batch_size,
            MAX_FACT_QUERIES,
            SEMANTIC_EMBEDDING_DIM,
            dtype=torch.float32,
        )
        fact_mask = torch.zeros(batch_size, MAX_FACT_QUERIES, dtype=torch.bool)
        positive_fact_mask = torch.zeros(
            batch_size,
            MAX_FACT_QUERIES,
            dtype=torch.bool,
        )
        plan_semantic = torch.zeros(
            batch_size,
            NUM_LANES,
            MAX_PLAN_NODES,
            PLAN_SEMANTIC_DIM,
            dtype=torch.float32,
        )
        plan_mask = torch.zeros(
            batch_size,
            NUM_LANES,
            MAX_PLAN_NODES,
            dtype=torch.bool,
        )
        fact_route = torch.zeros(
            batch_size,
            NUM_LANES,
            MAX_PLAN_NODES,
            MAX_FACT_QUERIES,
            dtype=torch.float32,
        )
        progress = torch.full(
            (batch_size, NUM_LANES, MAX_TARGET_BLOCKS),
            -100,
            dtype=torch.long,
        )
        fact_write = torch.full(
            (batch_size, NUM_LANES, MAX_TARGET_BLOCKS, MAX_FACT_QUERIES),
            -100,
            dtype=torch.long,
        )
        dependency_mask = torch.zeros(
            (
                batch_size,
                NUM_LANES,
                MAX_TARGET_BLOCKS,
                TARGET_BLOCK_TOKENS,
            ),
            dtype=torch.bool,
        )
        presentation = torch.empty(batch_size, NUM_LANES, dtype=torch.long)
        example_ids: list[str] = []

        for batch_index, (record, prompt) in enumerate(
            zip(batch, prompts, strict=True)
        ):
            example_ids.append(str(record["example_id"]))
            prompt_ids[batch_index, : len(prompt)] = torch.tensor(prompt)
            prompt_mask[batch_index, : len(prompt)] = True
            raw_facts = torch.tensor(record["fact_embeddings"], dtype=torch.float32)
            fact_count = raw_facts.size(0)
            positive_count_value = record.get("positive_fact_count")
            if type(positive_count_value) is not int:
                raise ValueError(
                    f"{example_ids[-1]} positive_fact_count must be an integer."
                )
            positive_count = positive_count_value
            fact_embeddings[batch_index, :fact_count] = raw_facts
            fact_mask[batch_index, :fact_count] = True
            positive_fact_mask[batch_index, :positive_count] = True

            permutation = torch.randperm(NUM_LANES, generator=self.generator)
            raw_plan_semantic = torch.tensor(
                record["plan_semantic_targets"],
                dtype=torch.float32,
            ).index_select(0, permutation)
            raw_plan_mask = torch.tensor(
                record["plan_node_mask"],
                dtype=torch.bool,
            ).index_select(0, permutation)
            raw_route = torch.tensor(
                record["fact_route_targets"],
                dtype=torch.float32,
            ).index_select(0, permutation)
            plan_semantic[batch_index] = raw_plan_semantic
            plan_mask[batch_index] = raw_plan_mask
            fact_route[batch_index, :, :, :fact_count] = raw_route
            raw_ranks = torch.tensor(
                record["presentation_rank_targets"],
                dtype=torch.long,
            ).index_select(0, permutation)
            presentation[batch_index] = raw_ranks
            lanes = record["lanes"]
            if not isinstance(lanes, list):
                raise ValueError(f"{example_ids[-1]} lanes must be a list.")
            for physical_lane, teacher_lane in enumerate(permutation.tolist()):
                lane = lanes[teacher_lane]
                if not isinstance(lane, Mapping):
                    raise ValueError(f"{example_ids[-1]} lane must be an object.")
                blocks = lane["target_block_ids"]
                masks = lane["target_block_attention_mask"]
                progress_rows = lane["outline_progress_targets"]
                write_rows = lane["fact_write_targets"]
                dependency_rows = lane["dependency_token_mask"]
                for block_index, (block, mask) in enumerate(
                    zip(blocks, masks, strict=True)  # type: ignore[arg-type]
                ):
                    values = torch.tensor(block, dtype=torch.long)
                    valid = torch.tensor(mask, dtype=torch.bool)
                    width = values.numel()
                    target_ids[batch_index, physical_lane, block_index, :width] = values
                    target_labels[
                        batch_index,
                        physical_lane,
                        block_index,
                        :width,
                    ] = values
                    target_mask[
                        batch_index,
                        physical_lane,
                        block_index,
                        :width,
                    ] = valid
                    dependency_mask[
                        batch_index,
                        physical_lane,
                        block_index,
                        :width,
                    ] = torch.tensor(
                        dependency_rows[block_index],  # type: ignore[index]
                        dtype=torch.bool,
                    )
                progress_count = len(progress_rows)  # type: ignore[arg-type]
                progress[
                    batch_index,
                    physical_lane,
                    :progress_count,
                ] = torch.tensor(progress_rows, dtype=torch.long)
                write_count = len(write_rows)  # type: ignore[arg-type]
                fact_write[
                    batch_index,
                    physical_lane,
                    :write_count,
                    :fact_count,
                ] = torch.tensor(write_rows, dtype=torch.long)
        return SampleBatch(
            example_ids=example_ids,
            planner_prompt_ids=prompt_ids,
            planner_prompt_attention_mask=prompt_mask,
            target_block_ids=target_ids,
            target_block_labels=target_labels,
            target_block_attention_mask=target_mask,
            fact_embeddings=fact_embeddings,
            fact_mask=fact_mask,
            positive_fact_mask=positive_fact_mask,
            plan_semantic_targets=plan_semantic,
            plan_node_mask=plan_mask,
            fact_route_targets=fact_route,
            outline_progress_targets=progress,
            fact_write_targets=fact_write,
            dependency_token_mask=dependency_mask,
            presentation_rank_targets=presentation,
            raw=list(batch),
        )


def _required_list(
    record: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> list[object]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{context} {field} must be a list.")
    return value


def _required_int_list(
    record: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> list[int]:
    values = _required_list(record, field, context=context)
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{context} {field} must contain non-negative integer IDs.")
    return cast(list[int], values)


def _validate_vector(value: object, *, width: int, context: str) -> None:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{context} expected vector width {width}.")
    if any(not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{context} embedding vectors must be numeric.")


def _validate_shape(value: object, shape: tuple[int, ...], *, context: str) -> None:
    if not shape:
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{context} must have shape {shape}.")
    for child in value:
        _validate_shape(child, shape[1:], context=context)
