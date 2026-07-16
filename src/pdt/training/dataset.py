"""Dataset and collator for the no-hash PDT benchmark schema."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from pdt.datasets.retokenize import validate_retokenized_record


LOGGER = logging.getLogger("pdt.training.dataset")


__all__ = ["PDTDependencyDataset", "PDTCollator", "SampleBatch"]


HASH_ERA_FIELDS = {
    "planner_ids",
    "notes_teacher",
    "notes_student",
    "raw_teacher_notes",
    "teacher_snapshots",
    "student_snapshots",
}


@dataclass(slots=True)
class SampleBatch:
    """One batch element is one full K-stream example."""

    example_ids: List[str]
    families: List[str]
    stream_labels: List[List[str]]
    planner_prompt_ids: torch.Tensor
    planner_prompt_attention_mask: torch.Tensor
    stream_prompt_ids: torch.Tensor
    stream_prompt_attention_mask: torch.Tensor
    block_transition_ids: torch.Tensor
    block_transition_attention_mask: torch.Tensor
    teacher_block_prompt_ids: torch.Tensor
    teacher_block_prompt_attention_mask: torch.Tensor
    target_block_ids: torch.Tensor
    target_block_labels: torch.Tensor
    target_block_attention_mask: torch.Tensor
    dependency_token_mask: torch.Tensor
    nondependency_token_mask: torch.Tensor
    raw: List[Mapping[str, object]]


class PDTDependencyDataset(Dataset):
    """Loads one canonical PDT example per JSONL row."""

    def __init__(self, path: str | Path, *, num_streams: int = 3) -> None:
        self.path = Path(path)
        self.num_streams = num_streams
        self._samples: List[Mapping[str, object]] = []
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._validate_record(rec, line_no=line_no)
                self._samples.append(rec)
        if not self._samples:
            raise ValueError(f"{self.path} contains no PDT examples.")
        LOGGER.info("Loaded %d PDT examples (K=%d) from %s", len(self), self.num_streams, self.path)

    def _validate_record(self, rec: Mapping[str, object], *, line_no: int) -> None:
        forbidden = sorted(HASH_ERA_FIELDS.intersection(rec))
        if forbidden:
            raise ValueError(f"{self.path}:{line_no} uses removed hash-era fields: {forbidden}.")
        if "stream_inputs" not in rec:
            raise ValueError(f"{self.path}:{line_no} missing required field 'stream_inputs'.")
        streams = rec["stream_inputs"]
        if not isinstance(streams, list) or len(streams) != self.num_streams:
            raise ValueError(
                f"{self.path}:{line_no} expected exactly {self.num_streams} stream_inputs."
            )
        lag_value = rec.get("visibility_lag_blocks", 1)
        if type(lag_value) is not int:
            raise ValueError(f"{self.path}:{line_no} visibility_lag_blocks must be an integer.")
        lag = lag_value
        for stream in streams:
            if not isinstance(stream, Mapping):
                raise ValueError(f"{self.path}:{line_no} stream_inputs entries must be objects.")
            blocks = stream.get("target_blocks")
            if not isinstance(blocks, list) or not blocks:
                raise ValueError(f"{self.path}:{line_no} each stream needs target_blocks.")
            if lag == 1 and len(blocks) < 2:
                raise ValueError(
                    f"{self.path}:{line_no} Delta=1 examples require at least two target blocks."
                )
        validate_retokenized_record(
            rec,
            line_ref=f"{self.path}:{line_no}",
            expected_streams=self.num_streams,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        return self._samples[idx]


class PDTCollator:
    """Pads canonical examples into fixed tensors for block rollout."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        num_streams: int = 3,
        max_planner_prompt_length: int = 256,
        max_stream_prompt_length: int = 512,
        max_block_transition_length: int = 64,
        max_teacher_prompt_length: int = 2048,
        max_blocks: int = 8,
        max_block_length: int = 32,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.num_streams = num_streams
        self.max_planner_prompt_length = max_planner_prompt_length
        self.max_stream_prompt_length = max_stream_prompt_length
        self.max_block_transition_length = max_block_transition_length
        self.max_teacher_prompt_length = max_teacher_prompt_length
        self.max_blocks = max_blocks
        self.max_block_length = max_block_length

    def __call__(self, batch: Sequence[Mapping[str, object]]) -> SampleBatch:
        if not batch:
            raise ValueError("PDTCollator cannot collate an empty batch.")
        bsz = len(batch)
        k_streams = self.num_streams
        max_planner = self.max_planner_prompt_length
        max_stream_prompt = self.max_stream_prompt_length
        max_transition = self.max_block_transition_length
        max_teacher = self.max_teacher_prompt_length
        max_blocks = self.max_blocks
        max_block = self.max_block_length

        example_ids: List[str] = []
        families: List[str] = []
        stream_labels: List[List[str]] = []
        planner_ids = torch.full((bsz, max_planner), self.pad_token_id, dtype=torch.long)
        planner_mask = torch.zeros((bsz, max_planner), dtype=torch.long)
        stream_prompt_ids = torch.full(
            (bsz, k_streams, max_stream_prompt), self.pad_token_id, dtype=torch.long
        )
        stream_prompt_mask = torch.zeros((bsz, k_streams, max_stream_prompt), dtype=torch.long)
        transition_ids = torch.full(
            (bsz, k_streams, max_blocks, max_transition),
            self.pad_token_id,
            dtype=torch.long,
        )
        transition_mask = torch.zeros(
            (bsz, k_streams, max_blocks, max_transition), dtype=torch.long
        )
        teacher_block_ids = torch.full(
            (bsz, max_blocks, max_teacher), self.pad_token_id, dtype=torch.long
        )
        teacher_block_mask = torch.zeros((bsz, max_blocks, max_teacher), dtype=torch.long)
        block_ids = torch.full(
            (bsz, k_streams, max_blocks, max_block),
            self.pad_token_id,
            dtype=torch.long,
        )
        block_labels = torch.full((bsz, k_streams, max_blocks, max_block), -100, dtype=torch.long)
        block_mask = torch.zeros((bsz, k_streams, max_blocks, max_block), dtype=torch.long)
        dep_mask = torch.zeros((bsz, k_streams, max_blocks, max_block), dtype=torch.bool)
        nondep_mask = torch.zeros((bsz, k_streams, max_blocks, max_block), dtype=torch.bool)

        for b, rec in enumerate(batch):
            example_ids.append(str(rec.get("example_id", "")))
            families.append(str(rec.get("family", "")))
            record_block_size_value = rec.get("block_size_tokens", -1)
            if type(record_block_size_value) is not int:
                raise ValueError(f"{example_ids[-1]} block_size_tokens must be an integer.")
            record_block_size = record_block_size_value
            if record_block_size != max_block:
                raise ValueError(
                    f"{example_ids[-1]} block_size_tokens={record_block_size}, but collator "
                    f"max_block_length={max_block}; train/runtime tau must match exactly."
                )
            planner = _ids(rec, "planner_prompt_ids", max_planner)
            planner_ids[b, : len(planner)] = torch.tensor(planner, dtype=torch.long)
            planner_mask[b, : len(planner)] = 1
            teacher_prompts = rec.get("teacher_block_prompt_ids")
            if not isinstance(teacher_prompts, list) or not teacher_prompts:
                raise ValueError("teacher_block_prompt_ids must be a non-empty list.")
            if len(teacher_prompts) > max_blocks:
                raise ValueError(
                    f"{example_ids[-1]} has {len(teacher_prompts)} teacher block prompts; "
                    f"max_blocks={max_blocks} would truncate them."
                )
            for block_idx, prompt_value in enumerate(teacher_prompts):
                prompt_record = {"prompt": prompt_value}
                prompt = _ids(prompt_record, "prompt", max_teacher)
                teacher_block_ids[b, block_idx, : len(prompt)] = torch.tensor(
                    prompt, dtype=torch.long
                )
                teacher_block_mask[b, block_idx, : len(prompt)] = 1

            streams_value = rec.get("stream_inputs")
            if not isinstance(streams_value, list):
                raise ValueError(f"{example_ids[-1]} stream_inputs must be a list.")
            streams = streams_value
            if len(streams) != k_streams:
                raise ValueError(
                    f"{example_ids[-1]} has {len(streams)} streams; expected {k_streams}."
                )
            labels: List[str] = []
            for k, stream in enumerate(streams):
                assert isinstance(stream, Mapping)
                labels.append(str(stream.get("stream_id", f"stream_{k}")))
                prompt = _ids(stream, "stream_prompt_ids", max_stream_prompt)
                stream_prompt_ids[b, k, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
                stream_prompt_mask[b, k, : len(prompt)] = 1

                target_blocks = stream.get("target_block_ids", [])
                if len(list(target_blocks)) > max_blocks:
                    raise ValueError(
                        f"{example_ids[-1]} {labels[-1]} has {len(list(target_blocks))} "
                        f"blocks; max_blocks={max_blocks} would truncate targets."
                    )
                transitions = stream.get("block_transition_ids")
                if not isinstance(transitions, list) or len(transitions) != len(
                    list(target_blocks)
                ):
                    raise ValueError(
                        f"{example_ids[-1]} {labels[-1]} must have one block transition "
                        "row per target block."
                    )
                for m, transition in enumerate(transitions):
                    ids = _ids_with_empty(
                        transition,
                        max_transition,
                        field=f"block_transition_ids[{m}]",
                    )
                    if m == 0 and ids:
                        raise ValueError("block transition 0 must be empty.")
                    if m > 0 and not ids:
                        raise ValueError(f"block transition {m} must be non-empty.")
                    if ids:
                        transition_ids[b, k, m, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                        transition_mask[b, k, m, : len(ids)] = 1
                for m, block in enumerate(list(target_blocks)):
                    ids = [int(x) for x in list(block)]
                    if len(ids) != max_block:
                        raise ValueError(
                            f"{example_ids[-1]} {labels[-1]} block {m} has {len(ids)} tokens; "
                            f"expected exactly tau={max_block}."
                        )
                    block_ids[b, k, m, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    block_labels[b, k, m, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    block_mask[b, k, m, : len(ids)] = 1

                _copy_bool_mask(stream, "dependency_token_mask", dep_mask[b, k])
                _copy_bool_mask(stream, "nondependency_token_mask", nondep_mask[b, k])

            stream_labels.append(labels)

        return SampleBatch(
            example_ids=example_ids,
            families=families,
            stream_labels=stream_labels,
            planner_prompt_ids=planner_ids,
            planner_prompt_attention_mask=planner_mask,
            stream_prompt_ids=stream_prompt_ids,
            stream_prompt_attention_mask=stream_prompt_mask,
            block_transition_ids=transition_ids,
            block_transition_attention_mask=transition_mask,
            teacher_block_prompt_ids=teacher_block_ids,
            teacher_block_prompt_attention_mask=teacher_block_mask,
            target_block_ids=block_ids,
            target_block_labels=block_labels,
            target_block_attention_mask=block_mask,
            dependency_token_mask=dep_mask,
            nondependency_token_mask=nondep_mask,
            raw=list(batch),
        )


def _ids(rec: Mapping[str, object], field: str, limit: int) -> List[int]:
    values = rec.get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list of token IDs after retokenization.")
    if not values:
        raise ValueError(f"{field} must contain at least one token ID.")
    if len(values) > limit:
        raise ValueError(
            f"{field} has {len(values)} tokens; configured limit {limit} would truncate it."
        )
    return [int(x) for x in values]


def _ids_with_empty(value: object, limit: int, *, field: str) -> List[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of token IDs after retokenization.")
    if len(value) > limit:
        raise ValueError(
            f"{field} has {len(value)} tokens; configured limit {limit} would truncate it."
        )
    return [int(item) for item in value]


def _copy_bool_mask(src: Mapping[str, object], field: str, target: torch.Tensor) -> None:
    values = src.get(field)
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list after retokenization.")
    for block_idx, row in enumerate(values):
        for token_idx, value in enumerate(list(row)):
            target[block_idx, token_idx] = bool(value)
