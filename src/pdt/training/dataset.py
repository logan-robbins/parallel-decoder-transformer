"""Dataset and collator for the no-hash PDT benchmark schema."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger("pdt.training.dataset")


__all__ = ["PDTDependencyDataset", "PDTKDDataset", "PDTCollator", "SampleBatch"]


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
    shared_ids: torch.Tensor
    shared_attention_mask: torch.Tensor
    local_ids: torch.Tensor
    local_attention_mask: torch.Tensor
    target_block_ids: torch.Tensor
    target_block_labels: torch.Tensor
    target_block_attention_mask: torch.Tensor
    dependency_token_mask: torch.Tensor
    nondependency_token_mask: torch.Tensor
    readiness_targets: torch.Tensor
    readiness_mask: torch.Tensor
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
        LOGGER.info("Loaded %d PDT examples (K=%d) from %s", len(self), self.num_streams, self.path)

    def _validate_record(self, rec: Mapping[str, object], *, line_no: int) -> None:
        forbidden = sorted(HASH_ERA_FIELDS.intersection(rec))
        if forbidden:
            raise ValueError(
                f"{self.path}:{line_no} uses removed hash-era fields: {forbidden}."
            )
        if "stream_inputs" not in rec:
            raise ValueError(f"{self.path}:{line_no} missing required field 'stream_inputs'.")
        streams = rec["stream_inputs"]
        if not isinstance(streams, list) or len(streams) != self.num_streams:
            raise ValueError(
                f"{self.path}:{line_no} expected exactly {self.num_streams} stream_inputs."
            )
        lag = int(rec.get("visibility_lag_blocks", 1))
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

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        return self._samples[idx]


# Backward-compatible import name; the schema is no longer KD.
PDTKDDataset = PDTDependencyDataset


class PDTCollator:
    """Pads canonical examples into fixed tensors for block rollout."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        num_streams: int = 3,
        max_shared_length: int = 256,
        max_local_length: int = 128,
        max_blocks: int = 4,
        max_block_length: int = 128,
        max_snapshots: int = 4,
        **_: object,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.num_streams = num_streams
        self.max_shared_length = max_shared_length
        self.max_local_length = max_local_length
        self.max_blocks = max_blocks
        self.max_block_length = max_block_length
        self.max_snapshots = max_snapshots

    def __call__(self, batch: Sequence[Mapping[str, object]]) -> SampleBatch:
        bsz = len(batch)
        k_streams = self.num_streams
        max_shared = self.max_shared_length
        max_local = self.max_local_length
        max_blocks = self.max_blocks
        max_block = self.max_block_length

        example_ids: List[str] = []
        families: List[str] = []
        stream_labels: List[List[str]] = []
        shared_ids = torch.full((bsz, max_shared), self.pad_token_id, dtype=torch.long)
        shared_mask = torch.zeros((bsz, max_shared), dtype=torch.long)
        local_ids = torch.full((bsz, k_streams, max_local), self.pad_token_id, dtype=torch.long)
        local_mask = torch.zeros((bsz, k_streams, max_local), dtype=torch.long)
        block_ids = torch.full(
            (bsz, k_streams, max_blocks, max_block),
            self.pad_token_id,
            dtype=torch.long,
        )
        block_labels = torch.full((bsz, k_streams, max_blocks, max_block), -100, dtype=torch.long)
        block_mask = torch.zeros((bsz, k_streams, max_blocks, max_block), dtype=torch.long)
        dep_mask = torch.zeros((bsz, k_streams, max_blocks, max_block), dtype=torch.bool)
        nondep_mask = torch.zeros((bsz, k_streams, max_blocks, max_block), dtype=torch.bool)
        readiness_targets = torch.zeros((bsz, self.max_snapshots), dtype=torch.float32)
        readiness_mask = torch.zeros((bsz, self.max_snapshots), dtype=torch.bool)

        for b, rec in enumerate(batch):
            example_ids.append(str(rec.get("example_id", "")))
            families.append(str(rec.get("family", "")))
            shared = _ids(rec, "shared_ids", max_shared)
            if shared:
                shared_ids[b, : len(shared)] = torch.tensor(shared, dtype=torch.long)
                shared_mask[b, : len(shared)] = 1

            streams = list(rec["stream_inputs"])[:k_streams]  # type: ignore[index]
            labels: List[str] = []
            for k, stream in enumerate(streams):
                assert isinstance(stream, Mapping)
                labels.append(str(stream.get("stream_id", f"stream_{k}")))
                local = _ids(stream, "local_ids", max_local)
                if local:
                    local_ids[b, k, : len(local)] = torch.tensor(local, dtype=torch.long)
                    local_mask[b, k, : len(local)] = 1

                target_blocks = stream.get("target_block_ids", [])
                for m, block in enumerate(list(target_blocks)[:max_blocks]):
                    ids = [int(x) for x in list(block)[:max_block]]
                    if not ids:
                        continue
                    block_ids[b, k, m, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    block_labels[b, k, m, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    block_mask[b, k, m, : len(ids)] = 1

                _copy_bool_mask(stream, "dependency_token_mask", dep_mask[b, k])
                explicit_non = _copy_bool_mask(stream, "nondependency_token_mask", nondep_mask[b, k])
                if not explicit_non:
                    nondep_mask[b, k] = block_mask[b, k].bool() & ~dep_mask[b, k]

            stream_labels.append(labels)

            ready = rec.get("readiness_targets", [])
            for i, value in enumerate(list(ready)[: self.max_snapshots]):
                readiness_targets[b, i] = float(value)
                readiness_mask[b, i] = True

        return SampleBatch(
            example_ids=example_ids,
            families=families,
            stream_labels=stream_labels,
            shared_ids=shared_ids,
            shared_attention_mask=shared_mask,
            local_ids=local_ids,
            local_attention_mask=local_mask,
            target_block_ids=block_ids,
            target_block_labels=block_labels,
            target_block_attention_mask=block_mask,
            dependency_token_mask=dep_mask,
            nondependency_token_mask=nondep_mask,
            readiness_targets=readiness_targets,
            readiness_mask=readiness_mask,
            raw=list(batch),
        )


def _ids(rec: Mapping[str, object], field: str, limit: int) -> List[int]:
    values = rec.get(field, [])
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list of token ids after retokenization.")
    return [int(x) for x in values[:limit]]


def _copy_bool_mask(src: Mapping[str, object], field: str, target: torch.Tensor) -> bool:
    values = src.get(field)
    if not values:
        return False
    for block_idx, row in enumerate(list(values)[: target.size(0)]):
        for token_idx, value in enumerate(list(row)[: target.size(1)]):
            target[block_idx, token_idx] = bool(value)
    return True
