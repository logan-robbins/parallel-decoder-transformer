"""Dataset + collator for the Qwen3 PDT KD JSONL.

The on-disk schema is produced by ``pdt.datasets.retokenize`` and contains:

- ``student_ids`` / ``student_labels``: Qwen3 token ids (variable length).
- ``planner_ids`` (length ~= 10): int in ``[0, V_p)``; ground-truth latent
  plan-slot targets.
- ``notes_student`` / ``notes_teacher``: ``(K, d_notes)`` lists of floats.
- ``teacher_snapshots`` / ``student_snapshots``: per-snapshot ``notes``.
- ``continuation_sufficiency_labels``: list[int], one per snapshot slot.
- ``plan_tokens``: canonical per-stream plan-item texts.
- ``stream_id``: which of the K streams this record corresponds to.
- ``metadata``: doc text, teacher plan, rollback flags, etc.

Each KD record corresponds to one (sample, stream) pair; the collator
gathers K records sharing a ``sample_id`` into a single batch element.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger("pdt.training.dataset")


__all__ = ["PDTKDDataset", "PDTCollator", "SampleBatch"]


@dataclass(slots=True)
class SampleBatch:
    """One batch element is one *sample* (all K streams), not one stream."""

    sample_ids: List[str]
    stream_labels: List[List[str]]  # per batch element: K names
    # (B, K, T)
    student_ids: torch.Tensor  # padded per-stream token ids
    student_labels: torch.Tensor
    attention_mask: torch.Tensor
    # (B, S)
    planner_targets: torch.Tensor  # planner_ids padded to S slots
    planner_mask: torch.Tensor
    # (B, K, d_notes)
    teacher_notes: torch.Tensor
    student_notes: torch.Tensor
    # (B, P)
    plan_item_ids: torch.Tensor  # hashed plan-item ids for coverage
    plan_item_mask: torch.Tensor
    # (B, M) M = max snapshots
    readiness_targets: torch.Tensor
    readiness_mask: torch.Tensor
    # Metadata, opaque.
    raw: List[Mapping[str, object]]


class PDTKDDataset(Dataset):
    """Loads KD JSONL, groups records by sample_id, and yields per-sample dicts."""

    def __init__(self, path: str | Path, *, num_streams: int = 3) -> None:
        self.path = Path(path)
        self.num_streams = num_streams
        self._samples: List[Dict[str, Mapping[str, object]]] = []
        self._load()

    def _load(self) -> None:
        by_sample: Dict[str, Dict[str, Mapping[str, object]]] = defaultdict(dict)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sample_id = str(rec.get("sample_id") or rec.get("example_id"))
                stream_id = str(rec.get("stream_id") or rec.get("stream"))
                if not sample_id or not stream_id:
                    continue
                by_sample[sample_id][stream_id] = rec
        for sample_id, bystream in by_sample.items():
            if len(bystream) < self.num_streams:
                # Only keep samples with exactly K records (paper contract).
                continue
            ordered = dict(sorted(bystream.items(), key=lambda kv: kv[0]))
            self._samples.append(ordered)
        LOGGER.info("Loaded %d samples (K=%d) from %s", len(self._samples), self.num_streams, self.path)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Mapping[str, object]]:
        return self._samples[idx]


class PDTCollator:
    """Pads per-sample records into fixed-shape SampleBatch tensors."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        num_slots: int = 16,
        num_streams: int = 3,
        notes_dim: int = 256,
        max_length: int = 2048,
        max_plan_items: int = 32,
        max_snapshots: int = 4,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.num_slots = num_slots
        self.num_streams = num_streams
        self.notes_dim = notes_dim
        self.max_length = max_length
        self.max_plan_items = max_plan_items
        self.max_snapshots = max_snapshots

    def __call__(self, batch: Sequence[Mapping[str, Mapping[str, object]]]) -> SampleBatch:
        B = len(batch)
        K = self.num_streams
        T = self.max_length
        S = self.num_slots
        P = self.max_plan_items
        M = self.max_snapshots
        d_notes = self.notes_dim

        sample_ids: List[str] = []
        stream_labels: List[List[str]] = []
        student_ids = torch.full((B, K, T), self.pad_token_id, dtype=torch.long)
        student_labels = torch.full((B, K, T), -100, dtype=torch.long)
        attention_mask = torch.zeros(B, K, T, dtype=torch.long)
        planner_targets = torch.zeros(B, S, dtype=torch.long)
        planner_mask = torch.zeros(B, S, dtype=torch.bool)
        teacher_notes = torch.zeros(B, K, d_notes, dtype=torch.float32)
        student_notes = torch.zeros(B, K, d_notes, dtype=torch.float32)
        plan_item_ids = torch.zeros(B, P, dtype=torch.long)
        plan_item_mask = torch.zeros(B, P, dtype=torch.bool)
        readiness_targets = torch.zeros(B, M, dtype=torch.float32)
        readiness_mask = torch.zeros(B, M, dtype=torch.bool)
        raw_metadata: List[Mapping[str, object]] = []

        for b, sample in enumerate(batch):
            streams = list(sample.keys())
            stream_labels.append(streams)
            sample_id = None
            # Per-stream token ids.
            for k, stream in enumerate(streams[:K]):
                rec = sample[stream]
                if sample_id is None:
                    sample_id = str(rec.get("sample_id") or rec.get("example_id"))
                ids = list(map(int, rec.get("student_ids", [])))[:T]
                lbls = list(map(int, rec.get("student_labels", [])))[:T]
                if ids:
                    student_ids[b, k, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[b, k, : len(ids)] = 1
                if lbls:
                    student_labels[b, k, : len(lbls)] = torch.tensor(lbls, dtype=torch.long)
                t_notes = rec.get("notes_teacher")
                s_notes = rec.get("notes_student")
                if isinstance(t_notes, list) and len(t_notes) > k:
                    teacher_notes[b, k] = torch.tensor(t_notes[k], dtype=torch.float32)
                if isinstance(s_notes, list) and len(s_notes) > k:
                    student_notes[b, k] = torch.tensor(s_notes[k], dtype=torch.float32)
            sample_ids.append(sample_id or "")

            # Per-sample planner targets (shared across streams -- take from
            # the first stream's record).
            first = sample[streams[0]]
            planner_ids = list(map(int, first.get("planner_ids", [])))[:S]
            if planner_ids:
                planner_targets[b, : len(planner_ids)] = torch.tensor(planner_ids, dtype=torch.long)
                planner_mask[b, : len(planner_ids)] = True

            # Canonical plan-item ids (for coverage supervision).
            plan_token_texts = list(first.get("plan_tokens", []))[:P]
            # Use the same bucketed plan ids as planner_ids for the first P items.
            plan_ids_for_cov = planner_ids[:P]
            for p, v in enumerate(plan_ids_for_cov):
                plan_item_ids[b, p] = int(v)
                plan_item_mask[b, p] = True

            # Readiness labels.
            ready = list(map(int, first.get("continuation_sufficiency_labels", [])))[:M]
            for i, v in enumerate(ready):
                readiness_targets[b, i] = float(v)
                readiness_mask[b, i] = True

            raw_metadata.append(first.get("metadata", {}) or {})

        return SampleBatch(
            sample_ids=sample_ids,
            stream_labels=stream_labels,
            student_ids=student_ids,
            student_labels=student_labels,
            attention_mask=attention_mask,
            planner_targets=planner_targets,
            planner_mask=planner_mask,
            teacher_notes=teacher_notes,
            student_notes=student_notes,
            plan_item_ids=plan_item_ids,
            plan_item_mask=plan_item_mask,
            readiness_targets=readiness_targets,
            readiness_mask=readiness_mask,
            raw=raw_metadata,
        )
