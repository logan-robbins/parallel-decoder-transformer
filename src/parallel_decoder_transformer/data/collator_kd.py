"""Two-branch collator wiring student and teacher tensors for KD training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .snapshots import SnapshotFeatures
from .teacher_provider import TeacherNotesProviderBase
from ..utils.plan_catalog import hash_plan_text


@dataclass(slots=True)
class TwoBranchKDCollatorConfig:
    pad_token_id: int
    label_pad_id: int = -100
    notes_dim: int = 2048
    max_length: int = 2048
    max_snapshots: int = 4
    commit_horizon: int = 0
    stream_to_id: Mapping[str, int] = field(
        default_factory=lambda: {"stream_1": 0, "stream_2": 1, "stream_3": 2}
    )
    plan_hash_buckets: int = 65536
    plan_hash_salt: str = "parallel-decoder-v1"
    dtype: str = "bfloat16"


class TwoBranchKnowledgeDistillationCollator:
    """Assembles student and teacher tensors in a single batch dictionary."""

    def __init__(
        self,
        config: TwoBranchKDCollatorConfig,
        *,
        teacher_provider: TeacherNotesProviderBase,
    ) -> None:
        if teacher_provider is None:
            raise ValueError(
                "TwoBranchKnowledgeDistillationCollator requires a teacher_provider. "
                "Teacher notes must be pre-generated during dataset pipeline (Stage 3: Notes Generation)."
            )
        self.config = config
        self._dtype = _resolve_dtype(config.dtype)
        self.teacher_provider = teacher_provider

    def __call__(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        examples = list(batch)
        if not examples:
            raise ValueError("TwoBranchKnowledgeDistillationCollator received an empty batch.")

        teacher_payloads = []
        for example in examples:
            payload = self.teacher_provider.fetch(example)
            teacher_payloads.append(payload)
            example["notes_teacher"] = payload.notes
            example["teacher_snapshots"] = payload.snapshots

        student_ids = self._pad_sequence(
            [self._ensure_list(example["student_ids"]) for example in examples],
            self.config.pad_token_id,
        )
        label_sequences: List[List[int]] = []
        labels_present = False
        for example in examples:
            if "student_labels" in example:
                labels_present = True
                label_sequences.append(self._ensure_list(example["student_labels"]))
            else:
                label_sequences.append(self._ensure_list(example["student_ids"]))
        student_labels = self._pad_sequence(
            label_sequences,
            self.config.label_pad_id,
        )
        label_masks = self._build_label_masks(examples, label_sequences)
        planner_ids = self._pad_sequence(
            [self._ensure_list(example["planner_ids"]) for example in examples],
            self.config.pad_token_id,
        )

        stream_indices = torch.tensor(
            [
                self._stream_index(example.get("stream_id") or example.get("stream"))
                for example in examples
            ],
            dtype=torch.long,
        )

        notes_student = torch.stack(
            [
                self._ensure_tensor(example["notes_student"]).to(dtype=self._dtype)
                for example in examples
            ]
        )
        notes_teacher = torch.stack(
            [payload.notes.to(dtype=self._dtype) for payload in teacher_payloads]
        )
        if notes_student.size() != notes_teacher.size():
            raise ValueError("Student and teacher notes must have identical shapes.")
        if notes_teacher.size(-1) != self.config.notes_dim:
            raise ValueError(
                f"Notes dim mismatch. Expected {self.config.notes_dim}, got {notes_teacher.size(-1)}."
            )

        attention_mask = (student_ids != self.config.pad_token_id).long()
        planner_mask = (planner_ids != self.config.pad_token_id).long()
        commit_mask = self._build_commit_mask(attention_mask)

        teacher_snapshots = self._collate_snapshot_block(
            examples,
            snapshots_key="teacher_snapshots",
            source="teacher",
        )
        student_snapshots = self._collate_snapshot_block(
            examples,
            snapshots_key="student_snapshots",
            source="student",
        )

        plan_texts = [list(example.get("plan_items", [])) for example in examples]
        notes_text = [example.get("notes_text", "") for example in examples]
        plan_catalogs = [
            list(example.get("plan_catalog", example.get("plan_items", []))) for example in examples
        ]
        plan_catalog_streams = []
        for index, catalog in enumerate(plan_catalogs):
            raw_streams = examples[index].get("plan_catalog_streams") or []
            stream_list = list(raw_streams)
            if stream_list and len(stream_list) != len(catalog):
                if len(stream_list) < len(catalog):
                    fallback = examples[index].get("stream_id") or examples[index].get("stream")
                    stream_list.extend([fallback] * (len(catalog) - len(stream_list)))
                else:
                    stream_list = stream_list[: len(catalog)]
            if not stream_list:
                fallback = examples[index].get("stream_id") or examples[index].get("stream")
                stream_list = [fallback] * len(catalog)
            plan_catalog_streams.append(stream_list)
        plan_item_ids, plan_item_mask, plan_item_stream_ids = self._encode_plan_items(
            plan_catalogs,
            plan_catalog_streams,
        )
        coverage_targets, coverage_mask = self._pad_float_sequences(
            [example.get("coverage_targets", []) for example in examples],
            plan_item_ids.size(1),
            mask_sequences=[example.get("coverage_supervision_mask") for example in examples],
        )

        agreement_labels, agreement_mask = self._pad_vector(
            [
                self._ensure_tensor(example.get("agreement_labels", []), dtype=torch.long)
                for example in examples
            ],
            teacher_snapshots.mask.size(1),
            pad_value=0,
        )
        metadata = [example.get("metadata", {}) for example in examples]
        example_ids = [example.get("example_id") for example in examples]
        sectional_flags = torch.tensor(
            [bool(example.get("sectional_independence", False)) for example in examples],
            dtype=torch.bool,
        )

        labels_mask_tensor = self._pad_bool_sequences(label_masks, student_labels.size(1))
        payload = {
            "input_ids": student_ids,
            "attention_mask": attention_mask,
            "commit_mask": commit_mask,
            "labels": student_labels,
            "planner_ids": planner_ids,
            "planner_mask": planner_mask,
            "notes_student": notes_student,
            "notes_teacher": notes_teacher,
            "stream_ids": stream_indices,
            "teacher_notes_bus": teacher_snapshots.notes,
            "teacher_bus_mask": teacher_snapshots.mask,
            "teacher_bus_stride": teacher_snapshots.stride,
            "teacher_bus_version": teacher_snapshots.version,
            "teacher_bus_stream_ids": teacher_snapshots.stream_ids,
            "teacher_bus_coverage": teacher_snapshots.coverage,
            "student_notes_bus": student_snapshots.notes,
            "student_bus_mask": student_snapshots.mask,
            "student_bus_stride": student_snapshots.stride,
            "student_bus_version": student_snapshots.version,
            "student_bus_stream_ids": student_snapshots.stream_ids,
            "student_bus_coverage": student_snapshots.coverage,
            "agreement_labels": agreement_labels,
            "agreement_mask": agreement_mask,
            "plan_item_ids": plan_item_ids,
            "plan_item_mask": plan_item_mask,
            "plan_item_stream_ids": plan_item_stream_ids,
            "coverage_targets": coverage_targets,
            "coverage_mask": coverage_mask,
            "plan_text": plan_texts,
            "notes_text": notes_text,
            "metadata": metadata,
            "example_ids": example_ids,
            "sectional_independence": sectional_flags,
        }
        if labels_present:
            base_mask = student_labels != self.config.label_pad_id
            payload["labels_mask"] = base_mask & labels_mask_tensor
        return payload

    def _pad_sequence(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        target_length = min(self.config.max_length, max(len(seq) for seq in sequences))
        batch = len(sequences)
        padded = torch.full((batch, target_length), pad_value, dtype=torch.long)
        for index, seq in enumerate(sequences):
            truncated = seq[:target_length]
            padded[index, : len(truncated)] = torch.tensor(truncated, dtype=torch.long)
        return padded

    def _stream_index(self, stream_id: Optional[str]) -> int:
        if stream_id is None:
            return 0
        key = str(stream_id).strip().lower()
        if not key:
            return 0
        try:
            return self.config.stream_to_id[key]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown stream token provided to collator: {stream_id!r}") from exc

    def _build_commit_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.commit_horizon <= 0:
            return torch.zeros_like(attention_mask, dtype=torch.bool)
        horizon = self.config.commit_horizon
        commit_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for index, mask_row in enumerate(attention_mask):
            active_tokens = int(mask_row.sum().item())
            if active_tokens == 0:
                continue
            start = max(0, active_tokens - horizon)
            commit_mask[index, start:active_tokens] = True
        return commit_mask

    def _collate_snapshot_block(
        self,
        examples: List[Mapping[str, Any]],
        *,
        snapshots_key: str,
        source: str,
    ) -> "SnapshotBatch":
        notes_dim = self.config.notes_dim
        stream_count = len(self.config.stream_to_id)

        normalized: List[List[SnapshotFeatures]] = []
        for example in examples:
            raw_snapshots = example.get(snapshots_key)
            if raw_snapshots is None:
                raise ValueError(f"Missing '{snapshots_key}' in example.")

            normalized.append(
                self._normalize_snapshots(
                    raw_snapshots,
                    streams=stream_count,
                    source=source,
                )
            )
        return self._stack_snapshots(normalized, streams=stream_count, notes_dim=notes_dim)

    def _normalize_snapshots(
        self,
        snapshots: Sequence[Any],
        *,
        streams: int,
        source: str,
    ) -> List[SnapshotFeatures]:
        if not snapshots:
            raise ValueError("Snapshots cannot be empty.")

        normalized: List[SnapshotFeatures] = []
        for index, snapshot in enumerate(snapshots):
            features = self._coerce_snapshot(snapshot, streams=streams, source=source, index=index)
            normalized.append(features)
        return normalized[: self.config.max_snapshots]

    def _coerce_snapshot(
        self,
        snapshot: Any,
        *,
        streams: int,
        source: str,
        index: int,
    ) -> SnapshotFeatures:
        if isinstance(snapshot, SnapshotFeatures):
            features = snapshot.to(dtype=self._dtype)
        elif isinstance(snapshot, Mapping):
            if "notes" not in snapshot:
                raise ValueError("Snapshot mapping must contain a 'notes' field.")
            notes = self._ensure_tensor(snapshot["notes"])
            stride = int(snapshot.get("stride", snapshot.get("stride_id", 0)))
            version = int(snapshot.get("version", snapshot.get("snapshot_id", index)))
            stream_id = snapshot.get("stream_id") or snapshot.get("stream")
            coverage_payload = snapshot.get("coverage_flags")
            coverage_tensor = None
            if coverage_payload is not None:
                coverage_tensor = self._ensure_tensor(coverage_payload).to(dtype=self._dtype)
            features = SnapshotFeatures(
                notes=notes.to(dtype=self._dtype),
                stride=stride,
                version=version,
                stream_id=str(stream_id).strip().lower() if stream_id else None,
                coverage=coverage_tensor,
                source=source,
            )
        else:
            tensor = self._ensure_tensor(snapshot).to(dtype=self._dtype)
            features = SnapshotFeatures(notes=tensor, source=source, stride=0, version=index)

        padded_notes = self._pad_notes(
            features.notes, streams=streams, notes_dim=self.config.notes_dim
        )
        padded_coverage = self._pad_coverage(features.coverage, streams=streams)
        stream_label = features.stream_id
        return SnapshotFeatures(
            notes=padded_notes,
            stride=features.stride,
            version=features.version,
            stream_id=stream_label,
            coverage=padded_coverage,
            source=features.source,
        )

    def _pad_notes(self, notes: torch.Tensor, *, streams: int, notes_dim: int) -> torch.Tensor:
        if notes.dim() == 1:
            notes = notes.unsqueeze(0)
        padded = torch.zeros((streams, notes_dim), dtype=notes.dtype)
        rows = min(streams, notes.size(0))
        cols = min(notes_dim, notes.size(-1))
        padded[:rows, :cols] = notes[:rows, :cols]
        return padded

    def _pad_coverage(
        self,
        coverage: Optional[torch.Tensor],
        *,
        streams: int,
    ) -> Optional[torch.Tensor]:
        if coverage is None:
            return None
        flat = coverage.view(-1)
        padded = torch.zeros(streams, dtype=coverage.dtype)
        length = min(streams, flat.numel())
        padded[:length] = flat[:length]
        return padded

    def _stack_snapshots(
        self,
        snapshots: Sequence[Sequence[SnapshotFeatures]],
        *,
        streams: int,
        notes_dim: int,
    ) -> "SnapshotBatch":
        batch_size = len(snapshots)
        max_snapshots = max(1, self.config.max_snapshots)
        notes_tensor = torch.zeros(
            (batch_size, max_snapshots, streams, notes_dim),
            dtype=self._dtype,
        )
        stride_tensor = torch.zeros((batch_size, max_snapshots), dtype=torch.long)
        version_tensor = torch.zeros((batch_size, max_snapshots), dtype=torch.long)
        coverage_tensor = torch.zeros((batch_size, max_snapshots, streams), dtype=self._dtype)
        mask_tensor = torch.zeros((batch_size, max_snapshots), dtype=torch.bool)
        stream_tensor = torch.full((batch_size, max_snapshots), fill_value=-1, dtype=torch.long)
        for batch_index, snapshot_list in enumerate(snapshots):
            truncated = list(snapshot_list)[:max_snapshots]
            for snapshot_index, snapshot in enumerate(truncated):
                mask_tensor[batch_index, snapshot_index] = True
                notes_tensor[batch_index, snapshot_index] = snapshot.notes
                stride_tensor[batch_index, snapshot_index] = snapshot.stride
                version_tensor[batch_index, snapshot_index] = snapshot.version
                if snapshot.coverage is not None:
                    coverage_tensor[batch_index, snapshot_index] = snapshot.coverage
                if snapshot.stream_id is not None:
                    try:
                        stream_tensor[batch_index, snapshot_index] = self._stream_index(
                            snapshot.stream_id
                        )
                    except ValueError:
                        stream_tensor[batch_index, snapshot_index] = -1
        return SnapshotBatch(
            notes=notes_tensor,
            stride=stride_tensor,
            version=version_tensor,
            coverage=coverage_tensor,
            mask=mask_tensor,
            stream_ids=stream_tensor,
        )

    def _pad_vector(
        self,
        tensors: Sequence[torch.Tensor],
        target_length: int,
        *,
        pad_value: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(tensors)
        padded = torch.full((batch_size, target_length), pad_value, dtype=torch.long)
        mask = torch.zeros((batch_size, target_length), dtype=torch.bool)
        for index, tensor in enumerate(tensors):
            vector = tensor.view(-1)
            length = min(target_length, vector.numel())
            if length == 0:
                continue
            padded[index, :length] = vector[:length]
            mask[index, :length] = True
        return padded, mask

    def _pad_bool_sequences(
        self,
        sequences: Sequence[Sequence[bool]],
        target_length: int,
    ) -> torch.Tensor:
        batch = len(sequences)
        padded = torch.zeros((batch, target_length), dtype=torch.bool)
        for index, seq in enumerate(sequences):
            truncated = list(seq)[:target_length]
            if not truncated:
                continue
            padded[index, : len(truncated)] = torch.tensor(truncated, dtype=torch.bool)
        return padded

    def _build_label_masks(
        self,
        examples: Sequence[Mapping[str, Any]],
        label_sequences: Sequence[Sequence[int]],
    ) -> list[list[bool]]:
        masks: list[list[bool]] = []
        for example, labels in zip(examples, label_sequences):
            seq_len = len(labels)
            mask = self._sectional_label_mask(example, seq_len)
            if mask is None:
                mask = [True] * seq_len
            masks.append(mask)
        return masks

    def _sectional_label_mask(
        self,
        example: Mapping[str, Any],
        seq_len: int,
    ) -> list[bool] | None:
        sectional_flag = bool(
            example.get("sectional_independence")
            or (
                isinstance(example.get("metadata"), Mapping)
                and example["metadata"].get("sectional_independence")
            )
        )
        if not sectional_flag:
            return None
        metadata = example.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        plan = metadata.get("teacher_plan")
        segments = plan.get("segments") if isinstance(plan, Mapping) else None
        role_lengths = metadata.get("role_surface_lengths")
        if not segments or not isinstance(role_lengths, Mapping):
            return None
        ranges = self._segment_ranges(segments, role_lengths, seq_len)
        stream_label = example.get("stream_id") or example.get("stream")
        start_end = self._lookup_range(ranges, stream_label)
        if start_end is None:
            return None
        start, end = start_end
        start = max(0, min(start, seq_len))
        end = max(start, min(end, seq_len))
        mask = [False] * seq_len
        for idx in range(start, end):
            mask[idx] = True
        if not any(mask):
            return None
        return mask

    def _segment_ranges(
        self,
        segments: Sequence[Mapping[str, Any]],
        role_lengths: Mapping[str, Any],
        seq_len: int,
    ) -> dict[str, tuple[int, int]]:
        entries: list[tuple[int, str, int]] = []
        for idx, segment in enumerate(segments):
            if not isinstance(segment, Mapping):
                continue
            stream_value = segment.get("stream") or segment.get("stream_id")
            normalized_stream = self._normalize_stream_label(stream_value)
            if normalized_stream is None:
                continue
            length = self._resolve_role_length(role_lengths, normalized_stream)
            if length is None or length <= 0:
                continue
            order = segment.get("paragraph_start")
            order_value = int(order) if isinstance(order, (int, float)) else idx
            entries.append((order_value, normalized_stream, int(length)))
        entries.sort(key=lambda item: item[0])
        ranges: dict[str, tuple[int, int]] = {}
        cursor = 0
        for _, stream_key, length in entries:
            if cursor >= seq_len:
                break
            start = cursor
            end = min(cursor + length, seq_len)
            for alias in self._stream_aliases(stream_key):
                ranges.setdefault(alias, (start, end))
            cursor += length
        return ranges

    def _lookup_range(
        self,
        ranges: Mapping[str, tuple[int, int]],
        stream_label: Any,
    ) -> tuple[int, int] | None:
        normalized = self._normalize_stream_label(stream_label)
        if normalized is None:
            return None
        for alias in self._stream_aliases(normalized):
            if alias in ranges:
                return ranges[alias]
        return None

    def _resolve_role_length(
        self,
        role_lengths: Mapping[str, Any],
        stream_label: str,
    ) -> int | None:
        for alias in self._stream_aliases(stream_label):
            if alias in role_lengths:
                try:
                    length = int(role_lengths[alias])
                except (TypeError, ValueError):
                    continue
                if length > 0:
                    return length
        return None

    def _normalize_stream_label(self, value: Any) -> str | None:
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key:
            return None
        return key

    def _stream_aliases(self, label: str) -> tuple[str, ...]:
        label = str(label).strip().lower()
        if not label:
            return tuple()
        aliases = [label]
        if label.startswith("stream_"):
            bare = label.split("stream_", 1)[-1]
            if bare:
                aliases.append(bare)
        else:
            aliases.append(f"stream_{label}")
        # Preserve order while removing duplicates
        seen: set[str] = set()
        ordered: list[str] = []
        for alias in aliases:
            if not alias or alias in seen:
                continue
            seen.add(alias)
            ordered.append(alias)
        return tuple(ordered)

    def _encode_plan_items(
        self,
        plan_texts: Sequence[Sequence[str]],
        plan_streams: Sequence[Sequence[Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(plan_texts)
        max_items = max((len(items) for items in plan_texts), default=0)
        if max_items == 0:
            return (
                torch.zeros((batch_size, 1), dtype=torch.long),
                torch.zeros((batch_size, 1), dtype=torch.bool),
                torch.full((batch_size, 1), fill_value=-1, dtype=torch.long),
            )
        plan_ids = torch.zeros((batch_size, max_items), dtype=torch.long)
        mask = torch.zeros((batch_size, max_items), dtype=torch.bool)
        stream_ids = torch.full((batch_size, max_items), fill_value=-1, dtype=torch.long)
        for batch_index, items in enumerate(plan_texts):
            for item_index, text in enumerate(items[:max_items]):
                mask[batch_index, item_index] = True
                plan_ids[batch_index, item_index] = hash_plan_text(
                    text,
                    self.config.plan_hash_buckets,
                    salt=self.config.plan_hash_salt,
                )
                stream_value = None
                if plan_streams and batch_index < len(plan_streams):
                    streams_for_batch = plan_streams[batch_index]
                    if item_index < len(streams_for_batch):
                        stream_value = streams_for_batch[item_index]
                stream_ids[batch_index, item_index] = self._plan_stream_to_id(stream_value)
        return plan_ids, mask, stream_ids

    def _pad_float_sequences(
        self,
        sequences: Sequence[Sequence[float]],
        target_length: int,
        *,
        mask_sequences: Optional[Sequence[Optional[Sequence[bool]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_length <= 0:
            target_length = 1
        batch_size = len(sequences)
        tensor = torch.zeros((batch_size, target_length), dtype=torch.float32)
        mask = torch.zeros((batch_size, target_length), dtype=torch.bool)
        for index, seq in enumerate(sequences):
            if not seq:
                continue
            length = min(target_length, len(seq))
            tensor[index, :length] = torch.tensor(seq[:length], dtype=torch.float32)
            mask_values: Optional[Sequence[bool]] = None
            if mask_sequences is not None and index < len(mask_sequences):
                mask_values = mask_sequences[index]
            for position in range(length):
                if mask_values and position < len(mask_values):
                    mask[index, position] = bool(mask_values[position])
                else:
                    mask[index, position] = True
        return tensor, mask

    def _plan_stream_to_id(self, stream_value: Any) -> int:
        if stream_value is None:
            return -1
        if isinstance(stream_value, int):
            return stream_value
        if isinstance(stream_value, str):
            stream_normalized = stream_value.strip().lower()
            if not stream_normalized:
                return -1
            try:
                return self._stream_index(stream_normalized)
            except ValueError:
                return -1
        return -1

    def _ensure_list(self, tensor_like: Any) -> List[int]:
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.view(-1).tolist()
        if isinstance(tensor_like, list):
            return tensor_like
        if isinstance(tensor_like, (tuple, range)):
            return list(tensor_like)
        raise TypeError(f"Unsupported sequence type: {type(tensor_like)!r}")

    def _ensure_tensor(self, value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype) if dtype else value
        tensor = torch.tensor(value, dtype=dtype or torch.float32)
        return tensor


@dataclass(slots=True)
class SnapshotBatch:
    """Packed snapshot tensors for either teacher or student branches."""

    notes: torch.Tensor
    stride: torch.Tensor
    version: torch.Tensor
    coverage: torch.Tensor
    mask: torch.Tensor
    stream_ids: torch.Tensor


def _resolve_dtype(alias: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[alias]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported dtype alias: {alias!r}") from exc


__all__ = [
    "TwoBranchKnowledgeDistillationCollator",
    "TwoBranchKDCollatorConfig",
    "SnapshotBatch",
]
