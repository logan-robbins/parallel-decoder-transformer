"""Exports Arrow/Parquet corpus rows into KD JSONL records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set

import pyarrow.parquet as pq
import torch

from parallel_decoder_transformer.data.teacher_provider import (
    _HashingEmbedder,
    _stringify_stream_notes,
)
from parallel_decoder_transformer.data.teacher_runner import normalize_stream_id

logger = logging.getLogger(__name__)

DEFAULT_STREAM_TO_ID = {"stream_1": 0, "stream_2": 1, "stream_3": 2}


@dataclass(slots=True)
class KDExportConfig:
    """Configuration for converting Stage-3 Parquet splits to KD JSONL."""

    dataset_dir: Path = Path("data/datasets/sample_run")
    output_dir: Path = Path("data/processed/sample_run")
    output_filename: str = "kd_{split}.jsonl"  # Changed: {split} placeholder for per-split files
    splits: Sequence[str] = ("train",)
    notes_dim: int = 2048
    stream_to_id: Mapping[str, int] = field(default_factory=lambda: DEFAULT_STREAM_TO_ID.copy())
    limit_per_split: int | None = None
    resume: bool = True
    force: bool = False


class KDExporter:
    """Reads Parquet corpus rows and emits KD JSONL entries."""

    def __init__(self, config: KDExportConfig) -> None:
        self.config = config
        self._embedder = _HashingStreamEmbedder(config.stream_to_id, config.notes_dim)

    def export(self) -> Mapping[str, int]:
        """Convert the configured Parquet splits into KD JSON lines."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}

        for split in self.config.splits:
            # Generate split-specific output path
            output_filename = self.config.output_filename.replace("{split}", split)
            output_path = (self.config.output_dir / output_filename).resolve()

            parquet_path = self.config.dataset_dir / f"{split}.parquet"
            if not parquet_path.exists():
                logger.warning("Skipping split '%s' (missing file %s)", split, parquet_path)
                continue

            # Handle existing files
            if output_path.exists():
                if self.config.force:
                    logger.info("Force rewriting existing KD dataset at %s", output_path)
                    output_path.unlink()
                elif not self.config.resume:
                    raise RuntimeError(
                        f"{output_path} exists. Pass --force to overwrite or enable resume semantics."
                    )

            # Load existing IDs for resume
            existing_ids = (
                self._load_existing_ids(output_path)
                if output_path.exists() and self.config.resume
                else set()
            )

            written_for_split = 0
            with output_path.open("a", encoding="utf-8") as handle:
                for row in self._iter_rows(parquet_path):
                    for record in self._records_from_row(row, split):
                        example_id = record["example_id"]
                        if example_id in existing_ids:
                            continue
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        existing_ids.add(example_id)
                        counts[split] = counts.get(split, 0) + 1
                        written_for_split += 1
                    if (
                        self.config.limit_per_split
                        and written_for_split >= self.config.limit_per_split
                    ):
                        logger.info(
                            "Reached limit %d for split '%s'; stopping early.",
                            self.config.limit_per_split,
                            split,
                        )
                        break

            logger.info("Exported %d records to %s", written_for_split, output_path)

        return counts

    def _load_existing_ids(self, path: Path) -> Set[str]:
        ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                example_id = payload.get("example_id")
                if isinstance(example_id, str):
                    ids.add(example_id)
        return ids

    def _iter_rows(self, parquet_path: Path) -> Iterable[Mapping[str, Any]]:
        dataset = pq.ParquetFile(parquet_path)
        for row_group in range(dataset.num_row_groups):
            table = dataset.read_row_group(row_group)
            for row in table.to_pylist():
                yield row

    def _records_from_row(self, row: Mapping[str, Any], split: str) -> List[Dict[str, Any]]:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            return []
        domain = str(row.get("domain") or "unknown")
        plan_entries = self._parse_json_array(row.get("plan_text"))
        true_notes = self._parse_json_array(row.get("notes_true"))
        speculative_variants = self._parse_json_array(row.get("notes_speculative"))
        versioned_notes = self._parse_json_array(row.get("notes_versioned"))

        teacher_notes_map = self._teacher_notes_map(true_notes, plan_entries)
        notes_teacher = self._embedder.embed_from_map(teacher_notes_map)
        notes_student_source = (
            speculative_variants[0].get("notes", []) if speculative_variants else true_notes
        )
        notes_student = self._embedder.embed_from_entries(notes_student_source)

        teacher_snapshots = self._snapshot_payload(versioned_notes, source="teacher")
        student_snapshots = self._student_snapshot_payload(
            speculative_variants, fallback=notes_student
        )
        metadata = self._metadata(row, plan_entries, versioned_notes, teacher_notes_map)

        student_ids = self._ensure_int_sequence(row.get("z_hat_tokens") or [])
        planner_ids = self._ensure_int_sequence(row.get("plan_tokens") or [])
        labels = self._ensure_int_sequence(row.get("z_n_tokens") or student_ids)
        sectional_flag = bool(row.get("sectional_independence", False))
        raw_teacher_notes = teacher_notes_map
        true_notes_payload = true_notes

        stream_records: list[dict[str, Any]] = []
        stream_labels = self._resolve_stream_labels(plan_entries)
        for stream_id in stream_labels:
            example_id = f"{sample_id}:{stream_id}"
            notes_text = " ".join(raw_teacher_notes.get(stream_id, []))
            record = {
                "example_id": example_id,
                "sample_id": sample_id,
                "domain": domain,
                "split": split,
                "stream_id": stream_id,
                "stream": stream_id,
                "student_ids": student_ids,
                "student_labels": labels,
                "planner_ids": planner_ids,
                "notes_student": notes_student.tolist(),
                "notes_teacher": notes_teacher.tolist(),
                "notes_schema_version": "2.0",
                "notes_text": notes_text,
                "metadata": metadata,
                "true_notes": true_notes_payload,
                "teacher_snapshots": teacher_snapshots,
                "student_snapshots": student_snapshots,
                "raw_teacher_notes": raw_teacher_notes,
                "sectional_independence": sectional_flag,
                "plan_tokens": planner_ids,
                "notes_tokens": [],
            }
            stream_records.append(record)
        return stream_records

    def _teacher_notes_map(
        self,
        notes_payload: Sequence[Mapping[str, Any]],
        plan_entries: Sequence[Mapping[str, Any]],
    ) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for idx, entry in enumerate(notes_payload):
            stream_value = entry.get("stream_id") or entry.get("stream") or f"stream_{idx + 1}"
            normalized = normalize_stream_id(stream_value)
            mapping[normalized] = _stringify_stream_notes(entry)
        if not mapping and plan_entries:
            for idx, entry in enumerate(plan_entries):
                stream_value = entry.get("stream_id") or entry.get("stream") or f"stream_{idx + 1}"
                normalized = normalize_stream_id(stream_value)
                mapping[normalized] = _stringify_stream_notes(entry)
        return mapping

    def _metadata(
        self,
        row: Mapping[str, Any],
        plan_entries: Sequence[Mapping[str, Any]],
        versioned_notes: Sequence[Mapping[str, Any]],
        teacher_notes_map: Mapping[str, Sequence[str]],
    ) -> Dict[str, Any]:
        document_text = str(row.get("x_text") or "")
        paragraphs = [paragraph for paragraph in document_text.split("\n\n") if paragraph]
        metadata: dict[str, Any] = {
            "document_text": document_text,
            "document_paragraphs": paragraphs,
            "teacher_plan": {"plan": plan_entries},
            "teacher_notes": teacher_notes_map,
            "notes_versioned": versioned_notes,
            "sectional_independence": bool(row.get("sectional_independence", False)),
            "coverage_provenance": {
                "method": "notes_json",
                "strength": "hint",
                "schema_version": "2.0",
                "confirmed": False,
            },
        }
        rollback_flags = row.get("rollback_flags")
        if isinstance(rollback_flags, str):
            try:
                metadata["rollback_flags"] = json.loads(rollback_flags)
            except json.JSONDecodeError:
                metadata["rollback_flags"] = rollback_flags
        metadata["lag_delta"] = row.get("lag_delta")
        metadata["note_cadence_M"] = row.get("note_cadence_M")
        z_true = row.get("z_n")
        if isinstance(z_true, str):
            metadata["z_n_text"] = z_true
        return metadata

    def _snapshot_payload(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        source: str,
    ) -> List[Mapping[str, Any]]:
        payload: list[Mapping[str, Any]] = []
        for index, entry in enumerate(entries):
            notes_block = entry.get("notes") or []
            tensor = self._embedder.embed_from_entries(notes_block)
            payload.append(
                {
                    "notes": tensor.tolist(),
                    "stride": int(entry.get("lag_delta", entry.get("stride", 0)) or 0),
                    "version": int(entry.get("snapshot_id", index)),
                    "source": entry.get("source", source),
                }
            )
        if not payload:
            payload.append(
                {
                    "notes": self._embedder.embed_from_map({}).tolist(),
                    "stride": 0,
                    "version": 0,
                    "source": source,
                }
            )
        return payload

    def _student_snapshot_payload(
        self,
        variants: Sequence[Mapping[str, Any]],
        *,
        fallback: torch.Tensor,
    ) -> List[Mapping[str, Any]]:
        snapshots: list[Mapping[str, Any]] = []
        for index, variant in enumerate(variants):
            notes_block = variant.get("notes") or []
            tensor = self._embedder.embed_from_entries(notes_block)
            snapshots.append(
                {
                    "notes": tensor.tolist(),
                    "stride": int(variant.get("lag_delta", 0) or 0),
                    "version": index,
                    "source": "student",
                }
            )
        if not snapshots:
            snapshots.append(
                {
                    "notes": fallback.tolist(),
                    "stride": 0,
                    "version": 0,
                    "source": "student",
                }
            )
        return snapshots

    def _resolve_stream_labels(self, plan_entries: Sequence[Mapping[str, Any]]) -> List[str]:
        labels: list[str] = []
        if not plan_entries:
            return list(DEFAULT_STREAM_TO_ID.keys())
        for idx, entry in enumerate(plan_entries):
            stream_value = entry.get("stream_id") or entry.get("stream") or f"stream_{idx + 1}"
            labels.append(normalize_stream_id(stream_value))
        return labels

    @staticmethod
    def _parse_json_array(value: Any) -> list[Mapping[str, Any]]:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
        else:
            parsed = value
        if not isinstance(parsed, Sequence):
            return []
        return [item for item in parsed if isinstance(item, Mapping)]

    @staticmethod
    def _ensure_int_sequence(values: Sequence[Any]) -> List[int]:
        sequence: list[int] = []
        for value in values:
            try:
                sequence.append(int(value))
            except (TypeError, ValueError):
                sequence.append(0)
        return sequence


class _HashingStreamEmbedder:
    """Wraps the hashing embedder to produce dense matrices per stream."""

    def __init__(self, stream_to_id: Mapping[str, int], notes_dim: int) -> None:
        self.stream_to_id = dict(stream_to_id)
        self.notes_dim = int(notes_dim)
        self._embedder = _HashingEmbedder()

    def embed_from_entries(self, entries: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        stream_map: MutableMapping[str, List[str]] = {}
        for idx, entry in enumerate(entries):
            stream_value = entry.get("stream_id") or entry.get("stream") or f"stream_{idx + 1}"
            normalized = normalize_stream_id(stream_value)
            stream_map[normalized] = _stringify_stream_notes(entry)
        return self.embed_from_map(stream_map)

    def embed_from_map(self, stream_map: Mapping[str, Sequence[str]]) -> torch.Tensor:
        tensor = torch.zeros(len(self.stream_to_id), self.notes_dim, dtype=torch.float32)
        for stream, index in self.stream_to_id.items():
            texts = self._resolve_texts(stream_map, stream)
            vector = self._embed_texts(texts)
            tensor[index] = vector
        return tensor

    def _resolve_texts(self, stream_map: Mapping[str, Sequence[str]], stream: str) -> Sequence[str]:
        if stream in stream_map:
            return stream_map[stream]
        if stream.startswith("stream_"):
            bare = stream.split("stream_", 1)[-1]
            if bare in stream_map:
                return stream_map[bare]
        return []

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        payload = [str(text) for text in texts if str(text).strip()]
        if not payload:
            payload = [""]
        vector = self._embedder.aggregate(payload, self.notes_dim)
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        return vector.to(dtype=torch.float32)


__all__ = ["KDExportConfig", "KDExporter"]
