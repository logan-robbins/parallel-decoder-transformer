"""Collation utilities for turning plan + notes artifacts into Arrow splits."""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

import numpy as np
from transformers import AutoTokenizer  # type: ignore
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CollateConfig:
    notes_dir: Path = Path("data/prep/notes")
    output_dir: Path = Path("data/datasets/pdt_corpus")
    tokenizer_path: Path = Path("gpt-oss-20b/tokenizer")
    max_seq_len: int = 2048
    augment_per_sample: int = 2
    splits: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    seed: int = 99


class DatasetCollator:
    """Produces Arrow splits from generated notes files."""

    def __init__(self, cfg: CollateConfig, tokenizer=None) -> None:
        self._cfg = cfg
        self._tokenizer = tokenizer or self._load_tokenizer(cfg)
        if getattr(self._tokenizer, "pad_token", None) is None and hasattr(
            self._tokenizer, "eos_token"
        ):  # pragma: no cover - HF detail
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._rng = random.Random(cfg.seed)

    def _process_file(self, path: Path) -> List[MutableMapping[str, Any]]:
        """Process a single notes file and return augmented records."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan_path = Path(payload.get("plan_path") or "")
        if not plan_path.exists():
            raise RuntimeError(f"Plan path {plan_path} missing for sample {path}")
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        base_record = self._build_record(payload, plan_payload)
        return list(self._augment_record(base_record))

    def collate(self) -> Mapping[str, Path]:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - heavy optional dep
            raise RuntimeError(
                "pyarrow is required to collate the dataset. Install pyarrow>=16."
            ) from exc

        note_files = sorted(self._cfg.notes_dir.rglob("*.json"))
        if not note_files:
            raise RuntimeError(
                f"No notes files found under {self._cfg.notes_dir}. Run generate_notes.py first."
            )

        logger.info("Processing %d notes files in parallel...", len(note_files))
        records: list[MutableMapping[str, Any]] = []

        # Process files in parallel with progress bar
        import multiprocessing as mp

        max_workers = min(mp.cpu_count(), 16)  # Cap at 16 to avoid memory issues

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_file, path): path for path in note_files}

            with tqdm(total=len(note_files), desc="collating", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        file_records = future.result()
                        records.extend(file_records)
                        pbar.update(1)
                    except Exception as exc:
                        path = futures[future]
                        logger.error("Failed to process %s: %s", path, exc)
                        raise

        logger.info("Prepared %d records (including augmentation)", len(records))
        splits = self._split_records(records)
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, Path] = {}
        for split_name, split_records in splits.items():
            if not split_records:
                continue
            table = pa.Table.from_pylist(split_records)
            path = self._cfg.output_dir / f"{split_name}.parquet"
            pq.write_table(table, path, compression="snappy")
            exported[split_name] = path
            logger.info("Wrote %s -> %s (%d rows)", split_name, path, len(split_records))
        manifest = {
            "counts": {split: len(rows) for split, rows in splits.items()},
            "config": {
                "max_seq_len": self._cfg.max_seq_len,
                "augment_per_sample": self._cfg.augment_per_sample,
            },
        }
        (self._cfg.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return exported

    def _build_record(
        self,
        notes_payload: Mapping[str, Any],
        plan_payload: Mapping[str, Any],
    ) -> MutableMapping[str, Any]:
        input_text = str(plan_payload.get("input_text", ""))
        plan_text = json.dumps(plan_payload.get("streams", []), ensure_ascii=False)
        true_notes = json.dumps(notes_payload.get("true_notes", []), ensure_ascii=False)
        spec_notes = json.dumps(notes_payload.get("speculative_notes", []), ensure_ascii=False)
        versioned_notes = json.dumps(notes_payload.get("versioned_notes", []), ensure_ascii=False)
        z_true = str(notes_payload.get("z_n", ""))
        z_hat_list = notes_payload.get("z_hat", [])
        z_hat_text = json.dumps(z_hat_list, ensure_ascii=False)
        rollback_flags = json.dumps(notes_payload.get("rollback", {}), ensure_ascii=False)
        sectional_independence = bool(
            notes_payload.get(
                "sectional_independence", plan_payload.get("sectional_independence", True)
            )
        )
        record: MutableMapping[str, Any] = {
            "sample_id": notes_payload.get("sample_id"),
            "domain": notes_payload.get("domain"),
            "x_text": input_text,
            "plan_text": plan_text,
            "notes_true": true_notes,
            "notes_speculative": spec_notes,
            "notes_versioned": versioned_notes,
            "z_n": z_true,
            "z_hat": z_hat_text,
            "rollback_flags": rollback_flags,
            "lag_delta": notes_payload.get("lag_delta"),
            "note_cadence_M": notes_payload.get("note_cadence_M"),
            "kl_divergence": notes_payload.get("kl_divergence", 1.0),
            "sectional_independence": sectional_independence,
        }
        record["x_tokens"] = self._encode(input_text)
        record["plan_tokens"] = self._encode(plan_text)
        record["z_n_tokens"] = self._encode(z_true)
        record["z_hat_tokens"] = self._encode(" ".join(z_hat_list))
        return record

    def _encode(self, text: str) -> list[int]:
        encoding = self._tokenizer(
            text,
            max_length=self._cfg.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        tokens = encoding["input_ids"][0].tolist()
        return tokens

    def _augment_record(self, record: MutableMapping[str, Any]) -> list[MutableMapping[str, Any]]:
        augmented = [dict(record)]
        for idx in range(self._cfg.augment_per_sample):
            clone = dict(record)
            clone["sample_id"] = f"{record['sample_id']}_aug{idx}"
            clone["lag_delta"] = max(
                1, int(record.get("lag_delta") or 1) + self._rng.choice([-1, 0, 1])
            )
            cadence = max(2, int(record.get("note_cadence_M") or 6) + self._rng.choice([-2, 0, 2]))
            clone["note_cadence_M"] = cadence
            augmented.append(clone)
        return augmented

    def _split_records(
        self, records: List[MutableMapping[str, Any]]
    ) -> Dict[str, List[MutableMapping[str, Any]]]:
        rng = random.Random(self._cfg.seed)
        buckets: dict[str, list[MutableMapping[str, Any]]] = {}
        for record in records:
            buckets.setdefault(str(record.get("domain", "unknown")), []).append(record)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        splits: dict[str, list[MutableMapping[str, Any]]] = {
            split: [] for split in self._cfg.splits
        }
        for domain_records in buckets.values():
            total = len(domain_records)
            offset = 0
            remaining_records = list(domain_records)
            for idx, (split_name, fraction) in enumerate(self._cfg.splits.items()):
                if idx == len(self._cfg.splits) - 1:
                    subset = remaining_records[offset:]
                else:
                    count = int(round(total * fraction))
                    subset = remaining_records[offset : offset + count]
                    offset += count
                splits.setdefault(split_name, []).extend(subset)
        return splits

    def _load_tokenizer(self, cfg: CollateConfig):
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
            return tokenizer
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning(
                "Failed to load tokenizer from %s (%s). Falling back to tiktoken cl100k_base.",
                cfg.tokenizer_path,
                exc,
            )
            try:
                import tiktoken  # type: ignore
            except ImportError as tiktoken_exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "Unable to load tokenizer and tiktoken is not installed for fallback use."
                ) from tiktoken_exc
            encoding = tiktoken.get_encoding("cl100k_base")
            return _TiktokenTokenizer(encoding)


class _TiktokenTokenizer:
    """Minimal wrapper so DatasetCollator can fall back to tiktoken encoders."""

    def __init__(self, encoding) -> None:
        self._encoding = encoding
        self.pad_token = 0
        self.eos_token = 0

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        truncation: bool,
        padding: str,
        return_tensors: str,
    ) -> Mapping[str, Any]:
        tokens = self._encoding.encode(text)
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        if padding == "max_length" and len(tokens) < max_length:
            tokens = tokens + [self.pad_token] * (max_length - len(tokens))
        array = np.array([tokens], dtype=np.int32)
        return {"input_ids": array}


__all__ = ["CollateConfig", "DatasetCollator"]
