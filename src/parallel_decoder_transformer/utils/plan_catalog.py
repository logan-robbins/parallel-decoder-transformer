"""Utilities for hashing and normalising plan catalog entries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class PlanHashParams:
    """Immutable bundle describing the hashed planner vocabulary."""

    vocab_size: int
    hash_buckets: int
    salt: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_vocab_size": int(self.vocab_size),
            "plan_hash_buckets": int(self.hash_buckets),
            "plan_hash_salt": self.salt,
        }


def hash_plan_text(text: str, bucket_count: int, *, salt: str = "") -> int:
    """Hash plan text into the plan vocab space used by the coverage head."""

    if bucket_count <= 1:
        raise ValueError("bucket_count must be greater than 1.")
    normalized = text.strip().lower()
    if not normalized:
        return 0
    payload = f"{salt}::{normalized}" if salt else normalized
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    hashed = int(digest, 16) % bucket_count
    return hashed if hashed != 0 else 1


def resolve_plan_hash_params(source: Any) -> PlanHashParams:
    """Infer plan hashing parameters from a model/config object."""

    config = getattr(source, "config", source)
    vocab_size = _coerce_positive_int(getattr(config, "plan_vocab_size", None))
    if vocab_size is None:
        embedding = getattr(getattr(source, "plan_embedding", None), "num_embeddings", None)
        vocab_size = _coerce_positive_int(embedding)
    if vocab_size is None:
        raise ValueError("Unable to resolve plan_vocab_size from the provided source.")
    collator_cfg = getattr(config, "collator", None)
    bucket_candidate = _coerce_positive_int(getattr(collator_cfg, "plan_hash_buckets", None))
    if bucket_candidate is None:
        bucket_candidate = vocab_size
    salt_value = getattr(collator_cfg, "plan_hash_salt", None)
    if not salt_value and hasattr(config, "plan_hash_salt"):
        salt_value = getattr(config, "plan_hash_salt")
    salt = str(salt_value or "")
    if vocab_size < bucket_candidate:
        bucket_candidate = vocab_size
    bucket_candidate = max(2, bucket_candidate)
    return PlanHashParams(vocab_size=vocab_size, hash_buckets=bucket_candidate, salt=salt)


def plan_hash_params_from_manifest(manifest: Mapping[str, Any]) -> PlanHashParams:
    """Extract plan hashing metadata from a manifest."""

    try:
        vocab_size = _coerce_positive_int(manifest["plan_vocab_size"])
        bucket_count = _coerce_positive_int(manifest["plan_hash_buckets"])
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Manifest missing plan hashing metadata.") from exc
    salt = str(manifest.get("plan_hash_salt", ""))
    if vocab_size is None or bucket_count is None:
        raise ValueError("Manifest provided invalid plan hash metadata.")
    if vocab_size < bucket_count:
        bucket_count = vocab_size
    bucket_count = max(2, bucket_count)
    return PlanHashParams(vocab_size=vocab_size, hash_buckets=bucket_count, salt=salt)


def plan_hash_fingerprint(params: PlanHashParams) -> str:
    """Compute a compact fingerprint for planner hash metadata."""

    return f"{params.vocab_size}:{params.hash_buckets}:{params.salt}"


def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def normalise_plan_map(
    plan_by_stream: Mapping[str, Sequence[str]],
    streams: Sequence[str],
) -> Dict[str, List[str]]:
    """Return stream-indexed plan lists aligned to the configured stream order."""

    normalized: Dict[str, List[str]] = {}
    for stream in streams:
        entries = plan_by_stream.get(stream)
        if entries is None:
            entries = plan_by_stream.get(stream.lower())
        if entries is None:
            entries = plan_by_stream.get(stream.upper())
        if entries is None:
            normalized[stream] = []
            continue
        cleaned = [
            str(item).strip() for item in entries if isinstance(item, str) and str(item).strip()
        ]
        normalized[stream] = cleaned
    return normalized


def flatten_plan_catalog(plan_map: Mapping[str, Sequence[str]]) -> Tuple[List[str], List[str]]:
    """Flatten a stream->plan list mapping into parallel arrays of streams and entries."""

    catalog: List[str] = []
    catalog_streams: List[str] = []
    for stream, entries in plan_map.items():
        for item in entries:
            catalog.append(item)
            catalog_streams.append(stream)
    return catalog, catalog_streams


def canonical_plan_catalog_entries(plan_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build a deterministic stream-tagged plan catalog from a plan contract payload."""

    raw_entries = plan_payload.get("plan")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raw_entries = plan_payload.get("streams")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        return []

    entries: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            continue
        stream = normalise_stream_label(raw_entry.get("stream_id") or raw_entry.get("stream"))
        if stream is None:
            stream = f"stream_{index + 1}"
        candidate_texts: List[str] = []
        for note in raw_entry.get("notes_contract") or raw_entry.get("notes") or []:
            text = _stringify_plan_item(note)
            if text:
                candidate_texts.append(text)
        summary = str(raw_entry.get("summary") or raw_entry.get("header") or "").strip()
        if summary:
            candidate_texts.append(summary)
        section_contract = raw_entry.get("section_contract")
        if isinstance(section_contract, Mapping) and section_contract:
            serialized = json.dumps(
                section_contract,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate_texts.append(f"SECTION::{serialized}")
        for text in candidate_texts:
            key = (stream, text)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"stream": stream, "text": text, "index": len(entries)})
    return entries


def hash_plan_catalog_entries(
    entries: Sequence[Mapping[str, Any]],
    bucket_count: int,
    *,
    salt: str = "",
) -> List[int]:
    """Hash canonical plan catalog entries into the latent planner vocabulary."""

    hashed: List[int] = []
    for entry in entries:
        value = hash_plan_entry(entry.get("stream"), entry.get("text"), bucket_count, salt=salt)
        if value == 0:
            continue
        hashed.append(value)
    return hashed


def hash_plan_entry(
    stream: Any,
    text: Any,
    bucket_count: int,
    *,
    salt: str = "",
) -> int:
    """Hash a stream-qualified plan entry into the shared latent plan vocabulary."""

    item_text = _stringify_plan_item(text)
    if not item_text:
        return 0
    stream_label = normalise_stream_label(stream)
    payload = item_text if stream_label is None else f"{stream_label}::{item_text}"
    return hash_plan_text(payload, bucket_count, salt=salt)


def pad_plan_ids(plan_ids: Sequence[int], target_length: int) -> Tuple[List[int], List[int]]:
    """Pad latent plan ids to a fixed planner slot count using 0 as the null slot."""

    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    values = [int(value) for value in plan_ids if int(value) >= 0]
    if len(values) > target_length:
        raise ValueError(
            f"Planner received {len(values)} plan ids but only {target_length} planner slots are configured."
        )
    padded = [0] * target_length
    mask = [0] * target_length
    for index, value in enumerate(values):
        padded[index] = value
        mask[index] = 1 if value != 0 else 0
    return padded, mask


def normalise_stream_label(value: Any) -> str | None:
    """Normalise a stream label into the canonical ``stream_*`` form."""

    if value is None:
        return None
    stream = str(value).strip().lower()
    if not stream:
        return None
    if not stream.startswith("stream_"):
        stream = f"stream_{stream}"
    return stream


def _stringify_plan_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


__all__ = [
    "PlanHashParams",
    "hash_plan_text",
    "normalise_plan_map",
    "flatten_plan_catalog",
    "canonical_plan_catalog_entries",
    "hash_plan_catalog_entries",
    "hash_plan_entry",
    "pad_plan_ids",
    "normalise_stream_label",
    "resolve_plan_hash_params",
    "plan_hash_params_from_manifest",
    "plan_hash_fingerprint",
]
