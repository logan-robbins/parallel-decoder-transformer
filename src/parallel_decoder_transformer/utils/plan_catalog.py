"""Utilities for hashing and normalising plan catalog entries."""

from __future__ import annotations

import hashlib
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


__all__ = [
    "PlanHashParams",
    "hash_plan_text",
    "normalise_plan_map",
    "flatten_plan_catalog",
    "resolve_plan_hash_params",
    "plan_hash_params_from_manifest",
    "plan_hash_fingerprint",
]
