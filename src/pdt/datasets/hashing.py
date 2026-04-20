"""SHA-256 byte-parity sign-hash embedder for teacher notes.

Deterministic: the output vector is a function only of (a) the input text
strings and (b) the target dimension. Tokenizer- and V_p-blind. Identical
semantics to the previous pipeline so the JSONL stays byte-compatible for
text-level fields.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, MutableMapping, Sequence

import torch


__all__ = ["aggregate_hash", "embed_stream_map"]


def aggregate_hash(texts: Sequence[str], target_dim: int) -> torch.Tensor:
    """Return a unit-vector ``(target_dim,)`` from a list of strings.

    Algorithm:
        - Start with a zero vector.
        - For each input text, SHA-256 the UTF-8 bytes, and walk the 32
          digest bytes; for byte index ``i``, the corresponding sign is
          +1 if byte is even else -1, contributing at position
          ``(i + text_index) % target_dim``.
        - L2-normalize the accumulated vector.

    This is the exact hashing contract from the v1 pipeline.
    """
    vector = torch.zeros(target_dim, dtype=torch.float32)
    for offset, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for index, byte in enumerate(digest):
            pos = (index + offset) % target_dim
            vector[pos] += 1.0 if byte % 2 == 0 else -1.0
    norm = torch.linalg.vector_norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def embed_stream_map(
    stream_map: Mapping[str, Sequence[str]],
    *,
    stream_to_id: Mapping[str, int],
    notes_dim: int,
) -> torch.Tensor:
    """Return ``(K, notes_dim)`` where K == len(stream_to_id)."""
    result = torch.zeros(len(stream_to_id), notes_dim, dtype=torch.float32)
    normalized: MutableMapping[str, Sequence[str]] = {}
    for raw_key, texts in stream_map.items():
        normalized[_normalize_stream_id(raw_key)] = texts
    for stream, idx in stream_to_id.items():
        key = _normalize_stream_id(stream)
        texts = normalized.get(key, ())
        if not isinstance(texts, Sequence):
            texts = ()
        if not texts:
            # Also try shortened key (e.g. "1" for "stream_1").
            if key.startswith("stream_"):
                texts = normalized.get(key.split("stream_", 1)[-1], ())
        result[idx] = aggregate_hash(list(texts), notes_dim)
    return result


def _normalize_stream_id(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("stream_"):
        return text
    if text.startswith("stream"):
        return f"stream_{text[len('stream'):]}"
    if text.isdigit():
        return f"stream_{text}"
    return text


def stringify_stream_notes(stream: Mapping[str, object]) -> list[str]:
    """Flatten ENT/FACT/COVERAGE entries into hash-ready text blocks."""
    parts: list[str] = []
    for entity in stream.get("ENT", []) or []:  # type: ignore[union-attr]
        if not isinstance(entity, Mapping):
            continue
        name = entity.get("name") or entity.get("id")
        entity_type = entity.get("type") or "entity"
        parts.append(f"ENT::{name}::{entity_type}")
    for fact in stream.get("FACT", []) or []:  # type: ignore[union-attr]
        if not isinstance(fact, Mapping):
            continue
        subj = fact.get("subj_id") or fact.get("subject")
        pred = fact.get("predicate") or "relates_to"
        obj = fact.get("object") or fact.get("object_id")
        parts.append(f"FACT::{subj}::{pred}::{obj}")
        span = fact.get("evidence_span")
        if isinstance(span, Mapping) and span.get("text"):
            parts.append(str(span.get("text")))
    for coverage in stream.get("COVERAGE", []) or []:  # type: ignore[union-attr]
        if not isinstance(coverage, Mapping):
            continue
        plan_item = coverage.get("plan_item_id")
        status = coverage.get("status")
        parts.append(f"COVER::{plan_item}::{status}")
    summary = stream.get("summary")
    if summary:
        parts.append(str(summary))
    return [p for p in parts if p]
