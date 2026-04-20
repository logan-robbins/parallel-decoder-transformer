"""Canonical plan catalog + V_p bucket hashing.

Given the teacher plan JSON (``plan_entries``), produces an ordered list of
stream-scoped plan items (header, summary, notes-contract bullets,
section-contract bounds, constraints) and hashes each into the V_p bucket
space. Deterministic across runs; the only dependency is the UTF-8 text
content.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


__all__ = [
    "canonical_plan_catalog_entries",
    "hash_plan_catalog_entries",
    "hash_plan_text",
]


def canonical_plan_catalog_entries(plan_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build an ordered per-stream catalog of plan-item texts.

    Accepts either ``{"plan": [...]}`` or a bare ``[...]`` list at the top
    level. Emits a flat list of ``{"stream", "text", "index"}`` dicts.
    """
    plan = plan_payload.get("plan") if isinstance(plan_payload, Mapping) else plan_payload
    if not isinstance(plan, Sequence):
        return []

    entries: list[dict[str, Any]] = []
    index = 0

    for stream_idx, entry in enumerate(plan):
        if not isinstance(entry, Mapping):
            continue
        stream_id = _normalize_stream_id(
            entry.get("stream_id") or entry.get("stream") or f"stream_{stream_idx + 1}"
        )
        summary = entry.get("summary")
        if isinstance(summary, str) and summary.strip():
            entries.append({"stream": stream_id, "text": summary.strip(), "index": index})
            index += 1
        header = entry.get("header")
        if isinstance(header, str) and header.strip() and header.strip() != summary:
            entries.append({"stream": stream_id, "text": header.strip(), "index": index})
            index += 1
        for item in entry.get("notes_contract", []) or []:
            if isinstance(item, str) and item.strip():
                entries.append({"stream": stream_id, "text": item.strip(), "index": index})
                index += 1
        section_contract = entry.get("section_contract")
        if isinstance(section_contract, Mapping):
            text = json.dumps(section_contract, sort_keys=True)
            entries.append({"stream": stream_id, "text": text, "index": index})
            index += 1
        for constraint in entry.get("constraints", []) or []:
            if isinstance(constraint, str) and constraint.strip():
                entries.append({"stream": stream_id, "text": constraint.strip(), "index": index})
                index += 1
    return entries


def hash_plan_text(text: str, bucket_count: int, *, salt: str = "") -> int:
    if bucket_count <= 1:
        raise ValueError("bucket_count must be > 1.")
    normalized = text.strip().lower()
    if not normalized:
        return 0
    payload = f"{salt}::{normalized}" if salt else normalized
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    hashed = int(digest, 16) % bucket_count
    return hashed if hashed != 0 else 1


def hash_plan_catalog_entries(
    entries: Sequence[Mapping[str, Any]],
    bucket_count: int,
    *,
    salt: str = "",
) -> list[int]:
    """Hash each ``(stream, text)`` pair into the V_p bucket space.

    Stream qualification means the same text under different streams hashes
    to different buckets, preserving ownership in the final planner-id
    sequence.
    """
    ids: list[int] = []
    for entry in entries:
        stream = entry.get("stream")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            ids.append(0)
            continue
        stream_label = _normalize_stream_id(stream)
        payload = (
            text if stream_label is None or not stream_label
            else f"{stream_label}::{text}"
        )
        ids.append(hash_plan_text(payload, bucket_count, salt=salt))
    return ids


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
