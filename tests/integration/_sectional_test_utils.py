from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

import parallel_decoder_transformer


def _normalize_stream_id(stream_id: str) -> str:
    return (
        stream_id.split("stream_", 1)[-1].lower()
        if stream_id.startswith("stream_")
        else stream_id.lower()
    )


def _stringify_stream_notes(note: Any) -> List[str]:
    parts: List[str] = []
    for entity in getattr(note, "entities", []) or []:
        parts.append(
            "ENT::{id}::{name}::{type}".format(
                id=entity.id,
                name=entity.name,
                type=getattr(entity, "type", "entity"),
            )
        )
    for fact in getattr(note, "facts", []) or []:
        parts.append(f"FACT::{fact.subj_id}::{fact.predicate}::{fact.object}")
        span = fact.evidence_span
        if span.text:
            parts.append(f"SPAN::{span.text}")
    for coverage in getattr(note, "coverage", []) or []:
        parts.append(f"COVER::{coverage.plan_item_id}::{coverage.status.value}")
    return parts or [note.stream_id]


def _embed_strings(strings: List[str], notes_dim: int) -> torch.Tensor:
    vector = torch.zeros(notes_dim, dtype=torch.float32)
    for offset, text in enumerate(strings):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for index, byte in enumerate(digest):
            position = (index + offset) % notes_dim
            vector[position] += 1.0 if byte % 2 == 0 else -1.0
    norm = torch.linalg.vector_norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def embed_plan_notes(notes: Iterable[Any], notes_dim: int) -> Dict[str, torch.Tensor]:
    vectors: Dict[str, torch.Tensor] = {}
    for note in notes:
        key = _normalize_stream_id(note.stream_id)
        strings = _stringify_stream_notes(note)
        vectors[key] = _embed_strings(strings, notes_dim)
    return vectors


def load_plan_contract_notes_module():
    """Ensure plan_contract_notes can be imported without pulling the full dataset package."""

    root = Path(__file__).resolve().parents[2]
    data_path = root / "src" / "parallel_decoder_transformer" / "data"
    datasets_path = root / "src" / "parallel_decoder_transformer" / "datasets"
    data_pkg_name = "parallel_decoder_transformer.data"
    datasets_pkg_name = "parallel_decoder_transformer.datasets"

    if data_pkg_name not in sys.modules:
        data_pkg = types.ModuleType(data_pkg_name)
        data_pkg.__path__ = [str(data_path)]
        sys.modules[data_pkg_name] = data_pkg
        setattr(parallel_decoder_transformer, "data", data_pkg)

    if datasets_pkg_name not in sys.modules:
        datasets_pkg = types.ModuleType(datasets_pkg_name)
        datasets_pkg.__path__ = [str(datasets_path)]
        sys.modules[datasets_pkg_name] = datasets_pkg
        setattr(parallel_decoder_transformer, "datasets", datasets_pkg)

    return importlib.import_module(f"{datasets_pkg_name}.plan_contract_notes")
