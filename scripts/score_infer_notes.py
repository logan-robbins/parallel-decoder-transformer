"""Score inference manifests for notes-centric contradiction and redundancy metrics."""

from __future__ import annotations

# ruff: noqa: E402

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from parallel_decoder_transformer.data.extraction import StreamNotes, load_stream_notes
from parallel_decoder_transformer.evaluation import compute_attribute_consistency
from parallel_decoder_transformer.utils import resolve_device
from parallel_decoder_transformer.utils.nli import NliScorer, NliScorerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute notes-centric metrics from inference manifests."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to inference manifest JSON (from scripts/infer.py).",
    )
    parser.add_argument(
        "--notes-key",
        default="reference_notes",
        help="Manifest key to read per-stream notes from (default: reference_notes).",
    )
    parser.add_argument(
        "--plan-text-file",
        type=Path,
        default=None,
        help="Optional JSON mapping { stream_id: [plan item strings...] } to use for NLI scoring.",
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default=None,
        help="Optional Hugging Face model name for contradiction scoring (e.g., 'facebook/bart-large-mnli').",
    )
    parser.add_argument(
        "--nli-margin",
        type=float,
        default=0.1,
        help="Margin applied when computing contradiction margin violations.",
    )
    parser.add_argument(
        "--redundancy-margin",
        type=float,
        default=0.7,
        help="Margin threshold for redundancy index (positive (cosine - margin)+).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "--write-manifest-metrics",
        action="store_true",
        help="Persist attribute consistency metrics back into the manifest under metrics.attribute_consistency.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stream_texts(manifest: Mapping[str, object], notes_key: str) -> Dict[str, List[str]]:
    notes_section = manifest.get(notes_key)
    if isinstance(notes_section, Mapping):
        result: Dict[str, List[str]] = {}
        for key, value in notes_section.items():
            strings = [str(item).strip() for item in value] if isinstance(value, Sequence) else []
            strings = [text for text in strings if text]
            result[str(key).lower()] = strings
        if result:
            return result
    streams_section = manifest.get("streams") or manifest.get("streams")
    if isinstance(streams_section, Mapping):
        result = {}
        for stream_id, payload in streams_section.items():
            notes_payload = None
            if isinstance(payload, Mapping):
                notes_payload = payload.get(notes_key) or payload.get("reference_notes")
            if isinstance(notes_payload, Sequence):
                cleaned = [str(item).strip() for item in notes_payload if isinstance(item, str)]
                cleaned = [text for text in cleaned if text]
                if cleaned:
                    result[str(stream_id).lower()] = cleaned
        if result:
            return result
    return {}


def load_plan_text(plan_text_file: Optional[Path]) -> Dict[str, List[str]]:
    if plan_text_file is None:
        return {}
    payload = json.loads(plan_text_file.read_text(encoding="utf-8"))
    plan_map: Dict[str, List[str]] = {}
    if not isinstance(payload, Mapping):
        raise ValueError(
            "--plan-text-file must contain a JSON object mapping stream_id -> list of plan strings."
        )
    for key, value in payload.items():
        if not isinstance(value, Sequence):
            raise ValueError("Plan text entries must be lists of strings.")
        entries = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
        plan_map[str(key).lower()] = entries
    return plan_map


def load_structured_stream_notes(manifest: Mapping[str, Any]) -> Dict[str, StreamNotes]:
    """Hydrate StreamNotes payloads from manifest sections when present."""

    structured: Dict[str, StreamNotes] = {}

    def _ingest(stream_name: str, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        normalized = dict(payload)
        normalized.setdefault("stream_id", stream_name)
        try:
            notes = load_stream_notes(normalized)
        except (ValueError, TypeError):
            return
        structured[str(stream_name).strip().lower()] = notes

    root = manifest.get("notes_structured")
    if isinstance(root, Mapping):
        for stream_name, payload in root.items():
            _ingest(str(stream_name), payload)

    streams_section = manifest.get("streams") or manifest.get("streams")
    if isinstance(streams_section, Mapping):
        for stream_name, payload in streams_section.items():
            if not isinstance(payload, Mapping):
                continue
            candidate = payload.get("notes_structured")
            if isinstance(candidate, Mapping):
                _ingest(str(stream_name), candidate)

    return structured


def gather_nli_pairs(
    notes_by_stream: Mapping[str, Sequence[str]],
    plan_by_stream: Mapping[str, Sequence[str]],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for stream_id, notes in notes_by_stream.items():
        plan_items = plan_by_stream.get(stream_id, [])
        if not plan_items:
            continue
        note_block = " ".join(note for note in notes if note)
        if not note_block:
            continue
        for item in plan_items:
            pairs.append((note_block, item))
    return pairs


def compute_nli_metrics(
    scorer: Optional[NliScorer],
    pairs: Sequence[Tuple[str, str]],
    margin: float,
) -> Dict[str, Optional[float]]:
    if scorer is None or not pairs:
        return {
            "contradiction_rate": None,
            "avg_margin_violation": None,
            "nli_pair_count": 0,
        }
    logits = scorer.score(pairs)
    if logits.size(0) == 0:
        return {
            "contradiction_rate": None,
            "avg_margin_violation": None,
            "nli_pair_count": 0,
        }
    contra_idx = scorer.label_index.get("contradiction", 0)
    neutral_idx = scorer.label_index.get("neutral", 1)
    predictions = logits.argmax(dim=-1)
    contradiction_total = int((predictions == contra_idx).sum().item())
    violations = torch.relu(logits[:, contra_idx] - logits[:, neutral_idx] - margin)
    violation_mean = float(violations.mean().item())
    return {
        "contradiction_rate": contradiction_total / float(logits.size(0)),
        "avg_margin_violation": violation_mean,
        "nli_pair_count": int(logits.size(0)),
    }


def build_tfidf_vectors(documents: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, float]]:
    tokenised: Dict[str, List[str]] = {}
    for stream_id, notes in documents.items():
        tokens = []
        for note in notes:
            tokens.extend(note.lower().split())
        tokenised[stream_id] = [token for token in tokens if token]
    vocab_documents = list(tokenised.values())
    idf: Dict[str, float] = {}
    if vocab_documents:
        df: MutableMapping[str, int] = {}
        for doc in vocab_documents:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        doc_count = len(vocab_documents)
        for token, freq in df.items():
            idf[token] = math.log((doc_count + 1) / (freq + 1)) + 1.0
    vectors: Dict[str, Dict[str, float]] = {}
    for stream_id, tokens in tokenised.items():
        if not tokens:
            vectors[stream_id] = {}
            continue
        tf: MutableMapping[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0
        total = sum(tf.values())
        if total <= 0.0:
            vectors[stream_id] = {}
            continue
        for token in tf:
            tf[token] /= total
        vectors[stream_id] = {token: tf[token] * idf.get(token, 1.0) for token in tf}
    return vectors


def cosine(vec_a: Mapping[str, float], vec_b: Mapping[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for token, weight in vec_a.items():
        dot += weight * vec_b.get(token, 0.0)
    norm_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_redundancy(
    notes_by_stream: Mapping[str, Sequence[str]],
    margin: float,
) -> Dict[str, Optional[float]]:
    stream_ids = list(notes_by_stream.keys())
    if len(stream_ids) < 2:
        return {"redundancy_index": None, "redundancy_pair_count": 0}
    vectors = build_tfidf_vectors(notes_by_stream)
    total = 0.0
    count = 0
    for stream_a, stream_b in combinations(stream_ids, 2):
        sim = cosine(vectors.get(stream_a, {}), vectors.get(stream_b, {}))
        excess = max(0.0, sim - margin)
        total += excess
        count += 1
    if count == 0:
        return {"redundancy_index": None, "redundancy_pair_count": 0}
    return {"redundancy_index": total / count, "redundancy_pair_count": count}


def print_table(metrics: Mapping[str, Optional[float]]) -> None:
    rows = [
        ("contradiction_rate", metrics.get("contradiction_rate")),
        ("avg_margin_violation", metrics.get("avg_margin_violation")),
        ("nli_pair_count", metrics.get("nli_pair_count")),
        ("redundancy_index", metrics.get("redundancy_index")),
        ("redundancy_pair_count", metrics.get("redundancy_pair_count")),
        ("attribute_cross_stream_rate", metrics.get("attribute_cross_stream_rate")),
        ("attribute_cross_stream_total", metrics.get("attribute_cross_stream_total")),
        ("attribute_cross_stream_violations", metrics.get("attribute_cross_stream_violations")),
        ("attribute_total_tuples", metrics.get("attribute_total_tuples")),
        ("attribute_source_counts", metrics.get("attribute_source_counts")),
        ("attribute_per_stream", metrics.get("attribute_per_stream")),
        ("attribute_time_per_stream", metrics.get("attribute_time_per_stream")),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
            text = json.dumps(value, sort_keys=True)
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
        elif value is None:
            text = "n/a"
        else:
            text = str(value)
        print(f"{name.ljust(width)} : {text}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    notes_by_stream = load_stream_texts(manifest, args.notes_key)
    if not notes_by_stream:
        raise ValueError(
            "No notes found in manifest. Provide --notes-text-file during inference or adjust --notes-key."
        )
    plan_by_stream = load_plan_text(args.plan_text_file)
    structured_notes = load_structured_stream_notes(manifest)
    scorer: Optional[NliScorer] = None
    if args.nli_model is not None:
        device = torch.device(resolve_device())
        scorer = NliScorer(NliScorerConfig(model_name=args.nli_model), device=device)
    nli_pairs = gather_nli_pairs(notes_by_stream, plan_by_stream)
    nli_metrics = compute_nli_metrics(scorer, nli_pairs, margin=args.nli_margin)
    redundancy_metrics = compute_redundancy(notes_by_stream, margin=args.redundancy_margin)
    attribute_result = compute_attribute_consistency(
        notes_by_stream,
        plan_by_stream=plan_by_stream,
        structured_notes=structured_notes,
    )
    attribute_payload = attribute_result.to_payload()
    attribute_metrics = {
        "attribute_cross_stream_rate": attribute_payload["cross_stream"]["consistency_rate"],
        "attribute_cross_stream_total": attribute_payload["cross_stream"]["total"],
        "attribute_cross_stream_violations": attribute_payload["cross_stream"]["violations"],
        "attribute_total_tuples": attribute_payload["total_tuples"],
        "attribute_source_counts": attribute_payload["source_counts"],
        "attribute_per_stream": attribute_payload["per_stream"],
        "attribute_time_per_stream": attribute_payload["time"],
    }
    metrics = {**nli_metrics, **redundancy_metrics, **attribute_metrics}
    metrics["attribute_consistency"] = attribute_payload
    if args.write_manifest_metrics:
        manifest.setdefault("metrics", {})
        manifest["metrics"]["attribute_consistency"] = attribute_payload
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print_table(metrics)


if __name__ == "__main__":
    main()
