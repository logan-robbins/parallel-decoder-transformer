# ruff: noqa: E402
"""Compare sequential baseline and parallel Parallel Decoder Transformer outputs."""

from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from parallel_decoder_transformer.utils.plan_catalog import (
    plan_hash_fingerprint,
    plan_hash_params_from_manifest,
)


@dataclass(slots=True)
class MetricResult:
    name: str
    value: Optional[float]
    detail: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequential baseline and parallel manifests."
    )
    parser.add_argument(
        "--seq-manifest",
        required=True,
        type=Path,
        help="Manifest JSON produced by infer.py with --baseline sequential.",
    )
    parser.add_argument(
        "--par-manifest",
        required=True,
        type=Path,
        help="Manifest JSON produced by standard Parallel Decoder Transformer inference.",
    )
    parser.add_argument(
        "--metrics",
        default="rouge,bertscore",
        help="Comma-separated list of metrics to attempt: rouge, bertscore, tfidf.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit metrics as JSON instead of a table.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_plan_hash_alignment(
    seq_manifest: Mapping[str, Any], par_manifest: Mapping[str, Any]
) -> None:
    seq_params = plan_hash_params_from_manifest(seq_manifest)
    par_params = plan_hash_params_from_manifest(par_manifest)
    if seq_params != par_params:
        raise ValueError(
            "Planner hash mismatch between manifests.\nsequential=%s\nparallel=%s"
            % (
                plan_hash_fingerprint(seq_params),
                plan_hash_fingerprint(par_params),
            )
        )


def extract_text(manifest: Dict[str, Any]) -> str:
    streams = manifest.get("streams", {})
    if not isinstance(streams, dict) or not streams:
        return manifest.get("generated_text", "")
    ordered_streams = sorted(streams.items(), key=lambda item: item[0])
    texts: List[str] = []
    for _, payload in ordered_streams:
        text = payload.get("text")
        if isinstance(text, str):
            texts.append(text.strip())
    return "\n\n".join(part for part in texts if part)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def compute_token_diff_pct(seq_tokens: List[str], par_tokens: List[str]) -> float:
    seq_len = len(seq_tokens)
    par_len = len(par_tokens)
    denom = max(seq_len, 1)
    return abs(seq_len - par_len) / denom


def _term_frequency(tokens: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {token: count / total for token, count in counts.items()}


def _inverse_document_frequency(documents: List[Iterable[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for doc in documents:
        seen = set(doc)
        for token in seen:
            df[token] = df.get(token, 0) + 1
    total_docs = len(documents)
    idf: Dict[str, float] = {}
    for token, count in df.items():
        idf[token] = math.log((total_docs + 1) / (count + 1)) + 1.0
    return idf


def compute_tfidf_cosine(a_tokens: List[str], b_tokens: List[str]) -> Optional[float]:
    if not a_tokens or not b_tokens:
        return None
    tf_a = _term_frequency(a_tokens)
    tf_b = _term_frequency(b_tokens)
    idf = _inverse_document_frequency([a_tokens, b_tokens])
    vocab = set(tf_a) | set(tf_b)
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for token in vocab:
        weight_a = tf_a.get(token, 0.0) * idf.get(token, 0.0)
        weight_b = tf_b.get(token, 0.0) * idf.get(token, 0.0)
        dot += weight_a * weight_b
        norm_a += weight_a * weight_a
        norm_b += weight_b * weight_b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return dot / math.sqrt(norm_a * norm_b)


def compute_rouge(seq_text: str, par_text: str) -> MetricResult:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except Exception:
        return MetricResult("rouge_l_f1", None, "rouge_score not installed")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(seq_text, par_text)
    return MetricResult("rouge_l_f1", float(score["rougeL"].fmeasure))


def compute_bertscore(seq_text: str, par_text: str) -> MetricResult:
    try:
        from bert_score import score as bert_score  # type: ignore
    except Exception:
        return MetricResult("bertscore_f1", None, "bert-score not installed")
    try:
        precision, recall, f1 = bert_score(
            [par_text],
            [seq_text],
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
    except Exception as err:
        return MetricResult("bertscore_f1", None, f"bert-score failed: {err}")
    return MetricResult("bertscore_f1", float(f1.mean()))


def print_table(results: List[MetricResult], token_diff_pct: float, tfidf: Optional[float]) -> None:
    rows: List[Tuple[str, Optional[float], Optional[str]]] = []
    for result in results:
        rows.append((result.name, result.value, result.detail))
    rows.append(("token_diff_pct", token_diff_pct, None))
    if tfidf is not None:
        rows.append(("tfidf_cosine", tfidf, None))
    width = max(len(name) for name, _, _ in rows)
    for name, value, detail in rows:
        if value is None:
            text = "n/a"
        else:
            text = f"{value:.4f}"
        if detail:
            text = f"{text} ({detail})"
        print(f"{name.ljust(width)} : {text}")


def main() -> None:
    args = parse_args()
    requested_metrics = [
        entry.strip().lower() for entry in args.metrics.split(",") if entry.strip()
    ]

    seq_manifest = load_manifest(args.seq_manifest)
    par_manifest = load_manifest(args.par_manifest)
    _validate_plan_hash_alignment(seq_manifest, par_manifest)

    seq_text = extract_text(seq_manifest)
    par_text = extract_text(par_manifest)

    seq_tokens = tokenize(seq_text)
    par_tokens = tokenize(par_text)

    token_diff_pct = compute_token_diff_pct(seq_tokens, par_tokens)
    tfidf_cosine = compute_tfidf_cosine(seq_tokens, par_tokens)

    results: List[MetricResult] = []
    for metric in requested_metrics:
        if metric == "rouge":
            results.append(compute_rouge(seq_text, par_text))
        elif metric == "bertscore":
            results.append(compute_bertscore(seq_text, par_text))
        elif metric == "tfidf":
            if tfidf_cosine is None:
                results.append(MetricResult("tfidf_cosine", None, "insufficient tokens"))
            else:
                results.append(MetricResult("tfidf_cosine", tfidf_cosine))
        else:
            results.append(MetricResult(metric, None, "unknown metric"))

    payload = {
        "rouge_l_f1": next((res.value for res in results if res.name == "rouge_l_f1"), None),
        "bertscore_f1": next((res.value for res in results if res.name == "bertscore_f1"), None),
        "tfidf_cosine": tfidf_cosine,
        "token_diff_pct": token_diff_pct,
        "requested_metrics": requested_metrics,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_table(results, token_diff_pct, tfidf_cosine)


if __name__ == "__main__":
    main()
