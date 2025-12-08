# ruff: noqa: E402
"""Compute runtime proxies (alpha, beta, S) from an inference manifest."""

from __future__ import annotations

import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping

from parallel_decoder_transformer.utils.plan_catalog import (
    plan_hash_fingerprint,
    plan_hash_params_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Parallel Decoder Transformer inference manifest telemetry."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest JSON emitted by scripts/infer.py.",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help="Optional sequential baseline manifest to compute relative speedup.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_plan_metadata(manifest: Mapping[str, Any], label: str) -> Any:
    try:
        return plan_hash_params_from_manifest(manifest)
    except ValueError as err:
        raise ValueError(f"{label} missing planner hash metadata: {err}") from err


def _enforce_matching_plan_hash(primary: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    primary_params = _ensure_plan_metadata(primary, "primary manifest")
    baseline_params = _ensure_plan_metadata(baseline, "baseline manifest")
    if primary_params != baseline_params:
        raise ValueError(
            "Planner hash mismatch between manifests.\nprimary=%s\nbaseline=%s"
            % (
                plan_hash_fingerprint(primary_params),
                plan_hash_fingerprint(baseline_params),
            )
        )


def _resolve_total_time(manifest: Dict[str, Any]) -> float:
    timings = manifest.get("timings", {})
    total_candidate = timings.get("total", 0.0)
    try:
        total = float(total_candidate or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total > 0.0:
        return total
    per_token = timings.get("per_token", [])
    if isinstance(per_token, list):
        total = 0.0
        for entry in per_token:
            try:
                total += float(entry.get("duration_s", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    if total > 0.0:
        return total
    stride_durations = timings.get("stride_durations", [])
    if isinstance(stride_durations, list):
        try:
            total = sum(float(value or 0.0) for value in stride_durations)
        except (TypeError, ValueError):
            total = 0.0
    return float(total)


def _resolve_stride_durations(manifest: Dict[str, Any]) -> List[float]:
    timings = manifest.get("timings", {})
    stride_durations = timings.get("stride_durations", [])
    durations: List[float] = []
    if isinstance(stride_durations, list) and stride_durations:
        for value in stride_durations:
            try:
                durations.append(float(value))
            except (TypeError, ValueError):
                continue
        if durations:
            return durations
    per_token = timings.get("per_token", [])
    if not isinstance(per_token, list) or not per_token:
        return durations
    aggregates: Dict[int, float] = defaultdict(float)
    for entry in per_token:
        try:
            stride_index = int(entry.get("stride_index"))
        except (TypeError, ValueError):
            continue
        try:
            aggregates[stride_index] += float(entry.get("duration_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    if not aggregates:
        return durations
    for key in sorted(aggregates):
        durations.append(float(aggregates[key]))
    return durations


def _resolve_sync_durations(manifest: Dict[str, Any]) -> List[float]:
    timings = manifest.get("timings", {})
    sync_entries = timings.get("stride_sync_durations", [])
    sync_list: List[float] = []
    if isinstance(sync_entries, list):
        for value in sync_entries:
            try:
                sync_list.append(float(value))
            except (TypeError, ValueError):
                continue
    return sync_list


def compute_metrics(
    manifest: Dict[str, Any], baseline: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    stride_durations = _resolve_stride_durations(manifest)
    total_time = _resolve_total_time(manifest)

    streams = manifest.get("streams", {})
    tokens_total = sum(len(data.get("token_ids", [])) for data in streams.values())
    rollbacks = manifest.get("rollbacks", [])
    rollback_events = len(rollbacks)
    rollback_tokens = sum(len(entry.get("tokens_removed", [])) for entry in rollbacks)
    strides = len(stride_durations)

    alpha = mean(stride_durations) if stride_durations else 0.0
    beta = (rollback_tokens / tokens_total) if tokens_total else 0.0
    speed = (tokens_total / total_time) if total_time > 0.0 else 0.0

    metrics: Dict[str, Any] = {
        "alpha_s_per_stride": alpha,
        "beta_rollback_token_fraction": beta,
        "S_tokens_per_second": speed,
        "total_time_s": total_time,
        "stride_count": strides,
        "rollback_events": rollback_events,
        "rollback_tokens": rollback_tokens,
        "token_count": tokens_total,
    }
    if baseline is not None:
        baseline_total = _resolve_total_time(baseline)
        if baseline_total > 0.0 and total_time > 0.0:
            metrics["speedup_vs_seq"] = baseline_total / total_time
        else:
            metrics["speedup_vs_seq"] = None

    sync_durations = _resolve_sync_durations(manifest)
    if sync_durations:
        overhead = sum(sync_durations)
        metrics["sync_overhead_s"] = overhead
        metrics["sync_mean_s_per_stride"] = overhead / len(sync_durations)
        metrics["sync_stride_samples"] = len(sync_durations)
    else:
        timings = manifest.get("timings", {})
        overhead_total = timings.get("sync_overhead_s")
        if isinstance(overhead_total, (int, float)):
            metrics["sync_overhead_s"] = float(overhead_total)

    per_token = manifest.get("timings", {}).get("per_token", [])
    if isinstance(per_token, list) and per_token:
        margins = [
            float(entry["top2_margin"])
            for entry in per_token
            if isinstance(entry, dict) and isinstance(entry.get("top2_margin"), (int, float))
        ]
        if margins:
            margins.sort()
            metrics["mean_top2_margin"] = sum(margins) / len(margins)
            metrics["p05_top2_margin"] = _percentile(margins, 5.0)
            metrics["p50_top2_margin"] = _percentile(margins, 50.0)
            metrics["p95_top2_margin"] = _percentile(margins, 95.0)
    coverage_counts = _collect_coverage_counts(manifest)
    if coverage_counts:
        metrics["coverage_status_counts"] = coverage_counts
    return metrics


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list.")
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * (q / 100.0)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value * (1.0 - weight) + upper_value * weight


def _collect_coverage_counts(manifest: Dict[str, Any]) -> Dict[str, Any]:
    streams_section = manifest.get("streams")
    if not isinstance(streams_section, dict):
        return {}
    totals = {"covered": 0, "partial": 0, "missing": 0}
    per_stream: Dict[str, Dict[str, int]] = {}
    for stream, payload in streams_section.items():
        coverage = payload.get("coverage") if isinstance(payload, dict) else None
        items = coverage.get("plan_items") if isinstance(coverage, dict) else None
        if not isinstance(items, list):
            continue
        counts = {"covered": 0, "partial": 0, "missing": 0}
        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).lower()
            if status in counts:
                counts[status] += 1
                totals[status] += 1
        if any(counts.values()):
            per_stream[str(stream)] = counts
    if not per_stream:
        return {}
    return {"total": totals, "per_stream": per_stream}


def print_table(metrics: Dict[str, Any]) -> None:
    rows = [
        ("alpha (s/stride)", metrics["alpha_s_per_stride"]),
        ("beta (rollback token frac)", metrics["beta_rollback_token_fraction"]),
        ("S (tokens/s)", metrics["S_tokens_per_second"]),
        ("stride count", metrics["stride_count"]),
        ("rollback events", metrics["rollback_events"]),
        ("rollback tokens", metrics["rollback_tokens"]),
        ("token count", metrics["token_count"]),
        ("total time (s)", metrics["total_time_s"]),
    ]
    if "speedup_vs_seq" in metrics:
        rows.insert(3, ("speedup vs seq", metrics["speedup_vs_seq"]))
    if "sync_overhead_s" in metrics:
        rows.append(("sync overhead (s)", metrics.get("sync_overhead_s")))
        if "sync_mean_s_per_stride" in metrics:
            rows.append(("sync mean (s/stride)", metrics.get("sync_mean_s_per_stride")))
        if "sync_stride_samples" in metrics:
            rows.append(("sync samples", metrics.get("sync_stride_samples")))
    if "mean_top2_margin" in metrics:
        rows.extend(
            [
                ("mean top2 margin", metrics.get("mean_top2_margin")),
                ("p05 top2 margin", metrics.get("p05_top2_margin")),
                ("p50 top2 margin", metrics.get("p50_top2_margin")),
                ("p95 top2 margin", metrics.get("p95_top2_margin")),
            ]
        )
    coverage = metrics.get("coverage_status_counts")
    if isinstance(coverage, dict):
        totals = coverage.get("total", {})
        rows.extend(
            [
                ("coverage covered (total)", totals.get("covered")),
                ("coverage partial (total)", totals.get("partial")),
                ("coverage missing (total)", totals.get("missing")),
            ]
        )
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        if value is None:
            text = "n/a"
        elif isinstance(value, float):
            text = f"{value:.4f}"
        else:
            text = str(value)
        print(f"{name.ljust(width)} : {text}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    baseline_manifest = (
        load_manifest(args.baseline_manifest) if args.baseline_manifest is not None else None
    )
    if baseline_manifest is not None:
        _enforce_matching_plan_hash(manifest, baseline_manifest)
    metrics = compute_metrics(manifest, baseline_manifest)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print_table(metrics)


if __name__ == "__main__":
    main()
