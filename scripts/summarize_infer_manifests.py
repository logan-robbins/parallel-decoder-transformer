#!/usr/bin/env python3
"""Summarize rollback, gate, and coverage telemetry across inference manifests."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from parallel_decoder_transformer.evaluation import aggregate_metrics, summarize_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize rollback/gate/coverage metrics for one or more inference manifests."
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        help="Manifest paths or glob patterns (JSON emitted by scripts/infer.py).",
    )
    parser.add_argument(
        "--plan-text-file",
        type=Path,
        default=None,
        help="Optional JSON file mapping stream -> list[str] to override plan text entries.",
    )
    parser.add_argument(
        "--coverage-partial-threshold",
        type=float,
        default=0.4,
        help="Token-overlap threshold used to treat plan coverage as partial when only text is available.",
    )
    parser.add_argument(
        "--gate-low-threshold",
        type=float,
        default=0.25,
        help="Gate value treated as 'low' when computing dwell/oscillation statistics.",
    )
    parser.add_argument(
        "--gate-high-threshold",
        type=float,
        default=0.75,
        help="Gate value treated as 'high' when computing dwell/oscillation statistics.",
    )
    parser.add_argument(
        "--gate-entropy-bins",
        type=int,
        default=20,
        help="Number of bins used when computing per-stream gate entropy.",
    )
    parser.add_argument(
        "--gate-series-max-points",
        type=int,
        default=256,
        help="Downsampled length for per-stream gate series in the JSON payload.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full summary as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary payload.",
    )
    return parser.parse_args()


def load_plan_override(path: Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("--plan-text-file must contain an object mapping stream -> list[str].")
    override: Dict[str, List[str]] = {}
    for stream, texts in payload.items():
        if not isinstance(texts, Sequence):
            raise ValueError("Plan override entries must be sequences of strings.")
        cleaned = [str(item).strip() for item in texts if isinstance(item, str) and item.strip()]
        override[str(stream).lower()] = cleaned
    return override


def expand_manifest_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
    deduped = []
    seen = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def print_manifest_summary(metrics) -> None:
    print(f"Manifest: {metrics.path}")
    rollback = metrics.rollback
    print(
        "  Rollbacks: events={events} tokens_removed={tokens} rate/stride={rate} len_p50={p50} len_p95={p95} corr={corr}".format(
            events=rollback.total_events,
            tokens=rollback.total_tokens_removed,
            rate=format_float(rollback.per_stride_rate),
            p50=format_float(rollback.length_p50),
            p95=format_float(rollback.length_p95),
            corr=format_float(rollback.agreement_correlation),
        )
    )
    for stream, count in sorted(rollback.per_stream_events.items()):
        rate = rollback.per_stream_rate.get(stream)
        print(
            f"    - {stream}: events={count} tokens_removed={rollback.per_stream_tokens.get(stream, 0)} rate/stride={format_float(rate)}"
        )
    if metrics.gate.per_stream:
        print("  Gate:")
        for stream in sorted(metrics.gate.per_stream):
            summary = metrics.gate.per_stream[stream]
            print(
                "    - {stream}: mean={mean} std={std} entropy={entropy} high={high} low={low} osc={osc}".format(
                    stream=stream,
                    mean=format_float(summary.mean),
                    std=format_float(summary.stddev),
                    entropy=format_float(summary.entropy),
                    high=format_float(summary.high_fraction),
                    low=format_float(summary.low_fraction),
                    osc=format_float(summary.oscillation_rate),
                )
            )
    coverage = metrics.coverage
    print(
        "  Coverage (source={source}): P={precision} R={recall} F1={f1} support={support}".format(
            source=coverage.source,
            precision=format_float(coverage.precision),
            recall=format_float(coverage.recall),
            f1=format_float(coverage.f1),
            support=coverage.support,
        )
    )
    for stream in sorted(coverage.per_stream):
        summary = coverage.per_stream[stream]
        print(
            "    - {stream}: P={precision} R={recall} F1={f1} eval={evaluated} support={support} missing_preds={missing}".format(
                stream=stream,
                precision=format_float(summary.precision),
                recall=format_float(summary.recall),
                f1=format_float(summary.f1),
                evaluated=summary.evaluated,
                support=summary.support,
                missing=summary.missing_predictions,
            )
        )


def print_aggregate_summary(aggregate) -> None:
    print("Aggregate Summary")
    print(
        "  Rollbacks: manifests={count} events={events} rate/stride={rate} len_p50={p50} len_p95={p95} corr={corr}".format(
            count=aggregate.count,
            events=aggregate.rollback.total_events,
            rate=format_float(aggregate.rollback.per_stride_rate),
            p50=format_float(aggregate.rollback.length_p50),
            p95=format_float(aggregate.rollback.length_p95),
            corr=format_float(aggregate.rollback.agreement_correlation),
        )
    )
    if aggregate.gate.per_stream:
        print("  Gate:")
        for stream in sorted(aggregate.gate.per_stream):
            summary = aggregate.gate.per_stream[stream]
            print(
                "    - {stream}: mean={mean} std={std} entropy={entropy} high={high} low={low} osc={osc}".format(
                    stream=stream,
                    mean=format_float(summary.mean),
                    std=format_float(summary.stddev),
                    entropy=format_float(summary.entropy),
                    high=format_float(summary.high_fraction),
                    low=format_float(summary.low_fraction),
                    osc=format_float(summary.oscillation_rate),
                )
            )
    coverage = aggregate.coverage
    print(
        "  Coverage (source={source}): P={precision} R={recall} F1={f1} support={support}".format(
            source=coverage.source,
            precision=format_float(coverage.precision),
            recall=format_float(coverage.recall),
            f1=format_float(coverage.f1),
            support=coverage.support,
        )
    )


def main() -> None:
    args = parse_args()
    paths = expand_manifest_paths(args.manifests)
    if not paths:
        raise SystemExit("No manifests matched the provided patterns.")
    plan_override = load_plan_override(args.plan_text_file)
    summaries = []
    for path in paths:
        manifest = load_manifest(path)
        summary = summarize_manifest(
            manifest,
            path=str(path),
            plan_text_override=plan_override if plan_override else None,
            coverage_partial_threshold=args.coverage_partial_threshold,
            gate_low_threshold=args.gate_low_threshold,
            gate_high_threshold=args.gate_high_threshold,
            gate_entropy_bins=args.gate_entropy_bins,
        )
        summaries.append(summary)
    aggregate = aggregate_metrics(summaries)
    payload = {
        "manifests": [
            summary.to_payload(gate_series_max_points=args.gate_series_max_points)
            for summary in summaries
        ],
        "aggregate": aggregate.to_payload(gate_series_max_points=args.gate_series_max_points),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for summary in summaries:
        print_manifest_summary(summary)
    print_aggregate_summary(aggregate)


if __name__ == "__main__":
    main()
