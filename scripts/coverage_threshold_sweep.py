# ruff: noqa: E402
"""Post-hoc coverage threshold sweep and PR curve generation.

Runs a threshold sweep over one or more saved inference manifests and
produces a precision-recall curve figure, a CSV table, and a stdout summary
of the best operating point.

Usage::

    uv run python scripts/coverage_threshold_sweep.py \\
        --manifest path/to/manifest.json [manifest2.json ...] \\
        --thresholds 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \\
        --bootstrap 1000 \\
        --output figures/coverage_pr_curve.png \\
        --csv results/coverage_threshold_sweep.csv
"""

from __future__ import annotations

import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from parallel_decoder_transformer.evaluation.manifest_metrics import (
    bootstrap_coverage_ci,
    compute_coverage_roc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coverage head threshold sweep with bootstrap CI and PR curve."
    )
    parser.add_argument(
        "--manifest",
        dest="manifests",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more inference manifest JSON files.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        metavar="TAU",
        help=(
            "Explicit threshold values to evaluate (e.g. 0.1 0.2 0.4).  "
            "Defaults to 19 evenly-spaced values from 0.05 to 0.95."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        metavar="N",
        help="Number of bootstrap resamples for CI estimation (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the PR curve figure (PNG).  Skipped if not supplied.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the per-threshold CSV table.  Skipped if not supplied.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.4,
        metavar="OV",
        help="Token-overlap ratio used for ground-truth labels (default: 0.4).",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        metavar="CI",
        help="Confidence level for bootstrap intervals (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="RNG seed for bootstrap resampling (default: 42).",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _merge_manifests(manifests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple manifests by concatenating stream coverage plan_items."""
    if len(manifests) == 1:
        return manifests[0]

    merged_streams: Dict[str, Any] = {}
    merged_plan_catalog: List[Any] = []

    for manifest in manifests:
        streams = manifest.get("streams", {})
        for stream_name, payload in streams.items():
            if stream_name not in merged_streams:
                merged_streams[stream_name] = {
                    "text": "",
                    "coverage": {"plan_items": []},
                    "token_ids": [],
                }
            if not isinstance(payload, dict):
                continue
            existing = merged_streams[stream_name]
            # Append text
            if payload.get("text"):
                existing["text"] = (existing["text"] + " " + payload["text"]).strip()
            # Append plan_items
            cov = payload.get("coverage", {})
            if isinstance(cov, dict):
                items = cov.get("plan_items", [])
                if isinstance(items, list):
                    existing["coverage"]["plan_items"].extend(items)
        # Merge plan catalog
        plan = manifest.get("plan", {})
        if isinstance(plan, dict):
            catalog = plan.get("catalog", [])
            if isinstance(catalog, list):
                merged_plan_catalog.extend(catalog)

    return {
        "streams": merged_streams,
        "plan": {"catalog": merged_plan_catalog},
        "config": manifests[0].get("config", {}),
    }


def _compute_sweep(
    manifest: Dict[str, Any],
    thresholds: Optional[List[float]],
    overlap_threshold: float,
    n_resamples: int,
    ci: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Compute ROC points and bootstrap CIs for each threshold."""
    roc_points = compute_coverage_roc(
        manifest,
        thresholds=thresholds,
        overlap_threshold=overlap_threshold,
    )

    rows: List[Dict[str, Any]] = []
    for point in roc_points:
        tau = point["tau"]
        ci_result = bootstrap_coverage_ci(
            manifest,
            threshold=tau,
            n_resamples=n_resamples,
            ci=ci,
            overlap_threshold=overlap_threshold,
            seed=seed,
        )
        rows.append(
            {
                "threshold": tau,
                "precision": point["precision"],
                "recall": point["recall"],
                "f1": point["f1"],
                "tp": point["tp"],
                "fp": point["fp"],
                "fn": point["fn"],
                "support": point["support"],
                "precision_lo": ci_result.get("precision_lo"),
                "precision_hi": ci_result.get("precision_hi"),
                "recall_lo": ci_result.get("recall_lo"),
                "recall_hi": ci_result.get("recall_hi"),
                "f1_lo": ci_result.get("f1_lo"),
                "f1_hi": ci_result.get("f1_hi"),
                "n_pairs": ci_result.get("n_pairs", 0),
            }
        )
    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "support",
        "precision_lo",
        "precision_hi",
        "recall_lo",
        "recall_hi",
        "f1_lo",
        "f1_hi",
        "n_pairs",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})
    print(f"CSV written: {path}")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_figure(rows: List[Dict[str, Any]], output_path: Path, operating_tau: float) -> None:
    """Generate precision-recall curve with bootstrap CI bands."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print(
            "WARNING: matplotlib not available — skipping figure generation. "
            "Install with: uv add matplotlib"
        )
        return

    precisions = [r["precision"] for r in rows]
    recalls = [r["recall"] for r in rows]
    f1s = [r["f1"] for r in rows]
    thresholds = [r["threshold"] for r in rows]

    p_lo = [r["precision_lo"] or r["precision"] for r in rows]
    p_hi = [r["precision_hi"] or r["precision"] for r in rows]
    r_lo = [r["recall_lo"] or r["recall"] for r in rows]
    r_hi = [r["recall_hi"] or r["recall"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: PR curve ---
    ax_pr = axes[0]
    ax_pr.plot(recalls, precisions, "b-o", linewidth=2, markersize=4, label="PR curve")
    ax_pr.fill_betweenx(
        precisions,
        r_lo,
        r_hi,
        alpha=0.15,
        color="blue",
        label=f"{int(100 * rows[0].get('n_pairs', 0) and 95)}% CI (recall)",
    )
    ax_pr.fill_between(
        recalls,
        p_lo,
        p_hi,
        alpha=0.15,
        color="orange",
        label="95% CI (precision)",
    )

    # Mark operating point at threshold=0.4 (or closest)
    op_row = min(rows, key=lambda r: abs(r["threshold"] - operating_tau))
    ax_pr.plot(
        op_row["recall"],
        op_row["precision"],
        "r*",
        markersize=14,
        zorder=5,
        label=f"τ={operating_tau:.2f} (operating)",
    )
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Coverage Head: Precision-Recall Curve", fontsize=13)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(fontsize=9)
    ax_pr.grid(True, alpha=0.3)

    # --- Right: F1 vs threshold ---
    ax_f1 = axes[1]
    ax_f1.plot(thresholds, f1s, "g-o", linewidth=2, markersize=4, label="F1")
    f1_lo = [r["f1_lo"] or r["f1"] for r in rows]
    f1_hi = [r["f1_hi"] or r["f1"] for r in rows]
    ax_f1.fill_between(thresholds, f1_lo, f1_hi, alpha=0.2, color="green", label="95% CI")
    best_row = max(rows, key=lambda r: r["f1"])
    ax_f1.axvline(
        best_row["threshold"],
        color="purple",
        linestyle="--",
        linewidth=1.5,
        label=f"Best τ={best_row['threshold']:.2f} (F1={best_row['f1']:.3f})",
    )
    ax_f1.axvline(
        operating_tau,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"Operating τ={operating_tau:.2f}",
    )
    ax_f1.set_xlabel("Threshold (τ)", fontsize=12)
    ax_f1.set_ylabel("F1 Score", fontsize=12)
    ax_f1.set_title("Coverage Head: F1 vs Threshold", fontsize=13)
    ax_f1.set_xlim(0.0, 1.0)
    ax_f1.set_ylim(-0.02, 1.02)
    ax_f1.legend(fontsize=9)
    ax_f1.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written: {output_path}")


def _print_summary(rows: List[Dict[str, Any]], operating_tau: float = 0.4) -> None:
    if not rows:
        print("No threshold points computed.")
        return

    best = max(rows, key=lambda r: r["f1"])
    op_row = min(rows, key=lambda r: abs(r["threshold"] - operating_tau))

    print("\n" + "=" * 60)
    print("COVERAGE HEAD THRESHOLD SWEEP SUMMARY")
    print("=" * 60)
    print(f"Total plan-item pairs evaluated: {rows[0].get('n_pairs', '?')}")
    print(f"Thresholds swept: {len(rows)}")

    print("\n--- Best F1 Operating Point ---")
    print(f"  Threshold:  {best['threshold']:.3f}")
    print(f"  Precision:  {best['precision']:.4f}  [{best['precision_lo'] or 0:.4f}, {best['precision_hi'] or 0:.4f}]")
    print(f"  Recall:     {best['recall']:.4f}  [{best['recall_lo'] or 0:.4f}, {best['recall_hi'] or 0:.4f}]")
    print(f"  F1:         {best['f1']:.4f}  [{best['f1_lo'] or 0:.4f}, {best['f1_hi'] or 0:.4f}]")
    print(f"  TP/FP/FN:   {int(best['tp'])}/{int(best['fp'])}/{int(best['fn'])}")

    print(f"\n--- Current Operating Point (τ={operating_tau:.2f}) ---")
    print(f"  Precision:  {op_row['precision']:.4f}  [{op_row['precision_lo'] or 0:.4f}, {op_row['precision_hi'] or 0:.4f}]")
    print(f"  Recall:     {op_row['recall']:.4f}  [{op_row['recall_lo'] or 0:.4f}, {op_row['recall_hi'] or 0:.4f}]")
    print(f"  F1:         {op_row['f1']:.4f}  [{op_row['f1_lo'] or 0:.4f}, {op_row['f1_hi'] or 0:.4f}]")
    print(f"  TP/FP/FN:   {int(op_row['tp'])}/{int(op_row['fp'])}/{int(op_row['fn'])}")

    print("\n--- Full Threshold Table ---")
    header = f"{'TAU':>6}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'SUPPORT':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        marker = " <-- best" if row["threshold"] == best["threshold"] else ""
        op_marker = " <-- operating" if row["threshold"] == op_row["threshold"] else marker
        print(
            f"{row['threshold']:6.3f}  "
            f"{row['precision']:7.4f}  "
            f"{row['recall']:7.4f}  "
            f"{row['f1']:7.4f}  "
            f"{int(row['tp']):6d}  "
            f"{int(row['fp']):6d}  "
            f"{int(row['fn']):6d}  "
            f"{int(row['support']):8d}"
            f"{op_marker}"
        )
    print("=" * 60)


def main() -> None:
    args = parse_args()

    print(f"Loading {len(args.manifests)} manifest(s)...")
    raw_manifests = [_load_manifest(Path(p)) for p in args.manifests]
    manifest = _merge_manifests(raw_manifests)

    # Determine operating threshold from manifest config (default 0.4)
    config = manifest.get("config", {})
    operating_tau: float = 0.4
    try:
        operating_tau = float(config.get("coverage_threshold", 0.4))
    except (TypeError, ValueError):
        pass

    print(f"Running threshold sweep (n_bootstrap={args.bootstrap}, ci={args.ci})...")
    rows = _compute_sweep(
        manifest,
        thresholds=args.thresholds,
        overlap_threshold=args.overlap_threshold,
        n_resamples=args.bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    if not rows:
        print("ERROR: No coverage data found in manifest(s). Check that manifests contain")
        print("       streams[*].coverage.plan_items with probability values.")
        raise SystemExit(1)

    _print_summary(rows, operating_tau=operating_tau)

    if args.csv_path is not None:
        _write_csv(rows, args.csv_path)

    if args.output is not None:
        _write_figure(rows, args.output, operating_tau=operating_tau)


if __name__ == "__main__":
    main()
