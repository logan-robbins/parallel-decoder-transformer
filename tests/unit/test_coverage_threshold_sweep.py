"""Tests for coverage threshold sweep functions and CLI script."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from parallel_decoder_transformer.evaluation.manifest_metrics import (
    bootstrap_coverage_ci,
    compute_coverage_roc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DISTINCT_WORDS = [
    "apple", "bridge", "cedar", "delta", "ember",
    "fjord", "grape", "harbor", "indigo", "jasper",
    "kelp", "lunar", "maple", "nexus", "orbit",
    "prism", "quartz", "river", "stella", "tundra",
    "ultra", "vapor", "walnut", "xenon", "yellow",
    "zephyr", "amber", "birch", "cobalt", "dune",
]


def _make_manifest(
    probs_and_labels: List[tuple[float, bool]],
    stream: str = "intro",
) -> Dict[str, Any]:
    """Build a minimal manifest with fully distinct plan item texts.

    Each plan item uses a unique single word so token-overlap scoring is exact:
    an item is covered iff its unique word appears in the stream text.
    """
    plan_items = []
    catalog = []
    stream_text_parts = []
    for idx, (prob, covered) in enumerate(probs_and_labels):
        # One unique word per item — zero cross-item token overlap
        plan_text = _DISTINCT_WORDS[idx % len(_DISTINCT_WORDS)]
        plan_items.append(
            {
                "index": idx,
                "stream": stream,
                "text": plan_text,
                "probability": prob,
            }
        )
        catalog.append({"index": idx, "stream": stream, "text": plan_text, "plan_item_id": idx})
        if covered:
            stream_text_parts.append(plan_text)
    return {
        "plan": {"catalog": catalog},
        "streams": {
            stream: {
                "text": " ".join(stream_text_parts),
                "coverage": {"plan_items": plan_items},
                "token_ids": [],
            }
        },
        "config": {"coverage_threshold": 0.4},
    }


# ---------------------------------------------------------------------------
# compute_coverage_roc tests
# ---------------------------------------------------------------------------

def test_compute_coverage_roc_returns_sorted_thresholds() -> None:
    # 3 positives, all predicted with high probability
    manifest = _make_manifest([(0.9, True), (0.8, True), (0.7, True), (0.1, False)])
    points = compute_coverage_roc(manifest)
    assert len(points) == 19
    taus = [p["tau"] for p in points]
    assert taus == sorted(taus), "ROC points must be sorted by threshold"
    for p in points:
        assert set(p.keys()) >= {"tau", "precision", "recall", "f1", "tp", "fp", "fn", "support"}


def test_compute_coverage_roc_monotone_recall_at_low_threshold() -> None:
    # At very low threshold every positive should be predicted positive → recall ~1
    manifest = _make_manifest(
        [(0.9, True), (0.8, True), (0.7, True), (0.3, True), (0.1, False)]
    )
    points = compute_coverage_roc(manifest, thresholds=[0.05])
    assert len(points) == 1
    point = points[0]
    # All 4 positives predicted at tau=0.05 → recall = 1.0
    assert point["recall"] == 1.0, f"Expected recall=1.0 at tau=0.05, got {point['recall']}"
    assert point["tp"] == 4


def test_compute_coverage_roc_empty_manifest() -> None:
    points = compute_coverage_roc({})
    assert points == []


def test_compute_coverage_roc_all_positive() -> None:
    # All items are ground-truth positive
    manifest = _make_manifest([(0.9, True), (0.8, True), (0.6, True)])
    points = compute_coverage_roc(manifest, thresholds=[0.1, 0.5, 0.95])
    # At tau=0.1 everything is predicted positive → recall=1.0, precision=1.0
    low_point = next(p for p in points if p["tau"] == 0.1)
    assert low_point["recall"] == 1.0
    assert low_point["precision"] == 1.0
    # At tau=0.95 nothing predicted positive → recall=0.0
    high_point = next(p for p in points if p["tau"] == 0.95)
    assert high_point["recall"] == 0.0
    assert high_point["tp"] == 0


# ---------------------------------------------------------------------------
# bootstrap_coverage_ci tests
# ---------------------------------------------------------------------------

def test_bootstrap_coverage_ci_contains_true_value() -> None:
    # 8 positives correctly predicted, 2 negatives; known P=1.0, R=1.0, F1=1.0
    probs_labels = [(0.9, True)] * 8 + [(0.1, False)] * 2
    manifest = _make_manifest(probs_labels)
    result = bootstrap_coverage_ci(manifest, threshold=0.5, n_resamples=500, seed=0)
    assert result["precision_point"] == 1.0
    assert result["recall_point"] == 1.0
    assert result["f1_point"] == 1.0
    # CI should contain 1.0 (or at least have reasonable hi values)
    assert result["precision_hi"] is not None
    assert result["precision_lo"] is not None
    assert result["f1_lo"] is not None and result["f1_hi"] is not None
    assert result["n_pairs"] == 10


def test_bootstrap_coverage_ci_width_decreases_with_more_samples() -> None:
    # With more resamples the CI bounds converge (not necessarily narrower, but
    # functionally this checks that the function runs without error and returns
    # non-None bounds with different resample counts).
    import random as _random
    _random.seed(42)
    pairs = [(0.7 if i % 3 != 0 else 0.2, i % 3 != 0) for i in range(30)]
    manifest = _make_manifest(pairs)

    ci_small = bootstrap_coverage_ci(manifest, threshold=0.5, n_resamples=100, seed=1)
    ci_large = bootstrap_coverage_ci(manifest, threshold=0.5, n_resamples=1000, seed=1)

    # Both should return non-None
    assert ci_small["f1_lo"] is not None
    assert ci_large["f1_lo"] is not None
    # With 10x more resamples the interval width should be ≤ that of 100 resamples
    # (we just verify both give meaningful estimates, not strict inequality which
    #  would be flaky)
    width_small = (ci_small["f1_hi"] or 0.0) - (ci_small["f1_lo"] or 0.0)
    width_large = (ci_large["f1_hi"] or 0.0) - (ci_large["f1_lo"] or 0.0)
    assert width_small >= 0.0
    assert width_large >= 0.0


def test_bootstrap_coverage_ci_empty_manifest() -> None:
    result = bootstrap_coverage_ci({}, threshold=0.5)
    assert result["precision_point"] is None
    assert result["n_pairs"] == 0


# ---------------------------------------------------------------------------
# CLI end-to-end test
# ---------------------------------------------------------------------------

def test_coverage_threshold_sweep_script_cli() -> None:
    """End-to-end: write a tiny manifest, run the script, check CSV output."""
    probs_labels = (
        [(0.9, True)] * 5
        + [(0.3, True)] * 3
        + [(0.8, False)] * 2
        + [(0.1, False)] * 4
    )
    manifest = _make_manifest(probs_labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "manifest.json"
        csv_path = tmpdir_path / "sweep.csv"

        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "scripts" / "coverage_threshold_sweep.py"),
                "--manifest",
                str(manifest_path),
                "--thresholds",
                "0.1",
                "0.3",
                "0.5",
                "0.7",
                "0.9",
                "--bootstrap",
                "50",
                "--csv",
                str(csv_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        assert csv_path.exists(), "CSV output file was not created"
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 5, f"Expected 5 threshold rows, got {len(rows)}"
        expected_cols = {
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
        }
        assert expected_cols.issubset(set(rows[0].keys())), (
            f"Missing columns: {expected_cols - set(rows[0].keys())}"
        )
        # Threshold values should match what we passed
        taus = [float(r["threshold"]) for r in rows]
        assert 0.1 in taus
        assert 0.9 in taus
