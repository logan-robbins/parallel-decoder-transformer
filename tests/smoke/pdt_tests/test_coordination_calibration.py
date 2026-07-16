"""Checkpoint-free calibration artifact tests."""

from __future__ import annotations

import csv
import json

from pdt.diagnostics.calibration import run_calibration, write_calibration


def test_calibration_matches_preregistered_signs_and_slopes():
    result, _ = run_calibration()

    expected_nonzero = {
        "snc.o_proj.weight",
        "snc.o_proj.bias",
        "adapter.up_proj",
    }
    for name, norm in result.gradient_norms.items():
        if name in expected_nonzero:
            assert norm > 0.0
        else:
            assert norm == 0.0
    assert all(0.85 <= slope <= 1.15 for slope in result.slopes.values())
    assert result.producer_permutation_delta_ce == 0.0
    assert result.delivery_reorder_delta_ce > 0.0
    assert 55.5 < result.suppression_ratio < 55.7
    assert 3080.0 < result.plateau_ratio < 3095.0


def test_calibration_writes_exact_appendix_c_artifacts(tmp_path):
    result = write_calibration(tmp_path)

    expected = {
        "table1_gradient_norms.csv",
        "figure1_escape_law.png",
        "table2_paired_nulls.csv",
        "calibration_summary.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected

    with (tmp_path / "table1_gradient_norms.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == set(result.gradient_norms)

    summary = json.loads((tmp_path / "calibration_summary.json").read_text())
    assert summary["slopes"] == result.slopes
