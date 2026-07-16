"""Document-paired blind/PDT/oracle quality-bound contracts."""

from __future__ import annotations

import copy

import pytest

from pdt.evaluation.quality_comparison import compare_quality_bounds


def _reports() -> tuple[dict[str, object], dict[str, object]]:
    baseline_rows = []
    pdt_rows = []
    blind_dependency = [3.0, 3.2, 2.8]
    pdt_dependency = [2.0, 2.1, 1.9]
    oracle_dependency = [1.0, 1.1, 0.9]
    for index in range(3):
        example_id = f"document-{index}"
        baseline_rows.append(
            {
                "example_id": example_id,
                "dependency_tokens": 10,
                "nondependency_tokens": 20,
                "blind_dependency_ce": blind_dependency[index],
                "blind_nondependency_ce": 1.0,
                "sequential_oracle_dependency_ce": oracle_dependency[index],
                "sequential_oracle_nondependency_ce": 0.9,
            }
        )
        pdt_rows.append(
            {
                "example_id": example_id,
                "dependency_tokens": 10,
                "nondependency_tokens": 20,
                "normal_dependency_ce": pdt_dependency[index],
                "normal_nondependency_ce": 1.0,
            }
        )
    baseline: dict[str, object] = {
        "trunk_profile": "qwen3_4b_instruct_2507",
        "evaluation": {
            "dataset_condition": "dependency",
            "documents": 3,
            "conditions": {
                "blind": {
                    "dependency_tokens": 30,
                    "nondependency_tokens": 60,
                    "dependency_ce": 3.0,
                    "nondependency_ce": 1.0,
                },
                "sequential_oracle": {
                    "dependency_tokens": 30,
                    "nondependency_tokens": 60,
                    "dependency_ce": 1.0,
                    "nondependency_ce": 0.9,
                },
            },
            "document_values": baseline_rows,
            "expected_outcome_passes": True,
        },
    }
    telemetry: dict[str, object] = {
        "global_step": 512,
        "trunk_profile": "qwen3_4b_instruct_2507",
        "coordination_source": "bus",
        "causal": {
            "documents": 3,
            "evidence_gate": {"passes": True},
            "gate_zero": {
                "dependency_tokens": 30,
                "nondependency_tokens": 60,
                "normal_dependency_ce": 2.0,
                "normal_nondependency_ce": 1.0,
                "document_inference": {"document_effects": pdt_rows},
            },
        },
    }
    return baseline, telemetry


def test_quality_bounds_use_paired_documents_and_strict_ordering() -> None:
    baseline, telemetry = _reports()
    result = compare_quality_bounds(
        baseline,
        telemetry,
        bootstrap_samples=2000,
        minimum_documents=3,
    )

    assert result.documents == 3
    assert result.blind_to_pdt_dependency_gain.mean == pytest.approx(1.0)
    assert result.blind_to_pdt_nondependency_gain.mean == pytest.approx(0.0)
    assert result.pdt_dependency_selectivity_gain.mean == pytest.approx(1.0)
    assert result.pdt_to_oracle_dependency_headroom.mean == pytest.approx(1.0)
    assert result.passes is True


def test_quality_bounds_reject_identity_and_checkpoint_schema_drift() -> None:
    baseline, telemetry = _reports()
    mismatched = copy.deepcopy(telemetry)
    effects = mismatched["causal"]["gate_zero"]["document_inference"]["document_effects"]
    effects[0]["example_id"] = "other-document"
    with pytest.raises(ValueError, match="document identities do not match"):
        compare_quality_bounds(baseline, mismatched, minimum_documents=3)

    legacy = copy.deepcopy(telemetry)
    del legacy["causal"]["gate_zero"]["document_inference"]["document_effects"][0][
        "normal_dependency_ce"
    ]
    with pytest.raises(ValueError, match="normal_dependency_ce"):
        compare_quality_bounds(baseline, legacy, minimum_documents=3)


def test_quality_bounds_fail_when_oracle_is_not_an_upper_control() -> None:
    baseline, telemetry = _reports()
    evaluation = baseline["evaluation"]
    for row in evaluation["document_values"]:
        row["sequential_oracle_dependency_ce"] = 2.5
    evaluation["conditions"]["sequential_oracle"]["dependency_ce"] = 2.5
    result = compare_quality_bounds(
        baseline,
        telemetry,
        bootstrap_samples=2000,
        minimum_documents=3,
    )

    assert result.oracle_headroom_ci_lower_positive is False
    assert result.passes is False
