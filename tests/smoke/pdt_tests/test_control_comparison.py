"""Contracts for the document-paired bus versus self-only comparison."""

from __future__ import annotations

import pytest

from pdt.evaluation.control_comparison import compare_self_only


def _telemetry(
    source: str,
    effects: tuple[float, ...],
    *,
    tokens: int = 24,
    step: int = 512,
):
    return {
        "global_step": step,
        "coordination_source": source,
        "causal": {
            "gate_zero": {
                "dependency_tokens": tokens,
                "document_inference": {
                    "document_effects": [
                        {"example_id": f"document-{index}", "dependency_ce_delta": effect}
                        for index, effect in enumerate(effects)
                    ]
                },
            }
        },
    }


def test_bus_advantage_uses_paired_document_effects() -> None:
    result = compare_self_only(
        _telemetry("bus", (0.8, 0.9, 1.0)),
        _telemetry("self_only", (0.2, 0.3, 0.4)),
        bootstrap_samples=2000,
        minimum_documents=3,
    )

    assert result.bus_advantage.mean == pytest.approx(0.6)
    assert result.bus_advantage.lower > 0.0
    assert result.passes is True
    assert result.to_dict()["dependency_tokens"] == 24


def test_overlapping_interval_or_too_few_documents_fails_evidence_gate() -> None:
    overlap = compare_self_only(
        _telemetry("bus", (0.8, 0.2, 0.5)),
        _telemetry("self_only", (0.2, 0.8, 0.5)),
        bootstrap_samples=2000,
        minimum_documents=3,
    )
    assert overlap.bus_advantage.lower <= 0.0
    assert overlap.passes is False

    too_small = compare_self_only(
        _telemetry("bus", (0.8, 0.9)),
        _telemetry("self_only", (0.2, 0.3)),
        bootstrap_samples=2000,
        minimum_documents=3,
    )
    assert too_small.enough_documents is False
    assert too_small.passes is False


@pytest.mark.parametrize(
    ("bus", "self_only", "message"),
    [
        (
            _telemetry("self_only", (0.8, 0.9)),
            _telemetry("self_only", (0.2, 0.3)),
            "Expected 'bus'",
        ),
        (
            _telemetry("bus", (0.8, 0.9), step=1),
            _telemetry("self_only", (0.2, 0.3)),
            "steps must match",
        ),
        (
            _telemetry("bus", (0.8, 0.9), tokens=12),
            _telemetry("self_only", (0.2, 0.3)),
            "counts must match",
        ),
        (
            _telemetry("bus", (0.8, 0.9)),
            _telemetry("self_only", (0.2,)),
            "document identities must match",
        ),
    ],
)
def test_misaligned_or_unidentified_telemetry_fails_fast(bus, self_only, message) -> None:
    with pytest.raises(ValueError, match=message):
        compare_self_only(bus, self_only, bootstrap_samples=2000, minimum_documents=2)
