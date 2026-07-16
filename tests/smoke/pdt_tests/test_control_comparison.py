"""Contracts for the preregistered bus versus self-only comparison."""

from __future__ import annotations

import pytest

from pdt.evaluation.control_comparison import compare_self_only_recovery


def _telemetry(source: str, gain: float, *, tokens: int = 24, step: int = 512):
    return {
        "global_step": step,
        "coordination_source": source,
        "causal": {
            "gate_zero": {
                "dependency_tokens": tokens,
                "dependency_ce_delta": gain,
            }
        },
    }


def test_recovery_fraction_uses_each_conditions_paired_gate_zero_gain() -> None:
    result = compare_self_only_recovery(
        _telemetry("bus", 0.8),
        _telemetry("self_only", 0.2),
    )

    assert result.recovery_fraction == pytest.approx(0.25)
    assert result.maximum_recovery_fraction == 0.5
    assert result.passes is True
    assert result.to_dict()["dependency_tokens"] == 24


def test_half_or_more_recovery_is_a_preregistered_failure() -> None:
    result = compare_self_only_recovery(
        _telemetry("bus", 0.8),
        _telemetry("self_only", 0.4),
    )
    assert result.recovery_fraction == pytest.approx(0.5)
    assert result.passes is False


@pytest.mark.parametrize(
    ("bus", "self_only", "message"),
    [
        (_telemetry("self_only", 0.8), _telemetry("self_only", 0.2), "Expected 'bus'"),
        (_telemetry("bus", 0.0), _telemetry("self_only", 0.2), "must be positive"),
        (_telemetry("bus", 0.8, step=1), _telemetry("self_only", 0.2), "steps must match"),
        (_telemetry("bus", 0.8, tokens=12), _telemetry("self_only", 0.2), "counts must match"),
    ],
)
def test_misaligned_or_unidentified_telemetry_fails_fast(bus, self_only, message) -> None:
    with pytest.raises(ValueError, match=message):
        compare_self_only_recovery(bus, self_only)
