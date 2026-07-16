"""Strict comparison of independently trained bus and self-only telemetry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


__all__ = ["SelfOnlyRecovery", "compare_self_only_recovery"]


@dataclass(frozen=True, slots=True)
class SelfOnlyRecovery:
    """Dependency-span capacity gains and the preregistered recovery test."""

    global_step: int
    dependency_tokens: int
    bus_dependency_gain: float
    self_only_dependency_gain: float
    recovery_fraction: float
    maximum_recovery_fraction: float
    passes: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def compare_self_only_recovery(
    bus_telemetry: Mapping[str, Any],
    self_only_telemetry: Mapping[str, Any],
    *,
    maximum_recovery_fraction: float = 0.5,
) -> SelfOnlyRecovery:
    """Compare paired gate-zero gains from matched, independently trained runs."""

    if not math.isfinite(maximum_recovery_fraction) or not 0 < maximum_recovery_fraction < 1:
        raise ValueError("maximum_recovery_fraction must be finite and in (0, 1).")
    _require_source(bus_telemetry, "bus")
    _require_source(self_only_telemetry, "self_only")
    bus_step = _require_nonnegative_int(bus_telemetry.get("global_step"), "bus.global_step")
    self_step = _require_nonnegative_int(
        self_only_telemetry.get("global_step"), "self_only.global_step"
    )
    if bus_step != self_step:
        raise ValueError(
            f"Control telemetry steps must match; bus={bus_step}, self_only={self_step}."
        )

    bus_gate = _gate_zero_metrics(bus_telemetry, "bus")
    self_gate = _gate_zero_metrics(self_only_telemetry, "self_only")
    bus_tokens = _require_positive_int(bus_gate.get("dependency_tokens"), "bus dependency_tokens")
    self_tokens = _require_positive_int(
        self_gate.get("dependency_tokens"), "self_only dependency_tokens"
    )
    if bus_tokens != self_tokens:
        raise ValueError(
            "Control telemetry dependency-token counts must match; "
            f"bus={bus_tokens}, self_only={self_tokens}."
        )
    bus_gain = _require_finite_float(bus_gate.get("dependency_ce_delta"), "bus dependency gain")
    self_gain = _require_finite_float(
        self_gate.get("dependency_ce_delta"), "self_only dependency gain"
    )
    if bus_gain <= 0:
        raise ValueError(
            "Bus dependency gain must be positive before self-only recovery is defined; "
            f"got {bus_gain}."
        )
    recovery = self_gain / bus_gain
    return SelfOnlyRecovery(
        global_step=bus_step,
        dependency_tokens=bus_tokens,
        bus_dependency_gain=bus_gain,
        self_only_dependency_gain=self_gain,
        recovery_fraction=recovery,
        maximum_recovery_fraction=maximum_recovery_fraction,
        passes=recovery < maximum_recovery_fraction,
    )


def _require_source(telemetry: Mapping[str, Any], expected: str) -> None:
    actual = telemetry.get("coordination_source")
    if actual != expected:
        raise ValueError(f"Expected {expected!r} coordination telemetry, got source={actual!r}.")


def _gate_zero_metrics(telemetry: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    causal = telemetry.get("causal")
    if not isinstance(causal, Mapping):
        raise ValueError(f"{label}.causal must be a mapping.")
    gate = causal.get("gate_zero")
    if not isinstance(gate, Mapping):
        raise ValueError(f"{label}.causal.gate_zero must be a mapping.")
    return gate


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return value


def _require_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}.")
    return result
