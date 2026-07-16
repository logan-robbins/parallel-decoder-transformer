"""Document-paired comparison of bus and self-only capacity controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

from pdt.evaluation.paired_causal import BootstrapMean, bootstrap_mean


__all__ = ["SelfOnlyComparison", "compare_self_only"]


@dataclass(frozen=True, slots=True)
class SelfOnlyComparison:
    """Paired bus advantage over a matched independently trained control."""

    global_step: int
    dependency_tokens: int
    documents: int
    bus_dependency_effect_mean: float
    self_only_dependency_effect_mean: float
    bus_advantage: BootstrapMean
    minimum_documents: int
    enough_documents: bool
    advantage_ci_lower_positive: bool
    passes: bool

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["bus_advantage"] = self.bus_advantage.to_dict()
        return result


def compare_self_only(
    bus_telemetry: Mapping[str, Any],
    self_only_telemetry: Mapping[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 1729,
    minimum_documents: int = 32,
) -> SelfOnlyComparison:
    """Test whether bus dependency effects exceed self-only effects by document."""

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
    bus_effects = _document_effects(bus_gate, "bus")
    self_effects = _document_effects(self_gate, "self_only")
    if set(bus_effects) != set(self_effects):
        missing_bus = sorted(set(self_effects) - set(bus_effects))
        missing_self = sorted(set(bus_effects) - set(self_effects))
        raise ValueError(
            "Control telemetry document identities must match; "
            f"missing_from_bus={missing_bus}, missing_from_self_only={missing_self}."
        )
    ordered_ids = sorted(bus_effects)
    advantages = [bus_effects[example_id] - self_effects[example_id] for example_id in ordered_ids]
    estimate = bootstrap_mean(
        advantages,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    if type(minimum_documents) is not int or minimum_documents <= 1:
        raise ValueError("minimum_documents must be an integer greater than one.")
    documents = len(ordered_ids)
    enough_documents = documents >= minimum_documents
    lower_positive = estimate.lower > 0.0
    return SelfOnlyComparison(
        global_step=bus_step,
        dependency_tokens=bus_tokens,
        documents=documents,
        bus_dependency_effect_mean=sum(bus_effects.values()) / documents,
        self_only_dependency_effect_mean=sum(self_effects.values()) / documents,
        bus_advantage=estimate,
        minimum_documents=minimum_documents,
        enough_documents=enough_documents,
        advantage_ci_lower_positive=lower_positive,
        passes=enough_documents and lower_positive,
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


def _document_effects(gate: Mapping[str, Any], label: str) -> dict[str, float]:
    inference = gate.get("document_inference")
    if not isinstance(inference, Mapping):
        raise ValueError(f"{label}.gate_zero.document_inference must be a mapping.")
    rows = inference.get("document_effects")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} document_effects must be a non-empty list.")
    effects: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} document_effects entries must be mappings.")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in effects:
            raise ValueError(f"{label} document_effects must have unique non-empty example_ids.")
        effects[example_id] = _require_finite_float(
            row.get("dependency_ce_delta"),
            f"{label} dependency effect for {example_id}",
        )
    return effects


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
