"""Document-paired comparison of PDT against blind and oracle controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

from pdt.evaluation.paired_causal import BootstrapMean, bootstrap_mean


__all__ = ["QualityBoundsComparison", "compare_quality_bounds"]


@dataclass(frozen=True, slots=True)
class QualityBoundsComparison:
    """Strict evidence gate for checkpoint quality within communication bounds."""

    trunk_profile: str
    global_step: int
    coordination_source: str
    documents: int
    dependency_tokens: int
    nondependency_tokens: int
    blind_dependency_ce: float
    pdt_dependency_ce: float
    oracle_dependency_ce: float
    blind_nondependency_ce: float
    pdt_nondependency_ce: float
    oracle_nondependency_ce: float
    blind_to_pdt_dependency_gain: BootstrapMean
    blind_to_pdt_nondependency_gain: BootstrapMean
    pdt_dependency_selectivity_gain: BootstrapMean
    pdt_to_oracle_dependency_headroom: BootstrapMean
    minimum_documents: int
    enough_documents: bool
    baseline_gate_passes: bool
    causal_gate_passes: bool
    dependency_gain_ci_lower_positive: bool
    selectivity_gain_ci_lower_positive: bool
    oracle_headroom_ci_lower_positive: bool
    passes: bool

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        for name in (
            "blind_to_pdt_dependency_gain",
            "blind_to_pdt_nondependency_gain",
            "pdt_dependency_selectivity_gain",
            "pdt_to_oracle_dependency_headroom",
        ):
            result[name] = getattr(self, name).to_dict()
        return result


def compare_quality_bounds(
    baseline_report: Mapping[str, Any],
    pdt_telemetry: Mapping[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 1729,
    minimum_documents: int = 32,
) -> QualityBoundsComparison:
    """Align per-document control CEs with normal PDT rollout CEs."""

    if type(minimum_documents) is not int or minimum_documents <= 1:
        raise ValueError("minimum_documents must be an integer greater than one.")
    profile = _require_text(baseline_report.get("trunk_profile"), "trunk_profile")
    evaluation = _mapping(baseline_report.get("evaluation"), "baseline.evaluation")
    if evaluation.get("dataset_condition") != "dependency":
        raise ValueError("PDT quality comparison requires a dependency control report.")
    baseline_gate = _boolean(
        evaluation.get("expected_outcome_passes"),
        "baseline.expected_outcome_passes",
    )
    conditions = _mapping(evaluation.get("conditions"), "baseline.conditions")
    blind = _condition(conditions, "blind")
    oracle = _condition(conditions, "sequential_oracle")
    baseline_rows = _baseline_documents(evaluation)
    baseline_documents = _positive_int(
        evaluation.get("documents"),
        "baseline.documents",
    )
    if baseline_documents != len(baseline_rows):
        raise ValueError("Baseline document count does not match document_values.")

    global_step = _nonnegative_int(pdt_telemetry.get("global_step"), "pdt.global_step")
    pdt_profile = _require_text(pdt_telemetry.get("trunk_profile"), "pdt.trunk_profile")
    if pdt_profile != profile:
        raise ValueError(
            f"Baseline and PDT trunk profiles do not match: {profile!r} != {pdt_profile!r}."
        )
    coordination_source = _require_text(
        pdt_telemetry.get("coordination_source"),
        "pdt.coordination_source",
    )
    if coordination_source not in {"bus", "self_only"}:
        raise ValueError(
            "pdt.coordination_source must be 'bus' or 'self_only', "
            f"got {coordination_source!r}."
        )
    causal = _mapping(pdt_telemetry.get("causal"), "pdt.causal")
    pdt_documents = _positive_int(causal.get("documents"), "pdt.causal.documents")
    evidence_gate = _mapping(causal.get("evidence_gate"), "pdt.causal.evidence_gate")
    causal_gate = _boolean(evidence_gate.get("passes"), "pdt.causal.evidence_gate.passes")
    gate_zero = _mapping(causal.get("gate_zero"), "pdt.causal.gate_zero")
    pdt_dependency_tokens = _positive_int(
        gate_zero.get("dependency_tokens"),
        "pdt dependency_tokens",
    )
    pdt_nondependency_tokens = _positive_int(
        gate_zero.get("nondependency_tokens"),
        "pdt nondependency_tokens",
    )
    if pdt_dependency_tokens != blind["dependency_tokens"]:
        raise ValueError("Baseline and PDT dependency-token counts do not match.")
    if pdt_nondependency_tokens != blind["nondependency_tokens"]:
        raise ValueError("Baseline and PDT nondependency-token counts do not match.")
    if blind["dependency_tokens"] != oracle["dependency_tokens"]:
        raise ValueError("Blind and oracle dependency-token counts do not match.")
    if blind["nondependency_tokens"] != oracle["nondependency_tokens"]:
        raise ValueError("Blind and oracle nondependency-token counts do not match.")

    inference = _mapping(
        gate_zero.get("document_inference"),
        "pdt.gate_zero.document_inference",
    )
    pdt_rows = _pdt_documents(inference)
    if pdt_documents != len(pdt_rows):
        raise ValueError("PDT document count does not match document_effects.")
    if set(baseline_rows) != set(pdt_rows):
        raise ValueError(
            "Baseline and PDT document identities do not match; "
            f"missing_from_baseline={sorted(set(pdt_rows) - set(baseline_rows))}, "
            f"missing_from_pdt={sorted(set(baseline_rows) - set(pdt_rows))}."
        )

    _validate_document_aggregates(
        baseline_rows,
        dependency_tokens=pdt_dependency_tokens,
        nondependency_tokens=pdt_nondependency_tokens,
        dependency_ce=blind["dependency_ce"],
        nondependency_ce=blind["nondependency_ce"],
        dependency_field="blind_dependency_ce",
        nondependency_field="blind_nondependency_ce",
        label="blind",
    )
    _validate_document_aggregates(
        baseline_rows,
        dependency_tokens=pdt_dependency_tokens,
        nondependency_tokens=pdt_nondependency_tokens,
        dependency_ce=oracle["dependency_ce"],
        nondependency_ce=oracle["nondependency_ce"],
        dependency_field="sequential_oracle_dependency_ce",
        nondependency_field="sequential_oracle_nondependency_ce",
        label="sequential_oracle",
    )
    _validate_document_aggregates(
        pdt_rows,
        dependency_tokens=pdt_dependency_tokens,
        nondependency_tokens=pdt_nondependency_tokens,
        dependency_ce=_finite(
            gate_zero.get("normal_dependency_ce"),
            "PDT dependency CE",
        ),
        nondependency_ce=_finite(
            gate_zero.get("normal_nondependency_ce"),
            "PDT nondependency CE",
        ),
        dependency_field="normal_dependency_ce",
        nondependency_field="normal_nondependency_ce",
        label="PDT",
    )

    dependency_gains: list[float] = []
    nondependency_gains: list[float] = []
    selectivity_gains: list[float] = []
    oracle_headrooms: list[float] = []
    for example_id in sorted(baseline_rows):
        control = baseline_rows[example_id]
        pdt = pdt_rows[example_id]
        if control["dependency_tokens"] != pdt["dependency_tokens"]:
            raise ValueError(
                f"Dependency-token count drifted for document {example_id!r}."
            )
        if control["nondependency_tokens"] != pdt["nondependency_tokens"]:
            raise ValueError(
                f"Nondependency-token count drifted for document {example_id!r}."
            )
        dep_gain = control["blind_dependency_ce"] - pdt["normal_dependency_ce"]
        non_gain = control["blind_nondependency_ce"] - pdt["normal_nondependency_ce"]
        dependency_gains.append(dep_gain)
        nondependency_gains.append(non_gain)
        selectivity_gains.append(dep_gain - non_gain)
        oracle_headrooms.append(
            pdt["normal_dependency_ce"] - control["sequential_oracle_dependency_ce"]
        )

    dependency_estimate = bootstrap_mean(
        dependency_gains,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    nondependency_estimate = bootstrap_mean(
        nondependency_gains,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 1,
    )
    selectivity_estimate = bootstrap_mean(
        selectivity_gains,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 2,
    )
    oracle_estimate = bootstrap_mean(
        oracle_headrooms,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 3,
    )
    documents = len(baseline_rows)
    enough_documents = documents >= minimum_documents
    dependency_positive = dependency_estimate.lower > 0.0
    selectivity_positive = selectivity_estimate.lower > 0.0
    oracle_positive = oracle_estimate.lower > 0.0
    passes = (
        enough_documents
        and baseline_gate
        and causal_gate
        and dependency_positive
        and selectivity_positive
        and oracle_positive
    )
    return QualityBoundsComparison(
        trunk_profile=profile,
        global_step=global_step,
        coordination_source=coordination_source,
        documents=documents,
        dependency_tokens=pdt_dependency_tokens,
        nondependency_tokens=pdt_nondependency_tokens,
        blind_dependency_ce=blind["dependency_ce"],
        pdt_dependency_ce=_finite(gate_zero.get("normal_dependency_ce"), "PDT dependency CE"),
        oracle_dependency_ce=oracle["dependency_ce"],
        blind_nondependency_ce=blind["nondependency_ce"],
        pdt_nondependency_ce=_finite(
            gate_zero.get("normal_nondependency_ce"),
            "PDT nondependency CE",
        ),
        oracle_nondependency_ce=oracle["nondependency_ce"],
        blind_to_pdt_dependency_gain=dependency_estimate,
        blind_to_pdt_nondependency_gain=nondependency_estimate,
        pdt_dependency_selectivity_gain=selectivity_estimate,
        pdt_to_oracle_dependency_headroom=oracle_estimate,
        minimum_documents=minimum_documents,
        enough_documents=enough_documents,
        baseline_gate_passes=baseline_gate,
        causal_gate_passes=causal_gate,
        dependency_gain_ci_lower_positive=dependency_positive,
        selectivity_gain_ci_lower_positive=selectivity_positive,
        oracle_headroom_ci_lower_positive=oracle_positive,
        passes=passes,
    )


def _condition(conditions: Mapping[str, Any], name: str) -> dict[str, int | float]:
    row = _mapping(conditions.get(name), f"baseline.conditions.{name}")
    return {
        "dependency_tokens": _positive_int(
            row.get("dependency_tokens"),
            f"{name}.dependency_tokens",
        ),
        "nondependency_tokens": _positive_int(
            row.get("nondependency_tokens"),
            f"{name}.nondependency_tokens",
        ),
        "dependency_ce": _finite(row.get("dependency_ce"), f"{name}.dependency_ce"),
        "nondependency_ce": _finite(
            row.get("nondependency_ce"),
            f"{name}.nondependency_ce",
        ),
    }


def _baseline_documents(evaluation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = evaluation.get("document_values")
    if not isinstance(rows, list) or not rows:
        raise ValueError("baseline.document_values must be a non-empty list.")
    required = (
        "blind_dependency_ce",
        "blind_nondependency_ce",
        "sequential_oracle_dependency_ce",
        "sequential_oracle_nondependency_ce",
    )
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        item = _mapping(row, "baseline document row")
        example_id = _require_text(item.get("example_id"), "baseline example_id")
        if example_id in result:
            raise ValueError(f"Duplicate baseline example_id {example_id!r}.")
        result[example_id] = {
            name: _finite(item.get(name), f"baseline {name} for {example_id}")
            for name in required
        }
        result[example_id]["dependency_tokens"] = float(
            _positive_int(
                item.get("dependency_tokens"),
                f"baseline dependency_tokens for {example_id}",
            )
        )
        result[example_id]["nondependency_tokens"] = float(
            _positive_int(
                item.get("nondependency_tokens"),
                f"baseline nondependency_tokens for {example_id}",
            )
        )
    return result


def _pdt_documents(inference: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = inference.get("document_effects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pdt document_effects must be a non-empty list.")
    required = ("normal_dependency_ce", "normal_nondependency_ce")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        item = _mapping(row, "PDT document row")
        example_id = _require_text(item.get("example_id"), "PDT example_id")
        if example_id in result:
            raise ValueError(f"Duplicate PDT example_id {example_id!r}.")
        result[example_id] = {
            name: _finite(item.get(name), f"PDT {name} for {example_id}")
            for name in required
        }
        result[example_id]["dependency_tokens"] = float(
            _positive_int(
                item.get("dependency_tokens"),
                f"PDT dependency_tokens for {example_id}",
            )
        )
        result[example_id]["nondependency_tokens"] = float(
            _positive_int(
                item.get("nondependency_tokens"),
                f"PDT nondependency_tokens for {example_id}",
            )
        )
    return result


def _validate_document_aggregates(
    rows: Mapping[str, Mapping[str, float]],
    *,
    dependency_tokens: int,
    nondependency_tokens: int,
    dependency_ce: float,
    nondependency_ce: float,
    dependency_field: str,
    nondependency_field: str,
    label: str,
) -> None:
    document_dependency_tokens = sum(int(row["dependency_tokens"]) for row in rows.values())
    document_nondependency_tokens = sum(
        int(row["nondependency_tokens"]) for row in rows.values()
    )
    if document_dependency_tokens != dependency_tokens:
        raise ValueError(f"{label} document dependency-token counts do not reconstruct aggregate.")
    if document_nondependency_tokens != nondependency_tokens:
        raise ValueError(
            f"{label} document nondependency-token counts do not reconstruct aggregate."
        )
    reconstructed_dependency_ce = sum(
        row[dependency_field] * row["dependency_tokens"] for row in rows.values()
    ) / dependency_tokens
    reconstructed_nondependency_ce = sum(
        row[nondependency_field] * row["nondependency_tokens"] for row in rows.values()
    ) / nondependency_tokens
    if not math.isclose(
        reconstructed_dependency_ce,
        dependency_ce,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label} document dependency CEs do not reconstruct aggregate.")
    if not math.isclose(
        reconstructed_nondependency_ce,
        nondependency_ce,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{label} document nondependency CEs do not reconstruct aggregate.")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text, got {value!r}.")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean, got {value!r}.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}.")
    return result
