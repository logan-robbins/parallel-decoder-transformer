"""Evaluation helpers for Parallel Decoder Transformer."""

from .attributes import AttributeConsistencyResult, compute_attribute_consistency
from .manifest_metrics import (
    AggregateMetrics,
    CoverageStreamSummary,
    CoverageSummary,
    GateStreamSummary,
    GateSummary,
    ManifestMetrics,
    PlanEntry,
    RollbackSummary,
    aggregate_metrics,
    compute_coverage_summary,
    compute_gate_summary,
    compute_rollback_summary,
    summarize_manifest,
)

__all__ = [
    "AggregateMetrics",
    "AttributeConsistencyResult",
    "CoverageStreamSummary",
    "CoverageSummary",
    "GateStreamSummary",
    "GateSummary",
    "ManifestMetrics",
    "PlanEntry",
    "RollbackSummary",
    "aggregate_metrics",
    "compute_attribute_consistency",
    "compute_coverage_summary",
    "compute_gate_summary",
    "compute_rollback_summary",
    "summarize_manifest",
]
