"""Evaluation contracts for paired PDT causal interventions."""

from pdt.evaluation.control_comparison import SelfOnlyComparison, compare_self_only
from pdt.evaluation.paired_causal import (
    CausalDocumentEvaluation,
    CausalDocumentEvaluator,
    PairedCausalEvaluation,
    PairedCausalEvaluator,
    TargetedMutationMetrics,
)
from pdt.evaluation.quality_comparison import QualityBoundsComparison, compare_quality_bounds
from pdt.evaluation.quality_controls import (
    DocumentQualityControl,
    QualityControlEvaluation,
    score_quality_control_records,
)
from pdt.evaluation.real_plan_generation import (
    GenerationEvaluationConfig,
    run_generation_evaluation,
)

__all__ = [
    "PairedCausalEvaluation",
    "PairedCausalEvaluator",
    "CausalDocumentEvaluation",
    "CausalDocumentEvaluator",
    "SelfOnlyComparison",
    "TargetedMutationMetrics",
    "DocumentQualityControl",
    "QualityBoundsComparison",
    "QualityControlEvaluation",
    "GenerationEvaluationConfig",
    "compare_self_only",
    "compare_quality_bounds",
    "score_quality_control_records",
    "run_generation_evaluation",
]
