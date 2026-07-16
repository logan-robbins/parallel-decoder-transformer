"""Strict paired-logit evaluation for PDT causal interventions.

Every update compares aligned teacher-forced logits from the same examples,
streams, blocks, and target positions.  Metrics are accumulated as token loss
sums and token counts; batch means are never averaged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from pdt.diagnostics.causal_metrics import (
    CausalAblationAccumulator,
    CausalAblationMetrics,
)


__all__ = [
    "PairedCausalEvaluation",
    "PairedCausalEvaluator",
    "TargetedMutationMetrics",
]


@dataclass(frozen=True, slots=True)
class TargetedMutationMetrics:
    """Baseline-to-mutation KL restricted to targeted dependency tokens."""

    mutation_dependency_tokens: int
    baseline_to_mutation_kl: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedCausalEvaluation:
    """Token-weighted results for the required causal interventions."""

    batches: int
    gate_zero: CausalAblationMetrics
    norm_scramble: CausalAblationMetrics
    targeted_mutation: TargetedMutationMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "batches": self.batches,
            "gate_zero": self.gate_zero.to_dict(),
            "norm_scramble": self.norm_scramble.to_dict(),
            "targeted_mutation": self.targeted_mutation.to_dict(),
        }


class PairedCausalEvaluator:
    """Aggregate aligned baseline/intervention teacher-forced logits.

    The token axes may contain any layout after the leading batch axis (for
    example ``[B, K, M, tau]``), but every input in an update must use the same
    layout.  Dependency and nondependency masks may leave formatting tokens
    unannotated, but they must be disjoint subsets of ``label_mask``.  The
    mutation dependency mask must be a non-empty subset of the dependency mask.
    """

    def __init__(self) -> None:
        self._gate_zero = CausalAblationAccumulator()
        self._norm_scramble = CausalAblationAccumulator()
        self._mutation_kl_sum = 0.0
        self._mutation_count = 0
        self._batches = 0

    @torch.no_grad()
    def update(
        self,
        *,
        baseline_logits: torch.Tensor,
        gate_zero_logits: torch.Tensor,
        norm_scramble_logits: torch.Tensor,
        mutation_logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        dependency_mask: torch.Tensor,
        nondependency_mask: torch.Tensor,
        mutation_dependency_mask: torch.Tensor,
    ) -> None:
        """Validate and add one aligned paired batch.

        Validation completes before any accumulator is updated, so a rejected
        batch cannot leave partial gate-zero, scramble, or mutation state.
        """

        _validate_paired_batch(
            baseline_logits=baseline_logits,
            gate_zero_logits=gate_zero_logits,
            norm_scramble_logits=norm_scramble_logits,
            mutation_logits=mutation_logits,
            labels=labels,
            label_mask=label_mask,
            dependency_mask=dependency_mask,
            nondependency_mask=nondependency_mask,
            mutation_dependency_mask=mutation_dependency_mask,
        )
        mutation_kl_sum, mutation_count = _targeted_mutation_batch_values(
            baseline_logits=baseline_logits,
            mutation_logits=mutation_logits,
            mutation_dependency_mask=mutation_dependency_mask,
        )
        shared = {
            "normal_logits": baseline_logits,
            "labels": labels,
            "label_mask": label_mask,
            "dependency_mask": dependency_mask,
            "nondependency_mask": nondependency_mask,
        }
        self._gate_zero.update_from_logits(
            ablated_logits=gate_zero_logits,
            **shared,
        )
        self._norm_scramble.update_from_logits(
            ablated_logits=norm_scramble_logits,
            **shared,
        )
        self._mutation_kl_sum += mutation_kl_sum
        self._mutation_count += mutation_count
        self._batches += 1

    def compute(self, *, epsilon: float = 1e-8) -> PairedCausalEvaluation:
        """Return token-weighted aggregate metrics for all accepted batches."""

        if self._batches == 0:
            raise RuntimeError("cannot compute paired causal evaluation: zero batches observed.")
        gate_zero = self._gate_zero.compute(epsilon=epsilon)
        norm_scramble = self._norm_scramble.compute(epsilon=epsilon)
        token_counts = {
            gate_zero.dependency_tokens,
            norm_scramble.dependency_tokens,
        }
        if len(token_counts) != 1:
            raise RuntimeError("paired causal accumulators have inconsistent dependency counts.")
        if self._mutation_count == 0:
            raise RuntimeError(
                "cannot compute targeted mutation metrics: zero mutation dependency tokens observed."
            )
        return PairedCausalEvaluation(
            batches=self._batches,
            gate_zero=gate_zero,
            norm_scramble=norm_scramble,
            targeted_mutation=TargetedMutationMetrics(
                mutation_dependency_tokens=self._mutation_count,
                baseline_to_mutation_kl=self._mutation_kl_sum / self._mutation_count,
            ),
        )


def _validate_paired_batch(
    *,
    baseline_logits: torch.Tensor,
    gate_zero_logits: torch.Tensor,
    norm_scramble_logits: torch.Tensor,
    mutation_logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    dependency_mask: torch.Tensor,
    nondependency_mask: torch.Tensor,
    mutation_dependency_mask: torch.Tensor,
) -> None:
    logits = {
        "baseline_logits": baseline_logits,
        "gate_zero_logits": gate_zero_logits,
        "norm_scramble_logits": norm_scramble_logits,
        "mutation_logits": mutation_logits,
    }
    if baseline_logits.ndim < 3:
        raise ValueError(
            "baseline_logits must have a batch axis, at least one token axis, "
            "and a vocabulary axis."
        )
    if baseline_logits.size(0) == 0:
        raise ValueError("paired causal evaluation batches must be non-empty.")
    if baseline_logits.size(-1) <= 1:
        raise ValueError("logit vocabulary width must be greater than one.")
    if not baseline_logits.is_floating_point():
        raise TypeError("baseline_logits must have a floating-point dtype.")

    expected_shape = baseline_logits.shape
    expected_device = baseline_logits.device
    expected_dtype = baseline_logits.dtype
    for name, tensor in logits.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}."
            )
        if tensor.device != expected_device:
            raise ValueError(f"{name} must be on {expected_device}, got {tensor.device}.")
        if tensor.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}.")

    token_shape = expected_shape[:-1]
    tensors = {
        "labels": labels,
        "label_mask": label_mask,
        "dependency_mask": dependency_mask,
        "nondependency_mask": nondependency_mask,
        "mutation_dependency_mask": mutation_dependency_mask,
    }
    for name, tensor in tensors.items():
        if tensor.shape != token_shape:
            raise ValueError(
                f"{name} must have shape {tuple(token_shape)}, got {tuple(tensor.shape)}."
            )
        if tensor.device != expected_device:
            raise ValueError(f"{name} must be on {expected_device}, got {tensor.device}.")

    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError("labels must have an integer dtype other than bool.")
    for name, mask in (
        ("label_mask", label_mask),
        ("dependency_mask", dependency_mask),
        ("nondependency_mask", nondependency_mask),
        ("mutation_dependency_mask", mutation_dependency_mask),
    ):
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}.")

    if bool((dependency_mask & nondependency_mask).any()):
        raise ValueError("dependency_mask and nondependency_mask must be disjoint.")
    if bool((mutation_dependency_mask & ~dependency_mask).any()):
        raise ValueError("mutation_dependency_mask must be a subset of dependency_mask.")
    if not bool(mutation_dependency_mask.any()):
        raise ValueError("mutation_dependency_mask must contain at least one token.")
    annotated = dependency_mask | nondependency_mask
    if bool((annotated & ~label_mask).any()):
        raise ValueError("annotated span tokens must be included in label_mask.")
    if not bool(annotated.any()):
        raise ValueError("each paired batch must contain at least one annotated token.")

    active_labels = labels[label_mask]
    if active_labels.numel() and (
        bool((active_labels < 0).any()) or bool((active_labels >= baseline_logits.size(-1)).any())
    ):
        raise ValueError(f"active labels must be in [0, {baseline_logits.size(-1)}).")

    for name, tensor in logits.items():
        annotated_logits = tensor[annotated]
        if annotated_logits.numel() and not bool(torch.isfinite(annotated_logits).all()):
            raise ValueError(f"{name} must be finite on every annotated token.")


def _targeted_mutation_batch_values(
    *,
    baseline_logits: torch.Tensor,
    mutation_logits: torch.Tensor,
    mutation_dependency_mask: torch.Tensor,
) -> tuple[float, int]:
    baseline_log_probs = F.log_softmax(baseline_logits.float(), dim=-1)
    mutation_log_probs = F.log_softmax(mutation_logits.float(), dim=-1)
    baseline_to_mutation_kl = (
        baseline_log_probs.exp() * (baseline_log_probs - mutation_log_probs)
    ).sum(dim=-1)
    selected = baseline_to_mutation_kl[mutation_dependency_mask]
    return (
        float(selected.detach().to(device="cpu", dtype=torch.float64).sum().item()),
        int(mutation_dependency_mask.sum().item()),
    )
