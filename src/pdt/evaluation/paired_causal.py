"""Document-paired causal evaluation for PDT interventions.

The scientific unit is a document, not a token or a dataloader batch.  Token-
weighted cross entropy remains available as a descriptive aggregate, while
acceptance evidence is formed from paired per-document effects and a
deterministic bootstrap over documents.  Difference-in-differences is used for
dependency selectivity so a near-zero nondependency effect cannot create an
unstable ratio.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from pdt.diagnostics.causal_metrics import (
    CausalAblationAccumulator,
    CausalAblationMetrics,
)


__all__ = [
    "BootstrapMean",
    "CausalDocumentEvaluation",
    "CausalDocumentEvaluator",
    "CausalEvidenceGate",
    "DocumentEffect",
    "PairedCausalEvaluation",
    "PairedCausalEvaluator",
    "TargetedMutationMetrics",
    "bootstrap_mean",
]


@dataclass(frozen=True, slots=True)
class BootstrapMean:
    """Mean and percentile interval from a document-level paired bootstrap."""

    mean: float
    lower: float
    upper: float
    confidence_level: float
    documents: int
    bootstrap_samples: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentEffect:
    """Normal-to-ablation loss changes for one complete document."""

    example_id: str
    dependency_ce_delta: float
    nondependency_ce_delta: float
    dependency_selectivity_difference: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentInference:
    """Bootstrap inference and auditable values for one intervention."""

    dependency_ce_delta: BootstrapMean
    nondependency_ce_delta: BootstrapMean
    dependency_selectivity_difference: BootstrapMean
    document_effects: tuple[DocumentEffect, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_ce_delta": self.dependency_ce_delta.to_dict(),
            "nondependency_ce_delta": self.nondependency_ce_delta.to_dict(),
            "dependency_selectivity_difference": (
                self.dependency_selectivity_difference.to_dict()
            ),
            "document_effects": [effect.to_dict() for effect in self.document_effects],
        }


@dataclass(frozen=True, slots=True)
class LagCausalEffect:
    """Dependency loss effect for one annotated source-to-target lag."""

    lag_blocks: int
    dependency_tokens: int
    normal_dependency_ce: float
    ablated_dependency_ce: float
    dependency_ce_delta: BootstrapMean

    def to_dict(self) -> dict[str, object]:
        return {
            "lag_blocks": self.lag_blocks,
            "dependency_tokens": self.dependency_tokens,
            "normal_dependency_ce": self.normal_dependency_ce,
            "ablated_dependency_ce": self.ablated_dependency_ce,
            "dependency_ce_delta": self.dependency_ce_delta.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CausalDocumentEvaluation:
    """Token aggregate plus document and lag inference for one intervention."""

    aggregate: CausalAblationMetrics
    document_inference: DocumentInference
    lag_effects: tuple[LagCausalEffect, ...]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            key: value for key, value in self.aggregate.to_dict().items()
        }
        result["document_inference"] = self.document_inference.to_dict()
        result["lag_effects"] = [effect.to_dict() for effect in self.lag_effects]
        return result


@dataclass(frozen=True, slots=True)
class MutationDocumentValue:
    """Targeted baseline-to-mutation KL for one document."""

    example_id: str
    baseline_to_mutation_kl: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TargetedMutationMetrics:
    """Mutation KL restricted to annotated future consumers of one write."""

    mutation_dependency_tokens: int
    baseline_to_mutation_kl: float
    document_bootstrap: BootstrapMean
    document_values: tuple[MutationDocumentValue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mutation_dependency_tokens": self.mutation_dependency_tokens,
            "baseline_to_mutation_kl": self.baseline_to_mutation_kl,
            "document_bootstrap": self.document_bootstrap.to_dict(),
            "document_values": [value.to_dict() for value in self.document_values],
        }


@dataclass(frozen=True, slots=True)
class CausalEvidenceGate:
    """Sign-based evidence gate with no arbitrary effect-size ratio."""

    documents: int
    minimum_documents: int
    enough_documents: bool
    dependency_ci_lower_positive: bool
    selectivity_ci_lower_positive: bool
    mutation_ci_lower_positive: bool
    passes: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedCausalEvaluation:
    """Results for gate zero, sibling scramble, and targeted mutation."""

    batches: int
    documents: int
    gate_zero: CausalDocumentEvaluation
    norm_scramble: CausalDocumentEvaluation
    targeted_mutation: TargetedMutationMetrics
    evidence_gate: CausalEvidenceGate

    def to_dict(self) -> dict[str, object]:
        return {
            "batches": self.batches,
            "documents": self.documents,
            "gate_zero": self.gate_zero.to_dict(),
            "norm_scramble": self.norm_scramble.to_dict(),
            "targeted_mutation": self.targeted_mutation.to_dict(),
            "evidence_gate": self.evidence_gate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _LagDocumentValue:
    example_id: str
    normal_sum: float
    ablated_sum: float
    tokens: int

    @property
    def delta(self) -> float:
        return (self.ablated_sum - self.normal_sum) / self.tokens


class CausalDocumentEvaluator:
    """Accumulate one aligned intervention with document and lag identity."""

    def __init__(self) -> None:
        self._aggregate = CausalAblationAccumulator()
        self._effects: list[DocumentEffect] = []
        self._lag_values: dict[int, list[_LagDocumentValue]] = {}
        self._seen_ids: set[str] = set()

    @torch.no_grad()
    def update(
        self,
        *,
        normal_logits: torch.Tensor,
        ablated_logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        dependency_mask: torch.Tensor,
        nondependency_mask: torch.Tensor,
        dependency_lag_masks: Mapping[int, torch.Tensor],
        example_ids: Sequence[str],
    ) -> None:
        _validate_causal_pair(
            normal_logits=normal_logits,
            ablated_logits=ablated_logits,
            labels=labels,
            label_mask=label_mask,
            dependency_mask=dependency_mask,
            nondependency_mask=nondependency_mask,
            dependency_lag_masks=dependency_lag_masks,
            example_ids=example_ids,
        )
        self._validate_new_ids(example_ids)
        self._update_validated(
            normal_logits=normal_logits,
            ablated_logits=ablated_logits,
            labels=labels,
            label_mask=label_mask,
            dependency_mask=dependency_mask,
            nondependency_mask=nondependency_mask,
            dependency_lag_masks=dependency_lag_masks,
            example_ids=example_ids,
        )

    def _validate_new_ids(self, example_ids: Sequence[str]) -> None:
        repeated = self._seen_ids.intersection(example_ids)
        if repeated:
            raise ValueError(f"causal evaluation example_ids were repeated: {sorted(repeated)}.")

    def _update_validated(
        self,
        *,
        normal_logits: torch.Tensor,
        ablated_logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        dependency_mask: torch.Tensor,
        nondependency_mask: torch.Tensor,
        dependency_lag_masks: Mapping[int, torch.Tensor],
        example_ids: Sequence[str],
    ) -> None:
        normal_nll, ablated_nll, mutation_kl = _token_values(
            normal_logits=normal_logits,
            ablated_logits=ablated_logits,
            labels=labels,
            label_mask=label_mask,
        )
        self._aggregate.update_token_values(
            normal_nll=normal_nll,
            ablated_nll=ablated_nll,
            dependency_mutation_kl=mutation_kl,
            label_mask=label_mask,
            dependency_mask=dependency_mask,
            nondependency_mask=nondependency_mask,
        )
        for document_index, example_id in enumerate(example_ids):
            dep = dependency_mask[document_index]
            non = nondependency_mask[document_index]
            if not bool(dep.any()) or not bool(non.any()):
                raise ValueError(
                    "every causal-evaluation document must contain dependency and "
                    f"nondependency tokens; example_id={example_id!r}."
                )
            dep_delta = _mean_float64(ablated_nll[document_index][dep]) - _mean_float64(
                normal_nll[document_index][dep]
            )
            non_delta = _mean_float64(ablated_nll[document_index][non]) - _mean_float64(
                normal_nll[document_index][non]
            )
            self._effects.append(
                DocumentEffect(
                    example_id=example_id,
                    dependency_ce_delta=dep_delta,
                    nondependency_ce_delta=non_delta,
                    dependency_selectivity_difference=dep_delta - non_delta,
                )
            )
            for lag, lag_mask in dependency_lag_masks.items():
                selected = lag_mask[document_index]
                count = int(selected.sum().item())
                if count == 0:
                    continue
                self._lag_values.setdefault(lag, []).append(
                    _LagDocumentValue(
                        example_id=example_id,
                        normal_sum=_sum_float64(normal_nll[document_index][selected]),
                        ablated_sum=_sum_float64(ablated_nll[document_index][selected]),
                        tokens=count,
                    )
                )
        self._seen_ids.update(example_ids)

    def compute(
        self,
        *,
        bootstrap_samples: int,
        confidence_level: float,
        seed: int,
    ) -> CausalDocumentEvaluation:
        if not self._effects:
            raise RuntimeError("cannot compute document causal evaluation: zero documents observed.")
        inference = DocumentInference(
            dependency_ce_delta=bootstrap_mean(
                [effect.dependency_ce_delta for effect in self._effects],
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed,
            ),
            nondependency_ce_delta=bootstrap_mean(
                [effect.nondependency_ce_delta for effect in self._effects],
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + 1,
            ),
            dependency_selectivity_difference=bootstrap_mean(
                [effect.dependency_selectivity_difference for effect in self._effects],
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + 2,
            ),
            document_effects=tuple(self._effects),
        )
        lag_effects: list[LagCausalEffect] = []
        for lag, values in sorted(self._lag_values.items()):
            tokens = sum(value.tokens for value in values)
            normal_sum = sum(value.normal_sum for value in values)
            ablated_sum = sum(value.ablated_sum for value in values)
            lag_effects.append(
                LagCausalEffect(
                    lag_blocks=lag,
                    dependency_tokens=tokens,
                    normal_dependency_ce=normal_sum / tokens,
                    ablated_dependency_ce=ablated_sum / tokens,
                    dependency_ce_delta=bootstrap_mean(
                        [value.delta for value in values],
                        samples=bootstrap_samples,
                        confidence_level=confidence_level,
                        seed=seed + 1000 + lag,
                    ),
                )
            )
        return CausalDocumentEvaluation(
            aggregate=self._aggregate.compute(),
            document_inference=inference,
            lag_effects=tuple(lag_effects),
        )


class PairedCausalEvaluator:
    """Aggregate all bus interventions over the same complete documents."""

    def __init__(self) -> None:
        self._gate_zero = CausalDocumentEvaluator()
        self._norm_scramble = CausalDocumentEvaluator()
        self._mutation_kl_sum = 0.0
        self._mutation_count = 0
        self._mutation_values: list[MutationDocumentValue] = []
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
        dependency_lag_masks: Mapping[int, torch.Tensor],
        example_ids: Sequence[str],
    ) -> None:
        """Validate all conditions before changing any accumulator state."""

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
            dependency_lag_masks=dependency_lag_masks,
            example_ids=example_ids,
        )
        self._gate_zero._validate_new_ids(example_ids)
        self._norm_scramble._validate_new_ids(example_ids)
        mutation_sum, mutation_count, mutation_values = _targeted_mutation_batch_values(
            baseline_logits=baseline_logits,
            mutation_logits=mutation_logits,
            mutation_dependency_mask=mutation_dependency_mask,
            example_ids=example_ids,
        )
        for evaluator, ablated_logits in (
            (self._gate_zero, gate_zero_logits),
            (self._norm_scramble, norm_scramble_logits),
        ):
            evaluator._update_validated(
                normal_logits=baseline_logits,
                ablated_logits=ablated_logits,
                labels=labels,
                label_mask=label_mask,
                dependency_mask=dependency_mask,
                nondependency_mask=nondependency_mask,
                dependency_lag_masks=dependency_lag_masks,
                example_ids=example_ids,
            )
        self._mutation_kl_sum += mutation_sum
        self._mutation_count += mutation_count
        self._mutation_values.extend(mutation_values)
        self._batches += 1

    def compute(
        self,
        *,
        bootstrap_samples: int,
        confidence_level: float,
        seed: int,
        minimum_documents: int,
    ) -> PairedCausalEvaluation:
        if self._batches == 0:
            raise RuntimeError("cannot compute paired causal evaluation: zero batches observed.")
        if type(minimum_documents) is not int or minimum_documents <= 1:
            raise ValueError("minimum_documents must be an integer greater than one.")
        gate_zero = self._gate_zero.compute(
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        norm_scramble = self._norm_scramble.compute(
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 10_000,
        )
        if self._mutation_count == 0 or not self._mutation_values:
            raise RuntimeError("cannot compute targeted mutation metrics without documents.")
        mutation_bootstrap = bootstrap_mean(
            [value.baseline_to_mutation_kl for value in self._mutation_values],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 20_000,
        )
        mutation = TargetedMutationMetrics(
            mutation_dependency_tokens=self._mutation_count,
            baseline_to_mutation_kl=self._mutation_kl_sum / self._mutation_count,
            document_bootstrap=mutation_bootstrap,
            document_values=tuple(self._mutation_values),
        )
        dependency_ci = gate_zero.document_inference.dependency_ce_delta
        selectivity_ci = gate_zero.document_inference.dependency_selectivity_difference
        documents = len(gate_zero.document_inference.document_effects)
        enough_documents = documents >= minimum_documents
        dependency_positive = dependency_ci.lower > 0.0
        selectivity_positive = selectivity_ci.lower > 0.0
        mutation_positive = mutation_bootstrap.lower > 0.0
        evidence_gate = CausalEvidenceGate(
            documents=documents,
            minimum_documents=minimum_documents,
            enough_documents=enough_documents,
            dependency_ci_lower_positive=dependency_positive,
            selectivity_ci_lower_positive=selectivity_positive,
            mutation_ci_lower_positive=mutation_positive,
            passes=(
                enough_documents
                and dependency_positive
                and selectivity_positive
                and mutation_positive
            ),
        )
        return PairedCausalEvaluation(
            batches=self._batches,
            documents=documents,
            gate_zero=gate_zero,
            norm_scramble=norm_scramble,
            targeted_mutation=mutation,
            evidence_gate=evidence_gate,
        )


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> BootstrapMean:
    """Return a deterministic percentile interval over independent documents."""

    if not values:
        raise ValueError("bootstrap values must be non-empty.")
    if type(samples) is not int or samples < 1000:
        raise ValueError("bootstrap samples must be an integer of at least 1000.")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1).")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer.")
    tensor = torch.tensor(tuple(values), dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("bootstrap values must all be finite.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=tensor.numel(),
        size=(samples, tensor.numel()),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    alpha = (1.0 - confidence_level) / 2.0
    bounds = torch.quantile(means, torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64))
    return BootstrapMean(
        mean=float(tensor.mean().item()),
        lower=float(bounds[0].item()),
        upper=float(bounds[1].item()),
        confidence_level=confidence_level,
        documents=tensor.numel(),
        bootstrap_samples=samples,
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
    dependency_lag_masks: Mapping[int, torch.Tensor],
    example_ids: Sequence[str],
) -> None:
    _validate_causal_pair(
        normal_logits=baseline_logits,
        ablated_logits=gate_zero_logits,
        labels=labels,
        label_mask=label_mask,
        dependency_mask=dependency_mask,
        nondependency_mask=nondependency_mask,
        dependency_lag_masks=dependency_lag_masks,
        example_ids=example_ids,
    )
    expected_shape = baseline_logits.shape
    for name, tensor in (
        ("norm_scramble_logits", norm_scramble_logits),
        ("mutation_logits", mutation_logits),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}.")
        if tensor.device != baseline_logits.device:
            raise ValueError(f"{name} must be on {baseline_logits.device}, got {tensor.device}.")
        if tensor.dtype != baseline_logits.dtype:
            raise TypeError(f"{name} must have dtype {baseline_logits.dtype}, got {tensor.dtype}.")
        annotated_logits = tensor[dependency_mask | nondependency_mask]
        if annotated_logits.numel() and not bool(torch.isfinite(annotated_logits).all()):
            raise ValueError(f"{name} must be finite on every annotated token.")
    if mutation_dependency_mask.shape != labels.shape:
        raise ValueError(
            "mutation_dependency_mask must have shape "
            f"{tuple(labels.shape)}, got {tuple(mutation_dependency_mask.shape)}."
        )
    if mutation_dependency_mask.device != baseline_logits.device:
        raise ValueError("mutation_dependency_mask must be on the logit device.")
    if mutation_dependency_mask.dtype != torch.bool:
        raise TypeError(
            "mutation_dependency_mask must have dtype torch.bool, "
            f"got {mutation_dependency_mask.dtype}."
        )
    if bool((mutation_dependency_mask & ~dependency_mask).any()):
        raise ValueError("mutation_dependency_mask must be a subset of dependency_mask.")
    if not bool(mutation_dependency_mask.any()):
        raise ValueError("mutation_dependency_mask must contain at least one token.")
    for document_index, example_id in enumerate(example_ids):
        if not bool(mutation_dependency_mask[document_index].any()):
            raise ValueError(
                "every document must contain future dependency tokens targeted by the "
                f"configured mutation; example_id={example_id!r}."
            )


def _validate_causal_pair(
    *,
    normal_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    dependency_mask: torch.Tensor,
    nondependency_mask: torch.Tensor,
    dependency_lag_masks: Mapping[int, torch.Tensor],
    example_ids: Sequence[str],
) -> None:
    if normal_logits.ndim < 3:
        raise ValueError("logits must have document, token, and vocabulary axes.")
    if normal_logits.size(0) == 0 or normal_logits.size(-1) <= 1:
        raise ValueError("causal evaluation needs documents and a vocabulary wider than one.")
    if not normal_logits.is_floating_point():
        raise TypeError("normal_logits must have a floating-point dtype.")
    if ablated_logits.shape != normal_logits.shape:
        raise ValueError(
            "ablated_logits must have shape "
            f"{tuple(normal_logits.shape)}, got {tuple(ablated_logits.shape)}."
        )
    if ablated_logits.device != normal_logits.device:
        raise ValueError("ablated_logits must be on the normal-logit device.")
    if ablated_logits.dtype != normal_logits.dtype:
        raise TypeError("ablated_logits must have the normal-logit dtype.")
    token_shape = normal_logits.shape[:-1]
    for name, tensor in (
        ("labels", labels),
        ("label_mask", label_mask),
        ("dependency_mask", dependency_mask),
        ("nondependency_mask", nondependency_mask),
    ):
        if tensor.shape != token_shape:
            raise ValueError(f"{name} must have shape {tuple(token_shape)}, got {tuple(tensor.shape)}.")
        if tensor.device != normal_logits.device:
            raise ValueError(f"{name} must be on {normal_logits.device}, got {tensor.device}.")
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError("labels must have an integer dtype other than bool.")
    for name, mask in (
        ("label_mask", label_mask),
        ("dependency_mask", dependency_mask),
        ("nondependency_mask", nondependency_mask),
    ):
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}.")
    if not isinstance(example_ids, Sequence) or isinstance(example_ids, (str, bytes)):
        raise TypeError("example_ids must be a sequence of document identifiers.")
    if len(example_ids) != normal_logits.size(0):
        raise ValueError(
            f"example_ids has {len(example_ids)} values for {normal_logits.size(0)} documents."
        )
    if any(not isinstance(value, str) or not value for value in example_ids):
        raise ValueError("every causal evaluation example_id must be a non-empty string.")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("causal evaluation example_ids must be unique within a batch.")
    if bool((dependency_mask & nondependency_mask).any()):
        raise ValueError("dependency_mask and nondependency_mask must be disjoint.")
    annotated = dependency_mask | nondependency_mask
    if bool((annotated & ~label_mask).any()):
        raise ValueError("annotated span tokens must be included in label_mask.")
    if not bool(annotated.any()):
        raise ValueError("each causal batch must contain annotated tokens.")
    for document_index, example_id in enumerate(example_ids):
        if not bool(dependency_mask[document_index].any()) or not bool(
            nondependency_mask[document_index].any()
        ):
            raise ValueError(
                "every causal-evaluation document must contain dependency and "
                f"nondependency tokens; example_id={example_id!r}."
            )
    if not dependency_lag_masks:
        raise ValueError("dependency_lag_masks must contain at least one annotated lag.")
    lag_union = torch.zeros_like(dependency_mask)
    for lag, mask in dependency_lag_masks.items():
        if type(lag) is not int or lag <= 0:
            raise ValueError(f"dependency lag keys must be positive integers, got {lag!r}.")
        if mask.shape != token_shape or mask.device != normal_logits.device:
            raise ValueError(f"dependency lag {lag} mask must match token shape and device.")
        if mask.dtype != torch.bool:
            raise TypeError(f"dependency lag {lag} mask must have dtype torch.bool.")
        if bool((lag_union & mask).any()):
            raise ValueError("dependency lag masks must be disjoint.")
        lag_union |= mask
    if not torch.equal(lag_union, dependency_mask):
        raise ValueError("dependency lag masks must form an exact partition of dependency_mask.")
    active_labels = labels[label_mask]
    if active_labels.numel() and (
        bool((active_labels < 0).any()) or bool((active_labels >= normal_logits.size(-1)).any())
    ):
        raise ValueError(f"active labels must be in [0, {normal_logits.size(-1)}).")
    for name, tensor in (("normal_logits", normal_logits), ("ablated_logits", ablated_logits)):
        selected = tensor[annotated]
        if selected.numel() and not bool(torch.isfinite(selected).all()):
            raise ValueError(f"{name} must be finite on every annotated token.")


def _targeted_mutation_batch_values(
    *,
    baseline_logits: torch.Tensor,
    mutation_logits: torch.Tensor,
    mutation_dependency_mask: torch.Tensor,
    example_ids: Sequence[str],
) -> tuple[float, int, tuple[MutationDocumentValue, ...]]:
    baseline_log_probs = F.log_softmax(baseline_logits.float(), dim=-1)
    mutation_log_probs = F.log_softmax(mutation_logits.float(), dim=-1)
    kl = (baseline_log_probs.exp() * (baseline_log_probs - mutation_log_probs)).sum(dim=-1)
    selected = kl[mutation_dependency_mask]
    values = tuple(
        MutationDocumentValue(
            example_id=example_id,
            baseline_to_mutation_kl=_mean_float64(
                kl[document_index][mutation_dependency_mask[document_index]]
            ),
        )
        for document_index, example_id in enumerate(example_ids)
    )
    return _sum_float64(selected), int(selected.numel()), values


def _token_values(
    *,
    normal_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_labels = labels.long().clone()
    safe_labels[~label_mask] = 0
    normal_log_probs = F.log_softmax(normal_logits.float(), dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits.float(), dim=-1)
    gather_index = safe_labels.unsqueeze(-1)
    normal_nll = -normal_log_probs.gather(-1, gather_index).squeeze(-1)
    ablated_nll = -ablated_log_probs.gather(-1, gather_index).squeeze(-1)
    mutation_kl = (
        normal_log_probs.exp() * (normal_log_probs - ablated_log_probs)
    ).sum(dim=-1)
    return normal_nll, ablated_nll, mutation_kl


def _sum_float64(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(device="cpu", dtype=torch.float64).sum().item())


def _mean_float64(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        raise ValueError("cannot average an empty causal token selection.")
    return float(tensor.detach().to(device="cpu", dtype=torch.float64).mean().item())
