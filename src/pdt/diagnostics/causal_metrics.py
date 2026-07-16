"""Token-weighted metrics for PDT causal interventions.

The paper-level quantities are defined over tokens, not over batches or
examples.  This module therefore stores loss sums and token counts and only
forms means when ``compute`` is called.  It is safe to update with batches of
different sizes and dependency-span lengths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


__all__ = [
    "CausalAblationAccumulator",
    "CausalAblationMetrics",
    "note_bandwidth_bytes",
]


@dataclass(frozen=True, slots=True)
class CausalAblationMetrics:
    """Final token-weighted metrics for one paired intervention."""

    dependency_tokens: int
    nondependency_tokens: int
    normal_dependency_ce: float
    ablated_dependency_ce: float
    dependency_ce_delta: float
    normal_nondependency_ce: float
    ablated_nondependency_ce: float
    nondependency_ce_delta: float
    dependency_selectivity_ratio: float
    dependency_mutation_kl: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CausalAblationAccumulator:
    """Accumulate paired normal/ablated loss and mutation statistics.

    ``dependency_mask`` and ``nondependency_mask`` must be disjoint.  Tokens
    outside both masks are permitted (for example formatting tokens), but
    annotated tokens must be included in ``label_mask``.
    """

    def __init__(self) -> None:
        self._dep_normal_sum = 0.0
        self._dep_ablated_sum = 0.0
        self._non_normal_sum = 0.0
        self._non_ablated_sum = 0.0
        self._dep_kl_sum = 0.0
        self._dep_count = 0
        self._non_count = 0

    @torch.no_grad()
    def update_from_logits(
        self,
        *,
        normal_logits: torch.Tensor,
        ablated_logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        dependency_mask: torch.Tensor,
        nondependency_mask: torch.Tensor,
    ) -> None:
        """Add a paired batch of logits with shapes ``(..., vocab)``."""

        if normal_logits.shape != ablated_logits.shape:
            raise ValueError(
                "normal_logits and ablated_logits must have identical shapes, "
                f"got {tuple(normal_logits.shape)} and {tuple(ablated_logits.shape)}."
            )
        if normal_logits.ndim < 2:
            raise ValueError("logits must have at least one token axis and one vocabulary axis.")
        token_shape = normal_logits.shape[:-1]
        masks = {
            "labels": labels,
            "label_mask": label_mask,
            "dependency_mask": dependency_mask,
            "nondependency_mask": nondependency_mask,
        }
        for name, tensor in masks.items():
            if tensor.shape != token_shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match token shape "
                    f"{tuple(token_shape)}."
                )

        active = label_mask.to(device=labels.device, dtype=torch.bool)
        active_labels = labels[active]
        vocab_size = normal_logits.size(-1)
        if active_labels.numel() and (
            bool((active_labels < 0).any()) or bool((active_labels >= vocab_size).any())
        ):
            raise ValueError(f"active labels must be in [0, {vocab_size}).")

        safe_labels = labels.to(device=normal_logits.device, dtype=torch.long).clone()
        safe_labels[~active.to(device=normal_logits.device)] = 0
        normal_log_probs = F.log_softmax(normal_logits.float(), dim=-1)
        ablated_log_probs = F.log_softmax(ablated_logits.float(), dim=-1)
        gather_index = safe_labels.unsqueeze(-1)
        normal_nll = -normal_log_probs.gather(-1, gather_index).squeeze(-1)
        ablated_nll = -ablated_log_probs.gather(-1, gather_index).squeeze(-1)
        normal_probs = normal_log_probs.exp()
        mutation_kl = (normal_probs * (normal_log_probs - ablated_log_probs)).sum(dim=-1)
        self.update_token_values(
            normal_nll=normal_nll,
            ablated_nll=ablated_nll,
            dependency_mutation_kl=mutation_kl,
            label_mask=active.to(device=normal_nll.device),
            dependency_mask=dependency_mask.to(device=normal_nll.device),
            nondependency_mask=nondependency_mask.to(device=normal_nll.device),
        )

    def update_token_values(
        self,
        *,
        normal_nll: torch.Tensor,
        ablated_nll: torch.Tensor,
        label_mask: torch.Tensor,
        dependency_mask: torch.Tensor,
        nondependency_mask: torch.Tensor,
        dependency_mutation_kl: torch.Tensor | None = None,
    ) -> None:
        """Add precomputed per-token values.

        This is also the canonical aggregation path for offline validators
        that already computed target-token cross entropy.
        """

        expected = normal_nll.shape
        tensors = {
            "ablated_nll": ablated_nll,
            "label_mask": label_mask,
            "dependency_mask": dependency_mask,
            "nondependency_mask": nondependency_mask,
        }
        if dependency_mutation_kl is not None:
            tensors["dependency_mutation_kl"] = dependency_mutation_kl
        for name, tensor in tensors.items():
            if tensor.shape != expected:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match normal_nll "
                    f"shape {tuple(expected)}."
                )

        active = label_mask.to(dtype=torch.bool, device=normal_nll.device)
        dep = dependency_mask.to(dtype=torch.bool, device=normal_nll.device)
        non = nondependency_mask.to(dtype=torch.bool, device=normal_nll.device)
        if bool((dep & non).any()):
            raise ValueError("dependency and nondependency masks must be disjoint.")
        if bool(((dep | non) & ~active).any()):
            raise ValueError("annotated span tokens must be included in label_mask.")
        dep &= active
        non &= active

        values = [normal_nll, ablated_nll]
        if dependency_mutation_kl is not None:
            values.append(dependency_mutation_kl)
        for value in values:
            selected = value[dep | non]
            if selected.numel() and not bool(torch.isfinite(selected).all()):
                raise ValueError("causal metric inputs must be finite on annotated tokens.")

        self._dep_normal_sum += _sum_float64(normal_nll[dep])
        self._dep_ablated_sum += _sum_float64(ablated_nll[dep])
        self._non_normal_sum += _sum_float64(normal_nll[non])
        self._non_ablated_sum += _sum_float64(ablated_nll[non])
        if dependency_mutation_kl is not None:
            self._dep_kl_sum += _sum_float64(dependency_mutation_kl[dep])
        self._dep_count += int(dep.sum().item())
        self._non_count += int(non.sum().item())

    def compute(self, *, epsilon: float = 1e-8) -> CausalAblationMetrics:
        if self._dep_count == 0:
            raise RuntimeError("cannot compute causal metrics: zero dependency tokens observed.")
        if self._non_count == 0:
            raise RuntimeError("cannot compute causal metrics: zero nondependency tokens observed.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")

        normal_dep = self._dep_normal_sum / self._dep_count
        ablated_dep = self._dep_ablated_sum / self._dep_count
        normal_non = self._non_normal_sum / self._non_count
        ablated_non = self._non_ablated_sum / self._non_count
        dep_delta = ablated_dep - normal_dep
        non_delta = ablated_non - normal_non
        return CausalAblationMetrics(
            dependency_tokens=self._dep_count,
            nondependency_tokens=self._non_count,
            normal_dependency_ce=normal_dep,
            ablated_dependency_ce=ablated_dep,
            dependency_ce_delta=dep_delta,
            normal_nondependency_ce=normal_non,
            ablated_nondependency_ce=ablated_non,
            nondependency_ce_delta=non_delta,
            dependency_selectivity_ratio=dep_delta / max(non_delta, epsilon),
            dependency_mutation_kl=self._dep_kl_sum / self._dep_count,
        )


def note_bandwidth_bytes(
    *,
    num_streams: int,
    notes_dim: int,
    bytes_per_element: int,
    num_blocks: int,
) -> int:
    """Return bytes written to the notes bus across synchronization blocks."""

    values = {
        "num_streams": num_streams,
        "notes_dim": notes_dim,
        "bytes_per_element": bytes_per_element,
        "num_blocks": num_blocks,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return num_streams * notes_dim * bytes_per_element * num_blocks


def _sum_float64(tensor: torch.Tensor) -> float:
    return float(tensor.detach().to(device="cpu", dtype=torch.float64).sum().item())
