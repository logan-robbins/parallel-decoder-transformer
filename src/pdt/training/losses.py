"""Losses for unordered plans and exact per-lane semantic supervision."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pdt.config.schemas import LossWeights


__all__ = [
    "LossBundle",
    "PermutationMatch",
    "compute_pdt_losses",
    "match_unordered_plans",
]


@dataclass(slots=True)
class PermutationMatch:
    """Predicted plan tensors reordered onto teacher physical-lane axes."""

    permutation: torch.Tensor
    nodes: torch.Tensor
    validity_logits: torch.Tensor
    presentation_order_logits: torch.Tensor
    semantic_loss: torch.Tensor


@dataclass(slots=True)
class LossBundle:
    total: torch.Tensor
    lm_ce: torch.Tensor
    plan_semantic: torch.Tensor
    fact_route: torch.Tensor
    outline_progress: torch.Tensor
    fact_write: torch.Tensor
    note_align: torch.Tensor
    presentation_order: torch.Tensor
    dynamic_vq_commit: torch.Tensor
    dynamic_vq_codebook: torch.Tensor
    dynamic_codebook_usage: torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            field: float(getattr(self, field).detach().float().item())
            for field in (
                "total",
                "lm_ce",
                "plan_semantic",
                "fact_route",
                "outline_progress",
                "fact_write",
                "note_align",
                "presentation_order",
                "dynamic_vq_commit",
                "dynamic_vq_codebook",
                "dynamic_codebook_usage",
            )
        }


def match_unordered_plans(
    *,
    predicted_nodes: torch.Tensor,
    validity_logits: torch.Tensor,
    presentation_order_logits: torch.Tensor,
    teacher_nodes: torch.Tensor,
    teacher_node_mask: torch.Tensor,
) -> PermutationMatch:
    """Enumerate all six lane permutations and choose minimum semantic cost."""

    if predicted_nodes.dim() != 4 or predicted_nodes.size(1) != 3:
        raise ValueError("predicted_nodes must have shape [B, 3, N, P].")
    if teacher_nodes.shape != predicted_nodes.shape:
        raise ValueError("teacher_nodes must exactly match predicted_nodes shape.")
    expected_mask_shape = predicted_nodes.shape[:-1]
    if validity_logits.shape != expected_mask_shape:
        raise ValueError("validity_logits must have shape [B, 3, N].")
    if teacher_node_mask.shape != expected_mask_shape:
        raise ValueError("teacher_node_mask must have shape [B, 3, N].")
    if presentation_order_logits.shape != predicted_nodes.shape[:2]:
        raise ValueError("presentation_order_logits must have shape [B, 3].")
    if bool((~teacher_node_mask.to(dtype=torch.bool).any(dim=-1)).any()):
        raise ValueError("Every teacher plan must contain at least one valid node.")

    permutations = tuple(itertools.permutations(range(3)))
    batch = predicted_nodes.size(0)
    detached_costs: list[torch.Tensor] = []
    for permutation in permutations:
        reordered_nodes = predicted_nodes[:, permutation]
        reordered_validity = validity_logits[:, permutation]
        cost = _per_example_plan_loss(
            reordered_nodes,
            reordered_validity,
            teacher_nodes,
            teacher_node_mask,
        )
        detached_costs.append(cost.detach())
    cost_matrix = torch.stack(detached_costs, dim=1)
    selected = cost_matrix.argmin(dim=1)
    permutation_tensor = torch.tensor(
        permutations,
        dtype=torch.long,
        device=predicted_nodes.device,
    ).index_select(0, selected)
    lane_index = permutation_tensor[:, :, None, None].expand_as(predicted_nodes)
    matched_nodes = predicted_nodes.gather(1, lane_index)
    validity_index = permutation_tensor[:, :, None].expand_as(validity_logits)
    matched_validity = validity_logits.gather(1, validity_index)
    matched_order = presentation_order_logits.gather(1, permutation_tensor)
    semantic_loss = _per_example_plan_loss(
        matched_nodes,
        matched_validity,
        teacher_nodes,
        teacher_node_mask,
    ).mean()
    if permutation_tensor.shape != (batch, 3):
        raise RuntimeError("Plan permutation matching returned the wrong shape.")
    return PermutationMatch(
        permutation=permutation_tensor,
        nodes=matched_nodes,
        validity_logits=matched_validity,
        presentation_order_logits=matched_order,
        semantic_loss=semantic_loss,
    )


def compute_pdt_losses(
    *,
    weights: LossWeights,
    lm_ce: Optional[torch.Tensor],
    plan_semantic: Optional[torch.Tensor],
    fact_route_logits: Optional[torch.Tensor],
    fact_route_targets: Optional[torch.Tensor],
    plan_node_mask: Optional[torch.Tensor],
    fact_mask: Optional[torch.Tensor],
    outline_progress_logits: Optional[torch.Tensor],
    outline_progress_targets: Optional[torch.Tensor],
    fact_write_logits: Optional[torch.Tensor],
    fact_write_targets: Optional[torch.Tensor],
    note_queries: Optional[torch.Tensor],
    note_keys: Optional[torch.Tensor],
    presentation_order_logits: Optional[torch.Tensor],
    presentation_rank_targets: Optional[torch.Tensor],
    dynamic_vq_commitment_loss: Optional[torch.Tensor],
    dynamic_vq_codebook_loss: Optional[torch.Tensor],
    dynamic_vq_logits: Optional[torch.Tensor],
) -> LossBundle:
    """Assemble every active real-data objective and reject missing supervision."""

    candidates = (
        lm_ce,
        plan_semantic,
        fact_route_logits,
        outline_progress_logits,
        fact_write_logits,
        note_queries,
        presentation_order_logits,
        dynamic_vq_commitment_loss,
        dynamic_vq_codebook_loss,
        dynamic_vq_logits,
    )
    zero = _zero_like(*candidates)
    loss_lm = _required_scalar("lm_ce", weights.lm_ce, lm_ce, zero)
    loss_plan = _required_scalar(
        "plan_semantic",
        weights.plan_semantic,
        plan_semantic,
        zero,
    )
    loss_route = _fact_route_loss(
        weight=weights.fact_route,
        logits=fact_route_logits,
        targets=fact_route_targets,
        plan_node_mask=plan_node_mask,
        fact_mask=fact_mask,
        zero=zero,
    )
    loss_progress = _cross_entropy_loss(
        name="outline_progress",
        weight=weights.outline_progress,
        logits=outline_progress_logits,
        targets=outline_progress_targets,
        zero=zero,
    )
    loss_write = _fact_write_loss(
        weight=weights.fact_write,
        logits=fact_write_logits,
        targets=fact_write_targets,
        zero=zero,
    )
    loss_note = _note_alignment_loss(
        weight=weights.note_align,
        queries=note_queries,
        keys=note_keys,
        zero=zero,
    )
    loss_order = _presentation_order_loss(
        weight=weights.presentation_order,
        logits=presentation_order_logits,
        rank_targets=presentation_rank_targets,
        zero=zero,
    )
    loss_dynamic_commit = _required_scalar(
        "dynamic_vq_commit",
        weights.dynamic_vq_commit,
        dynamic_vq_commitment_loss,
        zero,
    )
    loss_dynamic_codebook = _required_scalar(
        "dynamic_vq_codebook",
        weights.dynamic_vq_codebook,
        dynamic_vq_codebook_loss,
        zero,
    )
    if weights.dynamic_codebook_usage > 0:
        if dynamic_vq_logits is None:
            raise ValueError(
                "Positive dynamic_codebook_usage requires dynamic VQ assignment logits."
            )
        loss_dynamic_usage = _entropy_deficit(dynamic_vq_logits)
    else:
        loss_dynamic_usage = zero

    total = (
        weights.lm_ce * loss_lm
        + weights.plan_semantic * loss_plan
        + weights.fact_route * loss_route
        + weights.outline_progress * loss_progress
        + weights.fact_write * loss_write
        + weights.note_align * loss_note
        + weights.presentation_order * loss_order
        + weights.dynamic_vq_commit * loss_dynamic_commit
        + weights.dynamic_vq_codebook * loss_dynamic_codebook
        + weights.dynamic_codebook_usage * loss_dynamic_usage
    )
    return LossBundle(
        total=total,
        lm_ce=loss_lm,
        plan_semantic=loss_plan,
        fact_route=loss_route,
        outline_progress=loss_progress,
        fact_write=loss_write,
        note_align=loss_note,
        presentation_order=loss_order,
        dynamic_vq_commit=loss_dynamic_commit,
        dynamic_vq_codebook=loss_dynamic_codebook,
        dynamic_codebook_usage=loss_dynamic_usage,
    )


def _per_example_plan_loss(
    predicted: torch.Tensor,
    validity_logits: torch.Tensor,
    teacher: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    mask = teacher_mask.to(device=predicted.device, dtype=torch.bool)
    predicted_normalized = F.normalize(predicted.float(), dim=-1)
    teacher_normalized = F.normalize(teacher.float(), dim=-1)
    cosine_distance = 1.0 - (predicted_normalized * teacher_normalized).sum(dim=-1)
    semantic = (cosine_distance * mask).sum(dim=(1, 2)) / mask.sum(
        dim=(1, 2)
    ).clamp_min(1)
    validity = F.binary_cross_entropy_with_logits(
        validity_logits.float(),
        mask.to(dtype=torch.float32),
        reduction="none",
    ).mean(dim=(1, 2))
    return semantic + validity


def _fact_route_loss(
    *,
    weight: float,
    logits: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    plan_node_mask: Optional[torch.Tensor],
    fact_mask: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight <= 0:
        return zero
    if logits is None or targets is None or plan_node_mask is None or fact_mask is None:
        raise ValueError(
            "Positive fact_route requires logits, targets, plan_node_mask, and fact_mask."
        )
    if logits.shape != targets.shape:
        raise ValueError("fact_route logits and targets must have identical shapes.")
    if plan_node_mask.shape != logits.shape[:-1]:
        raise ValueError("plan_node_mask must align with fact_route plan nodes.")
    if fact_mask.shape != (logits.size(0), logits.size(-1)):
        raise ValueError("fact_mask must align with fact_route batch/fact axes.")
    active = plan_node_mask.to(dtype=torch.bool).unsqueeze(-1) & fact_mask[:, None, None, :]
    if not bool(active.any()):
        raise ValueError("fact_route supervision contains no active node/fact pair.")
    active_logits = logits.float()[active]
    active_targets = targets.to(
        device=logits.device,
        dtype=torch.float32,
    )[active]
    positive = active_targets == 1
    negative = active_targets == 0
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError(
            "fact_route requires both owned and non-owned node/fact pairs."
        )
    positive_loss = F.binary_cross_entropy_with_logits(
        active_logits[positive],
        active_targets[positive],
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        active_logits[negative],
        active_targets[negative],
    )
    return 0.5 * (positive_loss + negative_loss)


def _fact_write_loss(
    *,
    weight: float,
    logits: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight <= 0:
        return zero
    if logits is None or targets is None:
        raise ValueError("Positive fact_write requires aligned logits and targets.")
    if logits.shape[:-1] != targets.shape or logits.size(-1) != 3:
        raise ValueError(
            "fact_write targets must match logits [B, K, M, F, 3]."
        )
    active = targets != -100
    active_targets = targets.long()[active]
    active_logits = logits.float()[active]
    counts = torch.bincount(active_targets, minlength=3)
    if bool((counts == 0).any()):
        raise ValueError(
            "fact_write supervision must contain OWNER, REFERENCE, and ABSENT roles."
        )
    role_weights = active_targets.numel() / (3.0 * counts.float())
    return F.cross_entropy(
        active_logits,
        active_targets,
        weight=role_weights.to(device=active_logits.device),
    )


def _cross_entropy_loss(
    *,
    name: str,
    weight: float,
    logits: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight <= 0:
        return zero
    if logits is None or targets is None:
        raise ValueError(f"Positive {name} requires aligned logits and targets.")
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"{name} targets must match logits excluding class axis; "
            f"got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}."
        )
    active = targets != -100
    if not bool(active.any()):
        raise ValueError(f"{name} contains no active targets.")
    return F.cross_entropy(logits.float()[active], targets.long()[active])


def _note_alignment_loss(
    *,
    weight: float,
    queries: Optional[torch.Tensor],
    keys: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight <= 0:
        return zero
    if queries is None or keys is None:
        raise ValueError("Positive note_align requires note queries and keys.")
    if queries.shape != keys.shape or queries.dim() != 2:
        raise ValueError("note queries and keys must share shape [writes, notes_dim].")
    if queries.size(0) < 2:
        raise ValueError("note_align requires at least two writes for contrastive negatives.")
    queries = F.normalize(queries.float(), dim=-1)
    keys = F.normalize(keys.float(), dim=-1)
    logits = queries @ keys.t() / 0.07
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def _presentation_order_loss(
    *,
    weight: float,
    logits: Optional[torch.Tensor],
    rank_targets: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight <= 0:
        return zero
    if logits is None or rank_targets is None:
        raise ValueError(
            "Positive presentation_order requires lane scores and rank targets."
        )
    if logits.shape != rank_targets.shape or logits.shape[-1] != 3:
        raise ValueError("Presentation order tensors must share shape [B, 3].")
    if any(
        sorted(row.tolist()) != [0, 1, 2]
        for row in rank_targets.detach().cpu()
    ):
        raise ValueError("Every presentation rank row must be a permutation of 0,1,2.")
    order = rank_targets.argsort(dim=-1)
    ordered_logits = logits.gather(1, order).float()
    first = -F.log_softmax(ordered_logits, dim=1)[:, 0]
    second = -F.log_softmax(ordered_logits[:, 1:], dim=1)[:, 0]
    return (first + second).mean()


def _required_scalar(
    name: str,
    weight: float,
    value: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight <= 0:
        return zero
    if value is None:
        raise ValueError(f"Positive {name} requires its aligned scalar loss.")
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be one finite scalar tensor.")
    return value


def _entropy_deficit(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() < 3:
        raise ValueError("Dynamic VQ logits must end in [codebooks, codes].")
    probabilities = logits.softmax(dim=-1)
    codebook_axis = probabilities.dim() - 2
    reduce_axes = tuple(
        axis
        for axis in range(probabilities.dim() - 1)
        if axis != codebook_axis
    )
    marginal = probabilities.mean(dim=reduce_axes)
    entropy = -(marginal * marginal.clamp_min(1e-8).log()).sum(dim=-1).mean()
    maximum = math.log(probabilities.size(-1))
    if maximum <= 0:
        raise ValueError("Dynamic VQ usage requires at least two codes.")
    return (maximum - entropy) / maximum


def _zero_like(*candidates: Optional[torch.Tensor]) -> torch.Tensor:
    for candidate in candidates:
        if candidate is not None:
            return candidate.new_zeros(())
    return torch.tensor(0.0)
