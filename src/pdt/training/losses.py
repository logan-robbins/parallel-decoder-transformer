"""Loss assembly after hash-era supervision removal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pdt.config.schemas import LossWeights


__all__ = ["LossBundle", "compute_pdt_losses"]


@dataclass(slots=True)
class LossBundle:
    total: torch.Tensor
    lm_ce: torch.Tensor
    lm_ce_dependency: torch.Tensor
    lm_ce_nondependency: torch.Tensor
    kd_lm: torch.Tensor
    vq_commit: torch.Tensor
    vq_codebook: torch.Tensor
    codebook_usage: torch.Tensor
    coverage: torch.Tensor
    readiness: torch.Tensor
    stream_classifier: torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total.item()),
            "lm_ce": float(self.lm_ce.item()),
            "lm_ce_dependency": float(self.lm_ce_dependency.item()),
            "lm_ce_nondependency": float(self.lm_ce_nondependency.item()),
            "kd_lm": float(self.kd_lm.item()),
            "vq_commit": float(self.vq_commit.item()),
            "vq_codebook": float(self.vq_codebook.item()),
            "codebook_usage": float(self.codebook_usage.item()),
            "coverage": float(self.coverage.item()),
            "readiness": float(self.readiness.item()),
            "stream_classifier": float(self.stream_classifier.item()),
        }


def compute_pdt_losses(
    *,
    stage: int,
    weights: LossWeights,
    lm_logits: Optional[torch.Tensor] = None,
    lm_labels: Optional[torch.Tensor] = None,
    lm_label_mask: Optional[torch.Tensor] = None,
    dependency_mask: Optional[torch.Tensor] = None,
    nondependency_mask: Optional[torch.Tensor] = None,
    lm_teacher_logits: Optional[torch.Tensor] = None,
    kd_temperature_lm: float = 0.5,
    vq_commitment_loss: Optional[torch.Tensor] = None,
    vq_codebook_loss: Optional[torch.Tensor] = None,
    planner_logits: Optional[torch.Tensor] = None,
    coverage_logits: Optional[torch.Tensor] = None,
    coverage_targets: Optional[torch.Tensor] = None,
    coverage_mask: Optional[torch.Tensor] = None,
    readiness_logits: Optional[torch.Tensor] = None,
    readiness_targets: Optional[torch.Tensor] = None,
    readiness_mask: Optional[torch.Tensor] = None,
    stream_logits: Optional[torch.Tensor] = None,
    stream_targets: Optional[torch.Tensor] = None,
) -> LossBundle:
    del stage
    zero = _zero_like(
        lm_logits,
        vq_commitment_loss,
        vq_codebook_loss,
        planner_logits,
        coverage_logits,
        readiness_logits,
        stream_logits,
    )
    loss_lm_ce = zero
    loss_lm_dep = zero
    loss_lm_non = zero
    loss_kd_lm = zero
    loss_vq_commit = vq_commitment_loss if vq_commitment_loss is not None else zero
    loss_vq_codebook = vq_codebook_loss if vq_codebook_loss is not None else zero
    loss_usage = zero
    loss_coverage = zero
    loss_readiness = zero
    loss_stream_classifier = zero

    per_token_ce = None
    mask = None
    if lm_logits is not None and lm_labels is not None:
        mask = lm_label_mask
        if mask is None:
            mask = lm_labels >= 0
        logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
        labels_flat = lm_labels.reshape(-1)
        mask_flat = mask.reshape(-1)
        if mask_flat.any():
            ce_flat = F.cross_entropy(
                logits_flat[mask_flat],
                labels_flat[mask_flat].long(),
                reduction="none",
            )
            loss_lm_ce = ce_flat.mean()
            per_token_ce = torch.zeros_like(labels_flat, dtype=lm_logits.dtype)
            per_token_ce[mask_flat] = ce_flat
            per_token_ce = per_token_ce.view_as(lm_labels)

            if lm_teacher_logits is not None and kd_temperature_lm > 0:
                t = kd_temperature_lm
                student_log = F.log_softmax(lm_logits / t, dim=-1)
                teacher = F.softmax(lm_teacher_logits / t, dim=-1)
                kd_per_tok = F.kl_div(student_log, teacher, reduction="none").sum(dim=-1)
                loss_kd_lm = (kd_per_tok * mask.float()).sum() / mask.float().sum().clamp(
                    min=1.0
                ) * (t * t)

    if per_token_ce is not None and mask is not None:
        if dependency_mask is not None:
            dep = dependency_mask.to(device=mask.device, dtype=torch.bool) & mask
            if dep.any():
                loss_lm_dep = per_token_ce[dep].mean()
        if nondependency_mask is not None:
            non = nondependency_mask.to(device=mask.device, dtype=torch.bool) & mask
            if non.any():
                loss_lm_non = per_token_ce[non].mean()

    if planner_logits is not None:
        probs = planner_logits.softmax(dim=-1).mean(dim=(0, 1))
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
        max_entropy = torch.log(torch.tensor(probs.numel(), device=probs.device, dtype=probs.dtype))
        loss_usage = (max_entropy - entropy) / max_entropy.clamp_min(1.0)

    if coverage_logits is not None and coverage_targets is not None:
        cmask = coverage_mask
        if cmask is None:
            cmask = torch.ones_like(coverage_logits, dtype=torch.bool)
        if cmask.any():
            loss_coverage = F.binary_cross_entropy_with_logits(
                coverage_logits[cmask], coverage_targets[cmask].to(coverage_logits)
            )

    if readiness_logits is not None and readiness_targets is not None:
        rmask = readiness_mask
        if rmask is None:
            rmask = torch.ones_like(readiness_logits, dtype=torch.bool)
        if rmask.any():
            loss_readiness = F.binary_cross_entropy(
                readiness_logits[rmask].clamp(1e-6, 1 - 1e-6),
                readiness_targets[rmask].to(readiness_logits),
            )

    if stream_logits is not None and stream_targets is not None:
        loss_stream_classifier = F.cross_entropy(stream_logits, stream_targets.long())

    total = (
        weights.lm_ce * loss_lm_ce
        + weights.kd_lm * loss_kd_lm
        + weights.vq_commit * loss_vq_commit
        + weights.vq_codebook * loss_vq_codebook
        + weights.codebook_usage * loss_usage
        + weights.coverage * loss_coverage
        + weights.readiness * loss_readiness
        + 0.1 * loss_stream_classifier
    )

    return LossBundle(
        total=total,
        lm_ce=loss_lm_ce,
        lm_ce_dependency=loss_lm_dep,
        lm_ce_nondependency=loss_lm_non,
        kd_lm=loss_kd_lm,
        vq_commit=loss_vq_commit,
        vq_codebook=loss_vq_codebook,
        codebook_usage=loss_usage,
        coverage=loss_coverage,
        readiness=loss_readiness,
        stream_classifier=loss_stream_classifier,
    )


def _zero_like(*candidates: Optional[torch.Tensor]) -> torch.Tensor:
    for tensor in candidates:
        if tensor is not None:
            return tensor.new_tensor(0.0)
    return torch.tensor(0.0)
