"""Loss assembly matching the paper equation.

    L_total = L_plan + L_notes + 0.5 * L_spec + L_LM-CE
              + lambda_KD * L_KD-LM + lambda_cov * L_cov + lambda_ready * L_ready

Stage gating:
    - Stage 0: planner CE + notes MSE + plan_notes_proj MSE (if available).
    - Stage 1: + speculation MSE (0.5 weight) + stream classifier CE.
    - Stage 2: + LM CE + LM KD (temperature-scaled).
    - Stage 3: + coverage BCE + readiness BCE.

Each term's contribution is gated by stage; weights come from
``LossWeights``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pdt.config.schemas import LossWeights


__all__ = ["LossBundle", "compute_pdt_losses"]


@dataclass(slots=True)
class LossBundle:
    """All loss terms and the weighted total.

    Terms may be zero if their stage has not activated yet; ``total`` is
    the only value that should be ``backward()``'d.
    """

    total: torch.Tensor
    planner: torch.Tensor
    notes: torch.Tensor
    spec: torch.Tensor
    lm_ce: torch.Tensor
    kd_lm: torch.Tensor
    coverage: torch.Tensor
    readiness: torch.Tensor
    stream_classifier: torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total.item()),
            "planner": float(self.planner.item()),
            "notes": float(self.notes.item()),
            "spec": float(self.spec.item()),
            "lm_ce": float(self.lm_ce.item()),
            "kd_lm": float(self.kd_lm.item()),
            "coverage": float(self.coverage.item()),
            "readiness": float(self.readiness.item()),
            "stream_classifier": float(self.stream_classifier.item()),
        }


def compute_pdt_losses(
    *,
    stage: int,
    weights: LossWeights,
    # Planner supervision
    planner_logits: Optional[torch.Tensor] = None,  # (B, S, V_p)
    planner_targets: Optional[torch.Tensor] = None,  # (B, S) long
    planner_teacher_logits: Optional[torch.Tensor] = None,  # (B, S, V_p)
    kd_temperature_planner: float = 1.0,
    # Note alignment
    student_notes: Optional[torch.Tensor] = None,  # (B, K, d_notes)
    teacher_notes: Optional[torch.Tensor] = None,  # (B, K, d_notes)
    # Speculation alignment
    student_spec: Optional[torch.Tensor] = None,  # (B, K, d_notes)
    teacher_spec: Optional[torch.Tensor] = None,  # (B, K, d_notes)
    # Language model supervision
    lm_logits: Optional[torch.Tensor] = None,  # (B, T, V)
    lm_labels: Optional[torch.Tensor] = None,  # (B, T) long
    lm_label_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    lm_teacher_logits: Optional[torch.Tensor] = None,  # (B, T, V)
    kd_temperature_lm: float = 0.5,
    # Coverage supervision
    coverage_logits: Optional[torch.Tensor] = None,  # (B, P)
    coverage_targets: Optional[torch.Tensor] = None,  # (B, P) in [0, 1]
    coverage_mask: Optional[torch.Tensor] = None,  # (B, P) bool
    # Readiness supervision
    readiness_logits: Optional[torch.Tensor] = None,  # (B,) in [0, 1]
    readiness_targets: Optional[torch.Tensor] = None,  # (B,) in {0, 1}
    readiness_mask: Optional[torch.Tensor] = None,  # (B,) bool
    # Stream-classifier supervision
    stream_logits: Optional[torch.Tensor] = None,  # (B, K)
    stream_targets: Optional[torch.Tensor] = None,  # (B,) long in [0, K)
) -> LossBundle:
    zero = torch.tensor(0.0)
    loss_planner = zero
    loss_notes = zero
    loss_spec = zero
    loss_lm_ce = zero
    loss_kd_lm = zero
    loss_coverage = zero
    loss_readiness = zero
    loss_stream_classifier = zero

    # -- Planner (always active from stage 0) --
    if planner_logits is not None and planner_targets is not None:
        B, S, V = planner_logits.shape
        ce = F.cross_entropy(
            planner_logits.reshape(B * S, V),
            planner_targets.reshape(B * S).long(),
            reduction="mean",
        )
        loss_planner = ce
        if planner_teacher_logits is not None and kd_temperature_planner > 0:
            t = kd_temperature_planner
            student_log = F.log_softmax(planner_logits / t, dim=-1)
            teacher = F.softmax(planner_teacher_logits / t, dim=-1)
            kd = F.kl_div(student_log, teacher, reduction="batchmean") * (t * t)
            loss_planner = loss_planner + kd

    # -- Notes MSE (stage 0+) --
    if student_notes is not None and teacher_notes is not None:
        loss_notes = F.mse_loss(student_notes, teacher_notes.to(student_notes))

    # -- Speculation MSE (stage 1+) --
    if stage >= 1 and student_spec is not None and teacher_spec is not None:
        loss_spec = F.mse_loss(student_spec, teacher_spec.to(student_spec))

    # -- LM CE + KD (stage 2+) --
    if stage >= 2 and lm_logits is not None and lm_labels is not None:
        mask = lm_label_mask
        if mask is None:
            mask = torch.ones_like(lm_labels, dtype=torch.bool)
        logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
        labels_flat = lm_labels.reshape(-1)
        mask_flat = mask.reshape(-1)
        if mask_flat.any():
            ce = F.cross_entropy(
                logits_flat[mask_flat],
                labels_flat[mask_flat].long(),
                reduction="mean",
            )
            loss_lm_ce = ce
            if lm_teacher_logits is not None and kd_temperature_lm > 0:
                t = kd_temperature_lm
                student_log = F.log_softmax(lm_logits / t, dim=-1)
                teacher = F.softmax(lm_teacher_logits / t, dim=-1)
                # per-token KD with mask averaging
                kd_per_tok = F.kl_div(
                    student_log, teacher, reduction="none"
                ).sum(dim=-1)
                loss_kd_lm = (kd_per_tok * mask.float()).sum() / mask.float().sum().clamp(
                    min=1.0
                ) * (t * t)

    # -- Coverage BCE (stage 3+) --
    if stage >= 3 and coverage_logits is not None and coverage_targets is not None:
        cmask = coverage_mask
        if cmask is None:
            cmask = torch.ones_like(coverage_logits, dtype=torch.bool)
        if cmask.any():
            loss_coverage = F.binary_cross_entropy_with_logits(
                coverage_logits[cmask], coverage_targets[cmask].to(coverage_logits)
            )

    # -- Readiness BCE (stage 3+) --
    if stage >= 3 and readiness_logits is not None and readiness_targets is not None:
        rmask = readiness_mask
        if rmask is None:
            rmask = torch.ones_like(readiness_logits, dtype=torch.bool)
        if rmask.any():
            loss_readiness = F.binary_cross_entropy(
                readiness_logits[rmask].clamp(1e-6, 1 - 1e-6),
                readiness_targets[rmask].to(readiness_logits),
            )

    # -- Stream classifier CE (stage 1+; auxiliary) --
    if stage >= 1 and stream_logits is not None and stream_targets is not None:
        loss_stream_classifier = F.cross_entropy(
            stream_logits, stream_targets.long()
        )

    total = (
        weights.planner * loss_planner
        + weights.notes * loss_notes
        + weights.spec * loss_spec
        + weights.lm_ce * loss_lm_ce
        + weights.kd_lm * loss_kd_lm
        + weights.coverage * loss_coverage
        + weights.readiness * loss_readiness
        # Stream classifier rolls into planner weight by convention (minor aux).
        + 0.1 * loss_stream_classifier
    )

    return LossBundle(
        total=total,
        planner=loss_planner,
        notes=loss_notes,
        spec=loss_spec,
        lm_ce=loss_lm_ce,
        kd_lm=loss_kd_lm,
        coverage=loss_coverage,
        readiness=loss_readiness,
        stream_classifier=loss_stream_classifier,
    )
