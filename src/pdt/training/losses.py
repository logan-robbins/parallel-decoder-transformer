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
    kd_temperature_lm: float = 2.0,
    vq_commitment_loss: Optional[torch.Tensor] = None,
    vq_codebook_loss: Optional[torch.Tensor] = None,
    planner_logits: Optional[torch.Tensor] = None,
    stream_logits: Optional[torch.Tensor] = None,
    stream_targets: Optional[torch.Tensor] = None,
) -> LossBundle:
    del stage
    _validate_lm_loss_inputs(
        weights=weights,
        lm_logits=lm_logits,
        lm_labels=lm_labels,
        lm_label_mask=lm_label_mask,
        dependency_mask=dependency_mask,
        nondependency_mask=nondependency_mask,
        lm_teacher_logits=lm_teacher_logits,
        kd_temperature_lm=kd_temperature_lm,
    )
    _validate_auxiliary_loss_inputs(
        weights=weights,
        stream_logits=stream_logits,
        stream_targets=stream_targets,
    )
    zero = _zero_like(
        lm_logits,
        vq_commitment_loss,
        vq_codebook_loss,
        planner_logits,
        stream_logits,
    )
    loss_lm_ce = zero
    loss_lm_dep = zero
    loss_lm_non = zero
    loss_kd_lm = zero
    loss_vq_commit = vq_commitment_loss if vq_commitment_loss is not None else zero
    loss_vq_codebook = vq_codebook_loss if vq_codebook_loss is not None else zero
    loss_usage = zero
    loss_stream_classifier = zero

    per_token_ce = None
    mask = None
    if lm_logits is not None and lm_labels is not None:
        mask = lm_label_mask
        if mask is None:
            mask = lm_labels >= 0
        else:
            mask = mask.to(device=lm_logits.device, dtype=torch.bool)
        # Accumulate CE/KL in FP32 even though the frozen trunk emits BF16.
        # The cast remains differentiable and avoids low-precision reductions
        # over the full Qwen vocabulary.
        logits_flat = lm_logits.float().reshape(-1, lm_logits.size(-1))
        labels_flat = lm_labels.reshape(-1)
        mask_flat = mask.reshape(-1)
        if mask_flat.any():
            ce_flat = F.cross_entropy(
                logits_flat[mask_flat],
                labels_flat[mask_flat].long(),
                reduction="none",
            )
            loss_lm_ce = ce_flat.mean()
            per_token_ce = torch.zeros_like(labels_flat, dtype=torch.float32)
            per_token_ce[mask_flat] = ce_flat
            per_token_ce = per_token_ce.view_as(lm_labels)

            if weights.kd_lm > 0:
                assert lm_teacher_logits is not None
                assert dependency_mask is not None
                kd_mask = dependency_mask.to(device=mask.device, dtype=torch.bool) & mask
                if kd_mask.any():
                    t = kd_temperature_lm
                    student_log = F.log_softmax(lm_logits[kd_mask].float() / t, dim=-1)
                    teacher = F.softmax(
                        lm_teacher_logits.detach()[kd_mask].to(
                            device=lm_logits.device,
                            dtype=torch.float32,
                        )
                        / t,
                        dim=-1,
                    )
                    loss_kd_lm = F.kl_div(student_log, teacher, reduction="batchmean") * (t * t)

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

    if stream_logits is not None and stream_targets is not None:
        loss_stream_classifier = F.cross_entropy(stream_logits, stream_targets.long())

    total = (
        weights.lm_ce * loss_lm_ce
        + weights.kd_lm * loss_kd_lm
        + weights.vq_commit * loss_vq_commit
        + weights.vq_codebook * loss_vq_codebook
        + weights.codebook_usage * loss_usage
        + weights.stream_classifier * loss_stream_classifier
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
        stream_classifier=loss_stream_classifier,
    )


def _zero_like(*candidates: Optional[torch.Tensor]) -> torch.Tensor:
    for tensor in candidates:
        if tensor is not None:
            return tensor.new_tensor(0.0)
    return torch.tensor(0.0)


def _validate_lm_loss_inputs(
    *,
    weights: LossWeights,
    lm_logits: Optional[torch.Tensor],
    lm_labels: Optional[torch.Tensor],
    lm_label_mask: Optional[torch.Tensor],
    dependency_mask: Optional[torch.Tensor],
    nondependency_mask: Optional[torch.Tensor],
    lm_teacher_logits: Optional[torch.Tensor],
    kd_temperature_lm: float,
) -> None:
    """Fail fast on an incomplete or misaligned functional-KD batch."""

    if (lm_logits is None) != (lm_labels is None):
        raise ValueError("lm_logits and lm_labels must be supplied together.")
    if lm_logits is not None:
        expected_token_shape = lm_logits.shape[:-1]
        if lm_labels is None or lm_labels.shape != expected_token_shape:
            actual = None if lm_labels is None else tuple(lm_labels.shape)
            raise ValueError(
                "lm_labels must match lm_logits token dimensions: "
                f"expected {tuple(expected_token_shape)}, got {actual}."
            )
        for name, tensor in (
            ("lm_label_mask", lm_label_mask),
            ("dependency_mask", dependency_mask),
            ("nondependency_mask", nondependency_mask),
        ):
            if tensor is not None and tensor.shape != expected_token_shape:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_token_shape)}, "
                    f"got {tuple(tensor.shape)}."
                )

        active = (
            lm_labels >= 0
            if lm_label_mask is None
            else lm_label_mask.to(device=lm_labels.device, dtype=torch.bool)
        )
        if active.any():
            active_labels = lm_labels[active]
            if (active_labels < 0).any() or (active_labels >= lm_logits.size(-1)).any():
                raise ValueError("Active lm_labels must be valid vocabulary indices.")

        dep = None
        nondep = None
        if dependency_mask is not None:
            dep = dependency_mask.to(device=active.device, dtype=torch.bool)
            if (dep & ~active).any():
                raise ValueError("dependency_mask must be a subset of lm_label_mask.")
        if nondependency_mask is not None:
            nondep = nondependency_mask.to(device=active.device, dtype=torch.bool)
            if (nondep & ~active).any():
                raise ValueError("nondependency_mask must be a subset of lm_label_mask.")
        if dep is not None and nondep is not None and (dep & nondep).any():
            raise ValueError("dependency_mask and nondependency_mask must be disjoint.")

    if weights.kd_lm <= 0:
        return
    if lm_logits is None:
        raise ValueError("kd_lm > 0 requires student lm_logits and lm_labels.")
    if lm_teacher_logits is None:
        raise ValueError(
            "kd_lm > 0 requires privileged-context lm_teacher_logits; "
            "functional distillation may not silently disable itself."
        )
    if lm_teacher_logits.shape != lm_logits.shape:
        raise ValueError(
            "lm_teacher_logits must exactly match lm_logits: "
            f"expected {tuple(lm_logits.shape)}, got {tuple(lm_teacher_logits.shape)}."
        )
    if dependency_mask is None:
        raise ValueError("kd_lm > 0 requires dependency_mask.")
    if lm_labels is None:
        raise RuntimeError("Validated functional KD unexpectedly lost lm_labels.")
    active = (
        lm_labels >= 0
        if lm_label_mask is None
        else lm_label_mask.to(device=lm_labels.device, dtype=torch.bool)
    )
    dep = dependency_mask.to(device=active.device, dtype=torch.bool)
    if not (dep & active).any():
        raise ValueError("kd_lm > 0 requires at least one active dependency token in every batch.")
    if kd_temperature_lm != 2.0:
        raise ValueError(
            "The canonical same-trunk functional distillation temperature is 2.0; "
            f"got {kd_temperature_lm}."
        )


def _validate_auxiliary_loss_inputs(
    *,
    weights: LossWeights,
    stream_logits: Optional[torch.Tensor],
    stream_targets: Optional[torch.Tensor],
) -> None:
    """Reject positive objective weights whose aligned tensors are absent."""

    weight = weights.stream_classifier
    if weight < 0:
        raise ValueError(f"stream_classifier loss weight must be non-negative, got {weight}.")
    if weight > 0 and (stream_logits is None or stream_targets is None):
        raise ValueError(
            f"stream_classifier loss weight is {weight}, but aligned logits and targets "
            "were not both supplied. This objective may not silently disable itself."
        )
    if (stream_logits is None) != (stream_targets is None):
        raise ValueError("stream_classifier logits and targets must be supplied together.")
    if stream_logits is not None and stream_targets is not None:
        if stream_logits.shape[:-1] != stream_targets.shape:
            raise ValueError(
                "stream_classifier targets are not aligned with logits: "
                f"logits={tuple(stream_logits.shape)}, "
                f"targets={tuple(stream_targets.shape)}."
            )
