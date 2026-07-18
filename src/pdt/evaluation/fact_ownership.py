"""Claim-entailment, lane-exact fact ownership evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence

import torch


NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_MODEL_REVISION = "bab4bc7178836f731dcfd18c06ca9def0a137712"
NLI_MAX_LENGTH = 512

__all__ = [
    "FactOwnershipCounts",
    "NLI_MAX_LENGTH",
    "NLI_MODEL",
    "NLI_MODEL_REVISION",
    "NliEntailmentScorer",
    "calibrate_entailment_threshold",
    "fact_ownership_counts",
    "split_evidence_units",
]


@dataclass(frozen=True, slots=True)
class FactOwnershipCounts:
    """Additive counts for exact owner/reference/absence evaluation."""

    owner_hits: int
    owner_total: int
    reference_hits: int
    reference_total: int
    unauthorized_hits: int
    unauthorized_total: int
    hard_negative_hits: int
    hard_negative_total: int

    @property
    def owner_recall(self) -> float:
        return self.owner_hits / self.owner_total

    @property
    def reference_recall(self) -> float:
        return (
            self.reference_hits / self.reference_total
            if self.reference_total
            else 0.0
        )

    @property
    def unauthorized_leakage(self) -> float:
        return self.unauthorized_hits / self.unauthorized_total

    @property
    def hard_negative_rate(self) -> float:
        return self.hard_negative_hits / self.hard_negative_total

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = asdict(self)
        result.update(
            {
                "owner_recall": self.owner_recall,
                "reference_recall": self.reference_recall,
                "unauthorized_leakage": self.unauthorized_leakage,
                "hard_negative_rate": self.hard_negative_rate,
            }
        )
        return result

    def __add__(self, other: FactOwnershipCounts) -> FactOwnershipCounts:
        return FactOwnershipCounts(
            owner_hits=self.owner_hits + other.owner_hits,
            owner_total=self.owner_total + other.owner_total,
            reference_hits=self.reference_hits + other.reference_hits,
            reference_total=self.reference_total + other.reference_total,
            unauthorized_hits=self.unauthorized_hits + other.unauthorized_hits,
            unauthorized_total=self.unauthorized_total + other.unauthorized_total,
            hard_negative_hits=self.hard_negative_hits + other.hard_negative_hits,
            hard_negative_total=self.hard_negative_total + other.hard_negative_total,
        )


class NliEntailmentScorer:
    """Score atomic claims against sentence evidence with one pinned NLI model."""

    def __init__(
        self,
        *,
        device: str,
        batch_size: int = 64,
    ) -> None:
        if not device:
            raise ValueError("NLI device must be non-empty.")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("NLI batch_size must be a positive integer.")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            NLI_MODEL,
            revision=NLI_MODEL_REVISION,
            use_fast=True,
            local_files_only=True,
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Pinned NLI evaluation requires its fast tokenizer.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL,
            revision=NLI_MODEL_REVISION,
            local_files_only=True,
        )
        label2id = {
            str(label).casefold(): int(index)
            for label, index in self.model.config.label2id.items()
        }
        if set(label2id) != {"contradiction", "entailment", "neutral"}:
            raise RuntimeError(
                "Pinned NLI label mapping changed; expected contradiction, "
                f"entailment, neutral and got {label2id}."
            )
        self.entailment_index = label2id["entailment"]
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(
        self,
        lane_texts: Sequence[str],
        hypotheses: Sequence[str],
    ) -> torch.Tensor:
        """Return max sentence-level entailment probabilities `[3, hypotheses]`."""

        if len(lane_texts) != 3:
            raise ValueError("Fact ownership evaluation requires exactly three lane texts.")
        if not hypotheses or any(
            not isinstance(hypothesis, str) or not hypothesis.strip()
            for hypothesis in hypotheses
        ):
            raise ValueError("NLI hypotheses must contain non-empty claim text.")
        result = torch.zeros(3, len(hypotheses), dtype=torch.float32)
        premise_batch: list[str] = []
        hypothesis_batch: list[str] = []
        coordinates: list[tuple[int, int]] = []
        for lane_index, text in enumerate(lane_texts):
            if not isinstance(text, str):
                raise TypeError("Generated lane text must be a string.")
            if not text.strip():
                continue
            units = split_evidence_units(text)
            for hypothesis_index, hypothesis in enumerate(hypotheses):
                for unit in units:
                    premise_batch.append(unit)
                    hypothesis_batch.append(hypothesis)
                    coordinates.append((lane_index, hypothesis_index))
        for start in range(0, len(premise_batch), self.batch_size):
            stop = start + self.batch_size
            encoded = self.tokenizer(
                premise_batch[start:stop],
                hypothesis_batch[start:stop],
                padding=True,
                truncation="only_first",
                max_length=NLI_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
            output = self.model(**encoded)
            probabilities = output.logits.float().softmax(dim=-1)[
                :, self.entailment_index
            ].cpu()
            for coordinate, probability in zip(
                coordinates[start:stop],
                probabilities.tolist(),
                strict=True,
            ):
                lane_index, hypothesis_index = coordinate
                current = float(result[lane_index, hypothesis_index].item())
                result[lane_index, hypothesis_index] = max(current, float(probability))
        return result


def split_evidence_units(text: str) -> tuple[str, ...]:
    """Split prose into sentence-like NLI premises without short fragments."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Generated lane text must be non-empty.")
    paragraphs = tuple(
        paragraph.strip()
        for paragraph in re.split(r"\n{2,}", text)
        if paragraph.strip()
    )
    units: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if len(normalized) >= 20:
                units.append(normalized)
    if not units:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) < 20:
            raise ValueError(
                "Generated lane contains no evidence unit of at least 20 characters."
            )
        units.append(normalized)
    return tuple(units)


def calibrate_entailment_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Freeze a decision threshold from disjoint teacher-prose observations."""

    if scores.dim() != 1 or labels.shape != scores.shape:
        raise ValueError("Calibration scores and labels must share one-dimensional shape.")
    if labels.dtype != torch.bool:
        raise TypeError("Calibration labels must have dtype torch.bool.")
    if not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError("Calibration requires both present and absent fact observations.")
    if not bool(torch.isfinite(scores).all()) or bool(
        ((scores < 0.0) | (scores > 1.0)).any()
    ):
        raise ValueError("Entailment scores must be finite probabilities in [0, 1].")
    ordered = torch.unique(scores.detach().double().cpu(), sorted=True)
    boundaries = [
        max(0.0, float(ordered[0].item()) - 1e-6),
        *[
            float(((left + right) / 2).item())
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ],
        min(1.0, float(ordered[-1].item()) + 1e-6),
    ]
    truth = labels.detach().cpu()
    positive_total = int(truth.sum().item())
    negative_total = int((~truth).sum().item())
    best_threshold = boundaries[0]
    best_balanced_accuracy = -math.inf
    for threshold in boundaries:
        predicted = scores.detach().cpu() >= threshold
        true_positive_rate = int((predicted & truth).sum().item()) / positive_total
        true_negative_rate = int((~predicted & ~truth).sum().item()) / negative_total
        balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
        if (
            balanced_accuracy > best_balanced_accuracy
            or (
                balanced_accuracy == best_balanced_accuracy
                and threshold > best_threshold
            )
        ):
            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold
    return best_threshold


def fact_ownership_counts(
    scores: torch.Tensor,
    *,
    positive_fact_count: int,
    owner_lane_by_fact: torch.Tensor,
    reference_mask: torch.Tensor,
    absent_mask: torch.Tensor,
    threshold: float,
) -> FactOwnershipCounts:
    """Score entailed facts at assigned physical lanes, never against their union."""

    if scores.dim() != 2 or scores.size(0) != 3:
        raise ValueError("scores must have shape [3, fact_queries].")
    if type(positive_fact_count) is not int or positive_fact_count <= 0:
        raise ValueError("positive_fact_count must be a positive integer.")
    if scores.size(1) != 2 * positive_fact_count:
        raise ValueError("scores must contain one hard-negative query per positive fact.")
    if owner_lane_by_fact.shape != (positive_fact_count,):
        raise ValueError("owner_lane_by_fact must have shape [positive_facts].")
    if reference_mask.shape != (3, positive_fact_count):
        raise ValueError("reference_mask must have shape [3, positive_facts].")
    if absent_mask.shape != reference_mask.shape:
        raise ValueError("absent_mask must match reference_mask.")
    if reference_mask.dtype != torch.bool or absent_mask.dtype != torch.bool:
        raise TypeError("reference_mask and absent_mask must have dtype torch.bool.")
    if not bool(reference_mask.any()) or not bool(absent_mask.any()):
        raise ValueError(
            "Fact ownership evaluation requires reference and unauthorized-absence observations."
        )
    if (
        owner_lane_by_fact.dtype == torch.bool
        or owner_lane_by_fact.is_floating_point()
        or bool(((owner_lane_by_fact < 0) | (owner_lane_by_fact >= 3)).any())
    ):
        raise ValueError("owner_lane_by_fact must contain integer lane IDs in [0, 3).")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Fact entailment threshold must be finite and in [0, 1].")
    if not bool(torch.isfinite(scores).all()) or bool(
        ((scores < 0.0) | (scores > 1.0)).any()
    ):
        raise ValueError("Fact entailment scores must be finite probabilities in [0, 1].")

    positive_detected = scores[:, :positive_fact_count] >= threshold
    fact_index = torch.arange(positive_fact_count)
    owner_hits = int(
        positive_detected[owner_lane_by_fact.long(), fact_index].sum().item()
    )
    hard_negative_detected = scores[:, positive_fact_count:] >= threshold
    return FactOwnershipCounts(
        owner_hits=owner_hits,
        owner_total=positive_fact_count,
        reference_hits=int((positive_detected & reference_mask).sum().item()),
        reference_total=int(reference_mask.sum().item()),
        unauthorized_hits=int((positive_detected & absent_mask).sum().item()),
        unauthorized_total=int(absent_mask.sum().item()),
        hard_negative_hits=int(hard_negative_detected.sum().item()),
        hard_negative_total=int(hard_negative_detected.numel()),
    )
