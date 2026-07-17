"""Embedding-based, lane-exact fact ownership evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Sequence

import numpy as np
import torch


__all__ = [
    "FactOwnershipCounts",
    "calibrate_similarity_threshold",
    "fact_ownership_counts",
    "maximum_query_similarity",
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


def split_evidence_units(text: str) -> tuple[str, ...]:
    """Split prose into sentence-like evidence units without short fragments."""

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
            raise ValueError("Generated lane contains no evidence unit of at least 20 characters.")
        units.append(normalized)
    return tuple(units)


def maximum_query_similarity(
    lane_texts: Sequence[str],
    query_embeddings: torch.Tensor,
    *,
    embedder: Any,
) -> torch.Tensor:
    """Return maximum BGE cosine similarity with shape `[lanes, queries]`."""

    if len(lane_texts) != 3:
        raise ValueError("Fact ownership evaluation requires exactly three lane texts.")
    if query_embeddings.dim() != 2 or query_embeddings.size(0) == 0:
        raise ValueError("query_embeddings must have shape [queries, embedding_dim].")
    if not query_embeddings.is_floating_point():
        raise TypeError("query_embeddings must use a floating-point dtype.")
    if not bool(torch.isfinite(query_embeddings).all()):
        raise ValueError("query_embeddings must be finite.")
    normalized_queries = torch.nn.functional.normalize(
        query_embeddings.detach().float().cpu(),
        dim=-1,
    )
    lane_scores: list[torch.Tensor] = []
    for text in lane_texts:
        if not isinstance(text, str):
            raise TypeError("Generated lane text must be a string.")
        if not text.strip():
            lane_scores.append(torch.full((query_embeddings.size(0),), -1.0))
            continue
        units = list(split_evidence_units(text))
        encoded = embedder.encode(
            units,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        unit_embeddings = torch.from_numpy(np.asarray(encoded, dtype=np.float32))
        if unit_embeddings.shape != (len(units), query_embeddings.size(1)):
            raise RuntimeError(
                "Semantic embedder returned the wrong evidence shape: "
                f"expected {(len(units), query_embeddings.size(1))}, "
                f"got {tuple(unit_embeddings.shape)}."
            )
        lane_scores.append((unit_embeddings @ normalized_queries.t()).max(dim=0).values)
    return torch.stack(lane_scores)


def calibrate_similarity_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Choose a frozen threshold maximizing balanced accuracy on teacher prose."""

    if scores.dim() != 1 or labels.shape != scores.shape:
        raise ValueError("Calibration scores and labels must share one-dimensional shape.")
    if labels.dtype != torch.bool:
        raise TypeError("Calibration labels must have dtype torch.bool.")
    if not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError("Calibration requires both present and absent fact observations.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Calibration scores must be finite.")
    ordered = torch.unique(scores.detach().double().cpu(), sorted=True)
    boundaries = [
        float(ordered[0].item()) - 1e-6,
        *[
            float(((left + right) / 2).item())
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ],
        float(ordered[-1].item()) + 1e-6,
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
    """Score facts against their assigned physical lane, never against the union."""

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
    if not math.isfinite(threshold):
        raise ValueError("Fact similarity threshold must be finite.")

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
