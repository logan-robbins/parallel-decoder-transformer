"""Lane-exact embedded-fact evaluation contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pdt.evaluation.fact_ownership import (
    calibrate_similarity_threshold,
    fact_ownership_counts,
    maximum_query_similarity,
    split_evidence_units,
)
from pdt.evaluation.real_plan_generation import (
    long_form_statistics,
    remap_role_targets,
    swap_teacher_to_physical,
)


class _Embedder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for index, text in enumerate(texts):
            if "alpha" in text.lower():
                vectors[index, 0] = 1.0
            elif "beta" in text.lower():
                vectors[index, 1] = 1.0
            else:
                vectors[index, 2] = 1.0
        return vectors


def test_evidence_units_preserve_long_sentences_and_reject_empty_output() -> None:
    text = (
        "The first developed sentence contains enough material to stand alone. "
        "The second developed sentence also carries a complete historical claim."
    )
    assert len(split_evidence_units(text)) == 2
    with pytest.raises(ValueError, match="non-empty"):
        split_evidence_units(" ")


def test_maximum_similarity_is_computed_per_physical_lane() -> None:
    queries = torch.eye(4)[:3]
    scores = maximum_query_similarity(
        [
            "This sufficiently developed sentence discusses alpha in context.",
            "This sufficiently developed sentence discusses beta in context.",
            "This sufficiently developed sentence discusses gamma in context.",
        ],
        queries,
        embedder=_Embedder(),
    )
    assert scores.shape == (3, 3)
    torch.testing.assert_close(scores, torch.eye(3))


def test_threshold_is_calibrated_from_present_and_absent_teacher_observations() -> None:
    scores = torch.tensor([0.91, 0.84, 0.73, 0.22, 0.14, 0.05])
    labels = torch.tensor([True, True, True, False, False, False])
    threshold = calibrate_similarity_threshold(scores, labels)
    assert 0.22 < threshold <= 0.73


def test_owner_recall_never_grants_credit_from_the_wrong_lane() -> None:
    # Two positive queries followed by their two paired hard negatives.
    scores = torch.tensor(
        [
            [0.95, 0.90, 0.10, 0.10],
            [0.96, 0.20, 0.10, 0.80],
            [0.10, 0.92, 0.10, 0.10],
        ]
    )
    counts = fact_ownership_counts(
        scores,
        positive_fact_count=2,
        owner_lane_by_fact=torch.tensor([0, 2]),
        reference_mask=torch.tensor(
            [
                [False, False],
                [True, False],
                [False, False],
            ]
        ),
        absent_mask=torch.tensor(
            [
                [False, True],
                [False, True],
                [True, False],
            ]
        ),
        threshold=0.75,
    )
    assert counts.owner_recall == 1.0
    assert counts.reference_recall == 1.0
    assert counts.unauthorized_leakage == pytest.approx(1 / 3)
    assert counts.hard_negative_rate == pytest.approx(1 / 6)


def test_role_targets_follow_the_physical_plan_address() -> None:
    owner, references, absences = remap_role_targets(
        torch.tensor([0, 1, 2]),
        torch.tensor(
            [
                [False, True, False],
                [False, False, True],
                [True, False, False],
            ]
        ),
        torch.tensor(
            [
                [False, False, True],
                [True, False, False],
                [False, True, False],
            ]
        ),
        torch.tensor([2, 0, 1]),
    )
    assert owner.tolist() == [2, 0, 1]
    assert references.tolist() == [
        [False, False, True],
        [True, False, False],
        [False, True, False],
    ]
    assert absences.tolist() == [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]
    assert swap_teacher_to_physical(torch.tensor([2, 0, 1])).tolist() == [2, 1, 0]


def test_long_form_gate_rejects_short_choppy_output() -> None:
    developed = (
        "This developed historical sentence contains enough connected words to "
        "make a complete explanatory point."
    )
    long_text = "\n\n".join(
        f"{developed} {developed} {developed}"
        for _ in range(4)
    )
    assert long_form_statistics(long_text, token_count=700)["passes"] is True
    assert (
        long_form_statistics(
            "Too short. Still short. Not developed. No context.",
            token_count=700,
        )["passes"]
        is False
    )
