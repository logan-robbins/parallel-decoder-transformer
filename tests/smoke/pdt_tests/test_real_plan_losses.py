"""Exact active-loss and hard-negative supervision contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pdt.config.schemas import LossWeights
from pdt.training.losses import compute_pdt_losses


def _zero_weights() -> LossWeights:
    return LossWeights(
        lm_ce=0.0,
        plan_semantic=0.0,
        fact_route=0.0,
        outline_progress=0.0,
        fact_write=0.0,
        note_align=0.0,
        presentation_order=0.0,
        dynamic_vq_commit=0.0,
        dynamic_vq_codebook=0.0,
        dynamic_codebook_usage=0.0,
    )


def _loss(**overrides: object):
    values: dict[str, object] = {
        "weights": _zero_weights(),
        "lm_ce": None,
        "plan_semantic": None,
        "fact_route_logits": None,
        "fact_route_targets": None,
        "plan_node_mask": None,
        "fact_mask": None,
        "outline_progress_logits": None,
        "outline_progress_targets": None,
        "fact_write_logits": None,
        "fact_write_targets": None,
        "note_queries": None,
        "note_keys": None,
        "presentation_order_logits": None,
        "presentation_rank_targets": None,
        "dynamic_vq_commitment_loss": None,
        "dynamic_vq_codebook_loss": None,
        "dynamic_vq_logits": None,
    }
    values.update(overrides)
    return compute_pdt_losses(**values)


def test_positive_loss_weight_cannot_silently_disable_supervision() -> None:
    for field, expected in (
        ("lm_ce", "lm_ce"),
        ("plan_semantic", "plan_semantic"),
        ("note_align", "note_align"),
        ("presentation_order", "presentation_order"),
        ("dynamic_vq_commit", "dynamic_vq_commit"),
        ("dynamic_vq_codebook", "dynamic_vq_codebook"),
        ("dynamic_codebook_usage", "dynamic_codebook_usage"),
    ):
        with pytest.raises(ValueError, match=expected):
            _loss(weights=replace(_zero_weights(), **{field: 1.0}))


def test_fact_write_scores_paired_hard_negatives_as_absent() -> None:
    logits = torch.full((1, 3, 2, 4, 3), -8.0)
    targets = torch.tensor(
        [
            [
                [[0, 2, 2, 2], [2, 1, 2, 2]],
                [[2, 0, 2, 2], [1, 2, 2, 2]],
                [[2, 2, 2, 2], [0, 2, 2, 2]],
            ]
        ]
    )
    logits.scatter_(-1, targets.unsqueeze(-1), 8.0)
    losses = _loss(
        weights=replace(_zero_weights(), fact_write=1.0),
        fact_write_logits=logits,
        fact_write_targets=targets,
    )
    assert losses.fact_write < 1e-5

    incorrect = logits.clone()
    incorrect[..., 2:, 2] = -8.0
    incorrect[..., 2:, 0] = 8.0
    incorrect_losses = _loss(
        weights=replace(_zero_weights(), fact_write=1.0),
        fact_write_logits=incorrect,
        fact_write_targets=targets,
    )
    assert incorrect_losses.fact_write > 1.0


def test_dynamic_usage_penalizes_collapsed_finite_messages() -> None:
    balanced = torch.full((4, 2, 4), -10.0)
    collapsed = torch.full((4, 2, 4), -10.0)
    for sample in range(4):
        balanced[sample, :, sample] = 10.0
    collapsed[:, :, 0] = 10.0
    weights = replace(_zero_weights(), dynamic_codebook_usage=1.0)
    assert _loss(
        weights=weights,
        dynamic_vq_logits=balanced,
    ).dynamic_codebook_usage < 1e-6
    assert _loss(
        weights=weights,
        dynamic_vq_logits=collapsed,
    ).dynamic_codebook_usage > 0.99


def test_fact_route_masks_padded_nodes_and_queries() -> None:
    logits = torch.zeros(1, 3, 8, 6)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0, 0] = 1.0
    node_mask = torch.zeros(1, 3, 8, dtype=torch.bool)
    node_mask[:, :, :4] = True
    fact_mask = torch.tensor([[True, True, True, True, False, False]])
    result = _loss(
        weights=replace(_zero_weights(), fact_route=1.0),
        fact_route_logits=logits,
        fact_route_targets=targets,
        plan_node_mask=node_mask,
        fact_mask=fact_mask,
    )
    torch.testing.assert_close(result.fact_route, torch.log(torch.tensor(2.0)))
