"""Exact aggregation contracts for causal coordination metrics."""

from __future__ import annotations

import pytest
import torch

from pdt.diagnostics.causal_metrics import (
    CausalAblationAccumulator,
    note_bandwidth_bytes,
)


def test_metrics_are_token_weighted_across_unequal_updates():
    metrics = CausalAblationAccumulator()
    metrics.update_token_values(
        normal_nll=torch.tensor([1.0, 1.0]),
        ablated_nll=torch.tensor([3.0, 1.2]),
        dependency_mutation_kl=torch.tensor([0.4, 0.0]),
        label_mask=torch.tensor([True, True]),
        dependency_mask=torch.tensor([True, False]),
        nondependency_mask=torch.tensor([False, True]),
    )
    metrics.update_token_values(
        normal_nll=torch.tensor([2.0, 2.0, 2.0, 2.0]),
        ablated_nll=torch.tensor([2.0, 2.0, 2.0, 2.2]),
        dependency_mutation_kl=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        label_mask=torch.ones(4, dtype=torch.bool),
        dependency_mask=torch.tensor([True, True, True, False]),
        nondependency_mask=torch.tensor([False, False, False, True]),
    )

    result = metrics.compute()
    assert result.dependency_tokens == 4
    assert result.nondependency_tokens == 2
    assert result.dependency_ce_delta == pytest.approx(0.5)
    assert result.nondependency_ce_delta == pytest.approx(0.2)
    assert result.dependency_selectivity_ratio == pytest.approx(2.5)
    assert result.dependency_mutation_kl == pytest.approx(0.1)


def test_logit_path_matches_manual_target_cross_entropy():
    normal = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    ablated = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    labels = torch.tensor([[0, 1]])
    accumulator = CausalAblationAccumulator()
    accumulator.update_from_logits(
        normal_logits=normal,
        ablated_logits=ablated,
        labels=labels,
        label_mask=torch.ones_like(labels, dtype=torch.bool),
        dependency_mask=torch.tensor([[True, False]]),
        nondependency_mask=torch.tensor([[False, True]]),
    )

    result = accumulator.compute()
    expected_normal = torch.nn.functional.cross_entropy(normal[:, 0], labels[:, 0])
    expected_ablated = torch.nn.functional.cross_entropy(ablated[:, 0], labels[:, 0])
    assert result.dependency_ce_delta == pytest.approx(float(expected_ablated - expected_normal))
    assert result.nondependency_ce_delta == pytest.approx(0.0)
    assert result.dependency_mutation_kl > 0.0


def test_metric_masks_fail_fast_on_overlap_or_missing_span_types():
    accumulator = CausalAblationAccumulator()
    with pytest.raises(ValueError, match="must be disjoint"):
        accumulator.update_token_values(
            normal_nll=torch.ones(1),
            ablated_nll=torch.ones(1),
            label_mask=torch.tensor([True]),
            dependency_mask=torch.tensor([True]),
            nondependency_mask=torch.tensor([True]),
        )

    with pytest.raises(RuntimeError, match="zero dependency tokens"):
        CausalAblationAccumulator().compute()


def test_note_bandwidth_is_exact_and_rejects_invalid_dimensions():
    assert (
        note_bandwidth_bytes(
            num_streams=3,
            notes_dim=256,
            bytes_per_element=2,
            num_blocks=8,
        )
        == 12_288
    )
    with pytest.raises(ValueError, match="num_blocks"):
        note_bandwidth_bytes(
            num_streams=3,
            notes_dim=256,
            bytes_per_element=2,
            num_blocks=0,
        )
