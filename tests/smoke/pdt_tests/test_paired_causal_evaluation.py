"""Paired teacher-forced causal evaluation contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pdt.evaluation.paired_causal import PairedCausalEvaluator


def _batch() -> dict[str, object]:
    baseline = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, 2.0]],
            [[0.0, 2.0, -1.0], [3.0, 0.0, -1.0], [0.0, 1.0, 2.0]],
            [[2.0, 0.0, -1.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]],
        ]
    )
    gate_zero = baseline.clone()
    gate_zero[0, 0] = torch.tensor([0.0, 3.0, -1.0])
    gate_zero[1, :2] = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]])
    norm_scramble = baseline.clone()
    norm_scramble[:, 0] = torch.roll(norm_scramble[:, 0], shifts=1, dims=0)
    mutation = baseline.clone()
    mutation[0, 0] = torch.tensor([1.0, 2.0, -1.0])
    mutation[1, 0] = torch.tensor([2.0, 0.0, -1.0])
    mutation[2, 0] = torch.tensor([0.0, 2.0, -1.0])
    labels = torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 0]])
    label_mask = torch.ones_like(labels, dtype=torch.bool)
    dependency_mask = torch.tensor(
        [[True, False, False], [True, True, False], [True, False, False]]
    )
    nondependency_mask = torch.tensor(
        [[False, True, True], [False, False, True], [False, True, True]]
    )
    mutation_dependency_mask = torch.tensor(
        [[True, False, False], [True, False, False], [True, False, False]]
    )
    return {
        "baseline_logits": baseline,
        "gate_zero_logits": gate_zero,
        "norm_scramble_logits": norm_scramble,
        "mutation_logits": mutation,
        "labels": labels,
        "label_mask": label_mask,
        "dependency_mask": dependency_mask,
        "nondependency_mask": nondependency_mask,
        "mutation_dependency_mask": mutation_dependency_mask,
        "dependency_lag_masks": {1: dependency_mask.clone()},
        "example_ids": ["document-0", "document-1", "document-2"],
    }


def _update_by_rows(evaluator: PairedCausalEvaluator, batch: dict[str, object]) -> None:
    labels = batch["labels"]
    assert isinstance(labels, torch.Tensor)
    rows = labels.size(0)
    for row in range(rows):
        update: dict[str, object] = {}
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                update[name] = value[row : row + 1]
            elif name == "dependency_lag_masks":
                update[name] = {
                    lag: mask[row : row + 1] for lag, mask in value.items()
                }
            elif name == "example_ids":
                update[name] = value[row : row + 1]
        evaluator.update(**update)


def _compute(evaluator: PairedCausalEvaluator):
    return evaluator.compute(
        bootstrap_samples=2000,
        confidence_level=0.95,
        seed=17,
        minimum_documents=3,
    )


def test_results_are_batch_partition_invariant_and_token_weighted() -> None:
    batch = _batch()
    whole = PairedCausalEvaluator()
    whole.update(**batch)
    partitioned = PairedCausalEvaluator()
    _update_by_rows(partitioned, batch)

    whole_result = _compute(whole)
    partitioned_result = _compute(partitioned)
    assert whole_result.batches == 1
    assert partitioned_result.batches == 3
    assert whole_result.gate_zero.aggregate.to_dict() == pytest.approx(
        partitioned_result.gate_zero.aggregate.to_dict()
    )
    assert whole_result.norm_scramble.aggregate.to_dict() == pytest.approx(
        partitioned_result.norm_scramble.aggregate.to_dict()
    )
    assert whole_result.gate_zero.document_inference.dependency_ce_delta.to_dict() == (
        pytest.approx(
            partitioned_result.gate_zero.document_inference.dependency_ce_delta.to_dict()
        )
    )

    labels = batch["labels"]
    dep = batch["dependency_mask"]
    baseline_nll = F.cross_entropy(
        batch["baseline_logits"].reshape(-1, 3),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    gate_nll = F.cross_entropy(
        batch["gate_zero_logits"].reshape(-1, 3),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    expected_delta = float(gate_nll[dep].mean() - baseline_nll[dep].mean())
    assert whole_result.gate_zero.aggregate.dependency_tokens == int(dep.sum())
    assert whole_result.gate_zero.aggregate.dependency_ce_delta == pytest.approx(expected_delta)
    first_effect = whole_result.gate_zero.document_inference.document_effects[0]
    assert first_effect.normal_dependency_ce == pytest.approx(float(baseline_nll[0][dep[0]].mean()))
    assert first_effect.ablated_dependency_ce == pytest.approx(float(gate_nll[0][dep[0]].mean()))


def test_targeted_mutation_kl_uses_only_its_narrow_dependency_mask() -> None:
    batch = _batch()
    evaluator = PairedCausalEvaluator()
    evaluator.update(**batch)
    result = _compute(evaluator)

    mutation_dep = batch["mutation_dependency_mask"]
    baseline_log = F.log_softmax(batch["baseline_logits"].float(), dim=-1)
    mutation_log = F.log_softmax(batch["mutation_logits"].float(), dim=-1)
    expected = (baseline_log.exp() * (baseline_log - mutation_log)).sum(dim=-1)[mutation_dep].mean()
    assert result.gate_zero.aggregate.dependency_tokens == int(batch["dependency_mask"].sum())
    assert result.targeted_mutation.mutation_dependency_tokens == int(mutation_dep.sum())
    assert (
        result.targeted_mutation.mutation_dependency_tokens
        != result.gate_zero.aggregate.dependency_tokens
    )
    assert result.targeted_mutation.baseline_to_mutation_kl == pytest.approx(float(expected))


def test_alignment_failure_is_transactional() -> None:
    batch = _batch()
    invalid = dict(batch)
    invalid["norm_scramble_logits"] = batch["norm_scramble_logits"][:, :-1]
    evaluator = PairedCausalEvaluator()
    with pytest.raises(ValueError, match="norm_scramble_logits must have shape"):
        evaluator.update(**invalid)

    evaluator.update(**batch)
    result = _compute(evaluator)
    assert result.batches == 1
    assert result.gate_zero.aggregate.dependency_tokens == int(batch["dependency_mask"].sum())


@pytest.mark.parametrize(
    ("mutation_mask", "message"),
    [
        (torch.zeros((3, 3), dtype=torch.bool), "must contain at least one token"),
        (torch.ones((3, 2), dtype=torch.bool), "must have shape"),
        (
            torch.tensor([[False, True, False], [False, False, False], [False, False, False]]),
            "must be a subset of dependency_mask",
        ),
    ],
)
def test_invalid_mutation_mask_is_transactional(
    mutation_mask: torch.Tensor,
    message: str,
) -> None:
    batch = _batch()
    evaluator = PairedCausalEvaluator()
    evaluator.update(**batch)
    before = _compute(evaluator).to_dict()
    invalid = dict(batch)
    invalid["mutation_dependency_mask"] = mutation_mask

    with pytest.raises(ValueError, match=message):
        evaluator.update(**invalid)

    after = _compute(evaluator).to_dict()
    assert after == before
    assert after["batches"] == 1


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("labels", torch.ones((3, 3), dtype=torch.float32), "labels must have an integer"),
        ("label_mask", torch.ones((3, 3), dtype=torch.long), "label_mask must have dtype"),
        (
            "dependency_mask",
            torch.ones((3, 3), dtype=torch.long),
            "dependency_mask must have dtype",
        ),
        (
            "mutation_dependency_mask",
            torch.ones((3, 3), dtype=torch.long),
            "mutation_dependency_mask must have dtype",
        ),
    ],
)
def test_invalid_dtypes_fail_fast(
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    batch = _batch()
    batch[field] = replacement
    with pytest.raises(TypeError, match=message):
        PairedCausalEvaluator().update(**batch)


def test_invalid_masks_and_labels_fail_fast() -> None:
    overlapping = _batch()
    overlapping["nondependency_mask"] = overlapping["dependency_mask"].clone()
    with pytest.raises(ValueError, match="must be disjoint"):
        PairedCausalEvaluator().update(**overlapping)

    outside_labels = _batch()
    outside_labels["label_mask"][0, 0] = False
    with pytest.raises(ValueError, match="included in label_mask"):
        PairedCausalEvaluator().update(**outside_labels)

    invalid_label = _batch()
    invalid_label["labels"][0, 0] = invalid_label["baseline_logits"].size(-1)
    with pytest.raises(ValueError, match="active labels"):
        PairedCausalEvaluator().update(**invalid_label)


def test_compute_fails_without_both_span_types() -> None:
    with pytest.raises(RuntimeError, match="zero batches"):
        _compute(PairedCausalEvaluator())

    batch = _batch()
    batch["nondependency_mask"].zero_()
    with pytest.raises(ValueError, match="dependency and nondependency tokens"):
        PairedCausalEvaluator().update(**batch)


def test_evidence_gate_uses_positive_document_intervals_and_reports_lag() -> None:
    documents = 4
    baseline = torch.tensor([[[4.0, 0.0], [4.0, 0.0]]] * documents)
    gate_zero = baseline.clone()
    gate_zero[:, 0] = torch.tensor([0.0, 4.0])
    norm_scramble = gate_zero.clone()
    mutation = baseline.clone()
    mutation[:, 0] = torch.tensor([1.0, 3.0])
    dependency = torch.tensor([[True, False]] * documents)
    nondependency = ~dependency
    evaluator = PairedCausalEvaluator()
    evaluator.update(
        baseline_logits=baseline,
        gate_zero_logits=gate_zero,
        norm_scramble_logits=norm_scramble,
        mutation_logits=mutation,
        labels=torch.zeros((documents, 2), dtype=torch.long),
        label_mask=torch.ones((documents, 2), dtype=torch.bool),
        dependency_mask=dependency,
        nondependency_mask=nondependency,
        mutation_dependency_mask=dependency,
        dependency_lag_masks={4: dependency.clone()},
        example_ids=[f"positive-{index}" for index in range(documents)],
    )
    result = evaluator.compute(
        bootstrap_samples=2000,
        confidence_level=0.95,
        seed=31,
        minimum_documents=4,
    )

    assert result.evidence_gate.passes is True
    assert result.gate_zero.document_inference.dependency_ce_delta.lower > 0.0
    assert result.gate_zero.document_inference.dependency_selectivity_difference.lower > 0.0
    assert result.targeted_mutation.document_bootstrap.lower > 0.0
    assert [effect.lag_blocks for effect in result.gate_zero.lag_effects] == [4]
