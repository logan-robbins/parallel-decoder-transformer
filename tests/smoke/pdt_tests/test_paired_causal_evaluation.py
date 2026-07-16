"""Paired teacher-forced causal evaluation contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pdt.evaluation.paired_causal import PairedCausalEvaluator


def _batch() -> dict[str, torch.Tensor]:
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
    }


def _update_by_rows(evaluator: PairedCausalEvaluator, batch: dict[str, torch.Tensor]) -> None:
    rows = batch["labels"].size(0)
    for row in range(rows):
        evaluator.update(**{name: tensor[row : row + 1] for name, tensor in batch.items()})


def test_results_are_batch_partition_invariant_and_token_weighted() -> None:
    batch = _batch()
    whole = PairedCausalEvaluator()
    whole.update(**batch)
    partitioned = PairedCausalEvaluator()
    _update_by_rows(partitioned, batch)

    whole_result = whole.compute()
    partitioned_result = partitioned.compute()
    assert whole_result.batches == 1
    assert partitioned_result.batches == 3
    assert whole_result.gate_zero.to_dict() == pytest.approx(partitioned_result.gate_zero.to_dict())
    assert whole_result.norm_scramble.to_dict() == pytest.approx(
        partitioned_result.norm_scramble.to_dict()
    )
    assert whole_result.targeted_mutation.to_dict() == pytest.approx(
        partitioned_result.targeted_mutation.to_dict()
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
    assert whole_result.gate_zero.dependency_tokens == int(dep.sum())
    assert whole_result.gate_zero.dependency_ce_delta == pytest.approx(expected_delta)


def test_targeted_mutation_kl_uses_only_its_narrow_dependency_mask() -> None:
    batch = _batch()
    evaluator = PairedCausalEvaluator()
    evaluator.update(**batch)
    result = evaluator.compute()

    mutation_dep = batch["mutation_dependency_mask"]
    baseline_log = F.log_softmax(batch["baseline_logits"].float(), dim=-1)
    mutation_log = F.log_softmax(batch["mutation_logits"].float(), dim=-1)
    expected = (baseline_log.exp() * (baseline_log - mutation_log)).sum(dim=-1)[mutation_dep].mean()
    assert result.gate_zero.dependency_tokens == int(batch["dependency_mask"].sum())
    assert result.targeted_mutation.mutation_dependency_tokens == int(mutation_dep.sum())
    assert result.targeted_mutation.mutation_dependency_tokens != result.gate_zero.dependency_tokens
    assert result.targeted_mutation.baseline_to_mutation_kl == pytest.approx(float(expected))


def test_alignment_failure_is_transactional() -> None:
    batch = _batch()
    invalid = dict(batch)
    invalid["norm_scramble_logits"] = batch["norm_scramble_logits"][:, :-1]
    evaluator = PairedCausalEvaluator()
    with pytest.raises(ValueError, match="norm_scramble_logits must have shape"):
        evaluator.update(**invalid)

    evaluator.update(**batch)
    result = evaluator.compute()
    assert result.batches == 1
    assert result.gate_zero.dependency_tokens == int(batch["dependency_mask"].sum())


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
    before = evaluator.compute().to_dict()
    invalid = dict(batch)
    invalid["mutation_dependency_mask"] = mutation_mask

    with pytest.raises(ValueError, match=message):
        evaluator.update(**invalid)

    after = evaluator.compute().to_dict()
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
        PairedCausalEvaluator().compute()

    batch = _batch()
    batch["nondependency_mask"].zero_()
    evaluator = PairedCausalEvaluator()
    evaluator.update(**batch)
    with pytest.raises(RuntimeError, match="zero nondependency tokens"):
        evaluator.compute()
