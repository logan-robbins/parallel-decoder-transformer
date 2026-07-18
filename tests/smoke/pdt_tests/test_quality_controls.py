"""Frozen-trunk blind and sequential-oracle scoring contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from pdt.evaluation import quality_controls


class _CausalMarkerModel(nn.Module):
    """Tiny causal scorer that resolves token four only after oracle marker seven."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.register_buffer("anchor", torch.zeros(()))
        self.eval()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        assert position_ids.shape == input_ids.shape
        assert use_cache is False
        logits = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        for row in range(input_ids.size(0)):
            has_oracle_marker = bool((input_ids[row] == 7).any())
            if not has_oracle_marker:
                continue
            for position in range(input_ids.size(1) - 1):
                if int(input_ids[row, position + 1]) == 4:
                    logits[row, position, 4] = 4.0
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


def _record() -> dict[str, object]:
    return {
        "example_id": "document-0",
        "entropy_accounting": {"rho": 1.0},
        "stream_inputs": [
            {
                "stream_prompt_ids": [1],
                "target_block_ids": [[2, 3], [4, 5]],
                "block_transition_ids": [[], [6]],
                "full_text_oracle_block_prompt_ids": [[1], [1, 7]],
                "dependency_token_mask": [[False, False], [True, False]],
            }
        ],
    }


def test_scorer_accepts_local_blocks_without_dependency_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quality_controls,
        "validate_retokenized_record",
        lambda *args, **kwargs: None,
    )
    result = quality_controls.score_quality_control_records(
        [_record()],
        _CausalMarkerModel(),
        device=torch.device("cpu"),
        pad_token_id=0,
        batch_size=2,
        expected_tokenizer="test",
        expected_tokenizer_revision="revision",
        bootstrap_samples=1000,
        minimum_documents=2,
    )

    assert result.blind.dependency_tokens == 1
    assert result.scoring_tasks == 4
    assert result.maximum_sequence_tokens == 6
    assert result.blind.nondependency_tokens == 3
    assert result.oracle_dependency_advantage.mean > 0.0
    assert result.oracle_nondependency_advantage.mean == pytest.approx(0.0)
    assert result.enough_documents is False
    assert result.expected_outcome_passes is False


def test_scorer_rejects_a_trainable_or_training_trunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quality_controls,
        "validate_retokenized_record",
        lambda *args, **kwargs: None,
    )
    training = _CausalMarkerModel().train()
    with pytest.raises(ValueError, match="eval mode"):
        quality_controls.score_quality_control_records(
            [_record()],
            training,
            device=torch.device("cpu"),
            pad_token_id=0,
            batch_size=1,
            expected_tokenizer="test",
            expected_tokenizer_revision="revision",
        )

    trainable = _CausalMarkerModel()
    trainable.parameter = nn.Parameter(torch.zeros(()))
    with pytest.raises(ValueError, match="every trunk parameter"):
        quality_controls.score_quality_control_records(
            [_record()],
            trainable,
            device=torch.device("cpu"),
            pad_token_id=0,
            batch_size=1,
            expected_tokenizer="test",
            expected_tokenizer_revision="revision",
        )


def test_scorer_uses_the_real_qwen3_limited_logit_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        quality_controls,
        "validate_retokenized_record",
        lambda *args, **kwargs: None,
    )
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    model.requires_grad_(False)
    result = quality_controls.score_quality_control_records(
        [_record()],
        model,
        device=torch.device("cpu"),
        pad_token_id=0,
        batch_size=2,
        expected_tokenizer="test",
        expected_tokenizer_revision="revision",
        bootstrap_samples=1000,
        minimum_documents=2,
    )

    assert result.scoring_tasks == 4
    assert result.blind.dependency_tokens == 1

    tasks = quality_controls._record_tasks(_record(), document_index=0)
    limited_rows = quality_controls._score_task_batch(
        tasks,
        model,
        device=torch.device("cpu"),
        pad_token_id=0,
    )
    for task, limited in zip(tasks, limited_rows, strict=True):
        sequence = torch.tensor([task.context_ids + task.target_ids])
        full_logits = model(input_ids=sequence, use_cache=False).logits
        start = len(task.context_ids) - 1
        expected = F.cross_entropy(
            full_logits[0, start : start + len(task.target_ids)].float(),
            torch.tensor(task.target_ids),
            reduction="none",
        )
        assert torch.allclose(limited, expected, atol=1e-6, rtol=1e-6)
