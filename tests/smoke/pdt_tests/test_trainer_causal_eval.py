"""Canonical cached-rollout causal evaluation integration contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pdt.config import load_config
from pdt.training.dataset import SampleBatch
from pdt.training.trainer import (
    PDTTrainer,
    _RolloutIntervention,
    _StudentRollout,
    _mutation_dependency_mask,
)


def _batch() -> SampleBatch:
    targets = torch.tensor(
        [
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ]
        ]
    )
    active = torch.ones_like(targets)
    dependency = torch.zeros_like(targets, dtype=torch.bool)
    dependency[:, :, 1, 0] = True
    nondependency = active.bool() & ~dependency
    streams = []
    for receiver in range(3):
        source = (receiver + 1) % 3
        streams.append(
            {
                "stream_id": f"stream_{receiver}",
                "dependency_spans": [
                    {
                        "block_index": 1,
                        "source_stream": f"stream_{source}",
                        "source_block_index": 0,
                    }
                ],
            }
        )
    return SampleBatch(
        example_ids=["causal-row"],
        families=["test"],
        stream_labels=[["stream_0", "stream_1", "stream_2"]],
        planner_prompt_ids=torch.tensor([[50, 51]]),
        planner_prompt_attention_mask=torch.ones(1, 2, dtype=torch.long),
        stream_prompt_ids=torch.tensor([[[20, 21], [30, 31], [40, 41]]]),
        stream_prompt_attention_mask=torch.ones(1, 3, 2, dtype=torch.long),
        block_transition_ids=torch.tensor([[[[0], [90]], [[0], [91]], [[0], [92]]]]),
        block_transition_attention_mask=torch.tensor([[[[0], [1]], [[0], [1]], [[0], [1]]]]),
        teacher_block_prompt_ids=torch.ones(1, 2, 2, dtype=torch.long),
        teacher_block_prompt_attention_mask=torch.ones(1, 2, 2, dtype=torch.long),
        target_block_ids=targets,
        target_block_labels=targets.clone(),
        target_block_attention_mask=active,
        dependency_token_mask=dependency,
        nondependency_token_mask=nondependency,
        raw=[{"stream_inputs": streams}],
    )


class _FakeLayer:
    def __init__(self) -> None:
        self.context = None

    def set_runtime_context(self, context) -> None:
        self.context = context


class _FakeTrunk:
    def __init__(self, layer: _FakeLayer) -> None:
        self.layer = layer
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache,
        output_hidden_states,
    ):
        context = self.layer.context
        notes = None if context is None else context.notes.detach().clone()
        force_gate = None if context is None else context.snc_force_gate
        context_value = input_ids.new_zeros((input_ids.size(0), 1)).float()
        if notes is not None and force_gate is not False:
            mask = context.notes_mask.to(notes).unsqueeze(-1)
            context_value = (notes * mask).sum(dim=(1, 2)).unsqueeze(-1)
        prior = (
            input_ids.new_zeros((input_ids.size(0), 1)).float()
            if past_key_values is None
            else past_key_values
        )
        running = prior.unsqueeze(1) + input_ids.float().cumsum(dim=1).unsqueeze(-1)
        running = running + context_value.unsqueeze(1)
        hidden = torch.cat((input_ids.float().unsqueeze(-1), running), dim=-1)
        vocab_axis = torch.linspace(-0.4, 0.4, 128)
        logits = hidden.sum(dim=-1, keepdim=True) * vocab_axis.view(1, 1, -1)
        self.calls.append(
            {
                "ids": input_ids.detach().clone(),
                "notes": notes,
                "force_gate": force_gate,
                "stream": None if context is None else context.stream,
                "attention_length": attention_mask.size(1),
            }
        )
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden] if output_hidden_states else None,
            past_key_values=running[:, -1] if use_cache else None,
        )


class _FakePlanner(nn.Module):
    def forward(self, hidden, *, attention_mask):
        del attention_mask
        batch = hidden.size(0)
        quantized = hidden.new_tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]])
        quantized = quantized.expand(batch, -1, -1)
        zero = hidden.sum() * 0.0
        return SimpleNamespace(
            quantized=quantized,
            indices=torch.zeros((batch, 3), dtype=torch.long),
            logits=torch.zeros((batch, 3, 4)),
            commitment_loss=zero,
            codebook_loss=zero,
        )


class _FakePlanProjection(nn.Module):
    def forward(self, quantized, ownership):
        return torch.einsum("bks,bsh->bkh", ownership.to(quantized), quantized)


class _FakeSpeculation(nn.Module):
    def forward(self, hidden):
        return hidden


def _rollout_trainer() -> tuple[PDTTrainer, _FakeTrunk]:
    layer = _FakeLayer()
    trunk = _FakeTrunk(layer)
    trainer = object.__new__(PDTTrainer)
    trainer.model = SimpleNamespace(
        trunk_adapter=SimpleNamespace(forward=trunk.forward),
        instrumented_layers=[layer],
        sidecar=SimpleNamespace(
            planner_head=_FakePlanner(),
            plan_notes_proj=_FakePlanProjection(),
            speculation_head=_FakeSpeculation(),
        ),
    )
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        runtime=SimpleNamespace(
            streams=("stream_0", "stream_1", "stream_2"),
            block_size=2,
            notes_bus=SimpleNamespace(lag=1),
        ),
        sidecar=SimpleNamespace(planner_head=SimpleNamespace(num_slots=3)),
    )
    return trainer, trunk


def _run(
    trainer: PDTTrainer,
    trunk: _FakeTrunk,
    intervention: _RolloutIntervention,
) -> tuple[_StudentRollout, list[dict[str, object]]]:
    trunk.calls.clear()
    result = trainer._student_rollout(_batch(), intervention=intervention)
    return result, list(trunk.calls)


def test_all_interventions_share_call_order_and_block_major_alignment() -> None:
    trainer, trunk = _rollout_trainer()
    conditions = (
        _RolloutIntervention(),
        _RolloutIntervention(mode="gate_zero"),
        _RolloutIntervention(mode="norm_scramble", seed=19),
        _RolloutIntervention(
            mode="bus_mutation",
            mutation_producer="stream_0",
            mutation_block=0,
            mutation_magnitude=5.0,
        ),
    )
    runs = [_run(trainer, trunk, condition) for condition in conditions]
    expected_ids = [call["ids"].tolist() for call in runs[0][1]]
    for result, calls in runs[1:]:
        assert [call["ids"].tolist() for call in calls] == expected_ids
        assert torch.equal(result.lm_labels, runs[0][0].lm_labels)
        assert torch.equal(result.lm_label_mask, runs[0][0].lm_label_mask)
        assert torch.equal(result.dependency_mask, runs[0][0].dependency_mask)
    assert runs[0][0].lm_labels.tolist() == [
        [1, 2],
        [5, 6],
        [9, 10],
        [3, 4],
        [7, 8],
        [11, 12],
    ]
    assert all(call["force_gate"] is False for call in runs[1][1][1:])


def test_scramble_is_seeded_and_changes_only_sibling_dynamic_slots() -> None:
    trainer, trunk = _rollout_trainer()
    baseline, baseline_calls = _run(trainer, trunk, _RolloutIntervention())
    first, first_calls = _run(
        trainer,
        trunk,
        _RolloutIntervention(mode="norm_scramble", seed=23),
    )
    second, second_calls = _run(
        trainer,
        trunk,
        _RolloutIntervention(mode="norm_scramble", seed=23),
    )
    changed, changed_calls = _run(
        trainer,
        trunk,
        _RolloutIntervention(mode="norm_scramble", seed=24),
    )
    assert torch.equal(first.lm_logits, second.lm_logits)
    assert not torch.equal(first.lm_logits, changed.lm_logits)

    # Calls 7, 9, and 11 are the block-1 transition calls for streams 0, 1, 2.
    for stream_idx, call_idx in enumerate((7, 9, 11)):
        base_notes = baseline_calls[call_idx]["notes"]
        scrambled = first_calls[call_idx]["notes"]
        repeated = second_calls[call_idx]["notes"]
        other_seed = changed_calls[call_idx]["notes"]
        assert torch.equal(scrambled, repeated)
        assert torch.equal(scrambled[:, :3], base_notes[:, :3])
        assert torch.equal(scrambled[:, 3 + stream_idx], base_notes[:, 3 + stream_idx])
        sibling = torch.tensor([idx != stream_idx for idx in range(3)])
        assert torch.allclose(
            torch.linalg.vector_norm(scrambled[:, 3:][:, sibling], dim=-1),
            torch.linalg.vector_norm(base_notes[:, 3:][:, sibling], dim=-1),
        )
        assert not torch.equal(scrambled[:, 3:][:, sibling], other_seed[:, 3:][:, sibling])
    assert not torch.equal(baseline.lm_logits, first.lm_logits)


def test_targeted_mutation_changes_one_write_and_mask_selects_its_receivers() -> None:
    batch = _batch()
    mask = _mutation_dependency_mask(batch, producer="stream_0", source_block=0)
    expected = torch.zeros((6, 2), dtype=torch.bool)
    expected[5, 0] = True  # block 1, receiver stream_2, first dependency token.
    assert torch.equal(mask, expected)
    assert (
        bool(
            (
                mask
                & ~torch.cat(
                    [
                        batch.dependency_token_mask[:, stream_idx, block_idx]
                        for block_idx in range(2)
                        for stream_idx in range(3)
                    ],
                    dim=0,
                )
            ).any()
        )
        is False
    )

    trainer, trunk = _rollout_trainer()
    _, baseline_calls = _run(trainer, trunk, _RolloutIntervention())
    _, mutation_calls = _run(
        trainer,
        trunk,
        _RolloutIntervention(
            mode="bus_mutation",
            mutation_producer="stream_0",
            mutation_block=0,
            mutation_magnitude=5.0,
        ),
    )
    for call_idx in (7, 9, 11):
        baseline_note = baseline_calls[call_idx]["notes"][:, 3]
        mutation_note = mutation_calls[call_idx]["notes"][:, 3]
        assert torch.equal(mutation_note[:, 0], baseline_note[:, 0] + 5.0)
        assert torch.equal(mutation_note[:, 1:], baseline_note[:, 1:])


class _Stats:
    def to_dict(self):
        return {"perplexity": 1.0}

    def passes_stage0_gate(self):
        return False


class _Codebook:
    def __init__(self) -> None:
        self.reset_called = False

    def compute(self):
        return _Stats()

    def reset(self):
        self.reset_called = True


class _EvalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _FakeLayer()
        self.trunk_model = nn.Linear(1, 1).eval()
        self.trunk_adapter = SimpleNamespace(model=self.trunk_model)
        self.instrumented_layers = [self.layer]


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            tensor[:, stream_idx, block_idx]
            for block_idx in range(tensor.size(2))
            for stream_idx in range(tensor.size(1))
        ],
        dim=0,
    )


def test_eval_runs_four_aligned_conditions_and_writes_real_causal_telemetry(
    tmp_path,
) -> None:
    batch = _batch()
    trainer = object.__new__(PDTTrainer)
    trainer.model = _EvalModel().train()
    trainer.model.trunk_model.eval()
    trainer._eval_loader = [batch]
    trainer.device = torch.device("cpu")
    trainer.telemetry_dir = tmp_path
    trainer.global_step = 7
    trainer.codebook = _Codebook()
    trainer.curriculum = SimpleNamespace(
        current_stage=2,
        active_modules_snapshot=lambda: {"snc": True},
    )
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            causal_eval_seed=41,
            causal_eval_mutation_producer="stream_0",
            causal_eval_mutation_block=0,
            causal_eval_mutation_magnitude=2.0,
        )
    )
    modes: list[str] = []
    grad_enabled: list[bool] = []

    def fake_rollout(batch_value, *, intervention=None):
        mode = "normal" if intervention is None else intervention.mode
        modes.append(mode)
        grad_enabled.append(torch.is_grad_enabled())
        labels = _flat(batch_value.target_block_labels)
        label_mask = _flat(batch_value.target_block_attention_mask).bool()
        dependency = _flat(batch_value.dependency_token_mask)
        nondependency = _flat(batch_value.nondependency_token_mask)
        logits = torch.zeros((*labels.shape, 128))
        logits.scatter_(-1, labels.unsqueeze(-1), 2.0)
        if mode == "gate_zero":
            logits[dependency] = 0.0
        elif mode == "norm_scramble":
            logits[dependency, 0] = 1.0
        elif mode == "bus_mutation":
            logits[5, 0, 0] = 3.0
        return _StudentRollout(
            lm_logits=logits,
            lm_labels=labels,
            lm_label_mask=label_mask,
            dependency_mask=dependency,
            nondependency_mask=nondependency,
            classifier_hidden=torch.zeros(6, 2),
            planner=None,
            plan_snapshot=torch.zeros(1, 3, 2),
        )

    trainer._student_rollout = fake_rollout
    trainer.model.layer.context = object()
    trainer._eval()

    assert modes == ["normal", "gate_zero", "norm_scramble", "bus_mutation"]
    assert grad_enabled == [False] * 4
    assert trainer.model.training is True
    assert trainer.model.trunk_model.training is False
    assert trainer.model.layer.context is None
    assert trainer.codebook.reset_called is True
    telemetry = json.loads((tmp_path / "eval_0000007.json").read_text())
    assert telemetry["causal"]["batches"] == 1
    assert telemetry["causal"]["gate_zero"]["dependency_tokens"] == 3
    assert telemetry["causal"]["targeted_mutation"]["mutation_dependency_tokens"] == 1
    assert "passes" not in telemetry["causal"]


def test_eval_restores_modes_and_clears_contexts_when_a_rollout_fails(tmp_path) -> None:
    batch = _batch()
    trainer = object.__new__(PDTTrainer)
    trainer.model = _EvalModel().train()
    trainer.model.trunk_model.eval()
    trainer._eval_loader = [batch]
    trainer.device = torch.device("cpu")
    trainer.telemetry_dir = tmp_path
    trainer.global_step = 9
    trainer.codebook = _Codebook()
    trainer.curriculum = SimpleNamespace(current_stage=0)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            causal_eval_seed=1,
            causal_eval_mutation_producer="stream_0",
            causal_eval_mutation_block=0,
            causal_eval_mutation_magnitude=1.0,
        )
    )

    def failing_rollout(batch_value, *, intervention=None):
        if intervention is not None:
            trainer.model.layer.context = object()
            raise RuntimeError("injected causal rollout failure")
        labels = _flat(batch_value.target_block_labels)
        return _StudentRollout(
            lm_logits=torch.zeros((*labels.shape, 128)),
            lm_labels=labels,
            lm_label_mask=_flat(batch_value.target_block_attention_mask).bool(),
            dependency_mask=_flat(batch_value.dependency_token_mask),
            nondependency_mask=_flat(batch_value.nondependency_token_mask),
            classifier_hidden=torch.zeros(6, 2),
            planner=None,
            plan_snapshot=torch.zeros(1, 3, 2),
        )

    trainer._student_rollout = failing_rollout
    with pytest.raises(RuntimeError, match="injected causal rollout failure"):
        trainer._eval()

    assert trainer.model.training is True
    assert trainer.model.trunk_model.training is False
    assert trainer.model.layer.context is None
    assert not (tmp_path / "eval_0000009.json").exists()


def test_causal_eval_config_rejects_invalid_targeting() -> None:
    config = load_config("configs/pdt_qwen3_4b.yaml")
    config.training.causal_eval_seed = -1
    with pytest.raises(ValueError, match="causal_eval_seed"):
        config.validate()

    config = load_config("configs/pdt_qwen3_4b.yaml")
    config.training.causal_eval_mutation_producer = "missing"
    with pytest.raises(ValueError, match="mutation_producer"):
        config.validate()

    config = load_config("configs/pdt_qwen3_4b.yaml")
    config.training.causal_eval_mutation_block = 7
    with pytest.raises(ValueError, match="mutation_block"):
        config.validate()

    config = load_config("configs/pdt_qwen3_4b.yaml")
    config.training.causal_eval_mutation_magnitude = float("nan")
    with pytest.raises(ValueError, match="mutation_magnitude"):
        config.validate()
