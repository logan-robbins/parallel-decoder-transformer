"""Contracts for same-trunk privileged-context functional distillation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import pdt.training.trainer as trainer_module
from pdt.config.schemas import LossWeights, PDTConfig, TrainingConfig, TrunkConfig
from pdt.training.dataset import SampleBatch
from pdt.training.losses import compute_pdt_losses
from pdt.training.trainer import (
    PDTTrainer,
    _compact_single_prompt,
    _last_valid_hidden,
    _privileged_teacher_block_input,
    _require_cache_compatible_trunk,
    _validate_block_transitions,
    _validate_fixed_tau_blocks,
)


def _zero_aux_weights(*, kd_lm: float, lm_ce: float = 0.0) -> LossWeights:
    return LossWeights(
        lm_ce=lm_ce,
        kd_lm=kd_lm,
        vq_commit=0.0,
        vq_codebook=0.0,
        codebook_usage=0.0,
        stream_classifier=0.0,
    )


def _batch() -> SimpleNamespace:
    # B=1, K=2, M=2, T=3. Padding is deliberately present in every prefix.
    block_ids = torch.tensor([[[[30, 31, 0], [32, 33, 0]], [[40, 0, 0], [41, 42, 0]]]])
    block_mask = torch.tensor([[[[1, 1, 0], [1, 1, 0]], [[1, 0, 0], [1, 1, 0]]]])
    return SimpleNamespace(
        planner_prompt_ids=torch.tensor([[5, 6, 0]]),
        planner_prompt_attention_mask=torch.tensor([[1, 1, 0]]),
        stream_prompt_ids=torch.tensor([[[10, 11, 0], [20, 21, 0]]]),
        stream_prompt_attention_mask=torch.tensor([[[1, 1, 0], [1, 1, 0]]]),
        block_transition_ids=torch.tensor([[[[0, 0], [90, 91]], [[0, 0], [92, 93]]]]),
        block_transition_attention_mask=torch.tensor([[[[0, 0], [1, 1]], [[0, 0], [1, 1]]]]),
        teacher_block_prompt_ids=torch.tensor([[[1, 2, 3, 0, 0, 0], [1, 2, 3, 30, 31, 40]]]),
        teacher_block_prompt_attention_mask=torch.tensor(
            [[[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]]
        ),
        target_block_ids=block_ids,
        target_block_attention_mask=block_mask,
    )


def test_student_prefill_compacts_padding_for_incremental_cache():
    prompt_ids, prompt_mask = _compact_single_prompt(
        _batch().stream_prompt_ids[:, 0],
        _batch().stream_prompt_attention_mask[:, 0],
    )

    assert prompt_ids.tolist() == [[10, 11]]
    assert prompt_mask.tolist() == [[1, 1]]

    with pytest.raises(ValueError, match="at least one prompt token"):
        _compact_single_prompt(torch.tensor([[0, 0]]), torch.tensor([[0, 0]]))


def test_cached_training_rejects_hf_train_or_gradient_checkpointing_mode():
    model = SimpleNamespace(training=False, is_gradient_checkpointing=False)
    adapter = SimpleNamespace(model=model)
    _require_cache_compatible_trunk(adapter)

    model.training = True
    with pytest.raises(RuntimeError, match="frozen trunk in eval mode"):
        _require_cache_compatible_trunk(adapter)

    model.training = False
    model.is_gradient_checkpointing = True
    with pytest.raises(RuntimeError, match="incompatible with gradient checkpointing"):
        _require_cache_compatible_trunk(adapter)


def test_privileged_teacher_uses_faithful_retokenized_block_prefix():
    packed = _privileged_teacher_block_input(
        _batch(),
        stream_idx=1,
        block_idx=1,
        pad_token_id=0,
    )

    # The prefix was retokenized as a complete chat transcript. The trainer
    # must not reconstruct it by concatenating independently tokenized blocks.
    assert packed.input_ids.tolist() == [[1, 2, 3, 30, 31, 40, 41, 42]]
    assert packed.prediction_positions.tolist() == [[5, 6, 0]]


def test_functional_teacher_disables_all_instrumented_contexts():
    class FakeLayer:
        def __init__(self) -> None:
            self.context = object()

        def set_runtime_context(self, context) -> None:
            self.context = context

    layers = [FakeLayer(), FakeLayer()]

    class FakeTrunk:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def forward(self, *, input_ids, attention_mask, **kwargs):
            assert kwargs["output_hidden_states"] is False
            assert all(layer.context is None for layer in layers)
            self.calls.append(input_ids.detach().clone())
            logits = F.one_hot(input_ids.remainder(7), num_classes=7).float()
            return SimpleNamespace(logits=logits)

    trunk = FakeTrunk()
    trainer = object.__new__(PDTTrainer)
    trainer.model = SimpleNamespace(
        instrumented_layers=layers,
        trunk_adapter=trunk,
    )
    trainer.pad_token_id = 0

    logits = trainer._functional_teacher_logits(_batch())

    assert logits.shape == (4, 3, 7)
    assert logits.requires_grad is False
    assert len(trunk.calls) == 4
    assert all(layer.context is None for layer in layers)
    assert trunk.calls[3].tolist() == [[1, 2, 3, 30, 31, 40, 41, 42]]


def test_context_kd_is_temperature_two_dependency_only_and_teacher_detached():
    student_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[[-1.0, 2.0, 0.5], [3.0, -2.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[1, 2]])
    mask = torch.tensor([[True, True]])
    dependency = torch.tensor([[True, False]])

    losses = compute_pdt_losses(
        stage=0,
        weights=_zero_aux_weights(kd_lm=1.0),
        lm_logits=student_logits,
        lm_labels=labels,
        lm_label_mask=mask,
        dependency_mask=dependency,
        lm_teacher_logits=teacher_logits,
        kd_temperature_lm=2.0,
    )
    expected = (
        F.kl_div(
            F.log_softmax(student_logits[0, 0] / 2.0, dim=-1),
            F.softmax(teacher_logits[0, 0].detach() / 2.0, dim=-1),
            reduction="sum",
        )
        * 4.0
    )
    assert torch.allclose(losses.kd_lm, expected)

    losses.total.backward()
    assert student_logits.grad is not None
    assert torch.count_nonzero(student_logits.grad[0, 0]) > 0
    assert torch.count_nonzero(student_logits.grad[0, 1]) == 0
    assert teacher_logits.grad is None


def test_weighted_kd_fails_if_teacher_is_missing_or_temperature_drifts():
    kwargs = {
        "stage": 0,
        "weights": _zero_aux_weights(kd_lm=1.0),
        "lm_logits": torch.zeros(1, 1, 3),
        "lm_labels": torch.zeros(1, 1, dtype=torch.long),
        "lm_label_mask": torch.ones(1, 1, dtype=torch.bool),
        "dependency_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="lm_teacher_logits"):
        compute_pdt_losses(**kwargs)
    with pytest.raises(ValueError, match="temperature is 2.0"):
        compute_pdt_losses(
            **kwargs,
            lm_teacher_logits=torch.zeros(1, 1, 3),
            kd_temperature_lm=1.0,
        )


def test_bfloat16_trunk_logits_use_float32_ce_and_kd_reductions():
    student = torch.randn(1, 2, 5, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(1, 2, 5, dtype=torch.bfloat16)
    active = torch.ones((1, 2), dtype=torch.bool)
    losses = compute_pdt_losses(
        stage=0,
        weights=_zero_aux_weights(lm_ce=1.0, kd_lm=1.0),
        lm_logits=student,
        lm_labels=torch.tensor([[1, 2]]),
        lm_label_mask=active,
        dependency_mask=active,
        nondependency_mask=torch.zeros_like(active),
        lm_teacher_logits=teacher,
        kd_temperature_lm=2.0,
    )

    assert losses.lm_ce.dtype == torch.float32
    assert losses.kd_lm.dtype == torch.float32
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_lm_masks_are_boolean_disjoint_subsets_and_kd_cannot_be_empty():
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    labels = torch.tensor([[0, 0]])
    # Integer attention masks must be interpreted as booleans, never as row
    # indices into flattened logits.
    losses = compute_pdt_losses(
        stage=0,
        weights=_zero_aux_weights(kd_lm=0.0),
        lm_logits=logits,
        lm_labels=labels,
        lm_label_mask=torch.tensor([[1, 0]], dtype=torch.long),
        dependency_mask=torch.tensor([[False, False]]),
        nondependency_mask=torch.tensor([[True, False]]),
    )
    expected = F.cross_entropy(logits[:, :1].reshape(-1, 2), labels[:, :1].reshape(-1))
    assert torch.allclose(losses.lm_ce, expected)

    with pytest.raises(ValueError, match="must be disjoint"):
        compute_pdt_losses(
            stage=0,
            weights=_zero_aux_weights(kd_lm=0.0),
            lm_logits=logits,
            lm_labels=labels,
            lm_label_mask=torch.tensor([[True, True]]),
            dependency_mask=torch.tensor([[True, False]]),
            nondependency_mask=torch.tensor([[True, True]]),
        )
    with pytest.raises(ValueError, match="subset"):
        compute_pdt_losses(
            stage=0,
            weights=_zero_aux_weights(kd_lm=0.0),
            lm_logits=logits,
            lm_labels=labels,
            lm_label_mask=torch.tensor([[True, False]]),
            dependency_mask=torch.tensor([[False, True]]),
        )
    with pytest.raises(ValueError, match="active dependency token"):
        compute_pdt_losses(
            stage=0,
            weights=_zero_aux_weights(kd_lm=1.0),
            lm_logits=logits,
            lm_labels=labels,
            lm_label_mask=torch.tensor([[True, True]]),
            dependency_mask=torch.tensor([[False, False]]),
            lm_teacher_logits=logits.clone(),
            kd_temperature_lm=2.0,
        )


def test_bus_write_selects_each_rows_last_valid_generated_state():
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    selected = _last_valid_hidden(hidden, mask)
    assert torch.equal(selected, torch.stack((hidden[0, 1], hidden[1, 2])))

    with pytest.raises(ValueError, match="empty target rows"):
        _last_valid_hidden(hidden, torch.tensor([[False, False, False], [True, False, False]]))


def test_training_blocks_must_match_runtime_tau_exactly():
    batch = _batch()
    batch.target_block_attention_mask.fill_(1)
    _validate_fixed_tau_blocks(batch, tau=3)

    batch.target_block_attention_mask[0, 0, 1, -1] = 0
    with pytest.raises(ValueError, match="exactly tau valid tokens"):
        _validate_fixed_tau_blocks(batch, tau=3)
    with pytest.raises(ValueError, match="target width=3, tau=4"):
        _validate_fixed_tau_blocks(batch, tau=4)


def test_block_transitions_reveal_private_state_only_after_prefill():
    batch = _batch()
    _validate_block_transitions(batch)

    batch.block_transition_attention_mask[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="row 0 must be empty"):
        _validate_block_transitions(batch)

    batch = _batch()
    batch.block_transition_attention_mask[0, 1, 1].zero_()
    with pytest.raises(ValueError, match="must reveal a non-empty"):
        _validate_block_transitions(batch)


def test_student_rollout_prefills_once_and_consumes_each_block_once_with_grad_cache(
    monkeypatch,
):
    class FakeLayer:
        def __init__(self) -> None:
            self.context = None

        def set_runtime_context(self, context) -> None:
            self.context = context

    layer = FakeLayer()

    class FakeTrunk:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.prefill_caches: list[torch.Tensor] = []
            self.last_token = -1

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            past_key_values=None,
            use_cache,
            output_hidden_states,
        ):
            context = layer.context
            context_value = input_ids.new_zeros((input_ids.size(0), 1)).float()
            context_mask = None
            stream = None
            if context is not None:
                context_value = (
                    (context.notes * context.notes_mask.unsqueeze(-1).to(context.notes))
                    .sum(dim=(1, 2), keepdim=False)
                    .unsqueeze(-1)
                )
                context_mask = context.notes_mask.detach().clone()
                stream = context.stream
            prior = (
                input_ids.new_zeros((input_ids.size(0), 1)).float()
                if past_key_values is None
                else past_key_values
            )
            running = (
                prior.unsqueeze(1)
                + input_ids.float().cumsum(dim=1).unsqueeze(-1)
                + context_value.unsqueeze(1)
            )
            cache = running[:, -1]
            if use_cache and past_key_values is None and input_ids.size(1) > 1:
                cache.retain_grad()
                self.prefill_caches.append(cache)
            hidden = input_ids.float().unsqueeze(-1) + running
            vocab_axis = torch.linspace(-0.5, 0.5, 12, device=input_ids.device)
            logits = hidden * vocab_axis.view(1, 1, -1)
            self.last_token = int(input_ids[0, -1])
            self.calls.append(
                {
                    "ids": input_ids.detach().clone(),
                    "attention_length": attention_mask.size(1),
                    "has_past": past_key_values is not None,
                    "use_cache": use_cache,
                    "stream": stream,
                    "context_mask": context_mask,
                    "logits": logits.detach().clone(),
                }
            )
            return SimpleNamespace(
                logits=logits,
                hidden_states=[hidden] if output_hidden_states else None,
                past_key_values=cache if use_cache else None,
            )

    class FakePlanner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchors = nn.Parameter(torch.tensor([[[0.25], [0.5]]]))

        def forward(self, hidden, *, attention_mask):
            del attention_mask
            batch_size = hidden.size(0)
            quantized = self.anchors.expand(batch_size, -1, -1)
            zero = quantized.sum() * 0.0
            return SimpleNamespace(
                quantized=quantized,
                indices=torch.zeros((batch_size, 2), dtype=torch.long),
                logits=torch.zeros((batch_size, 2, 3)),
                commitment_loss=zero,
                codebook_loss=zero,
            )

    class FakePlanProjection(nn.Module):
        def forward(self, quantized, ownership):
            return torch.einsum("bks,bsh->bkh", ownership.to(quantized), quantized)

    class FakeSpeculation(nn.Module):
        def __init__(self, trunk) -> None:
            super().__init__()
            self.trunk = trunk
            self.final_tokens: list[int] = []

        def forward(self, hidden):
            self.final_tokens.append(self.trunk.last_token)
            return hidden

    class FakeClassifier(nn.Module):
        def forward(self, hidden):
            return torch.cat((-hidden, hidden), dim=-1)

    trunk = FakeTrunk()
    speculation = FakeSpeculation(trunk)
    sidecar = SimpleNamespace(
        planner_head=FakePlanner(),
        plan_notes_proj=FakePlanProjection(),
        speculation_head=speculation,
        stream_classifier=FakeClassifier(),
    )
    trainer = object.__new__(PDTTrainer)
    trainer.model = SimpleNamespace(
        trunk_adapter=SimpleNamespace(forward=trunk.forward),
        instrumented_layers=[layer],
        sidecar=sidecar,
    )
    trainer.device = torch.device("cpu")
    trainer.pad_token_id = 0
    trainer.config = SimpleNamespace(
        runtime=SimpleNamespace(
            streams=("stream_0", "stream_1"),
            block_size=2,
            notes_bus=SimpleNamespace(lag=1),
        ),
        sidecar=SimpleNamespace(planner_head=SimpleNamespace(num_slots=2)),
        training=SimpleNamespace(grad_accumulation=1, kd_temperature_lm=2.0),
    )
    weights = LossWeights(
        lm_ce=1.0,
        kd_lm=0.0,
        vq_commit=0.0,
        vq_codebook=0.0,
        codebook_usage=0.0,
        stream_classifier=0.0,
    )
    trainer.curriculum = SimpleNamespace(active_loss_weights=lambda stage: weights)
    trainer.codebook = SimpleNamespace(
        observe_selections=lambda value: None,
        observe_anchors=lambda value: None,
    )
    trainer._functional_teacher_logits = lambda batch: torch.zeros(4, 2, 12)
    captured_loss_inputs: dict[str, torch.Tensor] = {}

    def capture_loss_inputs(**kwargs):
        captured_loss_inputs["lm_logits"] = kwargs["lm_logits"].detach().clone()
        return compute_pdt_losses(**kwargs)

    monkeypatch.setattr(trainer_module, "compute_pdt_losses", capture_loss_inputs)

    targets = torch.tensor([[[[1, 2], [5, 6]], [[3, 4], [7, 8]]]])
    active = torch.ones_like(targets)
    batch = SampleBatch(
        example_ids=["cache-contract"],
        families=["test"],
        stream_labels=[["stream_0", "stream_1"]],
        planner_prompt_ids=torch.tensor([[5, 6, 0]]),
        planner_prompt_attention_mask=torch.tensor([[1, 1, 0]]),
        stream_prompt_ids=torch.tensor([[[10, 11, 0], [20, 21, 0]]]),
        stream_prompt_attention_mask=torch.tensor([[[1, 1, 0], [1, 1, 0]]]),
        block_transition_ids=torch.tensor([[[[0, 0], [9, 10]], [[0, 0], [9, 10]]]]),
        block_transition_attention_mask=torch.tensor([[[[0, 0], [1, 1]], [[0, 0], [1, 1]]]]),
        teacher_block_prompt_ids=torch.tensor([[[30, 31], [32, 33]]]),
        teacher_block_prompt_attention_mask=torch.ones(1, 2, 2, dtype=torch.long),
        target_block_ids=targets,
        target_block_labels=targets.clone(),
        target_block_attention_mask=active,
        dependency_token_mask=torch.zeros_like(targets, dtype=torch.bool),
        nondependency_token_mask=torch.ones_like(targets, dtype=torch.bool),
        raw=[{}],
    )

    trainer._train_step(batch, stage=0)

    assert [call["ids"].tolist() for call in trunk.calls] == [
        [[5, 6, 0]],
        [[10, 11]],
        [[20, 21]],
        [[1, 2]],
        [[3, 4]],
        [[9, 10]],
        [[5, 6]],
        [[9, 10]],
        [[7, 8]],
    ]
    assert [call["attention_length"] for call in trunk.calls[3:]] == [
        4,
        4,
        6,
        8,
        6,
        8,
    ]
    assert all(call["has_past"] for call in trunk.calls[3:])
    assert speculation.final_tokens == [2, 4, 6, 8]
    assert [call["context_mask"].tolist() for call in trunk.calls[1:3]] == [
        [[True, True, False, False]],
        [[True, True, False, False]],
    ]
    assert all(
        call["context_mask"].tolist() == [[True, True, False, False]] for call in trunk.calls[3:5]
    )
    assert all(
        call["context_mask"].tolist() == [[True, True, True, True]] for call in trunk.calls[5:]
    )
    expected_logits = torch.cat(
        (
            torch.cat(
                (trunk.calls[1]["logits"][:, -1:], trunk.calls[3]["logits"][:, :-1]),
                dim=1,
            ),
            torch.cat(
                (trunk.calls[2]["logits"][:, -1:], trunk.calls[4]["logits"][:, :-1]),
                dim=1,
            ),
            torch.cat(
                (trunk.calls[5]["logits"][:, -1:], trunk.calls[6]["logits"][:, :-1]),
                dim=1,
            ),
            torch.cat(
                (trunk.calls[7]["logits"][:, -1:], trunk.calls[8]["logits"][:, :-1]),
                dim=1,
            ),
        ),
        dim=0,
    )
    assert torch.equal(captured_loss_inputs["lm_logits"], expected_logits)
    assert len(trunk.prefill_caches) == 2
    assert all(cache.grad is not None for cache in trunk.prefill_caches)
    assert all(torch.count_nonzero(cache.grad) > 0 for cache in trunk.prefill_caches)
    assert sidecar.planner_head.anchors.grad is not None
    assert torch.count_nonzero(sidecar.planner_head.anchors.grad) > 0


def test_canonical_trunk_revision_and_kd_temperature_are_locked():
    trunk = TrunkConfig()
    assert trunk.base_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert trunk.revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert TrainingConfig().kd_temperature_lm == 2.0

    config = PDTConfig(training=replace(TrainingConfig(), kd_temperature_lm=1.0))
    with pytest.raises(ValueError, match="must be 2.0"):
        config.validate()

    config = PDTConfig(training=replace(TrainingConfig(), batch_size=2))
    with pytest.raises(ValueError, match="batch_size must be 1"):
        config.validate()


def test_positive_stream_classifier_weight_cannot_silently_disable_itself():
    weights = _zero_aux_weights(kd_lm=0.0)
    with pytest.raises(ValueError, match="stream_classifier loss weight"):
        compute_pdt_losses(
            stage=0,
            weights=replace(weights, stream_classifier=1.0),
        )
