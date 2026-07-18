"""Physical three-row packed training and causal-ablation plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from pdt.config.schemas import (
    PlanMemoryProjectionConfig,
    SpeculationHeadConfig,
)
from pdt.sidecar.heads.plan_memory import PlanMemoryProjection
from pdt.sidecar.heads.speculation import SpeculationHead
from pdt.training.dataset import SampleBatch
from pdt.training.trainer import PDTTrainer, _StepOutput


class _Cache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class _FakeTrunkAdapter:
    def __init__(self) -> None:
        self.embedding = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32)
        self.calls: list[tuple[int, int]] = []

    def frozen_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight]

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        self.calls.append(tuple(input_ids.shape))
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        cache = _Cache(attention_mask.size(1))
        hidden_states = (hidden,) if kwargs.get("output_hidden_states") else None
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=cache,
        )


class _FakePhysicalDecoder:
    def __init__(self) -> None:
        self.active = False

    def begin_compute_session(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        assert device.type == "cpu"
        assert dtype == torch.float32
        assert not self.active
        self.active = True

    def end_compute_session(self) -> None:
        assert self.active
        self.active = False


def _batch() -> SampleBatch:
    target = torch.arange(1, 1 + 3 * 2 * 32).reshape(1, 3, 2, 32) % 31
    target = target.long()
    target_mask = torch.ones_like(target, dtype=torch.bool)
    dependency = torch.zeros(1, 3, 32, 32, dtype=torch.bool)
    dependency[:, :, 1, :5] = True
    return SampleBatch(
        example_ids=["packed-example"],
        planner_prompt_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        planner_prompt_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        target_block_ids=target,
        target_block_labels=target.clone(),
        target_block_attention_mask=target_mask,
        fact_embeddings=torch.randn(1, 24, 12),
        fact_mask=torch.ones(1, 24, dtype=torch.bool),
        positive_fact_mask=torch.tensor(
            [[True] * 12 + [False] * 12],
            dtype=torch.bool,
        ),
        plan_semantic_targets=torch.randn(1, 3, 8, 4),
        plan_node_mask=torch.ones(1, 3, 8, dtype=torch.bool),
        fact_route_targets=torch.zeros(1, 3, 8, 24),
        outline_progress_targets=torch.zeros(1, 3, 32, dtype=torch.long),
        fact_write_targets=torch.full(
            (1, 3, 32, 24),
            2,
            dtype=torch.long,
        ),
        dependency_token_mask=dependency,
        presentation_rank_targets=torch.tensor([[0, 1, 2]]),
        raw=[],
    )


def _trainer() -> tuple[PDTTrainer, _FakeTrunkAdapter]:
    trunk = _FakeTrunkAdapter()
    sidecar = SimpleNamespace(
        plan_memory_proj=PlanMemoryProjection(
            PlanMemoryProjectionConfig(planner_width=4, notes_dim=4)
        ),
        speculation_head=SpeculationHead(
            SpeculationHeadConfig(
                hidden_size=8,
                notes_dim=4,
                num_codebooks=2,
                codes_per_codebook=4,
                dropout=0.0,
            )
        ),
    )
    trainer = object.__new__(PDTTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        instrumentation=SimpleNamespace(coordination_source="bus"),
        runtime=SimpleNamespace(
            block_size=32,
            streams=("stream_0", "stream_1", "stream_2"),
            notes_bus=SimpleNamespace(history_blocks=16),
        ),
        sidecar=SimpleNamespace(
            notes_dim=4,
            snc=SimpleNamespace(hidden_size=8),
        ),
    )
    trainer.model = SimpleNamespace(
        trunk_adapter=trunk,
        forward_frontier=trunk.forward,
        set_runtime_context=lambda _context: None,
        sidecar=sidecar,
        instrumented_layers=[],
        physical_decoder=_FakePhysicalDecoder(),
    )
    return trainer, trunk


def test_training_uses_one_three_row_call_per_synchronized_block() -> None:
    trainer, trunk = _trainer()
    batch = _batch()
    rollout = trainer._packed_rollout(
        batch,
        plan_nodes=batch.plan_semantic_targets,
        plan_mask=batch.plan_node_mask,
    )

    assert trunk.calls == [(3, 4), (3, 32), (3, 32)]
    assert rollout.block_hidden.shape == (1, 3, 2, 8)
    assert rollout.block_token_nll.shape == (1, 3, 2, 32)
    assert rollout.block_token_ce_sum.shape == (1, 3, 2)
    assert rollout.block_token_count.tolist() == [[[32, 32], [32, 32], [32, 32]]]
    assert torch.isfinite(rollout.lm_ce)
    assert rollout.note_queries.shape == (6, 4)


def test_dynamic_ablation_preserves_token_alignment_and_reports_targeted_effect() -> None:
    trainer, _ = _trainer()
    batch = _batch()
    normal = trainer._packed_rollout(
        batch,
        plan_nodes=batch.plan_semantic_targets,
        plan_mask=batch.plan_node_mask,
    )
    step_output = _StepOutput(
        losses=None,  # type: ignore[arg-type]
        rollout=normal,
        plan_nodes=batch.plan_semantic_targets,
        plan_mask=batch.plan_node_mask,
    )
    effect = trainer._dynamic_note_causal_effect(
        batch,
        step_output=step_output,
    )

    assert effect is not None
    assert effect.dependency_tokens == 3 * 5
    assert effect.nondependency_tokens == 3 * 2 * 32 - 3 * 5
    assert effect.dependency_delta_nats == 0.0
    assert effect.nondependency_delta_nats == 0.0
