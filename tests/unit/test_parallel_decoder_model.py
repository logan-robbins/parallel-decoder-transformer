from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from parallel_decoder_transformer.models import (
    ParallelDecoderModelConfig,
    ParallelDecoderTransformer,
    PlannerHeadConfig,
)


class FakeTrunk(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Minimal parameter so the adapter's freeze policy has something to touch.
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        hidden = torch.zeros(input_ids.size(0), input_ids.size(1), self.hidden_size)
        return SimpleNamespace(hidden_states=[hidden, hidden])


def test_parallel_decoder_transformer_forward_outputs() -> None:
    config = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(config)
    model.trunk_adapter.attach_model(FakeTrunk(config.hidden_size, config.vocab_size))

    input_ids = torch.ones(2, 3, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    hidden_states = model.encode(input_ids, attention_mask=attention_mask)
    notes = torch.zeros(2, 2, config.notes_dim)
    notes_mask = torch.ones(2, 2, dtype=torch.long)
    stream_ids = torch.tensor([0, 1])
    plan_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    plan_mask = torch.ones(2, 2, dtype=torch.bool)
    outputs = model(
        hidden_states,
        stream=stream_ids,
        notes=notes,
        notes_mask=notes_mask,
        attention_mask=attention_mask,
        plan_item_ids=plan_ids,
        plan_item_mask=plan_mask,
    )
    assert set(outputs.keys()) == {
        "planner_logits",
        "notes_logits",
        "speculative_notes",
        "agreement",
        "stream_logits",
        "coverage_logits",
        "lm_logits",
    }
    assert outputs["planner_logits"].shape[:2] == (2, config.planner_head.num_slots)
    assert outputs["planner_logits"].shape[-1] == config.plan_vocab_size
    assert outputs["coverage_logits"].shape[:2] == (2, 2)


def test_plan_notes_proj_exists_and_projects() -> None:
    config = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(config)

    proj = model.plan_notes_proj
    assert isinstance(proj, torch.nn.Linear)
    assert proj.weight.shape == (config.notes_dim, config.hidden_size)
    assert proj.bias is None

    # Verify it appears in adapter_state_dict
    model.trunk_adapter.attach_model(FakeTrunk(config.hidden_size, config.vocab_size))
    state = model.adapter_state_dict()
    assert "plan_notes_proj.weight" in state

    # Verify projection produces correct shape
    plan_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    emb = model.plan_embedding(plan_ids)  # (1, 3, hidden_size)
    projected = proj(emb)  # (1, 3, notes_dim)
    assert projected.shape == (1, 3, config.notes_dim)


def test_parallel_decoder_config_rejects_mismatched_plan_vocab() -> None:
    with pytest.raises(ValueError, match="planner_head.vocab_size must match plan_vocab_size"):
        ParallelDecoderModelConfig(
            hidden_size=8,
            vocab_size=16,
            notes_dim=4,
            num_heads=2,
            planner_head=PlannerHeadConfig(hidden_size=8, vocab_size=32),
        )
