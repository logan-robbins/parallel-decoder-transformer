"""VQ planner smoke tests."""

from __future__ import annotations

import torch

from pdt.config.schemas import PlannerHeadConfig
from pdt.sidecar.heads.planner import PlannerHead, PlannerOutput


def test_vq_planner_returns_quantized_vectors_and_losses():
    planner = PlannerHead(
        PlannerHeadConfig(hidden_size=16, vocab_size=32, num_slots=4)
    )
    hidden = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5)
    out = planner(hidden, attention_mask=mask)

    assert isinstance(out, PlannerOutput)
    assert out.logits.shape == (2, 4, 32)
    assert out.indices.shape == (2, 4)
    assert out.pre_quantized.shape == (2, 4, 16)
    assert out.quantized.shape == (2, 4, 16)
    assert out.commitment_loss.ndim == 0
    assert out.codebook_loss.ndim == 0


def test_vq_planner_straight_through_path_has_gradients():
    planner = PlannerHead(
        PlannerHeadConfig(hidden_size=16, vocab_size=32, num_slots=4)
    )
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    out = planner(hidden)
    loss = out.quantized.pow(2).mean() + out.commitment_loss + out.codebook_loss
    loss.backward()

    assert hidden.grad is not None
    assert planner.slot_projector.weight.grad is not None
    assert planner.codebook.weight.grad is not None
