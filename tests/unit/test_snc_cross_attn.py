from __future__ import annotations

import torch

from parallel_decoder_transformer.inference.snc_cross_attn import (
    SharedNotesCrossAttention,
    SharedNotesCrossAttentionConfig,
)
from parallel_decoder_transformer.models.snc_backend import PostTrunkSNC


def _set_identity(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        eye = torch.eye(linear.out_features, linear.in_features)
        linear.weight.copy_(eye)
        if linear.bias is not None:
            linear.bias.zero_()


def _build_attention(gating_init: float = -10.0) -> SharedNotesCrossAttention:
    config = SharedNotesCrossAttentionConfig(
        hidden_size=2,
        notes_dim=2,
        num_heads=1,
        gating_init=gating_init,
    )
    layer = SharedNotesCrossAttention(config)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    return layer


def test_shared_notes_cross_attention_force_gate_scalar() -> None:
    attention = _build_attention()
    hidden = torch.zeros(1, 1, 2)
    notes = torch.ones(1, 1, 2)

    baseline = attention(hidden, notes)
    forced = attention(hidden, notes, force_gate=True)

    assert torch.allclose(baseline, hidden, atol=1e-4)
    assert torch.allclose(forced, torch.ones_like(forced), atol=1e-4)


def test_post_trunk_snc_force_gate_per_batch() -> None:
    attention = _build_attention()
    backend = PostTrunkSNC(attention)
    hidden = torch.zeros(2, 1, 2)
    notes = torch.ones(2, 1, 2)

    outputs = backend.apply(hidden, notes, force_open=torch.tensor([True, False]))

    assert torch.allclose(outputs[0], torch.ones_like(outputs[0]), atol=1e-4)
    assert torch.allclose(outputs[1], hidden[1], atol=1e-4)


def test_shared_notes_cross_attention_tokenwise_mask() -> None:
    attention = _build_attention(gating_init=5.0)
    hidden = torch.zeros(1, 2, 2)
    notes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)

    outputs = attention(hidden, notes, notes_mask=mask, force_gate=True)

    assert torch.allclose(outputs[0, 0], torch.tensor([1.0, 0.0]), atol=1e-4)
    assert torch.allclose(outputs[0, 1], torch.tensor([0.0, 1.0]), atol=1e-4)
