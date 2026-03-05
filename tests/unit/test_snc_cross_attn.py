from __future__ import annotations

import torch

from parallel_decoder_transformer.inference.gate_utils import (
    migrate_scalar_gate_checkpoint,
)
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


# ---------------------------------------------------------------------------
# Per-head and per-head-dynamic gate mode tests
# ---------------------------------------------------------------------------


def test_per_head_gate_shape() -> None:
    """Gate tensor has shape (H,) in per_head mode."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2, gate_mode="per_head"
    )
    layer = SharedNotesCrossAttention(config)
    assert layer.gate.shape == (2,)


def test_per_head_gate_closed_at_init() -> None:
    """All per-head gates start near-closed when gating_init=-5."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head"
    )
    layer = SharedNotesCrossAttention(config)
    g = torch.sigmoid(layer.gate)
    assert (g < 0.01).all()


def test_per_head_gates_can_differ() -> None:
    """Different heads can have different gate values after manual assignment."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2, gate_mode="per_head", gating_init=0.0
    )
    layer = SharedNotesCrossAttention(config)
    with torch.no_grad():
        layer.gate[0] = -10.0   # head 0 nearly closed
        layer.gate[1] = 10.0    # head 1 nearly open
    hidden = torch.zeros(1, 1, 4)
    notes = torch.ones(1, 1, 4)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    # Output should be non-zero (head 1 is open) but less than force_gate=True
    out = layer(hidden, notes, force_gate=False)
    forced = layer(hidden, notes, force_gate=True)
    assert out.abs().sum() < forced.abs().sum()


def test_per_head_dynamic_gate_zero_at_init() -> None:
    """Dynamic component produces zero delta at initialization (W=0, b=0)."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    hidden = torch.randn(3, 7, 4)  # arbitrary non-zero hidden
    gate = layer._compute_gate(hidden, dtype=torch.float32, device=torch.device("cpu"))
    # Should match static gate since dynamic delta is zero
    expected = torch.sigmoid(layer.gate).view(1, 2, 1, 1).expand(3, 2, 1, 1)
    assert torch.allclose(gate, expected, atol=1e-6)


def test_per_head_dynamic_gate_varies_by_input() -> None:
    """After training a non-zero W_dyn, gate varies by input.

    Use gating_init=0.0 and non-uniform hidden states with a non-uniform
    weight matrix.  LayerNorm normalizes inputs to mean=0/std=1, so a uniform
    weight row (all 1s) would always produce ~0 dot-product.  Use an eye-like
    weight so that different LayerNorm outputs produce different projections.
    """
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=0.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    with torch.no_grad():
        # Non-uniform weight: each head selects different features.
        layer.gate_dyn_proj.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
    # Inputs with different variance profiles after mean-pooling.
    hidden_a = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    hidden_b = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    gate_a = layer._compute_gate(hidden_a, dtype=torch.float32, device=torch.device("cpu"))
    gate_b = layer._compute_gate(hidden_b, dtype=torch.float32, device=torch.device("cpu"))
    assert not torch.allclose(gate_a, gate_b)


def test_force_gate_overrides_per_head_dynamic() -> None:
    """force_gate=True sets all gates to 1.0 in dynamic mode."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    hidden = torch.zeros(1, 1, 4)
    notes = torch.ones(1, 1, 4)
    out_forced = layer(hidden, notes, force_gate=True)
    assert torch.allclose(out_forced, torch.ones_like(out_forced), atol=1e-4)


def test_force_gate_per_batch_per_head_dynamic() -> None:
    """Per-batch force_gate works in dynamic mode.

    Batch 0 is forced open so output equals notes (identity projections).
    Batch 1 uses the learned near-closed gate (gating_init=-5 -> sigmoid ~0.0067)
    so output is approximately hidden[1] but not exactly zero due to residual
    leakage through the near-closed gate.
    """
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    hidden = torch.zeros(2, 1, 4)
    notes = torch.ones(2, 1, 4)
    out = layer(hidden, notes, force_gate=torch.tensor([True, False]))
    assert torch.allclose(out[0], torch.ones_like(out[0]), atol=1e-4)
    # Batch 1 near-closed: output is hidden + gate^2 * notes  (gate applied twice
    # because context is gated before o_proj which is identity; gate * identity(gate * v)).
    # With identity projections and scalar gate g, output = hidden + g^2 * v.
    # sigmoid(-5) ~ 0.0067, so g^2 ~ 4.5e-5.  Use tolerance reflecting that.
    assert torch.allclose(out[1], hidden[1], atol=1e-2)


def test_scalar_mode_backward_compatible() -> None:
    """gate_mode='scalar' produces identical parameter structure to the original."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="scalar"
    )
    layer = SharedNotesCrossAttention(config)
    assert layer.gate.shape == (1,)
    assert layer.gate_pool_norm is None
    assert layer.gate_dyn_proj is None


def test_migrate_scalar_gate() -> None:
    """migrate_scalar_gate_checkpoint broadcasts (1,) to (H,)."""
    state = {"cross_attention.gate": torch.tensor([-5.0])}
    migrated = migrate_scalar_gate_checkpoint(state, "cross_attention.gate", num_heads=4)
    assert migrated["cross_attention.gate"].shape == (4,)
    assert (migrated["cross_attention.gate"] == -5.0).all()
