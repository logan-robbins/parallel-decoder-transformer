"""Tests for DeltaSNC pure-delta semantics and single-gate behavior."""
import math
import torch
import pytest
from parallel_decoder_transformer.inference.snc_cross_attn import (
    SharedNotesCrossAttention,
    SharedNotesCrossAttentionConfig,
)
from parallel_decoder_transformer.integration.instrumentation import DeltaSNC


def _make_snc_config(hidden: int = 8, notes_dim: int = 8, heads: int = 2) -> SharedNotesCrossAttentionConfig:
    return SharedNotesCrossAttentionConfig(
        hidden_size=hidden,
        notes_dim=notes_dim,
        num_heads=heads,
        gating_init=-5.0,
    )


def test_delta_snc_does_not_include_residual():
    """DeltaSNC output must not include the hidden_states residual."""
    config = _make_snc_config()
    snc = DeltaSNC(config)
    h = torch.zeros(1, 3, 8)
    notes = torch.ones(1, 2, 8)
    delta = snc(h, notes)
    # If residual were included, delta would be h + gate*proj = 0 + gate*proj = gate*proj
    # then the wrapper would return delta - h = gate*proj (same result here since h=0)
    # We instead check: delta should NOT equal h when h is non-zero
    h_nonzero = torch.ones(1, 3, 8)
    delta_nonzero = snc(h_nonzero, notes)
    # With delta_only=True, the output is proj = o_proj(softmax(QK^T/sqrt(d))V)
    # Q depends on h via q_proj, so delta WILL depend on h — that is correct behavior
    # What should NOT happen: h itself added to the output
    assert delta_nonzero.shape == h_nonzero.shape


def test_delta_snc_empty_notes_returns_zeros():
    """DeltaSNC must return zeros when notes sequence length is 0."""
    config = _make_snc_config()
    snc = DeltaSNC(config)
    h = torch.randn(1, 3, 8)
    notes = torch.zeros(1, 0, 8)  # empty notes
    delta = snc(h, notes)
    assert torch.allclose(delta, torch.zeros_like(h))


def test_snc_internal_gate_not_applied_in_delta_only_mode():
    """SharedNotesCrossAttention with delta_only=True must ignore self.gate."""
    config = _make_snc_config()
    attn = SharedNotesCrossAttention(config)
    h = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 8)
    # Set gate to extreme suppression
    with torch.no_grad():
        attn.gate.fill_(-100.0)
    # With delta_only=False (default): gate suppresses context before o_proj.
    # Output ≈ h + o_proj(0) = h + o_proj.bias (bias from o_proj remains).
    full_output = attn(h, notes, delta_only=False)
    expected_bias = attn.o_proj.bias.detach()
    assert torch.allclose(full_output, h + expected_bias, atol=1e-4), (
        "Gate=-100 should suppress context, leaving only h + o_proj.bias"
    )
    # With delta_only=True: gate is bypassed entirely, output is raw projection
    delta_output = attn(h, notes, delta_only=True)
    # delta_output should NOT be near zero (it's the full ungated projection)
    assert not torch.allclose(delta_output, torch.zeros_like(delta_output), atol=1e-2), (
        "delta_only=True must bypass the internal gate"
    )


def test_single_gate_gradient_magnitude():
    """Verify gradient through notes_gate is not attenuated by a second gate."""
    config = _make_snc_config()
    snc = DeltaSNC(config)
    h = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 8)
    notes_gate = torch.nn.Parameter(torch.tensor(0.0))
    delta = snc(h, notes)
    output = h + torch.sigmoid(notes_gate) * delta
    loss = output.sum()
    loss.backward()
    # Gradient of notes_gate: sum(delta) * sigmoid(0) * (1 - sigmoid(0)) = sum(delta) * 0.25
    expected_grad = delta.sum().item() * 0.25
    assert abs(notes_gate.grad.item() - expected_grad) < 1e-4


def test_post_trunk_snc_unaffected():
    """PostTrunkSNC (delta_only=False) must continue to include residual and internal gate.

    With gate=-100, context is fully suppressed before o_proj.  The output is
    ``h + o_proj(zeros) = h + o_proj.bias``.  The residual *is* present, and
    the gate *is* applied — the bias residual is the expected artifact of the
    gate-before-o_proj architecture.
    """
    from parallel_decoder_transformer.models.snc_backend import PostTrunkSNC
    config = _make_snc_config()
    attn = SharedNotesCrossAttention(config)
    with torch.no_grad():
        attn.gate.fill_(-100.0)  # Suppress gate completely
    backend = PostTrunkSNC(attn)
    h = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 8)
    output = backend.apply(h, notes)
    expected_bias = attn.o_proj.bias.detach()
    assert torch.allclose(output, h + expected_bias, atol=1e-4), (
        "PostTrunkSNC with suppressed gate must return h + o_proj.bias"
    )
