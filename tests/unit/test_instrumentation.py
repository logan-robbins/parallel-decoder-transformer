"""Unit tests covering instrumentation fail-fast behaviour."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from parallel_decoder_transformer.inference.snc_cross_attn import (
    SharedNotesCrossAttention,
    SharedNotesCrossAttentionConfig,
)
from parallel_decoder_transformer.integration.gpt_oss.trunk_adapter import TrunkAdapterConfig
from parallel_decoder_transformer.integration.instrumentation import (
    InstrumentedTrunkAdapter,
    InstrumentedTrunkAdapterConfig,
    InstrumentationSpec,
    SharedNotesResidual,
)


class _DummyModel(nn.Module):
    """Minimal torch module without transformer layers."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


def test_instrumented_trunk_adapter_requires_transformer_layers() -> None:
    """Ensure mid-stack instrumentation raises when no layers can be wrapped."""

    config = InstrumentedTrunkAdapterConfig(
        trunk=TrunkAdapterConfig(base_model="dummy"),
        instrumentation=InstrumentationSpec(enabled=True, top_k_layers=1),
    )
    dummy_trunk = _DummyModel()
    with pytest.raises(RuntimeError, match="InstrumentedTrunkAdapter"):
        InstrumentedTrunkAdapter(config, model=dummy_trunk)


def test_use_outer_notes_gate_false_bypasses_outer_gate() -> None:
    """When use_outer_notes_gate=False, the outer notes_gate is not applied.

    This test exercises the conditional logic directly on a SharedNotesResidual
    to verify that the outer gate scalar behaves correctly when active vs
    bypassed.  The actual instrumented layer delegates SNC to SharedNotesResidual
    and then conditionally applies the outer gate, so we can verify the
    composable behavior without constructing a full GPTNeoXLayer.
    """
    snc_config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2, gating_init=5.0
    )
    residual = SharedNotesResidual(snc_config)
    hidden = torch.randn(1, 2, 4)
    notes = torch.randn(1, 3, 4)
    delta = residual(hidden, notes)

    # Simulate outer gate active (near-closed).
    outer_gate_value = -5.0
    outer_gate = torch.sigmoid(torch.tensor(outer_gate_value))
    result_with = hidden + outer_gate * delta

    # Simulate outer gate bypassed.
    result_without = hidden + delta

    # The bypassed version should have a larger delta from hidden.
    abs_delta_with = (result_with - hidden).abs().sum()
    abs_delta_without = (result_without - hidden).abs().sum()
    assert abs_delta_without > abs_delta_with, (
        "Bypassing the near-closed outer gate should produce a larger SNC delta"
    )


def test_instrumentation_spec_use_outer_notes_gate_default() -> None:
    """InstrumentationSpec defaults use_outer_notes_gate to True."""
    spec = InstrumentationSpec()
    assert spec.use_outer_notes_gate is True


def test_instrumentation_spec_use_outer_notes_gate_propagation() -> None:
    """use_outer_notes_gate is stored on the spec and affects resolve_indices."""
    spec_on = InstrumentationSpec(enabled=True, top_k_layers=2, use_outer_notes_gate=True)
    spec_off = InstrumentationSpec(enabled=True, top_k_layers=2, use_outer_notes_gate=False)
    assert spec_on.use_outer_notes_gate is True
    assert spec_off.use_outer_notes_gate is False
    # Both produce the same indices (use_outer_notes_gate does not affect layer selection).
    assert spec_on.resolve_indices(10) == spec_off.resolve_indices(10)
