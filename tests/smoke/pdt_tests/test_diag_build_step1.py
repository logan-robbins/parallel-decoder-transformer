"""Diagnostic Build Step 1: Frozen trunk + per-stream adapters, K=2, no SNC, no planner.

Gate for Step 2. This test verifies:

- Instrumented layers actually execute inside the trunk forward graph (not
  silently detached, as the audit found was the case in the previous
  codebase).
- Two identical prompts through adapters `stream_0` and `stream_1` produce
  measurably different outputs purely from adapter divergence.

Uses a tiny on-the-fly Qwen3 model (no weight download) so the test is fast
and hermetic.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pdt.config.schemas import (
    InstrumentationConfig,
    SNCConfig,
    SidecarConfig,
    StreamAdapterConfig,
)
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.snc import SharedNotesCrossAttention
from pdt.trunk.instrumentation import (
    InstrumentedQwen3DecoderLayer,
    LayerRuntimeContext,
    instrument_trunk,
)


class _InlineTrunk:
    """Minimal adapter-shim that exposes the `layers` ModuleList and
    `replace_layer` identity guarantee without needing a real HF download."""

    def __init__(self, model: Qwen3ForCausalLM) -> None:
        self.model = model
        self._instrumented: tuple[int, ...] = ()

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers

    def num_layers(self) -> int:
        return len(self.layers)

    def replace_layer(self, index: int, replacement: torch.nn.Module) -> None:
        src = self.layers[index]
        device = next(src.parameters()).device
        dtype = next(src.parameters()).dtype
        replacement.to(device=device, dtype=dtype)
        self.layers[index] = replacement
        if self.layers[index] is not replacement:
            raise RuntimeError(f"Identity check failed at layer {index}.")

    def record_instrumented_indices(self, indices: tuple[int, ...]) -> None:
        self._instrumented = tuple(indices)


@pytest.fixture
def tiny_trunk():
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return _InlineTrunk(model)


def _instrument_adapters_only(trunk: _InlineTrunk) -> list[InstrumentedQwen3DecoderLayer]:
    instr = InstrumentationConfig(enabled=True, target_layers=(1, 3))
    side = SidecarConfig(
        hidden_size=64,
        notes_dim=32,
        plan_vocab_size=16,
        num_streams=2,
        snc=SNCConfig(hidden_size=64, notes_dim=32, num_heads=8),
        adapters=StreamAdapterConfig(
            hidden_size=64,
            bottleneck_size=16,
            streams=("stream_0", "stream_1"),
        ),
    )
    # Step 1 discipline: NO SNC. Pass None.
    def make_snc():
        return None  # instrument_trunk accepts None via make_snc returning None
    def make_adapter():
        return StreamAdapterLayer(side.adapters)

    # Workaround: instrument_trunk unconditionally creates via make_snc, so
    # we build the SNC but keep its gate closed and its outer gate too.
    # Alternatively, we set snc=None by intercepting after install -- cleaner
    # to just build and never supply notes context.
    def make_snc_real():
        return SharedNotesCrossAttention(side.snc, gating_init=-10.0)  # effectively 0
    return instrument_trunk(
        trunk,
        instr,
        side,
        make_snc=make_snc_real,
        make_adapter=make_adapter,
    )


def test_step1_instrumented_forward_executes(tiny_trunk):
    """Sanity: forward with instrumented layers runs and produces logits."""
    instrumented = _instrument_adapters_only(tiny_trunk)
    assert len(instrumented) == 2

    input_ids = torch.randint(0, 128, (1, 12))
    for layer in instrumented:
        layer.set_runtime_context(None)
    out = tiny_trunk.model(input_ids=input_ids, use_cache=False)
    assert out.logits.shape == (1, 12, 128)


def test_step1_k2_adapters_differentiate(tiny_trunk):
    """Core gate: two streams from identical prompts must produce different
    outputs when adapter deltas are enabled.

    We bypass the closed-gate initialization by (a) opening ``adapter_gate``
    and (b) giving the per-stream adapter up-projections non-zero weights,
    simulating what a few steps of training would produce.
    """
    instrumented = _instrument_adapters_only(tiny_trunk)

    # Open the adapter gates fully.
    with torch.no_grad():
        for layer in instrumented:
            layer.adapter_gate.fill_(10.0)  # sigmoid(10) ~ 1.0
            # Give stream adapters non-trivially different weights.
            adapters = layer.stream_adapter.adapters.adapters
            torch.manual_seed(0)
            adapters["stream_0"].up.weight.normal_(std=0.1)
            torch.manual_seed(1)
            adapters["stream_1"].up.weight.normal_(std=0.1)

    input_ids = torch.randint(0, 128, (1, 12))

    # Stream 0 forward (no notes context).
    ctx0 = LayerRuntimeContext(stream="stream_0", notes=None, notes_mask=None)
    for layer in instrumented:
        layer.set_runtime_context(ctx0)
    out0 = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    # Stream 1 forward.
    ctx1 = LayerRuntimeContext(stream="stream_1", notes=None, notes_mask=None)
    for layer in instrumented:
        layer.set_runtime_context(ctx1)
    out1 = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    delta = (out0 - out1).abs().max().item()
    assert delta > 1e-3, (
        f"Streams must differentiate, got max_delta={delta:.4e}. "
        f"This means the per-stream adapter contributions are not routing "
        f"or not landing in the trunk forward graph."
    )


def test_step1_closed_gate_is_no_op(tiny_trunk):
    """With gates at sigmoid(-4) and zero-init up-projection, adapter deltas
    are exactly zero -- trunk behavior is preserved at step 0."""
    instrumented = _instrument_adapters_only(tiny_trunk)
    input_ids = torch.randint(0, 128, (1, 12))

    # Baseline: no context.
    for layer in instrumented:
        layer.set_runtime_context(None)
    baseline = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    # With context, closed gates, zero-init up: must be bit-identical within
    # numerical noise.
    ctx = LayerRuntimeContext(stream="stream_0", notes=None, notes_mask=None)
    for layer in instrumented:
        layer.set_runtime_context(ctx)
    with_ctx = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    delta = (baseline - with_ctx).abs().max().item()
    assert delta < 1e-6, f"Closed-gate forward must be a no-op, got {delta:.4e}"
