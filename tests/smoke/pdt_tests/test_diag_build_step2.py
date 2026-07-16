"""Diagnostic Build Step 2: SNC with zero-init gates, no training.

Gate for Step 3. This test verifies:

- The SNC path actually runs inside the trunk forward graph when notes are
  supplied (closed gates give 0 delta, forcing gate open gives nonzero delta).
- The SNC delta is a function of the notes content -- swapping notes changes
  the output, which is the weakest possible form of the paper's coordination
  claim (SNC is not decorative).

Uses a tiny on-the-fly Qwen3 model (no weight download).
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
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        self.model = model

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
        pass


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


def _install(trunk: _InlineTrunk) -> list[InstrumentedQwen3DecoderLayer]:
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

    def make_snc():
        return SharedNotesCrossAttention(side.snc, gating_init=instr.snc_gate_init)

    def make_adapter():
        return StreamAdapterLayer(side.adapters)

    return instrument_trunk(trunk, instr, side, make_snc=make_snc, make_adapter=make_adapter)


def test_step2_closed_snc_gate_is_no_op(tiny_trunk):
    """With SNC gates closed (sigmoid(-4) \u2248 0.018) AND zero-init o_proj,
    supplying notes should not change the output at all."""
    instrumented = _install(tiny_trunk)

    input_ids = torch.randint(0, 128, (1, 12))

    # Baseline: no notes context.
    for layer in instrumented:
        layer.set_runtime_context(None)
    baseline = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    # With notes, closed gate + zero-init o_proj: no-op.
    ctx = LayerRuntimeContext(
        stream=None,
        notes=torch.randn(1, 4, 32),
        notes_mask=torch.ones(1, 4, dtype=torch.bool),
        snc_force_gate=None,
    )
    for layer in instrumented:
        layer.set_runtime_context(ctx)
    with_notes = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    delta = (baseline - with_notes).abs().max().item()
    assert delta < 1e-6, (
        f"Closed SNC gate must be a no-op, got {delta:.4e}. This means the "
        f"zero-init o_proj or closed gate is misconfigured."
    )


def test_step2_forced_open_snc_has_nonzero_effect(tiny_trunk):
    """Force SNC gates open + give SNC o_proj non-zero weights. Output must
    change vs no-notes baseline, proving the SNC path lands in forward."""
    instrumented = _install(tiny_trunk)

    # Give SNC non-trivial o_proj weights and force gate open.
    with torch.no_grad():
        for layer in instrumented:
            layer.snc.o_proj.weight.normal_(std=0.1)
            layer.notes_gate.fill_(10.0)  # sigmoid(10) ~ 1.0

    input_ids = torch.randint(0, 128, (1, 12))

    for layer in instrumented:
        layer.set_runtime_context(None)
    baseline = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    notes = torch.randn(1, 4, 32)
    ctx = LayerRuntimeContext(
        stream=None,
        notes=notes,
        notes_mask=torch.ones(1, 4, dtype=torch.bool),
        snc_force_gate=None,
    )
    for layer in instrumented:
        layer.set_runtime_context(ctx)
    with_notes = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    delta = (baseline - with_notes).abs().max().item()
    assert delta > 1e-3, (
        f"Open SNC gate must change outputs, got {delta:.4e}. SNC path is "
        f"silently detached from the trunk forward graph."
    )


def test_step2_snc_output_depends_on_notes_content(tiny_trunk):
    """The SNC delta must be a non-trivial function of notes content:
    swapping notes must change the output. This is the minimum evidence
    that SNC is reading notes at all (not just reacting to their existence)."""
    instrumented = _install(tiny_trunk)

    with torch.no_grad():
        for layer in instrumented:
            layer.snc.o_proj.weight.normal_(std=0.1)
            layer.notes_gate.fill_(10.0)

    input_ids = torch.randint(0, 128, (1, 12))

    notes_a = torch.randn(1, 4, 32)
    notes_b = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)

    ctx_a = LayerRuntimeContext(stream=None, notes=notes_a, notes_mask=mask)
    for layer in instrumented:
        layer.set_runtime_context(ctx_a)
    out_a = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    ctx_b = LayerRuntimeContext(stream=None, notes=notes_b, notes_mask=mask)
    for layer in instrumented:
        layer.set_runtime_context(ctx_b)
    out_b = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    delta = (out_a - out_b).abs().max().item()
    assert delta > 1e-3, (
        f"Different notes content must produce different outputs, got "
        f"{delta:.4e}. SNC is ignoring the notes tensor."
    )


def test_qwen3_kv_cache_preserves_gradient_from_prefill_notes(tiny_trunk):
    """A later token must backpropagate through Qwen's real prefill KV cache."""

    instrumented = _install(tiny_trunk)
    with torch.no_grad():
        for layer in instrumented:
            layer.snc.o_proj.weight.normal_(std=0.1)
            layer.notes_gate.fill_(10.0)

    notes = torch.randn(1, 4, 32, requires_grad=True)
    context = LayerRuntimeContext(
        stream="stream_0",
        notes=notes,
        notes_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    for layer in instrumented:
        layer.set_runtime_context(context)
    prefill = tiny_trunk.model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=True,
    )

    # Remove direct SNC access. Any gradient reaching `notes` from the next
    # token must traverse the key/value tensors produced by prefill.
    for layer in instrumented:
        layer.set_runtime_context(None)
    next_token = tiny_trunk.model(
        input_ids=torch.tensor([[4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        past_key_values=prefill.past_key_values,
        use_cache=True,
    )
    next_token.logits.square().mean().backward()

    assert notes.grad is not None
    assert torch.count_nonzero(notes.grad) > 0


def test_qwen3_cached_block_teacher_forcing_matches_full_causal_alignment(tiny_trunk):
    instrumented = _install(tiny_trunk)
    with torch.no_grad():
        for layer in instrumented:
            layer.snc.o_proj.weight.normal_(std=0.1)
            layer.notes_gate.fill_(2.0)

    context = LayerRuntimeContext(
        stream="stream_0",
        notes=torch.randn(1, 4, 32),
        notes_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    for layer in instrumented:
        layer.set_runtime_context(context)
    prompt = torch.tensor([[1, 2, 3]])
    target_block = torch.tensor([[4, 5]])
    full = tiny_trunk.model(
        input_ids=torch.cat((prompt, target_block), dim=1),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        use_cache=False,
        output_hidden_states=True,
    )
    prefill = tiny_trunk.model(
        input_ids=prompt,
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=True,
    )
    block = tiny_trunk.model(
        input_ids=target_block,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        past_key_values=prefill.past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )

    aligned = torch.cat((prefill.logits[:, -1:], block.logits[:, :-1]), dim=1)
    assert torch.allclose(aligned, full.logits[:, 2:4], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        block.hidden_states[-1],
        full.hidden_states[-1][:, 3:],
        atol=1e-5,
        rtol=1e-5,
    )


def test_step2_force_gate_override_works(tiny_trunk):
    """The ``snc_force_gate=False`` runtime override should zero out the
    SNC contribution even when the learned gate is open -- critical for
    Intervention A ablation on a trained checkpoint."""
    instrumented = _install(tiny_trunk)

    with torch.no_grad():
        for layer in instrumented:
            layer.snc.o_proj.weight.normal_(std=0.1)
            layer.notes_gate.fill_(10.0)  # learned gate fully open

    input_ids = torch.randint(0, 128, (1, 12))
    notes = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)

    # With learned gate open -> non-trivial output.
    ctx_open = LayerRuntimeContext(stream=None, notes=notes, notes_mask=mask, snc_force_gate=None)
    for layer in instrumented:
        layer.set_runtime_context(ctx_open)
    with_snc = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    # Force gate closed via runtime override.
    ctx_closed = LayerRuntimeContext(
        stream=None, notes=notes, notes_mask=mask, snc_force_gate=False
    )
    for layer in instrumented:
        layer.set_runtime_context(ctx_closed)
    without_snc = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    # No-notes baseline (expected to match the forced-closed case).
    for layer in instrumented:
        layer.set_runtime_context(None)
    baseline = tiny_trunk.model(input_ids=input_ids, use_cache=False).logits

    forced_delta = (baseline - without_snc).abs().max().item()
    open_delta = (baseline - with_snc).abs().max().item()
    assert forced_delta < 1e-6, f"snc_force_gate=False must match baseline, got {forced_delta:.4e}"
    assert open_delta > 1e-3, f"baseline check: learned-open delta {open_delta:.4e}"
