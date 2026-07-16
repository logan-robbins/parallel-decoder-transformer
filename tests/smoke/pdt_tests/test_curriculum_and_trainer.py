"""Smoke tests for the curriculum controller and trainer building blocks.

Uses a tiny on-the-fly Qwen3 -- no HF download, no real training.
Validates:
- CurriculumController.on_step correctly freezes/unfreezes the right
  modules at each stage boundary.
- Resolution of "snc" / "stream_adapters" / "plan_notes_proj" identifiers
  actually reaches the target parameters (the paper-level fix).
"""

from __future__ import annotations

import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from pdt.config.schemas import (
    InstrumentationConfig,
    NotesBusConfig,
    PDTConfig,
    PlannerHeadConfig,
    PlanNotesProjectionConfig,
    RuntimeConfig,
    SNCConfig,
    SidecarConfig,
    SpeculationHeadConfig,
    StreamAdapterConfig,
    StreamClassifierConfig,
    TrainingConfig,
    TrunkConfig,
)
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.snc import SharedNotesCrossAttention
from pdt.training.curriculum import CurriculumController
from pdt.trunk.instrumentation import instrument_trunk


# ---------- Minimal trunk shim + fake Model ---------- #


class _InlineTrunk:
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        self.model = model
        self._instrumented = ()
        self.tokenizer = None

    @property
    def layers(self):
        return self.model.model.layers

    def num_layers(self) -> int:
        return len(self.layers)

    def replace_layer(self, idx, replacement):
        src = self.layers[idx]
        device = next(src.parameters()).device
        dtype = next(src.parameters()).dtype
        replacement.to(device=device, dtype=dtype)
        self.layers[idx] = replacement
        if self.layers[idx] is not replacement:
            raise RuntimeError("Identity check failed")

    def record_instrumented_indices(self, idxs):
        self._instrumented = tuple(idxs)

    def frozen_parameters(self):
        return list(self.model.parameters())

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, **kwargs):
        return self.model(**kwargs)


class _FakePDTModel:
    """Mimic the PDTModel surface area that CurriculumController depends on."""

    def __init__(self, trunk, sidecar, instrumented_layers):
        self.trunk_adapter = trunk
        self.sidecar = sidecar
        self.instrumented_layers = instrumented_layers


def _build_tiny_model():
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
    trunk = _InlineTrunk(model)

    from pdt.model import Sidecar

    sidecar_cfg = SidecarConfig(
        hidden_size=64,
        notes_dim=32,
        plan_vocab_size=64,
        num_streams=3,
        snc=SNCConfig(hidden_size=64, notes_dim=32, num_heads=8),
        adapters=StreamAdapterConfig(
            hidden_size=64,
            bottleneck_size=16,
            streams=("stream_0", "stream_1", "stream_2"),
        ),
        planner_head=PlannerHeadConfig(
            hidden_size=64, planner_width=64, vocab_size=64, num_slots=16
        ),
        plan_notes_proj=PlanNotesProjectionConfig(planner_width=64, notes_dim=32),
        speculation_head=SpeculationHeadConfig(hidden_size=64, notes_dim=32),
        stream_classifier=StreamClassifierConfig(
            hidden_size=64, classifier_width=32, num_streams=3
        ),
    )
    sidecar = Sidecar(sidecar_cfg)

    instr_cfg = InstrumentationConfig(enabled=True, target_layers=(1, 3))

    def make_snc():
        return SharedNotesCrossAttention(
            sidecar_cfg.snc,
            num_producers=sidecar_cfg.num_streams,
            gating_init=instr_cfg.snc_gate_init,
        )

    def make_adapter():
        return StreamAdapterLayer(sidecar_cfg.adapters)

    instrumented = instrument_trunk(
        trunk, instr_cfg, sidecar_cfg, make_snc=make_snc, make_adapter=make_adapter
    )

    full_cfg = PDTConfig(
        trunk=TrunkConfig(),
        instrumentation=instr_cfg,
        sidecar=sidecar_cfg,
        runtime=RuntimeConfig(
            streams=("stream_0", "stream_1", "stream_2"),
            block_size=4,
            notes_bus=NotesBusConfig(snapshot_dim=32, lag=1),
        ),
        training=TrainingConfig(),
    )
    fake_model = _FakePDTModel(trunk, sidecar, instrumented)
    return fake_model, full_cfg, instrumented


def test_curriculum_resolves_all_names():
    model, config, _instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)

    for name in [
        "trunk",
        "planner_head",
        "plan_notes_proj",
        "speculation_head",
        "stream_classifier",
        "snc",
        "stream_adapters",
        "snc_gate",
        "adapter_gate",
    ]:
        handles = ctrl.resolve_handles(name)
        assert handles, f"Curriculum could not resolve identifier {name!r} -- papered ghost bug."
        for h in handles:
            assert h.parameters, f"Resolution of {name!r} returned zero parameters."


def test_instrumentation_preserves_frozen_trunk_parameter_flags():
    _model, _config, instrumented = _build_tiny_model()

    phi_prefixes = ("snc.", "stream_adapter.")
    phi_names = {"notes_gate", "adapter_gate"}
    for layer in instrumented:
        base_parameters = [
            parameter
            for name, parameter in layer.named_parameters()
            if not name.startswith(phi_prefixes) and name not in phi_names
        ]
        assert base_parameters
        assert not any(parameter.requires_grad for parameter in base_parameters)
        assert layer.notes_gate.requires_grad
        assert layer.adapter_gate.requires_grad
        assert all(parameter.requires_grad for parameter in layer.snc.parameters())
        assert all(parameter.requires_grad for parameter in layer.stream_adapter.parameters())


def test_curriculum_stage0_trains_diagnostic_bus_modules():
    model, config, instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)
    ctrl.on_step(global_step=0)  # Enter stage 0.

    # Diagnostic mechanism stage trains the bus path without planner targets.
    for name in (
        "stream_adapters",
        "snc",
        "snc_gate",
        "adapter_gate",
        "speculation_head",
    ):
        for h in ctrl.resolve_handles(name):
            for p in h.parameters:
                assert p.requires_grad, f"{name} must be trainable at stage 0"

    for name in (
        "planner_head",
        "plan_notes_proj",
        "stream_classifier",
    ):
        for h in ctrl.resolve_handles(name):
            for p in h.parameters:
                assert not p.requires_grad, f"{name} must be frozen at stage 0"


def test_curriculum_stage_transitions_are_applied():
    model, config, instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)
    thresholds = config.training.curriculum.stage_schedule

    # Stage 0 -> SNC/speculation trainable, planner frozen.
    ctrl.on_step(0)
    assert ctrl.current_stage == 0
    assert not any(p.requires_grad for p in model.sidecar.planner_head.parameters())
    for layer in instrumented:
        for p in layer.snc.parameters():
            assert p.requires_grad
        assert layer.notes_gate.requires_grad
        assert layer.adapter_gate.requires_grad

    # Stage 1 -> planner integration enables planner_head too.
    for layer in instrumented:
        layer.notes_gate.requires_grad_(False)
        layer.adapter_gate.requires_grad_(False)
    ctrl.on_step(thresholds[1])
    assert ctrl.current_stage == 1
    assert any(p.requires_grad for p in model.sidecar.planner_head.parameters())
    for layer in instrumented:
        for p in layer.snc.parameters():
            assert p.requires_grad
        for p in layer.stream_adapter.parameters():
            assert p.requires_grad
        assert layer.notes_gate.requires_grad
        assert layer.adapter_gate.requires_grad

    # Stage 2 exhaustively freezes the classifier instead of inheriting stage
    # 1 state, while both explicit outer-gate controls remain trainable.
    ctrl.on_step(thresholds[2])
    assert not any(p.requires_grad for p in model.sidecar.stream_classifier.parameters())
    assert all(layer.notes_gate.requires_grad for layer in instrumented)
    assert all(layer.adapter_gate.requires_grad for layer in instrumented)

    # Stage 3 explicitly re-enables the classifier.
    ctrl.on_step(thresholds[3])
    assert any(p.requires_grad for p in model.sidecar.stream_classifier.parameters())


def test_curriculum_trunk_handle_excludes_all_per_layer_phi_and_unknowns_fail():
    model, config, instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)

    trunk_ids = {
        id(parameter) for handle in ctrl.resolve_handles("trunk") for parameter in handle.parameters
    }
    phi_ids = {
        id(parameter)
        for identifier in ("snc", "stream_adapters", "snc_gate", "adapter_gate")
        for handle in ctrl.resolve_handles(identifier)
        for parameter in handle.parameters
    }
    assert trunk_ids.isdisjoint(phi_ids)
    assert phi_ids

    ctrl.on_step(0)
    assert not any(
        parameter.requires_grad for parameter in ctrl.resolve_handles("trunk")[0].parameters
    )
    assert all(
        parameter.requires_grad
        for identifier in ("snc", "stream_adapters", "snc_gate", "adapter_gate")
        for handle in ctrl.resolve_handles(identifier)
        for parameter in handle.parameters
    )
    telemetry = ctrl.active_modules_snapshot()
    assert telemetry["snc_gate"] is True
    assert telemetry["adapter_gate"] is True
    assert telemetry["trunk"] is False

    with pytest.raises(ValueError, match="Unknown curriculum identifier"):
        ctrl.resolve_handles("typo_module")
