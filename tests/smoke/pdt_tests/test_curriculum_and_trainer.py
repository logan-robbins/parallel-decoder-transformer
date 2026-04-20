"""Smoke tests for the curriculum controller and trainer building blocks.

Uses a tiny on-the-fly Qwen3 -- no HF download, no real training.
Validates:
- CurriculumController.on_step correctly freezes/unfreezes the right
  modules at each stage boundary.
- Resolution of "snc" / "stream_adapters" / "plan_notes_proj" identifiers
  actually reaches the target parameters (the paper-level fix).
- PDTCollator + PDTKDDataset correctly pack a JSONL into a SampleBatch.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from dataclasses import replace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pdt.config.schemas import (
    AgreementHeadConfig,
    CoverageHeadConfig,
    CurriculumConfig,
    InstrumentationConfig,
    LossWeights,
    NotesBusConfig,
    NotesHeadConfig,
    OptimizerConfig,
    PDTConfig,
    PlannerHeadConfig,
    PlanNotesProjectionConfig,
    RuntimeConfig,
    SNCConfig,
    SidecarConfig,
    SpeculationHeadConfig,
    StreamAdapterConfig,
    StreamClassifierConfig,
    TeacherCacheConfig,
    TrainingConfig,
    TrunkConfig,
)
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.snc import SharedNotesCrossAttention
from pdt.training.curriculum import CurriculumController
from pdt.training.dataset import PDTCollator, PDTKDDataset
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
            hidden_size=64, bottleneck_size=16,
            streams=("stream_0", "stream_1", "stream_2"),
        ),
        planner_head=PlannerHeadConfig(hidden_size=64, vocab_size=64, num_slots=16),
        plan_notes_proj=PlanNotesProjectionConfig(hidden_size=64, notes_dim=32),
        notes_head=NotesHeadConfig(hidden_size=64, notes_dim=32),
        speculation_head=SpeculationHeadConfig(hidden_size=64, notes_dim=32),
        coverage_head=CoverageHeadConfig(hidden_size=64, num_heads=4),
        agreement_head=AgreementHeadConfig(hidden_size=64, notes_dim=32, coverage_features=8),
        stream_classifier=StreamClassifierConfig(hidden_size=64, num_streams=3),
    )
    sidecar = Sidecar(sidecar_cfg)

    instr_cfg = InstrumentationConfig(enabled=True, target_layers=(1, 3))

    def make_snc():
        return SharedNotesCrossAttention(sidecar_cfg.snc, gating_init=instr_cfg.snc_gate_init)

    def make_adapter():
        return StreamAdapterLayer(sidecar_cfg.adapters)

    instrumented = instrument_trunk(trunk, instr_cfg, sidecar_cfg,
                                    make_snc=make_snc, make_adapter=make_adapter)

    full_cfg = PDTConfig(
        trunk=TrunkConfig(),
        instrumentation=instr_cfg,
        sidecar=sidecar_cfg,
        runtime=RuntimeConfig(
            streams=("stream_0", "stream_1", "stream_2"),
            block_size=4,
            commit_horizon=8,
            notes_bus=NotesBusConfig(snapshot_dim=32, max_snapshots=4, lag=1),
        ),
        training=TrainingConfig(),
    )
    fake_model = _FakePDTModel(trunk, sidecar, instrumented)
    return fake_model, full_cfg, instrumented


def test_curriculum_resolves_all_names():
    model, config, instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)

    for name in [
        "trunk", "planner_head", "plan_embedding", "plan_notes_proj",
        "notes_head", "speculation_head", "coverage_head", "agreement_head",
        "stream_classifier", "snc", "stream_adapters", "snc_gate", "adapter_gate",
    ]:
        handles = ctrl.resolve_handles(name)
        assert handles, f"Curriculum could not resolve identifier {name!r} -- papered ghost bug."
        for h in handles:
            assert h.parameters, f"Resolution of {name!r} returned zero parameters."


def test_curriculum_stage0_freezes_everything_except_planner_modules():
    model, config, instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)
    ctrl.on_step(global_step=0)  # Enter stage 0.

    # Must be trainable: planner_head, plan_embedding, plan_notes_proj, notes_head.
    for name in ("planner_head", "plan_embedding", "plan_notes_proj", "notes_head"):
        for h in ctrl.resolve_handles(name):
            for p in h.parameters:
                assert p.requires_grad, f"{name} must be trainable at stage 0"

    # Must be frozen: snc, stream_adapters, speculation_head, agreement_head,
    # coverage_head, stream_classifier, trunk.
    for name in ("snc", "stream_adapters", "speculation_head",
                 "agreement_head", "coverage_head", "stream_classifier"):
        for h in ctrl.resolve_handles(name):
            for p in h.parameters:
                assert not p.requires_grad, f"{name} must be frozen at stage 0"


def test_curriculum_stage_transitions_are_applied():
    model, config, instrumented = _build_tiny_model()
    ctrl = CurriculumController(model, config)
    thresholds = config.training.curriculum.stage_schedule

    # Stage 0 -> planner_head trainable, snc frozen.
    ctrl.on_step(0)
    assert ctrl.current_stage == 0
    assert any(p.requires_grad for p in model.sidecar.planner_head.parameters())
    for layer in instrumented:
        for p in layer.snc.parameters():
            assert not p.requires_grad

    # Stage 1 -> stream_adapters + snc trainable, planner_head frozen.
    ctrl.on_step(thresholds[1])
    assert ctrl.current_stage == 1
    for layer in instrumented:
        for p in layer.snc.parameters():
            assert p.requires_grad
        for p in layer.stream_adapter.parameters():
            assert p.requires_grad


def test_collator_packs_jsonl():
    with tempfile.TemporaryDirectory() as tdir:
        path = Path(tdir) / "kd.jsonl"
        with path.open("w") as f:
            for sample_id in ("sA", "sB"):
                for stream_id in ("stream_0", "stream_1", "stream_2"):
                    rec = {
                        "example_id": f"{sample_id}:{stream_id}",
                        "sample_id": sample_id,
                        "stream_id": stream_id,
                        "stream": stream_id,
                        "student_ids": [1, 2, 3, 4, 5],
                        "student_labels": [1, 2, 3, 4, 5],
                        "planner_ids": [7, 3, 11, 2],
                        "notes_teacher": [[0.1] * 32] * 3,
                        "notes_student": [[0.0] * 32] * 3,
                        "plan_tokens": ["p1", "p2", "p3", "p4"],
                        "continuation_sufficiency_labels": [1, 1, 0, 0],
                        "metadata": {},
                        "raw_teacher_notes": {},
                    }
                    f.write(json.dumps(rec) + "\n")

        ds = PDTKDDataset(path, num_streams=3)
        assert len(ds) == 2
        coll = PDTCollator(
            pad_token_id=0, num_slots=16, num_streams=3, notes_dim=32,
            max_length=16, max_plan_items=8, max_snapshots=4,
        )
        batch = coll([ds[0], ds[1]])
        assert batch.student_ids.shape == (2, 3, 16)
        assert batch.planner_targets.shape == (2, 16)
        assert batch.teacher_notes.shape == (2, 3, 32)
        assert batch.readiness_targets.shape == (2, 4)
        # Planner ids were [7, 3, 11, 2] -> padded to 16.
        assert batch.planner_targets[0, 0].item() == 7
        assert batch.planner_mask[0, 3].item() is True
        assert batch.planner_mask[0, 4].item() is False
