from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


if "parallel_decoder_transformer.data.teacher_runner" not in sys.modules:
    teacher_runner_stub = types.ModuleType("parallel_decoder_transformer.data.teacher_runner")

    @dataclass(slots=True)
    class DatasetTeacherConfig:
        cache_dir: str | None = None
        max_snapshots: int = 3
        id_field: str = "example_id"
        refresh_cache: bool = False
        refresh_cache: bool = False

    @dataclass(slots=True)
    class TeacherSnapshotText:
        version: int
        stride: int
        stream_notes: Mapping[str, Sequence[str]]
        coverage: Mapping[str, float] | None = None
        source: str = "teacher"

    @dataclass(slots=True)
    class TeacherRunResult:
        example_id: str
        stream_notes: Mapping[str, Sequence[str]]
        snapshots: Sequence[TeacherSnapshotText] = field(default_factory=tuple)
        teacher_plan: Mapping[str, Any] | None = None

    def normalize_stream_id(value: str) -> str:
        stream = str(value or "").strip().lower()
        if not stream:
            return "stream_unknown"
        if not stream.startswith("stream_"):
            stream = f"stream_{stream}"
        return stream

    def normalize_stream_notes(
        payload: Mapping[str, Sequence[str]] | None,
    ) -> Mapping[str, Sequence[str]]:
        return payload or {}

    teacher_runner_stub.DatasetTeacherConfig = DatasetTeacherConfig
    teacher_runner_stub.TeacherSnapshotText = TeacherSnapshotText
    teacher_runner_stub.TeacherRunResult = TeacherRunResult
    teacher_runner_stub.normalize_stream_id = normalize_stream_id
    teacher_runner_stub.normalize_stream_notes = normalize_stream_notes
    sys.modules["parallel_decoder_transformer.data.teacher_runner"] = teacher_runner_stub

from parallel_decoder_transformer.data.collator_kd import TwoBranchKDCollatorConfig
from parallel_decoder_transformer.data.teacher_provider import (
    TeacherNotes,
    TeacherNotesProviderBase,
)
from parallel_decoder_transformer.data.teacher_runner import DatasetTeacherConfig
from parallel_decoder_transformer.models import (
    ParallelDecoderModelConfig,
    ParallelDecoderTransformer,
)
from parallel_decoder_transformer.training import Trainer, TrainingConfig
from parallel_decoder_transformer.training import trainer as trainer_module
from parallel_decoder_transformer.training.trainer import StagePolicyConfig


class _StaticTeacherNotesProvider(TeacherNotesProviderBase):
    """Deterministic teacher provider for unit tests."""

    def __init__(self, *_, **__):
        pass

    def fetch(self, example: dict[str, object]) -> TeacherNotes:
        notes = example.get("notes_teacher")
        if not isinstance(notes, torch.Tensor):
            notes = torch.tensor(notes, dtype=torch.float32)
        return TeacherNotes(notes=notes.clone(), snapshots=[], raw_notes={})


trainer_module.DatasetTeacherNotesProvider = _StaticTeacherNotesProvider


class _StubParallelDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        linear = torch.nn.Linear(1, 1)
        self.trunk_adapter = types.SimpleNamespace(model=linear)
        identity = torch.nn.Identity()
        self.stream_adapters = identity
        self.cross_attention = identity
        self.planner_head = identity
        self.notes_head = identity
        self.speculation_head = identity
        self.agreement_head = identity
        self.coverage_head = identity
        self.stream_classifier = identity

    def iter_trainable_parameters(self):  # pragma: no cover - deterministic generator
        return (param for param in self.parameters())


def _training_config(**kwargs: object) -> TrainingConfig:
    cfg = TrainingConfig(**kwargs)
    cfg.dataset_teacher = DatasetTeacherConfig()
    cfg.device = "cpu"
    return cfg


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, _: int) -> dict[str, object]:
        return {
            "student_ids": torch.tensor([1, 2, 3]),
            "student_labels": torch.tensor([1, 2, 3]),
            "planner_ids": torch.tensor([4, 5, 6]),
            "notes_student": torch.zeros(3, 4),
            "notes_teacher": torch.zeros(3, 4),
            "plan_items": ["Cover the topic"],
            "coverage_targets": [1],
            "notes_text": "cover topic",
            "plan_tokens": ["core::Cover the topic"],
            "notes_tokens": ["cover", "topic"],
            "stream": "core",
        }


class FakeTrunk(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
    ) -> object:
        hidden = torch.zeros(input_ids.size(0), input_ids.size(1), self.hidden_size)
        return type("Outputs", (), {"hidden_states": [hidden, hidden]})()


def test_extract_notes_from_bus_preserves_window() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    streams = 3
    notes_dim = model_cfg.notes_dim
    max_snapshots = model_cfg.collator.max_snapshots
    snapshot_a = torch.arange(streams * notes_dim, dtype=torch.float32).view(streams, notes_dim)
    snapshot_b = (
        torch.arange(streams * notes_dim, dtype=torch.float32).view(streams, notes_dim) + 100.0
    )

    batch: dict[str, torch.Tensor] = {
        "notes_teacher": torch.zeros(1, streams, notes_dim, dtype=torch.float32),
        "teacher_notes_bus": torch.zeros(1, max_snapshots, streams, notes_dim, dtype=torch.float32),
        "teacher_bus_mask": torch.zeros(1, max_snapshots, dtype=torch.bool),
        "teacher_bus_coverage": torch.ones(1, max_snapshots, streams, dtype=torch.float32),
        "teacher_bus_streams": torch.zeros(1, max_snapshots, dtype=torch.long),
        "teacher_bus_stride": torch.arange(max_snapshots, dtype=torch.long).unsqueeze(0),
        "teacher_bus_version": torch.arange(max_snapshots, dtype=torch.long).unsqueeze(0),
    }

    batch["teacher_notes_bus"][0, 0] = snapshot_a
    batch["teacher_notes_bus"][0, 1] = snapshot_b
    batch["teacher_bus_mask"][0, :2] = True

    notes_full, coverage_full, streams_full, stride_full, version_full = (
        trainer._extract_notes_from_bus(
            batch,
            notes_bus_key="teacher_notes_bus",
            mask_key="teacher_bus_mask",
            fallback_key="notes_teacher",
            coverage_key="teacher_bus_coverage",
            streams_key="teacher_bus_streams",
            stride_key="teacher_bus_stride",
            version_key="teacher_bus_version",
            lag_override=0,
        )
    )

    assert notes_full.shape == (1, max_snapshots * streams, notes_dim)
    assert torch.allclose(notes_full[0, :streams], snapshot_a)
    assert torch.allclose(notes_full[0, streams : 2 * streams], snapshot_b)
    if coverage_full is not None:
        assert coverage_full.shape == (1, max_snapshots * streams)
        assert torch.allclose(
            coverage_full[0, : 2 * streams], torch.ones(2 * streams, dtype=coverage_full.dtype)
        )
    if streams_full is not None:
        assert streams_full.shape == (1, max_snapshots * streams)
    if stride_full is not None:
        assert torch.equal(stride_full[0, :streams], torch.zeros(streams, dtype=stride_full.dtype))
        assert torch.equal(
            stride_full[0, streams : 2 * streams],
            torch.full((streams,), 1, dtype=stride_full.dtype),
        )
    if version_full is not None:
        assert torch.equal(
            version_full[0, :streams], torch.zeros(streams, dtype=version_full.dtype)
        )
        assert torch.equal(
            version_full[0, streams : 2 * streams],
            torch.full((streams,), 1, dtype=version_full.dtype),
        )

    notes_lagged, coverage_lagged, _, _, _ = trainer._extract_notes_from_bus(
        batch,
        notes_bus_key="teacher_notes_bus",
        mask_key="teacher_bus_mask",
        fallback_key="notes_teacher",
        coverage_key="teacher_bus_coverage",
        streams_key="teacher_bus_streams",
        stride_key="teacher_bus_stride",
        version_key="teacher_bus_version",
        lag_override=1,
    )

    assert torch.allclose(notes_lagged[0, :streams], snapshot_a)
    assert torch.all(notes_lagged[0, streams : 2 * streams] == 0)
    if coverage_lagged is not None:
        assert torch.all(coverage_lagged[0, streams : 2 * streams] == 0)

    sectional_mask = torch.tensor([True], dtype=torch.bool, device=trainer.device)
    notes_sectional, _, _, _, _ = trainer._extract_notes_from_bus(
        batch,
        notes_bus_key="teacher_notes_bus",
        mask_key="teacher_bus_mask",
        fallback_key="notes_teacher",
        coverage_key="teacher_bus_coverage",
        streams_key="teacher_bus_streams",
        stride_key="teacher_bus_stride",
        version_key="teacher_bus_version",
        lag_override=1,
        sectional_mask=sectional_mask,
    )
    assert torch.allclose(notes_sectional[0, :streams], snapshot_a)
    assert torch.allclose(notes_sectional[0, streams : 2 * streams], snapshot_b)


def test_trainer_runs_single_step() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=1, log_interval=1, eval_interval=10)
    dataset = FakeDataset()
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=dataset,
        eval_dataset=None,
    )
    trainer.fit()
    assert trainer.state.global_step == 1


def test_plan_snapshot_freeze_prevents_plan_eviction() -> None:
    model = _StubParallelDecoder()
    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.curriculum.B = 2
    notes_dim = 4
    collator_cfg = TwoBranchKDCollatorConfig(pad_token_id=0, notes_dim=notes_dim, max_snapshots=2)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=collator_cfg,
        dataset=None,
        eval_dataset=None,
    )

    streams = len(collator_cfg.stream_to_id)
    max_snapshots = collator_cfg.max_snapshots
    plan_snapshot = torch.ones(streams, notes_dim)
    batch: dict[str, torch.Tensor] = {
        "student_notes_bus": torch.zeros(1, max_snapshots, streams, notes_dim, dtype=torch.float32),
        "student_bus_mask": torch.zeros(1, max_snapshots, dtype=torch.bool),
        "sectional_independence": torch.tensor([True]),
        "commit_mask": torch.zeros(1, collator_cfg.max_length, dtype=torch.bool),
    }
    batch["student_notes_bus"][0, 0] = plan_snapshot
    batch["student_bus_mask"][0, 0] = True

    trainer._ensure_plan_snapshot_freeze_state(batch)
    freeze_key = trainer_module._PLAN_SNAPSHOT_FREEZE_KEY
    assert batch[freeze_key].item() == trainer_cfg.curriculum.B

    new_notes = torch.full((1, streams, notes_dim), 2.0)
    trainer._update_student_bus(batch, new_notes, snapshot_streams=None, coverage=None)
    assert torch.allclose(batch["student_notes_bus"][0, 0], plan_snapshot)

    newer_notes = torch.full((1, streams, notes_dim), 3.0)
    trainer._update_student_bus(batch, newer_notes, snapshot_streams=None, coverage=None)
    assert torch.allclose(batch["student_notes_bus"][0, 0], plan_snapshot)

    trainer._advance_commit_mask(batch)
    assert batch[freeze_key].item() == 0

    newest_notes = torch.full((1, streams, notes_dim), 4.0)
    trainer._update_student_bus(batch, newest_notes, snapshot_streams=None, coverage=None)
    assert torch.allclose(batch["student_notes_bus"][0, 0], newer_notes[0])
    assert torch.allclose(batch["student_notes_bus"][0, 1], newest_notes[0])


def test_gradnorm_adjusts_scale() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.gradnorm.enabled = True
    trainer_cfg.gradnorm.target_ratio = 1.0
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    initial_scale = trainer.kd_scale
    trainer._maybe_adjust_gradnorm(torch.tensor(2.0), torch.tensor(1.0), stage=2)
    assert trainer.kd_scale < initial_scale
    decreased_scale = trainer.kd_scale
    trainer._maybe_adjust_gradnorm(torch.tensor(0.2), torch.tensor(1.0), stage=2)
    assert trainer.kd_scale > decreased_scale


def test_masked_labels_applies_pad_id() -> None:
    labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.tensor([[False, True, False, True]], dtype=torch.bool)
    pad_id = -100
    masked = Trainer._masked_labels(labels, mask, pad_id)
    assert masked.tolist() == [[pad_id, 2, pad_id, 4]]


def test_usage_regularizer_stage_gate() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.loss_weights.use = 0.5
    trainer_cfg.usage_min_stage = 4
    trainer_cfg.usage_margin = 0.1
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )
    weights = trainer.config.loss_weights
    assert not trainer._should_penalize_usage(stage=2, weights=weights)
    assert trainer._should_penalize_usage(stage=4, weights=weights)

    delta = torch.tensor(0.05, device=trainer.device)
    active_penalty = trainer._usage_penalty(delta, stage=4, weights=weights)
    assert torch.allclose(active_penalty, torch.tensor(0.05, device=trainer.device))

    suppressed_penalty = trainer._usage_penalty(delta, stage=2, weights=weights)
    assert suppressed_penalty.item() == 0.0


def test_mid_stack_lm_losses_drive_ratio() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    fake_trunk = FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size)
    model.trunk_adapter.attach_model(fake_trunk)

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    def _fake_encode_trunk(
        self: Trainer,
        model: ParallelDecoderTransformer,
        batch: dict[str, torch.Tensor],
        *,
        notes: torch.Tensor,
        notes_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = batch["input_ids"].shape
        hidden_size = model.config.hidden_size
        return torch.zeros(batch_size, seq_len, hidden_size, device=self.device)

    trainer._encode_trunk = types.MethodType(_fake_encode_trunk, trainer)
    trainer.state.stage_index = 3

    dataset = FakeDataset()
    batch = trainer.collator([dataset[0], dataset[1]])
    _loss, metrics = trainer._training_step(batch)

    assert metrics["lm_ce_loss"] > 0.0
    assert "lm_kd_loss" in metrics
    assert "kd_ce_ratio" in metrics
    assert metrics.get("kd_ce_ratio_source") in {"lm", "planner"}
    if metrics.get("kd_ce_ratio_source") == "lm":
        expected = metrics["lm_kd_loss"] / max(metrics["lm_ce_loss"], 1e-6)
        assert math.isclose(metrics["kd_ce_ratio"], expected, rel_tol=1e-6, abs_tol=1e-6)


def test_interhead_spec_kl_matches_expected() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    speculative = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32, device=trainer.device
    )
    mask = torch.tensor([[1, 1]], dtype=torch.long, device=trainer.device)

    loss = trainer._interhead_spec_kl(speculative, mask, temperature=1.0)

    log_probs = F.log_softmax(speculative[0], dim=-1)
    probs = log_probs.exp()
    kl_01 = torch.sum(probs[0] * (log_probs[0] - log_probs[1]))
    kl_10 = torch.sum(probs[1] * (log_probs[1] - log_probs[0]))
    expected = (kl_01 + kl_10) / 1.0

    assert torch.isclose(loss, expected, atol=1e-6)


def test_interhead_spec_kl_respects_overlap_threshold() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    speculative = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32, device=trainer.device
    )
    mask = torch.tensor([[1, 1]], dtype=torch.long, device=trainer.device)
    coverage = torch.tensor([[1.0, 1e-6]], dtype=torch.float32, device=trainer.device)

    loss = trainer._interhead_spec_kl(
        speculative,
        mask,
        temperature=1.0,
        coverage=coverage,
        min_overlap=1e-4,
    )

    assert torch.isclose(loss, torch.tensor(0.0, device=trainer.device), atol=1e-8)


def test_stability_and_rollback_metrics_emitted() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.curriculum.L = 1
    trainer_cfg.metrics.stability_every = 1
    dataset = FakeDataset()
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    batch = trainer.collator([dataset[0], dataset[1]])
    _, metrics = trainer._training_step(batch)

    assert "rollback_ratio" in metrics
    assert "stability_ratio" in metrics
    assert metrics.get("rollback_tokens", 0.0) > 0.0
    assert metrics.get("stability_tokens", 0.0) > 0.0
    assert "repair_error_rate" in metrics
    assert "stability_error_rate" in metrics


def test_stage_schedule_advances_as_configured() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.curriculum.steps_per_stage = 0
    trainer_cfg.curriculum.stage_schedule = (0, 2, 5, 7)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer.state.global_step = 0
    assert trainer._determine_stage() == 0
    trainer.state.global_step = 2
    assert trainer._determine_stage() == 1
    trainer.state.global_step = 5
    assert trainer._determine_stage() == 2
    trainer.state.global_step = 9
    assert trainer._determine_stage() == 3


def test_stage_policy_freeze_unfreeze_applied() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer_cfg.curriculum.stage_schedule = (0, 1)
    trainer_cfg.stage_policies = {
        0: StagePolicyConfig(freeze=("trunk",)),
        1: StagePolicyConfig(unfreeze=("trunk",)),
    }
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer.state.global_step = 0
    trainer._determine_stage()
    assert all(not param.requires_grad for param in model.trunk_adapter.model.parameters())

    trainer.state.global_step = 1
    trainer._determine_stage()
    assert any(param.requires_grad for param in model.trunk_adapter.model.parameters())


def test_negative_sampling_augments_plan_items_and_notes() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.negative_sampling.enabled = True
    trainer_cfg.negative_sampling.start_stage = 0
    trainer_cfg.negative_sampling.contradiction_ratio = 1.0
    trainer_cfg.negative_sampling.max_contradictions = 2
    trainer_cfg.negative_sampling.noise_ratio = 1.0
    trainer_cfg.negative_sampling.noise_std = 0.5
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    dataset = FakeDataset()
    batch = trainer.collator([dataset[0], dataset[1]])
    original_width = batch["plan_item_ids"].shape[1]
    original_notes = batch["notes_student"].clone()

    trainer._maybe_apply_negative_sampling(batch, stage=1)

    assert batch["plan_item_ids"].shape[1] > original_width
    neg_targets = batch["coverage_targets"][:, original_width:]
    assert neg_targets.numel() > 0
    assert torch.all(neg_targets == 0.0)
    assert torch.any(batch["notes_student"] != original_notes)
    if "plan_text" in batch:
        assert len(batch["plan_text"][0]) > len(dataset[0]["plan_items"])


def test_micro_rollout_updates_bus_and_commit_mask() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.parallel_micro_steps = 2
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )
    trainer.state.stage_index = 2

    dataset = FakeDataset()
    batch_cpu = trainer.collator([dataset[0], dataset[1]])
    batch = {
        key: value.to(trainer.device) if torch.is_tensor(value) else value
        for key, value in batch_cpu.items()
    }

    hidden_states = trainer.model.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
    teacher_branch = trainer._prepare_branch_inputs(batch, branch="teacher", stage=2)
    student_branch = trainer._prepare_branch_inputs(
        batch, branch="student", stage=2, teacher_branch=teacher_branch
    )

    initial_bus_mask = batch["student_bus_mask"].clone()
    initial_commit_mask = batch["commit_mask"].clone()

    plan_item_ids = batch.get("plan_item_ids")
    plan_item_mask = batch.get("plan_item_mask")

    _ = trainer._run_student_pass(
        hidden_states,
        batch,
        student_branch,
        stage=2,
        plan_item_ids=plan_item_ids,
        plan_item_mask=plan_item_mask,
    )

    updated_bus_mask = batch["student_bus_mask"]
    assert updated_bus_mask.sum().item() >= initial_bus_mask.sum().item()
    assert not torch.equal(initial_commit_mask, batch["commit_mask"].cpu())


def test_agreement_labels_autogenerated_when_missing() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=2, max_steps=0)
    trainer_cfg.loss_weights.agree = 1.0
    trainer_cfg.curriculum.L = 2
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer.state.stage_index = 4
    dataset = FakeDataset()
    batch = trainer.collator([dataset[0], dataset[1]])

    _, metrics = trainer._training_step(batch)

    assert "agreement_loss" in metrics
    assert "agreement_precision" in metrics
    assert metrics.get("agreement_auto") == 1.0


def test_generate_training_report_summarises_history() -> None:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))

    trainer_cfg = _training_config(batch_size=1, max_steps=0)
    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=None,
        eval_dataset=None,
    )

    trainer._log_metrics("train", {"loss": 1.0, "mask_ablation": 0.5, "kd_ce_ratio": 0.2})
    trainer.state.global_step = 5
    trainer._log_metrics(
        "train",
        {"loss": 0.8, "mask_ablation": 0.4, "kd_ce_ratio": 0.3, "agreement_precision": 0.9},
    )
    trainer._log_metrics("eval", {"eval_loss": 0.75})

    report = trainer.generate_training_report()

    assert report["global_step"] == trainer.state.global_step
    assert report["train_history_length"] == 2
    mask_summary = report["train_metrics"]["mask_ablation"]
    assert mask_summary["last"] == 0.4
    assert mask_summary["min"] == 0.4
    assert mask_summary["max"] == 0.5
    assert "eval_loss" in report["eval_metrics"]
