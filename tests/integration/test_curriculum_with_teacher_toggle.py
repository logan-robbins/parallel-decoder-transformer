from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest
import torch
from torch.utils.data import Dataset

from parallel_decoder_transformer.data.snapshots import SnapshotFeatures
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


class FakeTrunk(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.param = torch.nn.Parameter(torch.zeros(1))
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
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
        return type("Outputs", (), {"hidden_states": [hidden, hidden]})()


class CurriculumDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        notes_dim = 4
        streams = ["stream_1", "stream_2", "stream_3"]
        notes_student = torch.zeros((len(streams), notes_dim), dtype=torch.float32)
        notes_teacher = torch.full((len(streams), notes_dim), 0.25, dtype=torch.float32)
        coverage_vector = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        teacher_snapshot = {
            "notes": notes_teacher.clone(),
            "stride": 4,
            "version": 0,
            "coverage": coverage_vector.clone(),
            "source": "teacher",
        }
        metadata = {
            "teacher_plan": {
                "plan": [
                    {"stream_id": "stream_1", "summary": "Set context", "notes": ["Set context"]},
                    {
                        "stream_id": "stream_2",
                        "summary": "Provide evidence",
                        "notes": ["Provide evidence"],
                    },
                    {
                        "stream_id": "stream_3",
                        "summary": "Summarise outcome",
                        "notes": ["Summarise outcome"],
                    },
                ],
                "segments": [],
            },
            "teacher_notes": {
                "stream_1": ["Set context"],
                "stream_2": ["Provide evidence"],
                "stream_3": ["Summarise outcome"],
            },
        }
        return {
            "example_id": f"example-{index}",
            "student_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "student_labels": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "planner_ids": torch.tensor([5, 6, 7, 8], dtype=torch.long),
            "notes_student": notes_student,
            "notes_teacher": notes_teacher,
            "teacher_snapshots": [teacher_snapshot],
            "plan_items": ["Provide evidence"],
            "plan_catalog": ["Set context", "Provide evidence", "Summarise outcome"],
            "plan_catalog_streams": streams,
            "coverage_targets": [0, 1, 0],
            "notes_text": "Core evidence note",
            "plan_tokens": ["stream_2::Provide evidence"],
            "notes_tokens": ["stream_2 note"],
            "stream_id": "stream_2",
            "metadata": metadata,
        }


class StubTeacherProvider(TeacherNotesProviderBase):
    def __init__(self, *, notes_dim: int, stream_to_id: Mapping[str, int]) -> None:
        self.notes_dim = notes_dim
        self.stream_order = [
            stream for stream, _ in sorted(stream_to_id.items(), key=lambda item: item[1])
        ]
        self.calls = 0

    def fetch(self, example: Mapping[str, object]) -> TeacherNotes:
        self.calls += 1
        stream_count = len(self.stream_order)
        notes = torch.full((stream_count, self.notes_dim), 0.5, dtype=torch.float32)
        coverage = torch.ones(stream_count, dtype=torch.float32)
        snapshots = [
            SnapshotFeatures(
                notes=notes.clone(),
                stride=4,
                version=0,
                coverage=coverage.clone(),
                source="teacher",
            )
        ]
        raw_notes = {stream: [f"{stream}-note-{self.calls}"] for stream in self.stream_order}
        return TeacherNotes(notes=notes, snapshots=snapshots, raw_notes=raw_notes)


def _build_model() -> tuple[ParallelDecoderTransformer, ParallelDecoderModelConfig]:
    model_cfg = ParallelDecoderModelConfig(hidden_size=8, vocab_size=16, notes_dim=4, num_heads=2)
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.attach_model(FakeTrunk(model_cfg.hidden_size, model_cfg.vocab_size))
    return model, model_cfg


def _build_training_config() -> TrainingConfig:
    cfg = TrainingConfig(batch_size=1, max_steps=6, log_interval=1, eval_interval=10)
    cfg.curriculum.stage_schedule = (0, 1, 2, 3, 4)
    cfg.curriculum.steps_per_stage = 0
    cfg.metrics.stability_every = 1
    cfg.dataset_teacher = DatasetTeacherConfig()
    cfg.device = "cpu"
    return cfg


def _assert_all_stages_visited(trainer: Trainer) -> None:
    observed = {entry["stage"] for entry in trainer.state.stage_history}
    assert {0, 1, 2, 3, 4}.issubset(observed), f"stage history missing entries: {observed}"


def test_trainer_requires_teacher_runner_config() -> None:
    model, model_cfg = _build_model()
    trainer_cfg = _build_training_config()
    trainer_cfg.teacher_runner = None

    with pytest.raises(ValueError):
        Trainer(
            model,
            trainer_cfg,
            collator_config=model_cfg.collator,
            dataset=CurriculumDataset(),
            eval_dataset=None,
        )


def test_trainer_rejects_disabled_teacher_branch() -> None:
    model, model_cfg = _build_model()
    trainer_cfg = _build_training_config()
    trainer_cfg.teacher.enabled = False

    with pytest.raises(ValueError, match="teacher branch is mandatory"):
        Trainer(
            model,
            trainer_cfg,
            collator_config=model_cfg.collator,
            dataset=CurriculumDataset(),
            eval_dataset=None,
        )


def test_curriculum_progresses_with_teacher_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = CurriculumDataset()
    stub_instances: list[StubTeacherProvider] = []

    def _provider_factory(
        config: DatasetTeacherConfig,
        *,
        notes_dim: int,
        stream_to_id: Mapping[str, int],
        embedder: object | None = None,
        runner: object | None = None,
    ) -> StubTeacherProvider:
        provider = StubTeacherProvider(notes_dim=notes_dim, stream_to_id=stream_to_id)
        stub_instances.append(provider)
        return provider

    monkeypatch.setattr(
        "parallel_decoder_transformer.training.trainer.DatasetTeacherNotesProvider",
        _provider_factory,
    )

    model, model_cfg = _build_model()
    trainer_cfg = _build_training_config()
    trainer_cfg.teacher_runner.cache_dir = str(tmp_path)

    trainer = Trainer(
        model,
        trainer_cfg,
        collator_config=model_cfg.collator,
        dataset=dataset,
        eval_dataset=None,
    )
    trainer.fit()

    _assert_all_stages_visited(trainer)
    assert stub_instances and stub_instances[0].calls > 0
    assert any("coverage_f1" in metrics for metrics in trainer.metric_history.get("train", []))
