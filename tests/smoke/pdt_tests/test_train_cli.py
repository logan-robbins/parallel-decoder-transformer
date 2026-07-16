"""Training CLI resume forwarding contracts without constructing a real trunk."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pdt.cli.train as train_cli


def _install_training_fakes(monkeypatch, events: list[object]):
    config = SimpleNamespace(
        instrumentation=SimpleNamespace(coordination_source="bus"),
        training=SimpleNamespace(
            telemetry_dir="original",
            dataset_path="train.jsonl",
            eval_dataset_path="validation.jsonl",
            max_steps=50_000,
            grad_accumulation=16,
            save_every=2500,
            eval_interval=10_000,
            log_interval=25,
            optimizer=SimpleNamespace(warmup_steps=1250),
            curriculum=SimpleNamespace(stage_schedule=(0, 3750, 10_000, 25_000)),
        ),
        validate=lambda: None,
    )

    def fake_load_config(path: Path) -> object:
        events.append(("load_config", path))
        return config

    class FakeModel:
        def __init__(self, received_config: object) -> None:
            assert received_config is config
            events.append("model")

    class FakeTrainer:
        def __init__(self, model: FakeModel, received_config: object) -> None:
            assert isinstance(model, FakeModel)
            assert received_config is config
            events.append("trainer")

        def resume_from_checkpoint(self, path: Path) -> None:
            events.append(("resume", path))

        def train(self) -> None:
            events.append("train")

        def optimizer_probe(self) -> None:
            events.append("optimizer_probe")

        def evaluate(self) -> None:
            events.append("evaluate")

    monkeypatch.setattr(train_cli, "load_config", fake_load_config)
    monkeypatch.setattr(train_cli, "PDTModel", FakeModel)
    monkeypatch.setattr(train_cli, "PDTTrainer", FakeTrainer)
    return config


def test_train_cli_default_path_does_not_resume(tmp_path: Path, monkeypatch) -> None:
    events: list[object] = []
    _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    train_cli.main(["--config", str(config_path)])

    assert events == [
        ("load_config", config_path),
        "model",
        "trainer",
        "train",
    ]


def test_train_cli_forwards_resume_before_training(tmp_path: Path, monkeypatch) -> None:
    events: list[object] = []
    _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()

    train_cli.main(["--config", str(config_path), "--resume", str(checkpoint_path)])

    assert events == [
        ("load_config", config_path),
        "model",
        "trainer",
        ("resume", checkpoint_path),
        "train",
    ]


def test_train_cli_rejects_missing_resume_before_model_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []
    _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    missing = tmp_path / "missing.pt"

    with pytest.raises(SystemExit, match="2"):
        train_cli.main(["--config", str(config_path), "--resume", str(missing)])

    assert events == []


def test_train_cli_applies_isolated_condition_before_model_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []
    config = _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    telemetry_dir = tmp_path / "self-only"

    train_cli.main(
        [
            "--config",
            str(config_path),
            "--coordination-source",
            "self_only",
            "--telemetry-dir",
            str(telemetry_dir),
        ]
    )

    assert config.instrumentation.coordination_source == "self_only"
    assert config.training.telemetry_dir == str(telemetry_dir)
    assert events[-3:] == ["model", "trainer", "train"]


def test_train_cli_requires_isolated_directory_for_source_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []
    _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    with pytest.raises(SystemExit, match="2"):
        train_cli.main(
            [
                "--config",
                str(config_path),
                "--coordination-source",
                "self_only",
            ]
        )

    assert events == []


def test_train_cli_runs_fresh_optimizer_probe_in_isolated_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []
    config = _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    telemetry_dir = tmp_path / "probe"

    train_cli.main(
        [
            "--config",
            str(config_path),
            "--optimizer-probe",
            "--telemetry-dir",
            str(telemetry_dir),
        ]
    )

    assert config.training.grad_accumulation == 1
    assert events[-3:] == ["model", "trainer", "optimizer_probe"]


def test_train_cli_applies_locked_short_run_overrides_before_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[object] = []
    config = _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "validation.jsonl"
    train_path.touch()
    eval_path.touch()

    train_cli.main(
        [
            "--config",
            str(config_path),
            "--dataset-path",
            str(train_path),
            "--eval-dataset-path",
            str(eval_path),
            "--max-steps",
            "256",
            "--grad-accumulation",
            "1",
            "--warmup-steps",
            "16",
            "--stage-schedule",
            "0",
            "16",
            "64",
            "128",
            "--save-every",
            "64",
            "--eval-interval",
            "256",
            "--log-interval",
            "1",
        ]
    )

    assert config.training.dataset_path == str(train_path)
    assert config.training.eval_dataset_path == str(eval_path)
    assert config.training.max_steps == 256
    assert config.training.grad_accumulation == 1
    assert config.training.optimizer.warmup_steps == 16
    assert config.training.curriculum.stage_schedule == (0, 16, 64, 128)
    assert config.training.save_every == 64
    assert config.training.eval_interval == 256
    assert config.training.log_interval == 1


def test_train_cli_runs_checkpoint_evaluation_without_training(tmp_path: Path, monkeypatch) -> None:
    events: list[object] = []
    _install_training_fakes(monkeypatch, events)
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    eval_path = tmp_path / "validation.jsonl"
    telemetry_dir = tmp_path / "heldout-eval"
    for path in (config_path, checkpoint_path, eval_path):
        path.touch()

    train_cli.main(
        [
            "--config",
            str(config_path),
            "--resume",
            str(checkpoint_path),
            "--eval-only",
            "--eval-dataset-path",
            str(eval_path),
            "--telemetry-dir",
            str(telemetry_dir),
        ]
    )

    assert events[-4:] == ["model", "trainer", ("resume", checkpoint_path), "evaluate"]
