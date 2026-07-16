"""Training CLI resume forwarding contracts without constructing a real trunk."""

from __future__ import annotations

from pathlib import Path

import pytest

import pdt.cli.train as train_cli


def _install_training_fakes(monkeypatch, events: list[object]) -> None:
    config = object()

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

    monkeypatch.setattr(train_cli, "load_config", fake_load_config)
    monkeypatch.setattr(train_cli, "PDTModel", FakeModel)
    monkeypatch.setattr(train_cli, "PDTTrainer", FakeTrainer)


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
