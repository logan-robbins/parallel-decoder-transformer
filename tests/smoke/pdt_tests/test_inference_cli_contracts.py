"""Fail-fast inference and ablation CLI contracts."""

from __future__ import annotations

import json

import pytest

import pdt.cli.ablate as ablate_cli
import pdt.cli.infer as infer_cli


def test_infer_requires_a_checkpoint() -> None:
    with pytest.raises(SystemExit, match="2"):
        infer_cli.main(
            [
                "--config",
                "configs/pdt_qwen3_4b.yaml",
                "--prompt",
                "test",
            ]
        )


def test_infer_rejects_invalid_numeric_inputs_before_model_construction(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        infer_cli,
        "PDTModel",
        lambda _config: pytest.fail("model must not be constructed"),
    )

    with pytest.raises(SystemExit, match="2"):
        infer_cli.main(
            [
                "--config",
                "configs/pdt_qwen3_4b.yaml",
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                "test",
                "--max-new-tokens",
                "0",
            ]
        )


@pytest.mark.parametrize("payload", [[], 7, {"other": "missing"}, {"prompt": ""}])
def test_ablation_prompt_jsonl_is_strict(tmp_path, payload) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="Prompt JSONL"):
        ablate_cli._load_prompts(path)


def test_ablation_validates_prompts_before_model_construction(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "output.json"
    config.touch()
    checkpoint.touch()
    prompts.write_text("[]\n")
    monkeypatch.setattr(
        ablate_cli,
        "PDTModel",
        lambda _config: pytest.fail("model must not be constructed"),
    )

    with pytest.raises(ValueError, match="Prompt JSONL"):
        ablate_cli.main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--prompts-file",
                str(prompts),
                "--output",
                str(output),
            ]
        )
