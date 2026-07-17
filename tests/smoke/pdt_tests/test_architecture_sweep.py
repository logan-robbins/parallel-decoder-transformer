"""Contract tests for the deterministic architecture-screen compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdt.config import load_config
from pdt.datasets.historical_source import sha256_file
from pdt.experiments.sweep import (
    SWEEP_MANIFEST_SCHEMA,
    ArchitectureSweepManifest,
    compile_architecture_sweep,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCREEN_SPEC = PROJECT_ROOT / "configs" / "pdt_architecture_screen.yaml"


def test_production_architecture_screen_compiles_every_declared_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "architecture_screen"
    manifest = compile_architecture_sweep(SCREEN_SPEC, output)

    assert manifest.schema_version == SWEEP_MANIFEST_SCHEMA
    assert manifest.design == "one-factor-at-a-time-with-seeds"
    assert manifest.variants == 25
    assert len(manifest.runs) == 75
    assert len({run.run_id for run in manifest.runs}) == 75
    assert len({run.seed for run in manifest.runs}) == 3
    assert (output / "manifest.json").is_file()

    parsed = ArchitectureSweepManifest.model_validate_json(
        (output / "manifest.json").read_bytes()
    )
    assert parsed == manifest
    for run in manifest.runs:
        config_path = output / run.config_file
        assert sha256_file(config_path) == run.config_sha256
        config = load_config(config_path)
        assert config.training.seed == run.seed
        assert config.training.causal_eval_seed == run.seed
        assert run.run_id in config.training.telemetry_dir


def test_screen_publication_is_atomic_when_a_run_is_invalid(
    tmp_path: Path,
) -> None:
    invalid_spec = tmp_path / "invalid.yaml"
    spec_text = SCREEN_SPEC.read_text().replace(
        'instrumented_layer_count: [12, 6, 18]',
        'instrumented_layer_count: [12, 6, 100]',
    )
    spec_text = spec_text.replace(
        'base_config: "pdt_qwen3_4b.yaml"',
        f'base_config: "{(PROJECT_ROOT / "configs" / "pdt_qwen3_4b.yaml").as_posix()}"',
    )
    invalid_spec.write_text(spec_text)
    output = tmp_path / "unpublished"

    with pytest.raises(ValueError, match="instrumented_layer_count"):
        compile_architecture_sweep(invalid_spec, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".unpublished.*.tmp"))
