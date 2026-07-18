"""Compile a deterministic one-factor architecture screen into validated configs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Annotated

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pdt.config import load_config
from pdt.config.schemas import (
    PDTConfig,
    TRUNK_PROFILES,
    apply_trunk_profile,
    derive_instrumentation_layers,
)
from pdt.datasets.historical_source import canonical_json_bytes, sha256_file
from pdt.datasets.immutable_io import write_bytes_new


SWEEP_SPEC_SCHEMA = "pdt-architecture-screen-spec-v1"
SWEEP_MANIFEST_SCHEMA = "pdt-architecture-screen-manifest-v1"
Scalar = Annotated[int | float | str, Field(union_mode="left_to_right")]


class SweepParameter(StrEnum):
    TRUNK_PROFILE = "trunk_profile"
    INSTRUMENTED_LAYER_COUNT = "instrumented_layer_count"
    NOTES_DIM = "notes_dim"
    SNC_ATTENTION_WIDTH = "snc_attention_width"
    PLANNER_LAYER_COUNT = "planner_layer_count"
    PLANNER_FEEDFORWARD_WIDTH = "planner_feedforward_width"
    DYNAMIC_NUM_CODEBOOKS = "dynamic_num_codebooks"
    DYNAMIC_CODES_PER_CODEBOOK = "dynamic_codes_per_codebook"
    SNC_GATE_INIT = "snc_gate_init"
    PLAN_GATE_INIT = "plan_gate_init"
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"
    COORDINATION_SOURCE = "coordination_source"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchitectureSweepSpec(_StrictModel):
    schema_version: str = Field(pattern=rf"^{SWEEP_SPEC_SCHEMA}$")
    base_config: Path
    baseline: dict[SweepParameter, Scalar]
    axes: dict[SweepParameter, tuple[Scalar, ...]]
    seeds: tuple[int, ...] = Field(min_length=3)
    expected_variants: int = Field(gt=1)
    expected_runs: int = Field(gt=3)

    @model_validator(mode="after")
    def validate_screen(self) -> ArchitectureSweepSpec:
        if set(self.axes) != set(self.baseline):
            raise ValueError("Sweep baseline and axes must name exactly the same parameters.")
        if len(set(self.seeds)) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("Sweep seeds must be unique non-negative integers.")
        variant_count = 1
        for parameter, values in self.axes.items():
            if len(values) < 2 or len(set(values)) != len(values):
                raise ValueError(
                    f"Sweep axis {parameter.value!r} requires at least two unique values."
                )
            if self.baseline[parameter] not in values:
                raise ValueError(
                    f"Sweep baseline for {parameter.value!r} must occur in its axis."
                )
            variant_count += len(values) - 1
        if variant_count != self.expected_variants:
            raise ValueError(
                f"Sweep expected_variants={self.expected_variants}, computed {variant_count}."
            )
        run_count = variant_count * len(self.seeds)
        if run_count != self.expected_runs:
            raise ValueError(
                f"Sweep expected_runs={self.expected_runs}, computed {run_count}."
            )
        return self


class SweepRun(_StrictModel):
    run_id: str = Field(pattern=r"^[a-z0-9_]+-[0-9a-f]{12}-s[0-9]+$")
    seed: int = Field(ge=0)
    changed_parameter: str
    assignments: dict[str, Scalar]
    config_file: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArchitectureSweepManifest(_StrictModel):
    schema_version: str = Field(pattern=rf"^{SWEEP_MANIFEST_SCHEMA}$")
    design: str = Field(pattern=r"^one-factor-at-a-time-with-seeds$")
    spec_file: str
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_config_file: str
    base_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variants: int = Field(gt=1)
    runs: tuple[SweepRun, ...] = Field(min_length=1)


def compile_architecture_sweep(
    spec_path: Path,
    output_dir: Path,
) -> ArchitectureSweepManifest:
    """Publish all validated configs atomically; never skip an invalid combination."""

    if not spec_path.is_file():
        raise FileNotFoundError(f"Architecture sweep spec does not exist: {spec_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace architecture sweep: {output_dir}")
    loaded = OmegaConf.load(spec_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError("Architecture sweep spec must be a top-level mapping.")
    payload = OmegaConf.to_container(loaded, resolve=True)
    spec = ArchitectureSweepSpec.model_validate(payload)
    base_path = (
        spec.base_config
        if spec.base_config.is_absolute()
        else (spec_path.parent / spec.base_config).resolve()
    )
    base = load_config(base_path)
    variants = _screen_variants(spec)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    runs: list[SweepRun] = []
    try:
        for changed_parameter, assignments in variants:
            for seed in spec.seeds:
                config = deepcopy(base)
                for parameter, value in assignments.items():
                    _apply_assignment(config, parameter, value)
                config.training.seed = seed
                config.training.causal_eval_seed = seed
                assignment_payload = {
                    parameter.value: value
                    for parameter, value in sorted(
                        assignments.items(),
                        key=lambda row: row[0].value,
                    )
                }
                identity = json.dumps(
                    {
                        "changed_parameter": changed_parameter,
                        "assignments": assignment_payload,
                        "seed": seed,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                run_id = (
                    f"{changed_parameter.replace('baseline', 'base')}-"
                    f"{digest[:12]}-s{seed}"
                )
                config.training.telemetry_dir = str(
                    Path(config.training.telemetry_dir) / "architecture_screen" / run_id
                )
                config.validate()
                destination = temporary / f"{run_id}.yaml"
                yaml_text = OmegaConf.to_yaml(
                    OmegaConf.create(asdict(config)),
                    resolve=True,
                    sort_keys=False,
                )
                write_bytes_new(destination, yaml_text.encode("utf-8"))
                runs.append(
                    SweepRun(
                        run_id=run_id,
                        seed=seed,
                        changed_parameter=changed_parameter,
                        assignments=assignment_payload,
                        config_file=destination.name,
                        config_sha256=sha256_file(destination),
                    )
                )
        if len(runs) != spec.expected_runs:
            raise RuntimeError(
                f"Sweep compiler produced {len(runs)} runs; expected {spec.expected_runs}."
            )
        manifest = ArchitectureSweepManifest(
            schema_version=SWEEP_MANIFEST_SCHEMA,
            design="one-factor-at-a-time-with-seeds",
            spec_file=spec_path.name,
            spec_sha256=sha256_file(spec_path),
            base_config_file=base_path.name,
            base_config_sha256=sha256_file(base_path),
            variants=len(variants),
            runs=tuple(runs),
        )
        write_bytes_new(temporary / "manifest.json", canonical_json_bytes(manifest))
        os.rename(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def _screen_variants(
    spec: ArchitectureSweepSpec,
) -> tuple[tuple[str, dict[SweepParameter, Scalar]], ...]:
    baseline = dict(spec.baseline)
    variants: list[tuple[str, dict[SweepParameter, Scalar]]] = [
        ("baseline", baseline)
    ]
    for parameter in sorted(spec.axes, key=lambda item: item.value):
        for value in spec.axes[parameter]:
            if value == spec.baseline[parameter]:
                continue
            assignment = dict(baseline)
            assignment[parameter] = value
            variants.append((parameter.value, assignment))
    return tuple(variants)


def _apply_assignment(
    config: PDTConfig,
    parameter: SweepParameter,
    value: Scalar,
) -> None:
    if parameter is SweepParameter.TRUNK_PROFILE:
        if not isinstance(value, str) or value not in TRUNK_PROFILES:
            raise ValueError(f"Invalid trunk_profile sweep value {value!r}.")
        apply_trunk_profile(config, value)
    elif parameter is SweepParameter.INSTRUMENTED_LAYER_COUNT:
        count = _positive_int(value, parameter)
        config.instrumentation.instrumented_layer_count = count
        depth = TRUNK_PROFILES[config.trunk.profile].num_hidden_layers
        config.instrumentation.fork_layer = depth - count
        config.instrumentation.target_layers = derive_instrumentation_layers(depth, count)
    elif parameter is SweepParameter.NOTES_DIM:
        dimension = _positive_int(value, parameter)
        config.sidecar.notes_dim = dimension
        config.sidecar.snc.notes_dim = dimension
        config.sidecar.plan_memory_proj.notes_dim = dimension
        config.sidecar.speculation_head.notes_dim = dimension
        config.runtime.notes_bus.snapshot_dim = dimension
    elif parameter is SweepParameter.SNC_ATTENTION_WIDTH:
        config.sidecar.snc.attention_width = _positive_int(value, parameter)
    elif parameter is SweepParameter.PLANNER_LAYER_COUNT:
        config.sidecar.planner_head.num_layers = _positive_int(value, parameter)
    elif parameter is SweepParameter.PLANNER_FEEDFORWARD_WIDTH:
        config.sidecar.planner_head.feedforward_width = _positive_int(value, parameter)
    elif parameter is SweepParameter.DYNAMIC_NUM_CODEBOOKS:
        count = _positive_int(value, parameter)
        config.sidecar.speculation_head.num_codebooks = count
        config.runtime.notes_bus.num_codebooks = count
    elif parameter is SweepParameter.DYNAMIC_CODES_PER_CODEBOOK:
        count = _positive_int(value, parameter)
        config.sidecar.speculation_head.codes_per_codebook = count
        config.runtime.notes_bus.codes_per_codebook = count
    elif parameter is SweepParameter.SNC_GATE_INIT:
        config.instrumentation.snc_gate_init = _finite_float(value, parameter)
    elif parameter is SweepParameter.PLAN_GATE_INIT:
        config.instrumentation.plan_gate_init = _finite_float(value, parameter)
    elif parameter is SweepParameter.LEARNING_RATE:
        config.training.optimizer.learning_rate = _positive_float(value, parameter)
    elif parameter is SweepParameter.WEIGHT_DECAY:
        weight_decay = _finite_float(value, parameter)
        if weight_decay < 0:
            raise ValueError("weight_decay sweep values must be non-negative.")
        config.training.optimizer.weight_decay = weight_decay
    elif parameter is SweepParameter.COORDINATION_SOURCE:
        if value not in {"bus", "self_only"}:
            raise ValueError(f"Invalid coordination_source sweep value {value!r}.")
        config.instrumentation.coordination_source = value  # type: ignore[assignment]
    else:
        raise AssertionError(f"Unhandled sweep parameter {parameter!r}.")


def _positive_int(value: Scalar, parameter: SweepParameter) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{parameter.value} sweep values must be positive integers.")
    return value


def _finite_float(value: Scalar, parameter: SweepParameter) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{parameter.value} sweep values must be numeric.")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{parameter.value} sweep values must be finite.")
    return result


def _positive_float(value: Scalar, parameter: SweepParameter) -> float:
    result = _finite_float(value, parameter)
    if result <= 0:
        raise ValueError(f"{parameter.value} sweep values must be positive.")
    return result


__all__ = [
    "ArchitectureSweepManifest",
    "ArchitectureSweepSpec",
    "SWEEP_MANIFEST_SCHEMA",
    "SWEEP_SPEC_SCHEMA",
    "SweepParameter",
    "compile_architecture_sweep",
]
