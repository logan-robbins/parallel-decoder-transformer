"""Versioned, strict checkpoint persistence for PDT trainable state.

PDT checkpoints exclude the frozen shared lower trunk and retain every
trainable physical-decoder parameter bank plus the sidecar.  A checkpoint is
accepted only when its pinned trunk, fork, decoder count, branch depth, and
coordination condition match the instantiated model.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


CHECKPOINT_FORMAT_VERSION = 5

_ROOT_FIELDS = frozenset({"format_version", "identity", "phi", "training"})
_IDENTITY_FIELDS = frozenset(
    {
        "base_model",
        "revision",
        "instrumented_layers",
        "coordination_source",
        "fork_layer",
        "num_decoders",
    }
)
_PHI_FIELDS = frozenset({"sidecar", "physical_decoder"})
_TRAINING_FIELDS = frozenset(
    {
        "global_step",
        "stage",
        "optimizer_type",
        "optimizer",
        "optimizer_parameter_manifest",
        "scheduler_type",
        "scheduler",
    }
)
_OPTIMIZER_FIELDS = frozenset({"state", "param_groups"})
_OPTIMIZER_PARAMETER_FIELDS = frozenset({"name", "shape", "dtype"})

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointIOError",
    "CheckpointIdentity",
    "CheckpointMetadata",
    "CheckpointMismatchError",
    "load_checkpoint",
    "resume_checkpoint",
    "save_checkpoint",
]


class CheckpointError(RuntimeError):
    """Base class for checkpoint contract failures."""


class CheckpointIOError(CheckpointError):
    """The checkpoint could not be read or written."""


class CheckpointCorruptError(CheckpointError):
    """The file is unreadable or violates the versioned payload schema."""


class CheckpointMismatchError(CheckpointError):
    """The checkpoint is valid but incompatible with the target model."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Frozen trunk and instrumentation identity required by a checkpoint."""

    base_model: str
    revision: str
    instrumented_layers: tuple[int, ...]
    coordination_source: str
    fork_layer: int
    num_decoders: int


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Validated checkpoint metadata returned after a successful load/save."""

    format_version: int
    identity: CheckpointIdentity
    global_step: int
    stage: int


def save_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    global_step: int,
    stage: int,
) -> CheckpointMetadata:
    """Atomically save a complete, canonical PDT training checkpoint.

    The frozen trunk is represented by ``base_model`` and ``revision`` rather
    than duplicated weights.  ``os.replace`` publishes the completed file only
    after ``torch.save`` and ``fsync`` succeed in the destination directory.
    """

    identity, physical_decoder = _model_identity_and_layers(model)
    global_step = _validate_nonnegative_int(global_step, "global_step")
    stage = _validate_stage(stage)
    optimizer_state = optimizer.state_dict()
    optimizer_parameter_manifest = _optimizer_parameter_manifest(
        model, physical_decoder, optimizer
    )
    scheduler_state = scheduler.state_dict()
    _validate_optimizer_state(optimizer_state, "training.optimizer")
    _validate_optimizer_parameter_manifest(
        optimizer_parameter_manifest,
        optimizer_state,
        model,
        physical_decoder,
    )
    _validate_scheduler_state(scheduler_state, "training.scheduler")

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "identity": {
            "base_model": identity.base_model,
            "revision": identity.revision,
            "instrumented_layers": identity.instrumented_layers,
            "coordination_source": identity.coordination_source,
            "fork_layer": identity.fork_layer,
            "num_decoders": identity.num_decoders,
        },
        "phi": {
            "sidecar": _module_state_for_save(model.sidecar, "model.sidecar"),
            "physical_decoder": _module_state_for_save(
                physical_decoder,
                "model.physical_decoder",
            ),
        },
        "training": {
            "global_step": global_step,
            "stage": stage,
            "optimizer_type": _qualified_type_name(optimizer),
            "optimizer": optimizer_state,
            "optimizer_parameter_manifest": optimizer_parameter_manifest,
            "scheduler_type": _qualified_type_name(scheduler),
            "scheduler": scheduler_state,
        },
    }
    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return CheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        identity=identity,
        global_step=global_step,
        stage=stage,
    )


def load_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
) -> CheckpointMetadata:
    """Strictly restore phi for inference and return training metadata.

    The full training payload is still schema-validated.  Optimizer and
    scheduler objects are intentionally untouched; use :func:`resume_checkpoint`
    when continuing training.
    """

    payload, metadata, physical_decoder = _read_and_validate(path, model)
    _load_phi(payload["phi"], model, physical_decoder)
    return metadata


def resume_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> CheckpointMetadata:
    """Strictly restore phi, optimizer, and scheduler for training resume."""

    payload, metadata, physical_decoder = _read_and_validate(path, model)
    training = payload["training"]
    optimizer_type = _qualified_type_name(optimizer)
    if training["optimizer_type"] != optimizer_type:
        raise CheckpointMismatchError(
            "Optimizer type mismatch: "
            f"checkpoint={training['optimizer_type']!r}, runtime={optimizer_type!r}."
        )
    scheduler_type = _qualified_type_name(scheduler)
    if training["scheduler_type"] != scheduler_type:
        raise CheckpointMismatchError(
            "Scheduler type mismatch: "
            f"checkpoint={training['scheduler_type']!r}, runtime={scheduler_type!r}."
        )
    runtime_optimizer_manifest = _optimizer_parameter_manifest(
        model, physical_decoder, optimizer
    )
    _validate_runtime_optimizer_manifest(
        training["optimizer_parameter_manifest"], runtime_optimizer_manifest
    )
    try:
        optimizer.load_state_dict(training["optimizer"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointMismatchError(
            f"Checkpoint optimizer state is incompatible with {type(optimizer).__name__}: {exc}"
        ) from exc
    try:
        scheduler.load_state_dict(training["scheduler"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointMismatchError(
            f"Checkpoint scheduler state is incompatible with {type(scheduler).__name__}: {exc}"
        ) from exc
    _load_phi(payload["phi"], model, physical_decoder)
    return metadata


def _read_and_validate(
    path: str | os.PathLike[str],
    model: Any,
) -> tuple[dict[str, Any], CheckpointMetadata, nn.Module]:
    payload = _read_payload(Path(path))
    _require_exact_fields(payload, _ROOT_FIELDS, "checkpoint root")

    version = payload["format_version"]
    if type(version) is not int:
        raise CheckpointCorruptError("checkpoint.format_version must be an integer.")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointMismatchError(
            f"Unsupported checkpoint format_version={version}; "
            f"this runtime requires {CHECKPOINT_FORMAT_VERSION}."
        )

    saved_identity = _parse_identity(payload["identity"])
    expected_identity, physical_decoder = _model_identity_and_layers(model)
    if saved_identity.base_model != expected_identity.base_model:
        raise CheckpointMismatchError(
            "Frozen trunk base_model mismatch: "
            f"checkpoint={saved_identity.base_model!r}, model={expected_identity.base_model!r}."
        )
    if saved_identity.revision != expected_identity.revision:
        raise CheckpointMismatchError(
            "Frozen trunk revision mismatch: "
            f"checkpoint={saved_identity.revision!r}, model={expected_identity.revision!r}."
        )
    if saved_identity.instrumented_layers != expected_identity.instrumented_layers:
        raise CheckpointMismatchError(
            "Instrumentation layer mismatch: "
            f"checkpoint={saved_identity.instrumented_layers}, "
            f"model={expected_identity.instrumented_layers}."
        )
    if saved_identity.coordination_source != expected_identity.coordination_source:
        raise CheckpointMismatchError(
            "Coordination source mismatch: "
            f"checkpoint={saved_identity.coordination_source!r}, "
            f"model={expected_identity.coordination_source!r}."
        )
    if saved_identity.fork_layer != expected_identity.fork_layer:
        raise CheckpointMismatchError(
            "Physical decoder fork mismatch: "
            f"checkpoint={saved_identity.fork_layer}, model={expected_identity.fork_layer}."
        )
    if saved_identity.num_decoders != expected_identity.num_decoders:
        raise CheckpointMismatchError(
            "Physical decoder count mismatch: "
            f"checkpoint={saved_identity.num_decoders}, "
            f"model={expected_identity.num_decoders}."
        )

    phi = _require_mapping(payload["phi"], "checkpoint.phi")
    _require_exact_fields(phi, _PHI_FIELDS, "checkpoint.phi")
    _validate_module_state(phi["sidecar"], model.sidecar, "checkpoint.phi.sidecar")
    _validate_module_state(
        phi["physical_decoder"],
        physical_decoder,
        "checkpoint.phi.physical_decoder",
    )

    training = _require_mapping(payload["training"], "checkpoint.training")
    _require_exact_fields(training, _TRAINING_FIELDS, "checkpoint.training")
    global_step = _validate_nonnegative_int(
        training["global_step"], "checkpoint.training.global_step", corrupt=True
    )
    stage = _validate_stage(training["stage"], corrupt=True)
    _validate_identity_string(
        training["optimizer_type"], "checkpoint.training.optimizer_type", corrupt=True
    )
    _validate_optimizer_state(training["optimizer"], "checkpoint.training.optimizer")
    _validate_optimizer_parameter_manifest(
        training["optimizer_parameter_manifest"],
        training["optimizer"],
        model,
        physical_decoder,
    )
    _validate_identity_string(
        training["scheduler_type"], "checkpoint.training.scheduler_type", corrupt=True
    )
    _validate_scheduler_state(training["scheduler"], "checkpoint.training.scheduler")

    metadata = CheckpointMetadata(
        format_version=version,
        identity=saved_identity,
        global_step=global_step,
        stage=stage,
    )
    return payload, metadata, physical_decoder


def _read_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointIOError(f"Checkpoint file does not exist or is not a file: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointCorruptError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointCorruptError(
            f"Checkpoint root must be a dict, got {type(payload).__name__}."
        )
    return payload


def _parse_identity(value: Any) -> CheckpointIdentity:
    identity = _require_mapping(value, "checkpoint.identity")
    _require_exact_fields(identity, _IDENTITY_FIELDS, "checkpoint.identity")
    base_model = _validate_identity_string(identity["base_model"], "base_model", corrupt=True)
    revision = _validate_identity_string(identity["revision"], "revision", corrupt=True)
    layer_indices = _validate_layer_indices(
        identity["instrumented_layers"],
        "checkpoint.identity.instrumented_layers",
        corrupt=True,
    )
    coordination_source = _validate_coordination_source(
        identity["coordination_source"],
        "checkpoint.identity.coordination_source",
        corrupt=True,
    )
    fork_layer = _validate_nonnegative_int(
        identity["fork_layer"],
        "checkpoint.identity.fork_layer",
        corrupt=True,
    )
    num_decoders = _validate_nonnegative_int(
        identity["num_decoders"],
        "checkpoint.identity.num_decoders",
        corrupt=True,
    )
    if fork_layer <= 0:
        raise CheckpointCorruptError("checkpoint.identity.fork_layer must be positive.")
    if num_decoders != 3:
        raise CheckpointCorruptError(
            "checkpoint.identity.num_decoders must equal the canonical three."
        )
    return CheckpointIdentity(
        base_model,
        revision,
        layer_indices,
        coordination_source,
        fork_layer,
        num_decoders,
    )


def _model_identity_and_layers(model: Any) -> tuple[CheckpointIdentity, nn.Module]:
    try:
        trunk_config = model.config.trunk
        instrumentation_config = model.config.instrumentation
        sidecar = model.sidecar
        physical_decoder = model.physical_decoder
    except AttributeError as exc:
        raise CheckpointMismatchError(
            "Model must expose config.trunk, config.instrumentation, sidecar, "
            "and physical_decoder for checkpointing."
        ) from exc
    if not isinstance(sidecar, nn.Module):
        raise CheckpointMismatchError(
            f"model.sidecar must be torch.nn.Module, got {type(sidecar).__name__}."
        )
    if not isinstance(physical_decoder, nn.Module):
        raise CheckpointMismatchError(
            "model.physical_decoder must be a torch.nn.Module."
        )

    local_path = getattr(trunk_config, "local_path", None)
    if local_path is not None:
        raise CheckpointMismatchError(
            "Canonical PDT checkpoints require model.config.trunk.local_path to be None; "
            "local trunk contents are not represented by the checkpoint identity."
        )

    base_model = _validate_identity_string(trunk_config.base_model, "base_model")
    revision = _validate_identity_string(trunk_config.revision, "revision")
    coordination_source = _validate_coordination_source(
        getattr(instrumentation_config, "coordination_source", None),
        "model.config.instrumentation.coordination_source",
    )
    enabled = getattr(instrumentation_config, "enabled", None)
    if enabled is not True:
        raise CheckpointMismatchError(
            "Physical decoder checkpointing requires instrumentation.enabled=true."
        )
    configured_indices = _validate_layer_indices(
        instrumentation_config.target_layers,
        "model.config.instrumentation.target_layers",
    )
    actual_indices = tuple(getattr(physical_decoder, "layer_indices", ()))
    if actual_indices != configured_indices:
        raise CheckpointMismatchError(
            "Instantiated physical layers do not match model config: "
            f"actual={actual_indices}, configured={configured_indices}."
        )
    fork_layer = getattr(physical_decoder, "fork_layer", None)
    num_decoders = getattr(physical_decoder, "num_decoders", None)
    if type(fork_layer) is not int or fork_layer <= 0:
        raise CheckpointMismatchError("model.physical_decoder.fork_layer must be positive.")
    if fork_layer != getattr(instrumentation_config, "fork_layer", None):
        raise CheckpointMismatchError(
            "model.physical_decoder.fork_layer does not match model config."
        )
    if num_decoders != 3:
        raise CheckpointMismatchError(
            "model.physical_decoder.num_decoders must equal the canonical three."
        )
    return CheckpointIdentity(
        base_model,
        revision,
        actual_indices,
        coordination_source,
        fork_layer,
        num_decoders,
    ), physical_decoder


def _module_state_for_save(module: nn.Module, label: str) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key, value in module.state_dict().items():
        if not isinstance(value, Tensor):
            raise CheckpointMismatchError(
                f"{label}.state_dict()[{key!r}] must be a Tensor, got {type(value).__name__}."
            )
        result[key] = value.detach().cpu().clone()
    return result

def _validate_module_state(value: Any, module: nn.Module, label: str) -> None:
    saved = _require_mapping(value, label)
    expected = module.state_dict()
    _require_exact_fields(saved, frozenset(expected), label, mismatch=True)
    for key, target in expected.items():
        _validate_tensor(saved[key], target, f"{label}.{key}")


def _validate_tensor(value: Any, target: Tensor, label: str) -> None:
    if not isinstance(value, Tensor):
        raise CheckpointCorruptError(f"{label} must be a Tensor, got {type(value).__name__}.")
    if value.shape != target.shape:
        raise CheckpointMismatchError(
            f"{label} shape mismatch: checkpoint={tuple(value.shape)}, model={tuple(target.shape)}."
        )
    if value.dtype != target.dtype:
        raise CheckpointMismatchError(
            f"{label} dtype mismatch: checkpoint={value.dtype}, model={target.dtype}."
        )


def _load_phi(
    phi: Mapping[str, Any],
    model: Any,
    physical_decoder: nn.Module,
) -> None:
    # All key/shape/dtype checks have completed before mutation.  strict=True is
    # still used as the final guard against changes to module load semantics.
    try:
        model.sidecar.load_state_dict(phi["sidecar"], strict=True)
        physical_decoder.load_state_dict(phi["physical_decoder"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointMismatchError(f"Strict phi restore failed: {exc}") from exc


def _canonical_phi_parameters(
    model: Any, physical_decoder: nn.Module
) -> list[tuple[str, nn.Parameter]]:
    """Return every trainable model parameter once with a stable path."""

    result: list[tuple[str, nn.Parameter]] = []
    names: set[str] = set()
    parameter_names_by_id: dict[int, str] = {}

    def append(name: str, parameter: nn.Parameter) -> None:
        if name in names:
            raise CheckpointMismatchError(f"Duplicate canonical parameter name: {name!r}.")
        previous_name = parameter_names_by_id.get(id(parameter))
        if previous_name is not None:
            raise CheckpointMismatchError(
                "Canonical trainable parameters must not alias each other: "
                f"{previous_name!r} and {name!r} reference the same parameter."
            )
        names.add(name)
        parameter_names_by_id[id(parameter)] = name
        result.append((name, parameter))

    for local_name, parameter in physical_decoder.named_parameters():
        append(f"physical_decoder.{local_name}", parameter)
    for local_name, parameter in model.sidecar.named_parameters():
        append(f"sidecar.{local_name}", parameter)
    if not result:
        raise CheckpointMismatchError("Canonical PDT model exposes no trainable parameters.")
    return result


def _parameter_manifest_entry(name: str, parameter: nn.Parameter) -> dict[str, Any]:
    return {
        "name": name,
        "shape": tuple(parameter.shape),
        "dtype": str(parameter.dtype),
    }


def _optimizer_parameter_manifest(
    model: Any,
    physical_decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[list[dict[str, Any]]]:
    """Map canonical parameter identities exactly onto optimizer group order."""

    canonical = _canonical_phi_parameters(model, physical_decoder)
    canonical_by_id = {id(parameter): (name, parameter) for name, parameter in canonical}
    seen_parameter_ids: set[int] = set()
    manifest: list[list[dict[str, Any]]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list) or not parameters:
            raise CheckpointMismatchError(
                f"optimizer.param_groups[{group_index}].params must be a non-empty list."
            )
        manifest_group: list[dict[str, Any]] = []
        for parameter_index, parameter in enumerate(parameters):
            if not isinstance(parameter, nn.Parameter):
                raise CheckpointMismatchError(
                    f"optimizer.param_groups[{group_index}].params[{parameter_index}] "
                    f"must be torch.nn.Parameter, got {type(parameter).__name__}."
                )
            canonical_entry = canonical_by_id.get(id(parameter))
            if canonical_entry is None:
                raise CheckpointMismatchError(
                    f"optimizer.param_groups[{group_index}].params[{parameter_index}] "
                    "is not a canonical trainable parameter."
                )
            if id(parameter) in seen_parameter_ids:
                name = canonical_entry[0]
                raise CheckpointMismatchError(
                    f"Optimizer contains canonical parameter {name!r} more than once."
                )
            seen_parameter_ids.add(id(parameter))
            manifest_group.append(_parameter_manifest_entry(*canonical_entry))
        manifest.append(manifest_group)

    missing = [name for name, parameter in canonical if id(parameter) not in seen_parameter_ids]
    if missing:
        raise CheckpointMismatchError(f"Optimizer is missing canonical parameters: {missing}.")
    return manifest


def _validate_optimizer_parameter_manifest(
    value: Any,
    optimizer_state: Mapping[str, Any],
    model: Any,
    physical_decoder: nn.Module,
) -> None:
    """Validate saved optimizer ordering against its state and the target phi graph."""

    if not isinstance(value, list) or not value:
        raise CheckpointCorruptError(
            "checkpoint.training.optimizer_parameter_manifest must be a non-empty list."
        )
    state_groups = optimizer_state["param_groups"]
    if len(value) != len(state_groups):
        raise CheckpointCorruptError(
            "checkpoint.training.optimizer_parameter_manifest group count does not match "
            "checkpoint.training.optimizer.param_groups."
        )

    saved_entries: list[dict[str, Any]] = []
    for group_index, (manifest_group, state_group) in enumerate(zip(value, state_groups)):
        label = f"checkpoint.training.optimizer_parameter_manifest[{group_index}]"
        if not isinstance(manifest_group, list) or not manifest_group:
            raise CheckpointCorruptError(f"{label} must be a non-empty list.")
        if len(manifest_group) != len(state_group["params"]):
            raise CheckpointCorruptError(
                f"{label} length does not match "
                f"checkpoint.training.optimizer.param_groups[{group_index}].params."
            )
        for parameter_index, raw_entry in enumerate(manifest_group):
            entry_label = f"{label}[{parameter_index}]"
            entry = _require_mapping(raw_entry, entry_label)
            _require_exact_fields(entry, _OPTIMIZER_PARAMETER_FIELDS, entry_label)
            name = _validate_identity_string(entry["name"], f"{entry_label}.name", corrupt=True)
            shape = entry["shape"]
            if not isinstance(shape, tuple) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                raise CheckpointCorruptError(
                    f"{entry_label}.shape must be a tuple of non-negative integers."
                )
            dtype = _validate_identity_string(entry["dtype"], f"{entry_label}.dtype", corrupt=True)
            saved_entries.append({"name": name, "shape": shape, "dtype": dtype})

    saved_names = [entry["name"] for entry in saved_entries]
    if len(saved_names) != len(set(saved_names)):
        raise CheckpointCorruptError(
            "checkpoint.training.optimizer_parameter_manifest contains duplicate names."
        )

    expected = {
        name: _parameter_manifest_entry(name, parameter)
        for name, parameter in _canonical_phi_parameters(model, physical_decoder)
    }
    saved_by_name = {entry["name"]: entry for entry in saved_entries}
    missing = sorted(set(expected) - set(saved_by_name))
    extra = sorted(set(saved_by_name) - set(expected))
    if missing or extra:
        raise CheckpointMismatchError(
            "Optimizer parameter manifest does not cover the target model's canonical "
            f"parameters: missing={missing}, unexpected={extra}."
        )
    for name, expected_entry in expected.items():
        saved_entry = saved_by_name[name]
        if saved_entry["shape"] != expected_entry["shape"]:
            raise CheckpointMismatchError(
                f"Optimizer parameter manifest shape mismatch for {name!r}: "
                f"checkpoint={saved_entry['shape']}, model={expected_entry['shape']}."
            )
        if saved_entry["dtype"] != expected_entry["dtype"]:
            raise CheckpointMismatchError(
                f"Optimizer parameter manifest dtype mismatch for {name!r}: "
                f"checkpoint={saved_entry['dtype']!r}, model={expected_entry['dtype']!r}."
            )


def _validate_runtime_optimizer_manifest(
    saved: list[list[dict[str, Any]]],
    runtime: list[list[dict[str, Any]]],
) -> None:
    if saved == runtime:
        return
    if len(saved) != len(runtime):
        detail = f"group count checkpoint={len(saved)}, runtime={len(runtime)}"
    else:
        detail = "parameter ordering or group membership differs"
        for group_index, (saved_group, runtime_group) in enumerate(zip(saved, runtime)):
            if len(saved_group) != len(runtime_group):
                detail = (
                    f"group {group_index} length checkpoint={len(saved_group)}, "
                    f"runtime={len(runtime_group)}"
                )
                break
            for parameter_index, (saved_entry, runtime_entry) in enumerate(
                zip(saved_group, runtime_group)
            ):
                if saved_entry != runtime_entry:
                    detail = (
                        f"group {group_index} parameter {parameter_index} "
                        f"checkpoint={saved_entry}, runtime={runtime_entry}"
                    )
                    break
            if detail != "parameter ordering or group membership differs":
                break
    raise CheckpointMismatchError(f"Optimizer parameter manifest mismatch: {detail}.")


def _validate_optimizer_state(value: Any, label: str) -> None:
    state = _require_mapping(value, label)
    _require_exact_fields(state, _OPTIMIZER_FIELDS, label)
    if not isinstance(state["state"], dict):
        raise CheckpointCorruptError(f"{label}.state must be a dict.")
    param_groups = state["param_groups"]
    if not isinstance(param_groups, list) or not param_groups:
        raise CheckpointCorruptError(f"{label}.param_groups must be a non-empty list.")
    if any(not isinstance(group, dict) for group in param_groups):
        raise CheckpointCorruptError(f"Every entry in {label}.param_groups must be a dict.")
    parameter_ids: list[int] = []
    for group_index, group in enumerate(param_groups):
        if any(not isinstance(key, str) for key in group):
            raise CheckpointCorruptError(
                f"Every key in {label}.param_groups[{group_index}] must be a string."
            )
        params = group.get("params")
        if not isinstance(params, list) or not params:
            raise CheckpointCorruptError(
                f"{label}.param_groups[{group_index}].params must be a non-empty list."
            )
        if any(type(parameter_id) is not int or parameter_id < 0 for parameter_id in params):
            raise CheckpointCorruptError(
                f"{label}.param_groups[{group_index}].params must contain only "
                "non-negative integer parameter IDs."
            )
        parameter_ids.extend(params)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise CheckpointCorruptError(f"{label}.param_groups contain duplicate parameter IDs.")
    expected_ids = set(range(len(parameter_ids)))
    if set(parameter_ids) != expected_ids:
        raise CheckpointCorruptError(
            f"{label}.param_groups parameter IDs must be contiguous from 0; "
            f"expected={sorted(expected_ids)}, got={sorted(parameter_ids)}."
        )
    for parameter_id, parameter_state in state["state"].items():
        if type(parameter_id) is not int or parameter_id not in expected_ids:
            raise CheckpointCorruptError(
                f"Every key in {label}.state must reference a declared integer parameter ID."
            )
        if not isinstance(parameter_state, Mapping):
            raise CheckpointCorruptError(f"{label}.state[{parameter_id}] must be a mapping.")
        if any(not isinstance(key, str) for key in parameter_state):
            raise CheckpointCorruptError(
                f"Every key in {label}.state[{parameter_id}] must be a string."
            )


def _validate_scheduler_state(value: Any, label: str) -> None:
    state = _require_mapping(value, label)
    if not state:
        raise CheckpointCorruptError(f"{label} must be a non-empty mapping.")
    if any(not isinstance(key, str) for key in state):
        raise CheckpointCorruptError(f"Every key in {label} must be a string.")


def _validate_identity_string(value: Any, field: str, *, corrupt: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        error = CheckpointCorruptError if corrupt else CheckpointMismatchError
        raise error(f"{field} must be a non-empty string.")
    return value


def _validate_coordination_source(value: Any, field: str, *, corrupt: bool = False) -> str:
    if value not in ("bus", "self_only"):
        error = CheckpointCorruptError if corrupt else CheckpointMismatchError
        raise error(f"{field} must be 'bus' or 'self_only', got {value!r}.")
    return value


def _validate_layer_indices(value: Any, label: str, *, corrupt: bool = False) -> tuple[int, ...]:
    error = CheckpointCorruptError if corrupt else CheckpointMismatchError
    if not isinstance(value, tuple):
        raise error(f"{label} must be a tuple of non-negative integers.")
    if any(type(index) is not int or index < 0 for index in value):
        raise error(f"{label} must contain only non-negative integers.")
    if len(value) != len(set(value)):
        raise error(f"{label} contains duplicate layer indices.")
    return value


def _validate_nonnegative_int(value: Any, label: str, *, corrupt: bool = False) -> int:
    if type(value) is not int or value < 0:
        error = CheckpointCorruptError if corrupt else ValueError
        raise error(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _validate_stage(value: Any, *, corrupt: bool = False) -> int:
    if type(value) is not int or not 0 <= value <= 3:
        error = CheckpointCorruptError if corrupt else ValueError
        raise error(f"stage must be an integer in [0, 3], got {value!r}.")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCorruptError(f"{label} must be a mapping, got {type(value).__name__}.")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    label: str,
    *,
    mismatch: bool = False,
) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"unexpected={sorted(extra)}")
        error = CheckpointMismatchError if mismatch else CheckpointCorruptError
        raise error(f"{label} fields are invalid: {', '.join(details)}.")

def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointIOError(
            f"Could not create checkpoint directory {destination.parent}: {exc}"
        ) from exc

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except Exception as exc:
        raise CheckpointIOError(
            f"Could not atomically save checkpoint {destination}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
