"""YAML -> PDTConfig loader.

Uses OmegaConf for merging (so env/CLI overrides compose cleanly) but
materializes back into typed dataclasses so downstream code never touches
``DictConfig`` directly.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Union, cast

from omegaconf import DictConfig, OmegaConf

from pdt.config.schemas import (
    CurriculumConfig,
    InstrumentationConfig,
    LossWeights,
    NotesBusConfig,
    OptimizerConfig,
    PDTConfig,
    PlanMemoryProjectionConfig,
    PlannerHeadConfig,
    RuntimeConfig,
    SemanticSupervisionConfig,
    SNCConfig,
    SidecarConfig,
    SpeculationHeadConfig,
    StagePolicy,
    PlanAdapterConfig,
    TrainingConfig,
    TrunkConfig,
)

__all__ = ["load_config"]


def load_config(path: Union[str, Path]) -> PDTConfig:
    """Load and validate a PDT config from a YAML file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Config at {path} must be a mapping at the top level.")
    raw = cast(Any, OmegaConf.to_container(loaded, resolve=True))
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError(f"Config at {path} must use string keys at the top level.")
    config = _materialize(PDTConfig, cast(Mapping[str, Any], raw))
    config.validate()
    return config


def _materialize(dataclass_type: type, payload: Mapping[str, Any]) -> Any:
    """Recursively materialize a dataclass tree from a plain-dict payload.

    Unknown keys are rejected (typo safety). Missing keys fall back to the
    dataclass default.
    """

    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass.")

    valid_fields = {f.name: f for f in fields(dataclass_type)}
    unknown = set(payload) - set(valid_fields)
    if unknown:
        raise ValueError(
            f"Unknown keys for {dataclass_type.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_fields)}."
        )

    kwargs: dict[str, Any] = {}
    for name, field_ in valid_fields.items():
        if name not in payload:
            continue
        value = payload[name]
        kwargs[name] = _coerce(field_.type, value, field_name=name)
    return dataclass_type(**kwargs)


_DATACLASS_MAP: dict[str, type] = {
    "TrunkConfig": TrunkConfig,
    "InstrumentationConfig": InstrumentationConfig,
    "SNCConfig": SNCConfig,
    "PlanAdapterConfig": PlanAdapterConfig,
    "PlannerHeadConfig": PlannerHeadConfig,
    "PlanMemoryProjectionConfig": PlanMemoryProjectionConfig,
    "SemanticSupervisionConfig": SemanticSupervisionConfig,
    "SpeculationHeadConfig": SpeculationHeadConfig,
    "SidecarConfig": SidecarConfig,
    "NotesBusConfig": NotesBusConfig,
    "RuntimeConfig": RuntimeConfig,
    "LossWeights": LossWeights,
    "OptimizerConfig": OptimizerConfig,
    "StagePolicy": StagePolicy,
    "CurriculumConfig": CurriculumConfig,
    "TrainingConfig": TrainingConfig,
    "PDTConfig": PDTConfig,
}


def _coerce(annotation: Any, value: Any, *, field_name: str) -> Any:
    """Coerce a raw YAML value into the target dataclass type."""

    # Normalize string annotations (forward refs from `from __future__ import annotations`).
    if isinstance(annotation, str):
        target = _DATACLASS_MAP.get(annotation)
        if target is not None and isinstance(value, Mapping):
            return _materialize(target, value)
        # Handle Tuple[...] / Optional[...] / Dict[...] fallbacks by just returning the value.
        return _coerce_container(value)

    # Dataclass types: recurse.
    if is_dataclass(annotation):
        dataclass_type = cast(type, annotation)
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Field {field_name!r}: expected mapping for {dataclass_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return _materialize(dataclass_type, value)

    return _coerce_container(value)


def _coerce_container(value: Any) -> Any:
    """Turn OmegaConf-materialized lists into tuples where the schema expects
    ``Tuple[...]``. For nested stage policies we detect the pattern and
    materialize StagePolicy explicitly."""

    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        # ``curriculum.stages`` is Dict[int, StagePolicy]. Detect by presence
        # of a ``name`` key at the next level.
        materialized: dict[Any, Any] = {}
        for k, v in value.items():
            try:
                key_int = int(k)
            except (TypeError, ValueError):
                key_int = k
            if isinstance(v, Mapping) and "name" in v:
                materialized[key_int] = _materialize_stage_policy(v)
            else:
                materialized[key_int] = v
        return materialized
    return value


def _materialize_stage_policy(payload: Mapping[str, Any]) -> StagePolicy:
    """Special-case handler for StagePolicy since it contains a nested LossWeights."""
    valid_fields = {field_.name: field_ for field_ in fields(StagePolicy)}
    unknown = set(payload) - set(valid_fields)
    if unknown:
        raise ValueError(
            f"Unknown keys for StagePolicy: {sorted(unknown)}. Valid keys: {sorted(valid_fields)}."
        )
    kwargs: dict[str, Any] = {}
    for name, field_ in valid_fields.items():
        if name not in payload:
            continue
        raw_value = payload[name]
        if name == "loss_weights" and isinstance(raw_value, Mapping):
            kwargs[name] = _materialize(LossWeights, raw_value)
        elif isinstance(raw_value, list):
            kwargs[name] = tuple(raw_value)
        else:
            kwargs[name] = raw_value
    return StagePolicy(**kwargs)
