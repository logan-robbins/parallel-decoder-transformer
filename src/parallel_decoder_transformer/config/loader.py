"""Shared config loading helpers used by training entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from ..config import ModelConfig, TrainingConfig
from ..models.stream_adapters import StreamAdapterConfig
from ..models.heads import (
    PlannerHeadConfig,
    NotesHeadConfig,
    SpeculationHeadConfig,
    AgreementHeadConfig,
    CoverageHeadConfig,
    StreamClassifierConfig,
)
from ..inference.dnb_bus import DynamicNotesBusConfig
from ..inference.snc_cross_attn import SharedNotesCrossAttentionConfig
from ..integration.gpt_oss import TrunkAdapterConfig
from ..integration.instrumentation import InstrumentationSpec
from ..training import KDJsonlDataset
from ..training.trainer import (
    CurriculumConfig,
    LossWeights,
    MetricsConfig,
    NotesNoiseConfig,
    StagePolicyConfig,
    TeacherBranchConfig,
)
from ..data.teacher_runner import DatasetTeacherConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-OSS with Parallel Decoder Transformer adapters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical.yaml"),
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def maybe_create_dataset(path: str | None) -> KDJsonlDataset | None:
    if not path:
        return None
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    return KDJsonlDataset(dataset_path)


def _coerce_model_config(payload: Dict[str, Any]) -> ModelConfig:
    data = dict(payload)
    if isinstance(data.get("trunk"), dict):
        data["trunk"] = TrunkAdapterConfig(**data["trunk"])
    if isinstance(data.get("stream_adapters"), dict):
        data["stream_adapters"] = StreamAdapterConfig(**data["stream_adapters"])
    if isinstance(data.get("notes_bus"), dict):
        data["notes_bus"] = DynamicNotesBusConfig(**data["notes_bus"])
    if isinstance(data.get("cross_attention"), dict):
        data["cross_attention"] = SharedNotesCrossAttentionConfig(**data["cross_attention"])
    if isinstance(data.get("planner_head"), dict):
        planner_dict = dict(data["planner_head"])
        if "vocab_size" not in planner_dict:
            planner_dict["vocab_size"] = data.get("plan_vocab_size", 65536)
        data["planner_head"] = PlannerHeadConfig(**planner_dict)
    if isinstance(data.get("notes_head"), dict):
        data["notes_head"] = NotesHeadConfig(**data["notes_head"])
    if isinstance(data.get("speculation_head"), dict):
        data["speculation_head"] = SpeculationHeadConfig(**data["speculation_head"])
    if isinstance(data.get("agreement_head"), dict):
        data["agreement_head"] = AgreementHeadConfig(**data["agreement_head"])
    if "coverage_head" not in data or data["coverage_head"] is None:
        data["coverage_head"] = CoverageHeadConfig(
            hidden_size=data.get("hidden_size", 4096), dropout=0.0
        )
    elif isinstance(data.get("coverage_head"), dict):
        data["coverage_head"] = CoverageHeadConfig(**data["coverage_head"])
    if "stream_classifier_head" not in data or data["stream_classifier_head"] is None:
        num_streams = 3
        if isinstance(data.get("stream_adapters"), StreamAdapterConfig):
            num_streams = len(data["stream_adapters"].roles)
        elif isinstance(data.get("stream_adapters"), dict):
            roles = data["stream_adapters"].get("roles", [])
            num_streams = len(roles) if roles else 3
        data["stream_classifier_head"] = StreamClassifierConfig(
            hidden_size=data.get("hidden_size", 4096), num_streams=num_streams, dropout=0.0
        )
    elif isinstance(data.get("stream_classifier_head"), dict):
        data["stream_classifier_head"] = StreamClassifierConfig(**data["stream_classifier_head"])
    if isinstance(data.get("instrumentation"), dict):
        instrumentation_payload = dict(data["instrumentation"])
        stream_cfg = instrumentation_payload.get("stream_adapters")
        if isinstance(stream_cfg, dict):
            instrumentation_payload["stream_adapters"] = StreamAdapterConfig(**stream_cfg)
        cross_cfg = instrumentation_payload.get("cross_attention")
        if isinstance(cross_cfg, dict):
            instrumentation_payload["cross_attention"] = SharedNotesCrossAttentionConfig(
                **cross_cfg
            )
        spec_cfg = instrumentation_payload.get("speculation")
        if isinstance(spec_cfg, dict):
            instrumentation_payload["speculation"] = SpeculationHeadConfig(**spec_cfg)
        data["instrumentation"] = InstrumentationSpec(**instrumentation_payload)
    return ModelConfig(**data)


def _coerce_training_config(payload: Dict[str, Any]) -> TrainingConfig:
    data = dict(payload)
    if isinstance(data.get("teacher"), dict):
        data["teacher"] = TeacherBranchConfig(**data["teacher"])
    if isinstance(data.get("dataset_teacher"), dict):
        data["dataset_teacher"] = DatasetTeacherConfig(**data["dataset_teacher"])
    if isinstance(data.get("curriculum"), dict):
        data["curriculum"] = CurriculumConfig(**data["curriculum"])
    if isinstance(data.get("loss_weights"), dict):
        data["loss_weights"] = LossWeights(**data["loss_weights"])
    if isinstance(data.get("notes_noise"), dict):
        data["notes_noise"] = NotesNoiseConfig(**data["notes_noise"])
    if isinstance(data.get("metrics"), dict):
        data["metrics"] = MetricsConfig(**data["metrics"])
    if isinstance(data.get("stage_policies"), dict):
        policies: Dict[int, StagePolicyConfig] = {}
        for key, policy_payload in data["stage_policies"].items():
            try:
                index = int(key)
            except (TypeError, ValueError) as err:
                raise ValueError(f"Stage policy keys must be integers, received {key!r}.") from err
            payload = dict(policy_payload or {})
            if isinstance(payload.get("notes_noise"), dict):
                payload["notes_noise"] = NotesNoiseConfig(**payload["notes_noise"])
            policies[index] = StagePolicyConfig(**payload)
        data["stage_policies"] = policies
    return TrainingConfig(**data)
