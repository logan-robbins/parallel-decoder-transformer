"""Configuration schemas aligned with the GPT-OSS backed model stack."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..integration.gpt_oss import TrunkAdapterConfig
from ..models import ParallelDecoderModelConfig
from ..training.trainer import TrainingConfig as TrainerRuntimeConfig


ModelConfig = ParallelDecoderModelConfig
TrainingConfig = TrainerRuntimeConfig


@dataclass(slots=True)
class RunConfig:
    """Convenience wrapper bundling model and training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


__all__ = ["ModelConfig", "TrainingConfig", "RunConfig", "TrunkAdapterConfig"]
