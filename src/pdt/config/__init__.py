"""PDT configuration schemas and YAML loader."""

from pdt.config.schemas import (
    CurriculumConfig,
    InstrumentationConfig,
    LossWeights,
    PDTConfig,
    RuntimeConfig,
    SidecarConfig,
    StagePolicy,
    TrainingConfig,
    TrunkProfile,
    TrunkConfig,
    TRUNK_PROFILES,
    apply_trunk_profile,
    derive_instrumentation_layers,
)
from pdt.config.loader import load_config

__all__ = [
    "CurriculumConfig",
    "InstrumentationConfig",
    "LossWeights",
    "PDTConfig",
    "RuntimeConfig",
    "SidecarConfig",
    "StagePolicy",
    "TrainingConfig",
    "TrunkProfile",
    "TrunkConfig",
    "TRUNK_PROFILES",
    "apply_trunk_profile",
    "derive_instrumentation_layers",
    "load_config",
]
