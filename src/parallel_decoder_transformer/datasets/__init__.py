"""High-level dataset modules for the Parallel Decoder Transformer fine-tuning corpus."""

from .collation import CollateConfig, DatasetCollator
from .example import DatasetExample, ExampleNotes, PlanPayload, PlanSection, SectionDecomposition
from .kd_export import KDExportConfig, KDExporter
from .processors import ArticleProcessor
from .config import (
    AnthropicConfig,
    DatasetBuildConfig,
    ExecutionConfig,
    GenerationConfig,
    LLMConfig,
    LocalLLMConfig,
    OpenAIConfig,
    QualityFilterConfig,
    resolve_notes_llm_config,
    SpeculativeNotesNoiseConfig,
    SyntheticAugmentationConfig,
    WikipediaSourceConfig,
)
from .notes_generation import NotesGenerationConfig, NotesGenerator
from .plan_generation import PlanGenerationConfig, PlanGenerator

__all__ = [
    "ArticleProcessor",
    "DatasetCollator",
    "DatasetExample",
    "CollateConfig",
    "ExampleNotes",
    "PlanPayload",
    "PlanSection",
    "SectionDecomposition",
    "KDExportConfig",
    "KDExporter",
    "AnthropicConfig",
    "DatasetBuildConfig",
    "ExecutionConfig",
    "GenerationConfig",
    "LLMConfig",
    "LocalLLMConfig",
    "OpenAIConfig",
    "QualityFilterConfig",
    "resolve_notes_llm_config",
    "SpeculativeNotesNoiseConfig",
    "SyntheticAugmentationConfig",
    "WikipediaSourceConfig",
    "PlanGenerationConfig",
    "PlanGenerator",
    "NotesGenerationConfig",
    "NotesGenerator",
]
