"""Configuration schema for the fine-tuning dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OpenAIConfig:
    """Configuration for the OpenAI client."""

    api_key: str | None = None
    org_id: str | None = None
    model: str = "gpt-5.1"
    service_tier: str | None = None
    client_timeout: float | None = None
    reasoning_effort: str | None = None  # "low", "medium", "high" for reasoning models


@dataclass(slots=True)
class AnthropicConfig:
    """Configuration for the Anthropic client."""

    api_key: str | None = None
    model: str = "claude-3-opus-20240229"


@dataclass(slots=True)
class LocalLLMConfig:
    """Configuration for the local LLM client."""

    model: str
    tokenizer: str | None = None
    tensor_parallel: int = 8
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    revision: str | None = None
    max_model_len: int = 16384


@dataclass(slots=True)
class OllamaConfig:
    """Configuration for the Ollama client."""

    model: str = "gpt-oss:120b"
    base_url: str = "http://localhost:11434"
    client_timeout: float = 1800.0


@dataclass(slots=True)
class LLMConfig:
    """Configuration for all LLM clients."""

    backend: str = "openai"  # options: openai, anthropic, local, ollama, vllm
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    local: LocalLLMConfig | None = None
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    # vllm uses openai config (OpenAI-compatible API)


@dataclass(slots=True)
class GenerationConfig:
    """Sampling configuration shared across pipeline stages."""

    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int | None = None
    max_new_tokens: int = 512
    min_tokens: int = 0
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass(slots=True)
class WikipediaSourceConfig:
    """Parameters controlling Wikipedia sourcing and filtering."""

    enabled: bool = True
    dataset: str = "wikimedia/structured-wikipedia"
    config_name: str = "20231101.en"
    split: str = "train"
    data_files: str | None = None  # For local files
    min_article_tokens: int = 2000
    min_sections: int = 3
    max_articles: int | None = None
    shuffle_buffer: int = 100_000
    random_seed: int = 13
    article_id_field: str = "id"
    text_field: str = "text"
    title_field: str = "title"
    section_field: str | None = "sections"
    abstract_field: str | None = "abstract"


@dataclass(slots=True)
class SpeculativeNotesNoiseConfig:
    """Controls noise injection for speculative notes."""

    paraphrase_ratio: float = 0.15
    drop_ratio: float = 0.05
    hallucination_ratio: float = 0.05
    shuffle_notes: bool = True


@dataclass(slots=True)
class QualityFilterConfig:
    """Thresholds for dataset quality control."""

    length_balance_variance: float = 0.2
    notes_kl_threshold: float = 0.5
    contradiction_threshold: float = 0.2
    max_speculative_noise_distance: float = 0.35
    max_missing_trace_ratio: float = 0.8
    sample_for_manual_audit: float = 0.01
    random_seed: int = 17
    min_section_tokens: int = 500
    allow_short_sections: bool = False
    max_length_ratio: float = 3.0
    # Flags to disable expensive local model checks
    use_nli_filter: bool = False  # Disable by default (requires local models)
    use_embedding_filter: bool = False  # Disable by default (requires local models)


@dataclass(slots=True)
class SyntheticAugmentationConfig:
    """Generation parameters for synthetic augmentation."""

    enabled: bool = True
    target_examples: int = 30_000
    batch_topics: int = 512
    negative_ratio: float = 0.1
    topic_temperature: float = 0.8
    article_temperature: float = 0.6
    reuse_wikipedia_prompt_seed: bool = False
    max_article_tokens: int = 5500


@dataclass(slots=True)
class SimpleTaskConfig:
    """Configuration for simple balanced task generation."""

    enabled: bool = True
    fraction: float = 0.6
    reasoning_gym_count: int = 5000
    max_output_tokens: int = 150
    templates_per_category: int = 5
    categories: list[str] = field(
        default_factory=lambda: ["lists", "sequences", "facts", "definitions"]
    )


@dataclass(slots=True)
class ExportConfig:
    """Controls dataset export format and location."""

    output_dir: Path = Path("data/fine_tuning")
    run_id: str = "parallel decoder transformer_dataset_v1"
    compression: str = "snappy"
    write_manifest: bool = True
    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    parquet_chunk_size: int = 2048


@dataclass(slots=True)
class DatasetBuildConfig:
    """Aggregated configuration for the dataset build pipeline."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    notes_llm: LLMConfig | None = None
    wikipedia: WikipediaSourceConfig = field(default_factory=WikipediaSourceConfig)
    generation_prompt: GenerationConfig = field(default_factory=GenerationConfig)
    generation_plan: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.2, max_new_tokens=256)
    )
    generation_decomposition: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.2, max_new_tokens=1300)
    )
    generation_true_notes: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.1, max_new_tokens=512)
    )
    generation_speculative_notes: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(temperature=0.6, max_new_tokens=384)
    )
    speculative_noise: SpeculativeNotesNoiseConfig = field(
        default_factory=SpeculativeNotesNoiseConfig
    )
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    synthetic: SyntheticAugmentationConfig = field(default_factory=SyntheticAugmentationConfig)
    simple_tasks: SimpleTaskConfig = field(default_factory=SimpleTaskConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    execution: "ExecutionConfig" = field(default_factory=lambda: ExecutionConfig())
    total_target_examples: int = 100_000
    wikipedia_fraction: float = 0.4
    enable_traces: bool = True
    enable_negative_pairs: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_output_dir(self) -> Path:
        return (self.export.output_dir / self.export.run_id).resolve()


def resolve_notes_llm_config(cfg: DatasetBuildConfig) -> LLMConfig:
    """Return the LLM configuration that should back NotesGenerator."""
    if cfg.notes_llm is not None:
        return cfg.notes_llm
    return cfg.llm


@dataclass(slots=True)
class ExecutionConfig:
    """Execution parameters for distributed dataset generation."""

    use_ray: bool = True
    ray_actor_count: int = 8
    ray_gpus_per_actor: float = 1.0
    llm_tensor_parallel_per_actor: int = 1
