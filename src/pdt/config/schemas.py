"""Dataclass-based configuration schemas for PDT.

One source of truth: a single ``PDTConfig`` dataclass tree that maps 1:1 to
the canonical YAML. All runtime code consumes subtrees of this type; no
subsystem is allowed to read YAML directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Dict, List, Literal, Optional, Tuple


CANONICAL_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CANONICAL_QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN3_14B_MODEL = "Qwen/Qwen3-14B"
QWEN3_14B_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


@dataclass(frozen=True, slots=True)
class TrunkProfile:
    """Pinned dense-Qwen3 architecture and checkpoint identity."""

    name: str
    base_model: str
    revision: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


TRUNK_PROFILES: Dict[str, TrunkProfile] = {
    "qwen3_4b_instruct_2507": TrunkProfile(
        name="qwen3_4b_instruct_2507",
        base_model=CANONICAL_QWEN_MODEL,
        revision=CANONICAL_QWEN_REVISION,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "qwen3_14b": TrunkProfile(
        name="qwen3_14b",
        base_model=QWEN3_14B_MODEL,
        revision=QWEN3_14B_REVISION,
        hidden_size=5120,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
    ),
}
DEFAULT_TRUNK_PROFILE = "qwen3_4b_instruct_2507"


def derive_instrumentation_layers(
    num_hidden_layers: int,
    instrumented_layer_count: int,
) -> Tuple[int, ...]:
    """Return the consecutive upper layers owned by the physical decoders."""

    if type(num_hidden_layers) is not int or num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be a positive integer.")
    if (
        type(instrumented_layer_count) is not int
        or instrumented_layer_count <= 0
        or instrumented_layer_count > num_hidden_layers
    ):
        raise ValueError(
            "instrumented_layer_count must be a positive integer no larger than "
            "num_hidden_layers."
        )
    fork_layer = num_hidden_layers - instrumented_layer_count
    return tuple(range(fork_layer, num_hidden_layers))


# --------------------------------------------------------------------------- #
# Trunk
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrunkConfig:
    """Frozen Qwen3 trunk loader configuration."""

    profile: str = DEFAULT_TRUNK_PROFILE
    base_model: str = CANONICAL_QWEN_MODEL
    revision: str = CANONICAL_QWEN_REVISION
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = None
    attn_implementation: str = "pdt_gqa_sdpa"
    # Local weight override. If set, loader uses `from_pretrained(local_path)`.
    local_path: Optional[str] = None


@dataclass(slots=True)
class InstrumentationConfig:
    """Physical decoder fork and persistent-memory gate initialization."""

    enabled: bool = True
    # The scientific condition is part of model/checkpoint identity. ``bus``
    # reads delayed sibling messages; ``self_only`` replaces every SNC read
    # with an exactly parameter-matched receiver-history read.
    coordination_source: Literal["bus", "self_only"] = "bus"
    instrumented_layer_count: int = 12
    fork_layer: int = 24
    target_layers: Tuple[int, ...] = field(
        default_factory=lambda: derive_instrumentation_layers(36, 12)
    )
    # Initial pre-sigmoid gates for SNC and persistent plan attention. -4.0 gives
    # sigmoid(-4) \u2248 0.0180 so at step 0 the instrumented deltas contribute
    # near-zero; training opens the gates as the auxiliary paths become
    # reliable.
    snc_gate_init: float = -4.0
    plan_gate_init: float = -4.0


# --------------------------------------------------------------------------- #
# Planner, semantic supervision, and communication extensions
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SNCConfig:
    hidden_size: int = 2560
    notes_dim: int = 256
    attention_width: int = 512
    num_heads: int = 8  # 512 // 8 = head_dim 64
    dropout: float = 0.0


@dataclass(slots=True)
class PlannerHeadConfig:
    hidden_size: int = 2560
    planner_width: int = 512
    num_streams: int = 3
    max_nodes_per_stream: int = 8
    num_layers: int = 2
    num_heads: int = 8
    feedforward_width: int = 2048
    dropout: float = 0.0


@dataclass(slots=True)
class PlanMemoryProjectionConfig:
    """Project structured continuous plan nodes into persistent SNC memory."""

    planner_width: int = 512
    notes_dim: int = 256


@dataclass(slots=True)
class SemanticSupervisionConfig:
    """Fact routing, fact writing, outline progress, and render-order heads."""

    hidden_size: int = 2560
    planner_width: int = 512
    fact_embedding_dim: int = 1024
    attention_width: int = 512
    num_fact_roles: int = 3
    max_facts: int = 128
    max_nodes_per_stream: int = 8
    dropout: float = 0.0


@dataclass(slots=True)
class SpeculationHeadConfig:
    hidden_size: int = 2560
    notes_dim: int = 256
    num_codebooks: int = 4
    codes_per_codebook: int = 256
    dropout: float = 0.0


@dataclass(slots=True)
class CoverageHeadConfig:
    hidden_size: int = 2560
    num_heads: int = 8
    dropout: float = 0.0
    sentence_window: int = 32
    learn_temperature: bool = True


@dataclass(slots=True)
class AgreementHeadConfig:
    """Per paper \u00a72, AgreementHead consumes (hidden, W_v, c_v, \u00f1_v)."""

    hidden_size: int = 2560
    notes_dim: int = 256
    # Per-plan-item coverage projector output size. If 0, coverage is mean-pooled
    # before concatenation.
    coverage_features: int = 64
    dropout: float = 0.0
    # gamma threshold is tuned from ROC sweeps, not a learned parameter.
    gamma_init: float = 0.5


@dataclass(slots=True)
class SidecarConfig:
    """Planner and semantic/communication extension configuration."""

    hidden_size: int = 2560  # Must match trunk hidden_size.
    notes_dim: int = 256  # d_notes
    num_streams: int = 3  # K
    snc: SNCConfig = field(default_factory=SNCConfig)
    planner_head: PlannerHeadConfig = field(default_factory=PlannerHeadConfig)
    plan_memory_proj: PlanMemoryProjectionConfig = field(
        default_factory=PlanMemoryProjectionConfig
    )
    semantic_supervision: SemanticSupervisionConfig = field(
        default_factory=SemanticSupervisionConfig
    )
    speculation_head: SpeculationHeadConfig = field(default_factory=SpeculationHeadConfig)


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class NotesBusConfig:
    snapshot_dim: int = 256  # Must match sidecar.notes_dim.
    lag: int = 1  # \u0394
    dtype: str = "bfloat16"
    num_codebooks: int = 4
    codes_per_codebook: int = 256
    history_blocks: int = 16


@dataclass(slots=True)
class RuntimeConfig:
    streams: Tuple[str, ...] = ("stream_0", "stream_1", "stream_2")
    # \u03c4: tokens per provisional block between synchronization decisions.
    block_size: int = 32
    notes_bus: NotesBusConfig = field(default_factory=NotesBusConfig)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LossWeights:
    """Canonical real-data structured-outline objective coefficients."""

    lm_ce: float = 1.0
    plan_semantic: float = 1.0
    fact_route: float = 1.0
    outline_progress: float = 0.5
    fact_write: float = 1.0
    note_align: float = 0.25
    presentation_order: float = 0.1
    dynamic_vq_commit: float = 0.25
    dynamic_vq_codebook: float = 1.0
    dynamic_codebook_usage: float = 0.1


CURRICULUM_IDENTIFIERS: Tuple[str, ...] = (
    "trunk",
    "decoder_branches",
    "planner_head",
    "plan_memory_proj",
    "semantic_heads",
    "speculation_head",
    "snc",
    "plan_attention",
    "snc_gate",
    "plan_gate",
)


@dataclass(slots=True)
class StagePolicy:
    """Per-stage freeze/unfreeze policy.

    Module identifiers here are RESOLVED by the name resolver in
    ``pdt.training.curriculum`` to one of:
    - ``"trunk"``              \u2192 the frozen shared lower Qwen3 model
    - ``"decoder_branches"``   \u2192 three independent upper Qwen parameter banks
    - ``"planner_head"``       \u2192 ``sidecar.planner_head``
    - ``"plan_memory_proj"``   \u2192 ``sidecar.plan_memory_proj``
    - ``"semantic_heads"``     \u2192 ``sidecar.semantic_heads``
    - ``"speculation_head"``   \u2192 ``sidecar.speculation_head``
    - ``"plan_attention"``     \u2192 persistent hard-routed plan reads
    - ``"snc"``                \u2192 per-layer SharedNotesCrossAttention inside every
                                    physical decoder layer
    - ``"snc_gate"``           \u2192 per-layer outer SNC residual gates
    - ``"plan_gate"``          \u2192 per-layer outer plan-attention residual gates
    """

    name: str
    freeze: Tuple[str, ...] = field(default_factory=tuple)
    unfreeze: Tuple[str, ...] = field(default_factory=tuple)
    # Optional per-stage loss-weight override. Entries that are None fall back
    # to the global ``LossWeights``.
    loss_weights: Optional[LossWeights] = None


@dataclass(slots=True)
class CurriculumConfig:
    """Staged curriculum schedule. Stage index is monotone in global_step."""

    # Optimizer-update count at which each stage becomes active. Microbatches
    # accumulated within one optimizer update do not advance global_step.
    stage_schedule: Tuple[int, ...] = (0, 3750, 10000, 25000)
    stages: Dict[int, StagePolicy] = field(
        default_factory=lambda: {
            0: StagePolicy(
                name="oracle_outline_executor",
                freeze=(
                    "trunk",
                    "planner_head",
                ),
                unfreeze=(
                    "decoder_branches",
                    "plan_memory_proj",
                    "semantic_heads",
                    "plan_attention",
                    "snc",
                    "snc_gate",
                    "plan_gate",
                    "speculation_head",
                ),
                loss_weights=LossWeights(
                    lm_ce=1.0,
                    plan_semantic=0.0,
                    fact_route=1.0,
                    outline_progress=0.5,
                    fact_write=1.0,
                    note_align=0.25,
                    presentation_order=0.0,
                    dynamic_vq_commit=0.25,
                    dynamic_vq_codebook=1.0,
                    dynamic_codebook_usage=0.1,
                ),
            ),
            1: StagePolicy(
                name="planner_distillation",
                freeze=(
                    "trunk",
                    "decoder_branches",
                    "speculation_head",
                    "plan_memory_proj",
                    "semantic_heads",
                    "plan_attention",
                    "snc",
                    "snc_gate",
                    "plan_gate",
                ),
                unfreeze=("planner_head",),
                loss_weights=LossWeights(
                    lm_ce=0.0,
                    plan_semantic=1.0,
                    fact_route=1.0,
                    outline_progress=0.0,
                    fact_write=0.0,
                    note_align=0.0,
                    presentation_order=0.1,
                    dynamic_vq_commit=0.0,
                    dynamic_vq_codebook=0.0,
                    dynamic_codebook_usage=0.0,
                ),
            ),
            2: StagePolicy(
                name="joint_packed_rollout",
                freeze=("trunk",),
                unfreeze=(
                    "decoder_branches",
                    "planner_head",
                    "plan_memory_proj",
                    "semantic_heads",
                    "plan_attention",
                    "snc",
                    "snc_gate",
                    "plan_gate",
                    "speculation_head",
                ),
            ),
            3: StagePolicy(
                name="late_joint_training",
                freeze=("trunk",),
                unfreeze=(
                    "decoder_branches",
                    "planner_head",
                    "plan_memory_proj",
                    "semantic_heads",
                    "plan_attention",
                    "snc",
                    "snc_gate",
                    "plan_gate",
                    "speculation_head",
                ),
            ),
        }
    )


@dataclass(slots=True)
class OptimizerConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1250
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: str = (
        "data/processed/real_plan/qwen3_4b_instruct_2507/train.jsonl"
    )
    eval_dataset_path: str = (
        "data/processed/real_plan/qwen3_4b_instruct_2507/validation.jsonl"
    )
    telemetry_dir: str = "experiments/qwen3_4b"
    batch_size: int = 1
    max_planner_prompt_length: int = 16384
    max_stream_prompt_length: int = 16384
    max_block_transition_length: int = 64
    max_teacher_prompt_length: int = 1024
    max_blocks: int = 32
    grad_accumulation: int = 16
    max_steps: int = 50_000
    save_every: int = 2500
    log_interval: int = 25
    eval_interval: int = 10_000
    seed: int = 1729
    causal_eval_seed: int = 1729
    causal_eval_bootstrap_samples: int = 10_000
    causal_eval_confidence_level: float = 0.95
    causal_eval_min_documents: int = 32
    causal_eval_mutation_producer: str = "stream_0"
    causal_eval_mutation_block: int = 0
    causal_eval_mutation_code_offset: int = 1
    device: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PDTConfig:
    """Top-level PDT configuration. Loaded from YAML via ``load_config``."""

    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)
    sidecar: SidecarConfig = field(default_factory=SidecarConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """Cross-subtree consistency checks that cannot live in a single subtree."""

        profile = TRUNK_PROFILES.get(self.trunk.profile)
        if profile is None:
            raise ValueError(
                f"trunk.profile must name one of {tuple(TRUNK_PROFILES)}, "
                f"got {self.trunk.profile!r}."
            )
        if self.trunk.base_model != profile.base_model:
            raise ValueError(
                f"trunk.base_model must match profile {profile.name!r}: "
                f"expected {profile.base_model!r}, "
                f"got {self.trunk.base_model!r}."
            )
        if self.trunk.revision != profile.revision:
            raise ValueError(
                f"trunk.revision must match profile {profile.name!r}: "
                f"expected {profile.revision!r}, "
                f"got {self.trunk.revision!r}."
            )
        if self.trunk.local_path is not None:
            raise ValueError(
                "trunk.local_path is not allowed for canonical training because "
                "checkpoint identity requires the pinned model and revision."
            )
        if self.trunk.attn_implementation != "pdt_gqa_sdpa":
            raise ValueError(
                "trunk.attn_implementation must be the canonical masked native-GQA "
                f"SDPA path 'pdt_gqa_sdpa', got {self.trunk.attn_implementation!r}."
            )
        if not self.instrumentation.enabled:
            raise ValueError("instrumentation.enabled must be true for canonical PDT.")
        if self.instrumentation.coordination_source not in ("bus", "self_only"):
            raise ValueError(
                "instrumentation.coordination_source must be 'bus' or 'self_only'; "
                f"got {self.instrumentation.coordination_source!r}."
            )
        layer_count = self.instrumentation.instrumented_layer_count
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError(
                "instrumentation.instrumented_layer_count must be a positive integer."
            )
        target_layers = tuple(self.instrumentation.target_layers)
        if not target_layers:
            raise ValueError("instrumentation.target_layers must be non-empty.")
        if len(set(target_layers)) != len(target_layers):
            raise ValueError("instrumentation.target_layers must be unique.")
        if any(type(layer_idx) is not int or layer_idx < 0 for layer_idx in target_layers):
            raise ValueError("instrumentation.target_layers must contain non-negative integers.")
        expected_layers = derive_instrumentation_layers(
            profile.num_hidden_layers,
            layer_count,
        )
        expected_fork = profile.num_hidden_layers - layer_count
        if self.instrumentation.fork_layer != expected_fork:
            raise ValueError(
                "instrumentation.fork_layer must equal trunk depth minus physical "
                f"decoder depth; expected {expected_fork}, "
                f"got {self.instrumentation.fork_layer}."
            )
        if target_layers != expected_layers:
            raise ValueError(
                "instrumentation.target_layers must be the consecutive physical layers "
                f"profile depth; expected {expected_layers}, got {target_layers}."
            )
        if self.runtime.block_size != 32:
            raise ValueError(
                f"runtime.block_size must equal canonical tau=32, got {self.runtime.block_size}."
            )
        if self.runtime.notes_bus.lag != 1:
            raise ValueError(
                f"runtime.notes_bus.lag must equal canonical Delta=1, "
                f"got {self.runtime.notes_bus.lag}."
            )
        if self.runtime.notes_bus.history_blocks != 16:
            raise ValueError(
                "runtime.notes_bus.history_blocks must equal the canonical long-form "
                f"horizon 16, got {self.runtime.notes_bus.history_blocks}."
            )

        runtime_streams = tuple(self.runtime.streams)
        if not runtime_streams or any(
            not isinstance(stream, str) or not stream.strip() for stream in runtime_streams
        ):
            raise ValueError("runtime.streams must contain non-empty stream identifiers.")
        if len(set(runtime_streams)) != len(runtime_streams):
            raise ValueError("runtime.streams must be unique.")

        positive_dimensions = (
            ("sidecar.hidden_size", self.sidecar.hidden_size),
            ("sidecar.notes_dim", self.sidecar.notes_dim),
            ("sidecar.num_streams", self.sidecar.num_streams),
            ("sidecar.snc.hidden_size", self.sidecar.snc.hidden_size),
            ("sidecar.snc.notes_dim", self.sidecar.snc.notes_dim),
            ("sidecar.snc.attention_width", self.sidecar.snc.attention_width),
            ("sidecar.snc.num_heads", self.sidecar.snc.num_heads),
            ("sidecar.planner_head.hidden_size", self.sidecar.planner_head.hidden_size),
            ("sidecar.planner_head.planner_width", self.sidecar.planner_head.planner_width),
            ("sidecar.planner_head.num_streams", self.sidecar.planner_head.num_streams),
            (
                "sidecar.planner_head.max_nodes_per_stream",
                self.sidecar.planner_head.max_nodes_per_stream,
            ),
            ("sidecar.planner_head.num_layers", self.sidecar.planner_head.num_layers),
            ("sidecar.planner_head.num_heads", self.sidecar.planner_head.num_heads),
            (
                "sidecar.planner_head.feedforward_width",
                self.sidecar.planner_head.feedforward_width,
            ),
            (
                "sidecar.plan_memory_proj.planner_width",
                self.sidecar.plan_memory_proj.planner_width,
            ),
            ("sidecar.plan_memory_proj.notes_dim", self.sidecar.plan_memory_proj.notes_dim),
            (
                "sidecar.semantic_supervision.hidden_size",
                self.sidecar.semantic_supervision.hidden_size,
            ),
            (
                "sidecar.semantic_supervision.planner_width",
                self.sidecar.semantic_supervision.planner_width,
            ),
            (
                "sidecar.semantic_supervision.fact_embedding_dim",
                self.sidecar.semantic_supervision.fact_embedding_dim,
            ),
            (
                "sidecar.semantic_supervision.attention_width",
                self.sidecar.semantic_supervision.attention_width,
            ),
            (
                "sidecar.semantic_supervision.num_fact_roles",
                self.sidecar.semantic_supervision.num_fact_roles,
            ),
            (
                "sidecar.semantic_supervision.max_facts",
                self.sidecar.semantic_supervision.max_facts,
            ),
            (
                "sidecar.semantic_supervision.max_nodes_per_stream",
                self.sidecar.semantic_supervision.max_nodes_per_stream,
            ),
            (
                "sidecar.speculation_head.hidden_size",
                self.sidecar.speculation_head.hidden_size,
            ),
            ("sidecar.speculation_head.notes_dim", self.sidecar.speculation_head.notes_dim),
            (
                "sidecar.speculation_head.num_codebooks",
                self.sidecar.speculation_head.num_codebooks,
            ),
            (
                "sidecar.speculation_head.codes_per_codebook",
                self.sidecar.speculation_head.codes_per_codebook,
            ),
            ("runtime.notes_bus.snapshot_dim", self.runtime.notes_bus.snapshot_dim),
            ("runtime.notes_bus.num_codebooks", self.runtime.notes_bus.num_codebooks),
            (
                "runtime.notes_bus.codes_per_codebook",
                self.runtime.notes_bus.codes_per_codebook,
            ),
            ("runtime.notes_bus.history_blocks", self.runtime.notes_bus.history_blocks),
        )
        for name, value in positive_dimensions:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.sidecar.snc.attention_width % self.sidecar.snc.num_heads != 0:
            raise ValueError(
                "sidecar.snc.attention_width must be divisible by sidecar.snc.num_heads; "
                f"got attention_width={self.sidecar.snc.attention_width}, "
                f"num_heads={self.sidecar.snc.num_heads}."
            )
        planner = self.sidecar.planner_head
        if planner.planner_width % planner.num_heads != 0:
            raise ValueError(
                "sidecar.planner_head.planner_width must be divisible by num_heads; "
                f"got planner_width={planner.planner_width}, num_heads={planner.num_heads}."
            )
        if planner.num_streams != 3:
            raise ValueError(
                "sidecar.planner_head.num_streams must equal the canonical three lanes."
            )
        if planner.max_nodes_per_stream != 8:
            raise ValueError(
                "sidecar.planner_head.max_nodes_per_stream must equal the canonical eight."
            )
        if self.sidecar.semantic_supervision.num_fact_roles != 3:
            raise ValueError(
                "semantic_supervision.num_fact_roles must encode OWNER, REFERENCE, ABSENT."
            )
        if self.sidecar.semantic_supervision.max_facts < 96:
            raise ValueError(
                "semantic_supervision.max_facts must accommodate 48 source facts plus "
                "one paired hard-negative query per fact (at least 96 queries)."
            )
        if self.sidecar.semantic_supervision.fact_embedding_dim != 1024:
            raise ValueError(
                "semantic_supervision.fact_embedding_dim must match pinned BGE-large width 1024."
            )
        note_quantizer = self.sidecar.speculation_head
        if note_quantizer.notes_dim % note_quantizer.num_codebooks != 0:
            raise ValueError(
                "sidecar.speculation_head.notes_dim must be divisible by num_codebooks; "
                f"got notes_dim={note_quantizer.notes_dim}, "
                f"num_codebooks={note_quantizer.num_codebooks}."
            )
        code_count = note_quantizer.codes_per_codebook
        if code_count < 2 or code_count & (code_count - 1):
            raise ValueError(
                "sidecar.speculation_head.codes_per_codebook must be a power of two "
                "greater than one."
            )
        if self.runtime.notes_bus.num_codebooks != note_quantizer.num_codebooks:
            raise ValueError(
                "runtime.notes_bus.num_codebooks must match sidecar.speculation_head.num_codebooks."
            )
        if self.runtime.notes_bus.codes_per_codebook != note_quantizer.codes_per_codebook:
            raise ValueError(
                "runtime.notes_bus.codes_per_codebook must match "
                "sidecar.speculation_head.codes_per_codebook."
        )
        for name, dropout in (
            ("sidecar.snc.dropout", self.sidecar.snc.dropout),
            ("sidecar.planner_head.dropout", self.sidecar.planner_head.dropout),
            (
                "sidecar.semantic_supervision.dropout",
                self.sidecar.semantic_supervision.dropout,
            ),
            ("sidecar.speculation_head.dropout", self.sidecar.speculation_head.dropout),
        ):
            if not math.isfinite(dropout) or not 0 <= dropout < 1:
                raise ValueError(f"{name} must be finite and in [0, 1), got {dropout}.")

        # notes_dim must match everywhere it appears.
        dims: List[Tuple[str, int]] = [
            ("sidecar.notes_dim", self.sidecar.notes_dim),
            ("sidecar.snc.notes_dim", self.sidecar.snc.notes_dim),
            ("sidecar.speculation_head.notes_dim", self.sidecar.speculation_head.notes_dim),
            (
                "sidecar.plan_memory_proj.notes_dim",
                self.sidecar.plan_memory_proj.notes_dim,
            ),
            ("runtime.notes_bus.snapshot_dim", self.runtime.notes_bus.snapshot_dim),
        ]
        canonical = dims[0][1]
        for name, value in dims[1:]:
            if value != canonical:
                raise ValueError(
                    f"notes_dim mismatch: {name}={value} != sidecar.notes_dim={canonical}"
                )

        # hidden_size must match everywhere.
        hs: List[Tuple[str, int]] = [
            ("sidecar.hidden_size", self.sidecar.hidden_size),
            ("sidecar.snc.hidden_size", self.sidecar.snc.hidden_size),
            ("sidecar.planner_head.hidden_size", self.sidecar.planner_head.hidden_size),
            (
                "sidecar.speculation_head.hidden_size",
                self.sidecar.speculation_head.hidden_size,
            ),
            (
                "sidecar.semantic_supervision.hidden_size",
                self.sidecar.semantic_supervision.hidden_size,
            ),
        ]
        canonical_h = hs[0][1]
        if canonical_h != profile.hidden_size:
            raise ValueError(
                "sidecar.hidden_size must match the selected trunk profile; "
                f"expected {profile.hidden_size}, got {canonical_h}."
            )
        for name, value in hs[1:]:
            if value != canonical_h:
                raise ValueError(
                    f"hidden_size mismatch: {name}={value} != sidecar.hidden_size={canonical_h}"
                )

        if self.sidecar.plan_memory_proj.planner_width != planner.planner_width:
            raise ValueError(
                "plan_memory_proj.planner_width must equal planner_head.planner_width."
            )
        semantic = self.sidecar.semantic_supervision
        if semantic.planner_width != planner.planner_width:
            raise ValueError(
                "semantic_supervision.planner_width must equal planner_head.planner_width."
            )
        if semantic.max_nodes_per_stream != planner.max_nodes_per_stream:
            raise ValueError(
                "semantic_supervision.max_nodes_per_stream must equal "
                "planner_head.max_nodes_per_stream."
            )

        # K must match between planner, sidecar, and runtime.
        if planner.num_streams != self.sidecar.num_streams:
            raise ValueError("planner_head.num_streams must equal sidecar.num_streams")
        if len(self.runtime.streams) != self.sidecar.num_streams:
            raise ValueError(
                f"len(runtime.streams)={len(self.runtime.streams)} != "
                f"sidecar.num_streams={self.sidecar.num_streams}"
            )

        # Training counters are optimizer-update counts. Every periodic event
        # must be positive, and all curriculum stages must be reachable before
        # the final optimizer update.
        for name in (
            "grad_accumulation",
            "max_steps",
            "save_every",
            "log_interval",
            "eval_interval",
        ):
            value = getattr(self.training, name)
            if value <= 0:
                raise ValueError(f"training.{name} must be positive, got {value}.")
        learning_rate = self.training.optimizer.learning_rate
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(
                "training.optimizer.learning_rate must be finite and positive, "
                f"got {learning_rate}."
            )
        weight_decay = self.training.optimizer.weight_decay
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                "training.optimizer.weight_decay must be finite and non-negative, "
                f"got {weight_decay}."
            )
        if self.training.optimizer.lr_scheduler not in ("cosine", "linear", "constant"):
            raise ValueError(
                "training.optimizer.lr_scheduler must be one of cosine, linear, constant; "
                f"got {self.training.optimizer.lr_scheduler!r}."
            )
        warmup_steps = self.training.optimizer.warmup_steps
        if not 0 <= warmup_steps < self.training.max_steps:
            raise ValueError(
                "training.optimizer.warmup_steps must be in [0, max_steps), "
                f"got warmup_steps={warmup_steps}, max_steps={self.training.max_steps}."
            )
        if type(self.training.seed) is not int or self.training.seed < 0:
            raise ValueError("training.seed must be a non-negative integer.")
        if type(self.training.causal_eval_seed) is not int or self.training.causal_eval_seed < 0:
            raise ValueError("training.causal_eval_seed must be a non-negative integer.")
        bootstrap_samples = self.training.causal_eval_bootstrap_samples
        if type(bootstrap_samples) is not int or bootstrap_samples < 1000:
            raise ValueError(
                "training.causal_eval_bootstrap_samples must be an integer of at least 1000."
            )
        confidence_level = self.training.causal_eval_confidence_level
        if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError(
                "training.causal_eval_confidence_level must be finite and in (0, 1)."
            )
        minimum_documents = self.training.causal_eval_min_documents
        if type(minimum_documents) is not int or minimum_documents <= 1:
            raise ValueError(
                "training.causal_eval_min_documents must be an integer greater than one."
            )
        mutation_producer = self.training.causal_eval_mutation_producer
        if mutation_producer not in self.runtime.streams:
            raise ValueError(
                "training.causal_eval_mutation_producer must name a runtime stream; "
                f"got {mutation_producer!r}, expected one of {self.runtime.streams}."
            )
        mutation_block = self.training.causal_eval_mutation_block
        latest_visible_source = self.training.max_blocks - self.runtime.notes_bus.lag - 1
        if type(mutation_block) is not int or not 0 <= mutation_block <= latest_visible_source:
            raise ValueError(
                "training.causal_eval_mutation_block must identify a write that becomes "
                f"visible within max_blocks; got {mutation_block}, latest is "
                f"{latest_visible_source}."
            )
        mutation_offset = self.training.causal_eval_mutation_code_offset
        code_count = self.sidecar.speculation_head.codes_per_codebook
        if type(mutation_offset) is not int or not 0 < mutation_offset < code_count:
            raise ValueError(
                "training.causal_eval_mutation_code_offset must be an integer in "
                f"[1, {code_count}); got {mutation_offset!r}."
            )
        _validate_nonnegative_loss_weights(
            self.training.loss_weights,
            label="training.loss_weights",
        )

        # Curriculum schedule sanity.
        sched = self.training.curriculum.stage_schedule
        if len(sched) != 4:
            raise ValueError(f"stage_schedule must have 4 entries, got {len(sched)}.")
        if sched[0] != 0:
            raise ValueError("stage_schedule must start at 0.")
        if any(sched[i + 1] <= sched[i] for i in range(len(sched) - 1)):
            raise ValueError("stage_schedule must be strictly increasing.")
        if sched[-1] >= self.training.max_steps:
            raise ValueError(
                "Every curriculum stage must be reachable before max_steps; "
                f"last threshold={sched[-1]}, max_steps={self.training.max_steps}."
            )
        stage_indices = set(self.training.curriculum.stages)
        expected_stage_indices = set(range(4))
        if stage_indices != expected_stage_indices:
            raise ValueError(
                f"Curriculum stages must be exactly 0, 1, 2, 3; got {sorted(stage_indices)}."
            )
        expected_identifiers = set(CURRICULUM_IDENTIFIERS)
        for stage_idx, policy in self.training.curriculum.stages.items():
            if policy.loss_weights is not None:
                _validate_nonnegative_loss_weights(
                    policy.loss_weights,
                    label=f"training.curriculum.stages[{stage_idx}].loss_weights",
                )
            frozen = tuple(policy.freeze)
            unfrozen = tuple(policy.unfreeze)
            declared = frozen + unfrozen
            if len(set(declared)) != len(declared):
                raise ValueError(
                    f"Curriculum stage {stage_idx} contains duplicate freeze/unfreeze identifiers."
                )
            actual_identifiers = set(declared)
            if actual_identifiers != expected_identifiers:
                missing = sorted(expected_identifiers - actual_identifiers)
                unknown = sorted(actual_identifiers - expected_identifiers)
                raise ValueError(
                    f"Curriculum stage {stage_idx} must exhaustively control every "
                    f"identifier; missing={missing}, unknown={unknown}."
                )
            if "trunk" not in frozen:
                raise ValueError(
                    f"Curriculum stage {stage_idx} must keep the frozen trunk in policy.freeze."
                )

        if self.training.batch_size != 1:
            raise ValueError(
                "training.batch_size must be 1 for the canonical differentiable "
                "cached rollout; use training.grad_accumulation for a larger "
                "effective batch."
            )
        for name, value in (
            ("max_planner_prompt_length", self.training.max_planner_prompt_length),
            ("max_stream_prompt_length", self.training.max_stream_prompt_length),
            ("max_block_transition_length", self.training.max_block_transition_length),
            ("max_teacher_prompt_length", self.training.max_teacher_prompt_length),
            ("max_blocks", self.training.max_blocks),
        ):
            if value <= 0:
                raise ValueError(f"training.{name} must be positive, got {value}.")
        if self.training.max_blocks != 32:
            raise ValueError(
                "training.max_blocks must equal the canonical long-form horizon of 32, "
                f"got {self.training.max_blocks}."
            )


def apply_trunk_profile(config: PDTConfig, profile_name: str) -> None:
    """Materialize one pinned trunk scale into the shared architecture config."""

    profile = TRUNK_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(
            f"Unknown trunk profile {profile_name!r}; expected one of {tuple(TRUNK_PROFILES)}."
        )
    config.trunk.profile = profile.name
    config.trunk.base_model = profile.base_model
    config.trunk.revision = profile.revision
    config.instrumentation.target_layers = derive_instrumentation_layers(
        profile.num_hidden_layers,
        config.instrumentation.instrumented_layer_count,
    )
    config.instrumentation.fork_layer = (
        profile.num_hidden_layers - config.instrumentation.instrumented_layer_count
    )
    config.sidecar.hidden_size = profile.hidden_size
    config.sidecar.snc.hidden_size = profile.hidden_size
    config.sidecar.planner_head.hidden_size = profile.hidden_size
    config.sidecar.semantic_supervision.hidden_size = profile.hidden_size
    config.sidecar.speculation_head.hidden_size = profile.hidden_size
    processed_root = f"data/processed/real_plan/{profile.name}"
    config.training.dataset_path = f"{processed_root}/train.jsonl"
    config.training.eval_dataset_path = f"{processed_root}/validation.jsonl"
    config.training.telemetry_dir = f"experiments/{profile.name}"


def _validate_nonnegative_loss_weights(weights: LossWeights, *, label: str) -> None:
    for field_ in fields(LossWeights):
        value = getattr(weights, field_.name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label}.{field_.name} must be finite and non-negative, got {value}.")


__all__ = [
    "AgreementHeadConfig",
    "CURRICULUM_IDENTIFIERS",
    "CoverageHeadConfig",
    "CurriculumConfig",
    "InstrumentationConfig",
    "LossWeights",
    "NotesBusConfig",
    "OptimizerConfig",
    "PDTConfig",
    "PlanMemoryProjectionConfig",
    "PlannerHeadConfig",
    "RuntimeConfig",
    "SemanticSupervisionConfig",
    "SNCConfig",
    "SidecarConfig",
    "SpeculationHeadConfig",
    "StagePolicy",
    "TrainingConfig",
    "TrunkProfile",
    "TrunkConfig",
    "TRUNK_PROFILES",
    "apply_trunk_profile",
    "derive_instrumentation_layers",
]
