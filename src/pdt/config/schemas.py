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
    """Place instrumentation at the rounded end of equal-depth trunk bands."""

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
    layers = tuple(
        ((band * num_hidden_layers + instrumented_layer_count // 2) // instrumented_layer_count)
        - 1
        for band in range(1, instrumented_layer_count + 1)
    )
    if len(set(layers)) != instrumented_layer_count:
        raise RuntimeError("Equal-depth instrumentation produced duplicate layer indices.")
    return layers


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
    attn_implementation: str = "sdpa"
    # Local weight override. If set, loader uses `from_pretrained(local_path)`.
    local_path: Optional[str] = None


@dataclass(slots=True)
class InstrumentationConfig:
    """Which decoder layers to instrument, and how gates are initialized."""

    enabled: bool = True
    # The scientific condition is part of model/checkpoint identity. ``bus``
    # reads delayed sibling messages; ``self_only`` replaces every SNC read
    # with an exactly parameter-matched receiver-history read.
    coordination_source: Literal["bus", "self_only"] = "bus"
    instrumented_layer_count: int = 12
    # Materialized from trunk depth by ``apply_trunk_profile``. Keeping the
    # resolved indices in config makes checkpoint identity explicit.
    target_layers: Tuple[int, ...] = field(
        default_factory=lambda: derive_instrumentation_layers(36, 12)
    )
    # Initial pre-sigmoid gate logits for SNC and stream adapters. -4.0 gives
    # sigmoid(-4) \u2248 0.0180 so at step 0 the instrumented deltas contribute
    # near-zero; training opens the gates as the auxiliary paths become
    # reliable.
    snc_gate_init: float = -4.0
    adapter_gate_init: float = -4.0


# --------------------------------------------------------------------------- #
# Sidecar (\u03c6)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SNCConfig:
    hidden_size: int = 2560
    notes_dim: int = 256
    attention_width: int = 512
    num_heads: int = 8  # 512 // 8 = head_dim 64
    dropout: float = 0.0


@dataclass(slots=True)
class StreamAdapterConfig:
    hidden_size: int = 2560
    bottleneck_size: int = 512
    streams: Tuple[str, ...] = ("stream_0", "stream_1", "stream_2")
    activation: str = "gelu"
    dropout: float = 0.0


@dataclass(slots=True)
class PlannerHeadConfig:
    hidden_size: int = 2560
    planner_width: int = 512
    vocab_size: int = 8192  # V_p
    num_slots: int = 16  # S
    dropout: float = 0.0


@dataclass(slots=True)
class PlanNotesProjectionConfig:
    """Per-stream projector from quantized planner-slot vectors to notes_dim."""

    planner_width: int = 512
    notes_dim: int = 256


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
class StreamClassifierConfig:
    hidden_size: int = 2560
    classifier_width: int = 512
    num_streams: int = 3
    dropout: float = 0.0


@dataclass(slots=True)
class SidecarConfig:
    """Top-level \u03c6 config. All trainable modules live here."""

    hidden_size: int = 2560  # Must match trunk hidden_size.
    notes_dim: int = 256  # d_notes
    plan_vocab_size: int = 8192  # V_p
    num_streams: int = 3  # K
    snc: SNCConfig = field(default_factory=SNCConfig)
    adapters: StreamAdapterConfig = field(default_factory=StreamAdapterConfig)
    planner_head: PlannerHeadConfig = field(default_factory=PlannerHeadConfig)
    plan_notes_proj: PlanNotesProjectionConfig = field(default_factory=PlanNotesProjectionConfig)
    speculation_head: SpeculationHeadConfig = field(default_factory=SpeculationHeadConfig)
    stream_classifier: StreamClassifierConfig = field(default_factory=StreamClassifierConfig)


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
    """Loss coefficients after removing hash-era supervision.

    L_total = L_LM-CE + lambda_KD*L_KD-LM
              + beta_plan_commit*L_plan_commit + beta_plan_codebook*L_plan_codebook
              + beta_note_commit*L_note_commit + beta_note_codebook*L_note_codebook
              + lambda_plan_usage*L_plan_usage + lambda_note_usage*L_note_usage
              + lambda_stream*L_stream
    """

    lm_ce: float = 1.0
    kd_lm: float = 2.0  # \u03bb_KD
    planner_vq_commit: float = 0.25
    planner_vq_codebook: float = 1.0
    dynamic_vq_commit: float = 0.25
    dynamic_vq_codebook: float = 1.0
    planner_codebook_usage: float = 0.0
    dynamic_codebook_usage: float = 0.0
    stream_classifier: float = 0.1


CURRICULUM_IDENTIFIERS: Tuple[str, ...] = (
    "trunk",
    "planner_head",
    "plan_notes_proj",
    "speculation_head",
    "stream_classifier",
    "snc",
    "stream_adapters",
    "snc_gate",
    "adapter_gate",
)


@dataclass(slots=True)
class StagePolicy:
    """Per-stage freeze/unfreeze policy.

    Module identifiers here are RESOLVED by the name resolver in
    ``pdt.training.curriculum`` to one of:
    - ``"trunk"``              \u2192 the frozen Qwen3 base model
    - ``"planner_head"``       \u2192 ``sidecar.planner_head``
    - ``"plan_notes_proj"``    \u2192 ``sidecar.plan_notes_proj``
    - ``"speculation_head"``   \u2192 ``sidecar.speculation_head``
    - ``"stream_classifier"``  \u2192 ``sidecar.stream_classifier``
    - ``"stream_adapters"``    \u2192 per-layer StreamAdapterLayer inside every
                                    instrumented Qwen3 decoder layer
    - ``"snc"``                \u2192 per-layer SharedNotesCrossAttention inside every
                                    instrumented Qwen3 decoder layer
    - ``"snc_gate"``           \u2192 per-layer outer SNC residual gates
    - ``"adapter_gate"``       \u2192 per-layer outer adapter residual gates
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
                name="diagnostic_mechanism",
                freeze=(
                    "trunk",
                    "planner_head",
                    "plan_notes_proj",
                    "stream_classifier",
                ),
                unfreeze=(
                    "stream_adapters",
                    "snc",
                    "snc_gate",
                    "adapter_gate",
                    "speculation_head",
                ),
            ),
            1: StagePolicy(
                name="vq_planner_integration",
                freeze=("trunk",),
                unfreeze=(
                    "planner_head",
                    "plan_notes_proj",
                    "stream_adapters",
                    "snc",
                    "snc_gate",
                    "adapter_gate",
                    "speculation_head",
                    "stream_classifier",
                ),
            ),
            2: StagePolicy(
                name="integrated_block_rollout",
                freeze=(
                    "trunk",
                    "stream_classifier",
                ),
                unfreeze=(
                    "planner_head",
                    "plan_notes_proj",
                    "stream_adapters",
                    "snc",
                    "snc_gate",
                    "adapter_gate",
                    "speculation_head",
                ),
            ),
            3: StagePolicy(
                name="late_mechanism_training",
                freeze=("trunk",),
                unfreeze=(
                    "planner_head",
                    "plan_notes_proj",
                    "stream_adapters",
                    "snc",
                    "snc_gate",
                    "adapter_gate",
                    "speculation_head",
                    "stream_classifier",
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
        "data/processed/long_form_dependency/qwen3_4b_instruct_2507/train.jsonl"
    )
    eval_dataset_path: str = (
        "data/processed/long_form_dependency/qwen3_4b_instruct_2507/validation.jsonl"
    )
    telemetry_dir: str = "experiments/qwen3_4b"
    batch_size: int = 1
    max_planner_prompt_length: int = 256
    max_stream_prompt_length: int = 512
    max_block_transition_length: int = 64
    max_teacher_prompt_length: int = 1024
    max_blocks: int = 32
    grad_accumulation: int = 16
    max_steps: int = 50_000
    save_every: int = 2500
    log_interval: int = 25
    eval_interval: int = 10_000
    causal_eval_seed: int = 1729
    causal_eval_bootstrap_samples: int = 10_000
    causal_eval_confidence_level: float = 0.95
    causal_eval_min_documents: int = 32
    causal_eval_mutation_producer: str = "stream_0"
    causal_eval_mutation_block: int = 0
    causal_eval_mutation_code_offset: int = 1
    kd_temperature_lm: float = 2.0
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
        if target_layers != expected_layers:
            raise ValueError(
                "instrumentation.target_layers must be derived from the selected trunk "
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
        adapter_streams = tuple(self.sidecar.adapters.streams)
        if not runtime_streams or any(
            not isinstance(stream, str) or not stream.strip() for stream in runtime_streams
        ):
            raise ValueError("runtime.streams must contain non-empty stream identifiers.")
        if len(set(runtime_streams)) != len(runtime_streams):
            raise ValueError("runtime.streams must be unique.")
        if runtime_streams != adapter_streams:
            raise ValueError(
                "runtime.streams must exactly match sidecar.adapters.streams in order: "
                f"runtime={runtime_streams}, adapters={adapter_streams}."
            )

        positive_dimensions = (
            ("sidecar.hidden_size", self.sidecar.hidden_size),
            ("sidecar.notes_dim", self.sidecar.notes_dim),
            ("sidecar.plan_vocab_size", self.sidecar.plan_vocab_size),
            ("sidecar.num_streams", self.sidecar.num_streams),
            ("sidecar.snc.hidden_size", self.sidecar.snc.hidden_size),
            ("sidecar.snc.notes_dim", self.sidecar.snc.notes_dim),
            ("sidecar.snc.attention_width", self.sidecar.snc.attention_width),
            ("sidecar.snc.num_heads", self.sidecar.snc.num_heads),
            ("sidecar.adapters.hidden_size", self.sidecar.adapters.hidden_size),
            ("sidecar.adapters.bottleneck_size", self.sidecar.adapters.bottleneck_size),
            ("sidecar.planner_head.hidden_size", self.sidecar.planner_head.hidden_size),
            ("sidecar.planner_head.planner_width", self.sidecar.planner_head.planner_width),
            ("sidecar.planner_head.vocab_size", self.sidecar.planner_head.vocab_size),
            ("sidecar.planner_head.num_slots", self.sidecar.planner_head.num_slots),
            (
                "sidecar.plan_notes_proj.planner_width",
                self.sidecar.plan_notes_proj.planner_width,
            ),
            ("sidecar.plan_notes_proj.notes_dim", self.sidecar.plan_notes_proj.notes_dim),
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
            (
                "sidecar.stream_classifier.hidden_size",
                self.sidecar.stream_classifier.hidden_size,
            ),
            (
                "sidecar.stream_classifier.classifier_width",
                self.sidecar.stream_classifier.classifier_width,
            ),
            (
                "sidecar.stream_classifier.num_streams",
                self.sidecar.stream_classifier.num_streams,
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
            ("sidecar.adapters.dropout", self.sidecar.adapters.dropout),
            ("sidecar.planner_head.dropout", self.sidecar.planner_head.dropout),
            ("sidecar.speculation_head.dropout", self.sidecar.speculation_head.dropout),
            ("sidecar.stream_classifier.dropout", self.sidecar.stream_classifier.dropout),
        ):
            if not math.isfinite(dropout) or not 0 <= dropout < 1:
                raise ValueError(f"{name} must be finite and in [0, 1), got {dropout}.")

        # notes_dim must match everywhere it appears.
        dims: List[Tuple[str, int]] = [
            ("sidecar.notes_dim", self.sidecar.notes_dim),
            ("sidecar.snc.notes_dim", self.sidecar.snc.notes_dim),
            ("sidecar.speculation_head.notes_dim", self.sidecar.speculation_head.notes_dim),
            (
                "sidecar.plan_notes_proj.notes_dim",
                self.sidecar.plan_notes_proj.notes_dim,
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
            ("sidecar.adapters.hidden_size", self.sidecar.adapters.hidden_size),
            ("sidecar.planner_head.hidden_size", self.sidecar.planner_head.hidden_size),
            (
                "sidecar.speculation_head.hidden_size",
                self.sidecar.speculation_head.hidden_size,
            ),
            (
                "sidecar.stream_classifier.hidden_size",
                self.sidecar.stream_classifier.hidden_size,
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

        # V_p must match between planner head and top-level sidecar.
        if self.sidecar.planner_head.vocab_size != self.sidecar.plan_vocab_size:
            raise ValueError(
                "planner_head.vocab_size must equal sidecar.plan_vocab_size "
                "(planner logits index into the same latent planner codebook)."
            )
        if self.sidecar.plan_notes_proj.planner_width != self.sidecar.planner_head.planner_width:
            raise ValueError(
                "plan_notes_proj.planner_width must equal planner_head.planner_width."
            )

        # K must match between sidecar, adapters, stream_classifier, runtime.
        if len(self.sidecar.adapters.streams) != self.sidecar.num_streams:
            raise ValueError(
                f"len(adapters.streams)={len(self.sidecar.adapters.streams)} != "
                f"sidecar.num_streams={self.sidecar.num_streams}"
            )
        if self.sidecar.stream_classifier.num_streams != self.sidecar.num_streams:
            raise ValueError("stream_classifier.num_streams must equal sidecar.num_streams")
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

        if self.training.kd_temperature_lm != 2.0:
            raise ValueError(
                "training.kd_temperature_lm must be 2.0 for the canonical "
                "same-trunk functional distillation path."
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
    config.sidecar.hidden_size = profile.hidden_size
    config.sidecar.snc.hidden_size = profile.hidden_size
    config.sidecar.adapters.hidden_size = profile.hidden_size
    config.sidecar.planner_head.hidden_size = profile.hidden_size
    config.sidecar.speculation_head.hidden_size = profile.hidden_size
    config.sidecar.stream_classifier.hidden_size = profile.hidden_size
    processed_root = f"data/processed/long_form_dependency/{profile.name}"
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
    "PlanNotesProjectionConfig",
    "PlannerHeadConfig",
    "RuntimeConfig",
    "SNCConfig",
    "SidecarConfig",
    "SpeculationHeadConfig",
    "StagePolicy",
    "StreamAdapterConfig",
    "StreamClassifierConfig",
    "TrainingConfig",
    "TrunkProfile",
    "TrunkConfig",
    "TRUNK_PROFILES",
    "apply_trunk_profile",
    "derive_instrumentation_layers",
]
