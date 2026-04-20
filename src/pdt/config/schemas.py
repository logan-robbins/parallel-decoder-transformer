"""Dataclass-based configuration schemas for PDT.

One source of truth: a single ``PDTConfig`` dataclass tree that maps 1:1 to
the canonical YAML. All runtime code consumes subtrees of this type; no
subsystem is allowed to read YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


# --------------------------------------------------------------------------- #
# Trunk
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TrunkConfig:
    """Frozen Qwen3 trunk loader configuration."""

    base_model: str = "Qwen/Qwen3-4B-Base"
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = None
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    # Local weight override. If set, loader uses `from_pretrained(local_path)`.
    local_path: Optional[str] = None
    # List of extra special tokens to add to the tokenizer + embedding matrix.
    extra_special_tokens: Tuple[str, ...] = (
        "<plan>",
        "<notes>",
        "<rollback>",
        "<commit>",
    )


@dataclass(slots=True)
class InstrumentationConfig:
    """Which decoder layers to instrument, and how gates are initialized."""

    enabled: bool = True
    # Explicit layer indices to instrument. For Qwen3-4B (36 layers) every 3rd
    # layer corresponds to [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35].
    target_layers: Tuple[int, ...] = (
        2,
        5,
        8,
        11,
        14,
        17,
        20,
        23,
        26,
        29,
        32,
        35,
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
    num_heads: int = 16  # 2560 // 16 = head_dim 160
    dropout: float = 0.0
    spectral_norm: bool = False


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
    vocab_size: int = 8192  # V_p
    num_slots: int = 16  # S
    dropout: float = 0.0


@dataclass(slots=True)
class PlanNotesProjectionConfig:
    """Per-stream projector from active planner-slot embeddings to notes_dim."""

    hidden_size: int = 2560  # Input: pooled plan-slot embeddings
    notes_dim: int = 256  # Output: snapshot-0 vector per stream


@dataclass(slots=True)
class NotesHeadConfig:
    hidden_size: int = 2560
    notes_dim: int = 256
    dropout: float = 0.0
    gated: bool = True


@dataclass(slots=True)
class SpeculationHeadConfig:
    hidden_size: int = 2560
    notes_dim: int = 256
    dropout: float = 0.0
    teacher_scale: float = 1.0


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
    num_streams: int = 3
    dropout: float = 0.0


@dataclass(slots=True)
class SidecarConfig:
    """Top-level \u03c6 config. All trainable modules live here."""

    hidden_size: int = 2560  # Must match trunk hidden_size.
    notes_dim: int = 256  # d_notes
    plan_vocab_size: int = 8192  # V_p
    num_streams: int = 3  # K
    plan_hash_salt: str = "parallel-decoder-v1"
    snc: SNCConfig = field(default_factory=SNCConfig)
    adapters: StreamAdapterConfig = field(default_factory=StreamAdapterConfig)
    planner_head: PlannerHeadConfig = field(default_factory=PlannerHeadConfig)
    plan_notes_proj: PlanNotesProjectionConfig = field(
        default_factory=PlanNotesProjectionConfig
    )
    notes_head: NotesHeadConfig = field(default_factory=NotesHeadConfig)
    speculation_head: SpeculationHeadConfig = field(default_factory=SpeculationHeadConfig)
    coverage_head: CoverageHeadConfig = field(default_factory=CoverageHeadConfig)
    agreement_head: AgreementHeadConfig = field(default_factory=AgreementHeadConfig)
    stream_classifier: StreamClassifierConfig = field(default_factory=StreamClassifierConfig)


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class NotesBusConfig:
    snapshot_dim: int = 256  # Must match sidecar.notes_dim.
    max_snapshots: int = 4  # B
    lag: int = 1  # \u0394
    dtype: str = "bfloat16"


@dataclass(slots=True)
class RuntimeConfig:
    streams: Tuple[str, ...] = ("stream_0", "stream_1", "stream_2")
    topology: Literal["all_to_all"] = "all_to_all"
    # \u03c4: tokens per provisional block between synchronization decisions.
    block_size: int = 32
    # Commit horizon: how many tokens can be rolled back.
    commit_horizon: int = 64
    # Self-only warmup: first N tokens only see own snapshots (helps stabilize).
    self_only_tokens: int = 0
    # Agreement threshold (\u03b3). Initial value; can be tuned offline via ROC.
    agreement_threshold: float = 0.5
    notes_bus: NotesBusConfig = field(default_factory=NotesBusConfig)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LossWeights:
    """Loss coefficients matching the paper equation (Appendix A).

    L_total = L_plan + L_notes + 0.5*L_spec + L_LM-CE + \u03bb_KD*L_KD-LM
              + \u03bb_cov*L_cov + \u03bb_ready*L_ready
    """

    planner: float = 1.0
    notes: float = 1.0
    spec: float = 0.5  # Fixed 0.5 per paper.
    lm_ce: float = 1.0
    kd_lm: float = 2.0  # \u03bb_KD
    coverage: float = 1.0  # \u03bb_cov
    readiness: float = 1.0  # \u03bb_ready


@dataclass(slots=True)
class StagePolicy:
    """Per-stage freeze/unfreeze policy.

    Module identifiers here are RESOLVED by the name resolver in
    ``pdt.training.curriculum`` to one of:
    - ``"trunk"``              \u2192 the frozen Qwen3 base model
    - ``"planner_head"``       \u2192 ``sidecar.planner_head``
    - ``"plan_notes_proj"``    \u2192 ``sidecar.plan_notes_proj``
    - ``"plan_embedding"``     \u2192 ``sidecar.plan_embedding``
    - ``"notes_head"``         \u2192 ``sidecar.notes_head``
    - ``"speculation_head"``   \u2192 ``sidecar.speculation_head``
    - ``"coverage_head"``      \u2192 ``sidecar.coverage_head``
    - ``"agreement_head"``     \u2192 ``sidecar.agreement_head``
    - ``"stream_classifier"``  \u2192 ``sidecar.stream_classifier``
    - ``"stream_adapters"``    \u2192 per-layer StreamAdapterLayer inside every
                                    instrumented Qwen3 decoder layer
    - ``"snc"``                \u2192 per-layer SharedNotesCrossAttention inside every
                                    instrumented Qwen3 decoder layer
    """

    name: str
    freeze: Tuple[str, ...] = field(default_factory=tuple)
    unfreeze: Tuple[str, ...] = field(default_factory=tuple)
    bus_mix_prob: float = 0.0
    stream_dropout_prob: float = 0.0
    # Optional per-stage loss-weight override. Entries that are None fall back
    # to the global ``LossWeights``.
    loss_weights: Optional[LossWeights] = None


@dataclass(slots=True)
class CurriculumConfig:
    """Staged curriculum schedule. Stage index is monotone in global_step."""

    # Global step at which each stage becomes active. Length must be 4.
    stage_schedule: Tuple[int, ...] = (0, 3750, 10000, 25000)
    stages: Dict[int, StagePolicy] = field(
        default_factory=lambda: {
            0: StagePolicy(
                name="planner_pretrain",
                freeze=(
                    "trunk",
                    "stream_adapters",
                    "snc",
                    "speculation_head",
                    "agreement_head",
                    "coverage_head",
                    "stream_classifier",
                ),
                unfreeze=(
                    "planner_head",
                    "plan_embedding",
                    "plan_notes_proj",
                    "notes_head",
                ),
            ),
            1: StagePolicy(
                name="stream_bootstrap",
                freeze=(
                    "trunk",
                    "speculation_head",
                    "agreement_head",
                    "coverage_head",
                ),
                unfreeze=(
                    "stream_adapters",
                    "snc",
                    "stream_classifier",
                ),
            ),
            2: StagePolicy(
                name="notes_bus_enable",
                freeze=(
                    "trunk",
                    "agreement_head",
                    "coverage_head",
                ),
                unfreeze=("speculation_head",),
            ),
            3: StagePolicy(
                name="commit_control",
                freeze=("trunk",),
                unfreeze=("agreement_head", "coverage_head"),
                bus_mix_prob=0.35,
                stream_dropout_prob=0.15,
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
class TeacherCacheConfig:
    """Per-example .pt cache of hashed teacher-note tensors.

    Cache is invalidated by any change to ``notes_dim`` or to the stream
    ordering; point ``cache_dir`` at a fresh directory when either changes.
    """

    cache_dir: str = "data/teacher_cache_qwen3_4b"
    max_snapshots: int = 4
    id_field: str = "example_id"
    refresh_cache: bool = False


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: str = "data/processed/pdt_10k_qwen3_4b/kd_train.jsonl"
    eval_dataset_path: str = "data/processed/pdt_10k_qwen3_4b/kd_validation.jsonl"
    telemetry_dir: str = "experiments/qwen3_4b"
    batch_size: int = 1
    grad_accumulation: int = 16
    max_steps: int = 50_000
    save_every: int = 2500
    log_interval: int = 25
    eval_interval: int = 10_000
    kd_temperature_planner: float = 1.0
    kd_temperature_lm: float = 0.5
    coverage_threshold: float = 0.4
    device: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    teacher_cache: TeacherCacheConfig = field(default_factory=TeacherCacheConfig)
    # Dataset-side knobs for in-batch masking.
    bus_mix_prob: float = 0.0
    stream_dropout_prob: float = 0.0


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

        # notes_dim must match everywhere it appears.
        dims: List[Tuple[str, int]] = [
            ("sidecar.notes_dim", self.sidecar.notes_dim),
            ("sidecar.snc.notes_dim", self.sidecar.snc.notes_dim),
            ("sidecar.notes_head.notes_dim", self.sidecar.notes_head.notes_dim),
            ("sidecar.speculation_head.notes_dim", self.sidecar.speculation_head.notes_dim),
            (
                "sidecar.plan_notes_proj.notes_dim",
                self.sidecar.plan_notes_proj.notes_dim,
            ),
            ("runtime.notes_bus.snapshot_dim", self.runtime.notes_bus.snapshot_dim),
            ("sidecar.agreement_head.notes_dim", self.sidecar.agreement_head.notes_dim),
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
                "sidecar.plan_notes_proj.hidden_size",
                self.sidecar.plan_notes_proj.hidden_size,
            ),
            ("sidecar.notes_head.hidden_size", self.sidecar.notes_head.hidden_size),
            (
                "sidecar.speculation_head.hidden_size",
                self.sidecar.speculation_head.hidden_size,
            ),
            ("sidecar.coverage_head.hidden_size", self.sidecar.coverage_head.hidden_size),
            (
                "sidecar.agreement_head.hidden_size",
                self.sidecar.agreement_head.hidden_size,
            ),
            (
                "sidecar.stream_classifier.hidden_size",
                self.sidecar.stream_classifier.hidden_size,
            ),
        ]
        canonical_h = hs[0][1]
        for name, value in hs[1:]:
            if value != canonical_h:
                raise ValueError(
                    f"hidden_size mismatch: {name}={value} != sidecar.hidden_size={canonical_h}"
                )

        # V_p must match between planner head and top-level sidecar.
        if self.sidecar.planner_head.vocab_size != self.sidecar.plan_vocab_size:
            raise ValueError(
                "planner_head.vocab_size must equal sidecar.plan_vocab_size "
                "(planner logits index into the same latent plan vocabulary as plan_embedding)."
            )

        # K must match between sidecar, adapters, stream_classifier, runtime.
        if len(self.sidecar.adapters.streams) != self.sidecar.num_streams:
            raise ValueError(
                f"len(adapters.streams)={len(self.sidecar.adapters.streams)} != "
                f"sidecar.num_streams={self.sidecar.num_streams}"
            )
        if self.sidecar.stream_classifier.num_streams != self.sidecar.num_streams:
            raise ValueError(
                "stream_classifier.num_streams must equal sidecar.num_streams"
            )
        if len(self.runtime.streams) != self.sidecar.num_streams:
            raise ValueError(
                f"len(runtime.streams)={len(self.runtime.streams)} != "
                f"sidecar.num_streams={self.sidecar.num_streams}"
            )

        # Curriculum schedule sanity.
        sched = self.training.curriculum.stage_schedule
        if len(sched) != 4:
            raise ValueError(f"stage_schedule must have 4 entries, got {len(sched)}.")
        if sched[0] != 0:
            raise ValueError("stage_schedule must start at 0.")
        if any(sched[i + 1] < sched[i] for i in range(len(sched) - 1)):
            raise ValueError("stage_schedule must be non-decreasing.")
        for stage_idx in range(4):
            if stage_idx not in self.training.curriculum.stages:
                raise ValueError(f"Missing stage policy for stage {stage_idx}")


__all__ = [
    "AgreementHeadConfig",
    "CoverageHeadConfig",
    "CurriculumConfig",
    "InstrumentationConfig",
    "LossWeights",
    "NotesBusConfig",
    "NotesHeadConfig",
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
    "TeacherCacheConfig",
    "TrainingConfig",
    "TrunkConfig",
]
