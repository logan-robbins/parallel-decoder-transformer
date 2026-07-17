"""Strict contracts for the canonical PDT configuration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pdt.config.schemas import (
    CurriculumConfig,
    InstrumentationConfig,
    LossWeights,
    NotesBusConfig,
    OptimizerConfig,
    PDTConfig,
    RuntimeConfig,
    SidecarConfig,
    StagePolicy,
    TrainingConfig,
    TRUNK_PROFILES,
    TrunkConfig,
    apply_trunk_profile,
    derive_instrumentation_layers,
)


@pytest.mark.parametrize(
    "trunk",
    (
        replace(TrunkConfig(), base_model="Qwen/Qwen3-4B-Base"),
        replace(TrunkConfig(), revision="main"),
    ),
)
def test_trunk_model_and_revision_are_exactly_pinned(trunk: TrunkConfig) -> None:
    with pytest.raises(ValueError, match="must match profile"):
        PDTConfig(trunk=trunk).validate()


def test_trunk_attention_is_the_canonical_masked_native_gqa_path() -> None:
    with pytest.raises(ValueError, match="pdt_gqa_sdpa"):
        PDTConfig(trunk=replace(TrunkConfig(), attn_implementation="sdpa")).validate()


def test_14b_profile_materializes_one_shared_architecture() -> None:
    config = PDTConfig()
    apply_trunk_profile(config, "qwen3_14b")
    config.validate()

    profile = TRUNK_PROFILES["qwen3_14b"]
    assert config.trunk.base_model == profile.base_model
    assert config.trunk.revision == profile.revision
    assert config.sidecar.hidden_size == 5120
    assert config.instrumentation.target_layers == derive_instrumentation_layers(40, 12)
    assert config.sidecar.snc.attention_width == 512


def test_runtime_timing_is_exactly_tau_32_and_delta_1() -> None:
    with pytest.raises(ValueError, match="tau=32"):
        PDTConfig(runtime=replace(RuntimeConfig(), block_size=16)).validate()
    notes_bus = replace(NotesBusConfig(), lag=0)
    with pytest.raises(ValueError, match="Delta=1"):
        PDTConfig(runtime=replace(RuntimeConfig(), notes_bus=notes_bus)).validate()


@pytest.mark.parametrize(
    "streams",
    (
        (),
        ("stream_0", "stream_0", "stream_2"),
    ),
)
def test_runtime_streams_are_nonempty_and_unique(
    streams: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="runtime.streams"):
        PDTConfig(runtime=replace(RuntimeConfig(), streams=streams)).validate()


@pytest.mark.parametrize(
    "instrumentation",
    (
        replace(InstrumentationConfig(), enabled=False),
        replace(InstrumentationConfig(), target_layers=()),
        replace(InstrumentationConfig(), target_layers=(2, 2)),
        replace(InstrumentationConfig(), target_layers=(-1, 2)),
        replace(InstrumentationConfig(), target_layers=("2", 5)),  # type: ignore[arg-type]
        replace(
            InstrumentationConfig(),
            coordination_source="siblings_plus_self",  # type: ignore[arg-type]
        ),
    ),
)
def test_instrumentation_is_enabled_with_valid_unique_layers(
    instrumentation: InstrumentationConfig,
) -> None:
    with pytest.raises(ValueError, match="instrumentation"):
        PDTConfig(instrumentation=instrumentation).validate()


@pytest.mark.parametrize(
    "dimension_path",
    (
        "hidden_size",
        "notes_dim",
        "num_streams",
        "snc.num_heads",
        "adapters.bottleneck_size",
        "planner_head.num_layers",
        "planner_head.feedforward_width",
        "plan_memory_proj.notes_dim",
        "semantic_supervision.attention_width",
        "speculation_head.notes_dim",
    ),
)
def test_core_dimensions_must_be_positive(dimension_path: str) -> None:
    sidecar = SidecarConfig()
    owner = sidecar
    parts = dimension_path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], 0)
    with pytest.raises(ValueError, match="must be positive"):
        PDTConfig(sidecar=sidecar).validate()


def test_snc_attention_width_must_be_head_divisible() -> None:
    sidecar = SidecarConfig()
    sidecar.snc.attention_width = 511
    with pytest.raises(ValueError, match="must be divisible"):
        PDTConfig(sidecar=sidecar).validate()


@pytest.mark.parametrize(
    "dropout_path",
    (
        "snc.dropout",
        "adapters.dropout",
        "planner_head.dropout",
        "semantic_supervision.dropout",
        "speculation_head.dropout",
    ),
)
@pytest.mark.parametrize("value", (-0.1, 1.0, float("nan")))
def test_canonical_dropouts_are_finite_probabilities(
    dropout_path: str,
    value: float,
) -> None:
    sidecar = SidecarConfig()
    owner_name, field_name = dropout_path.split(".")
    setattr(getattr(sidecar, owner_name), field_name, value)
    with pytest.raises(ValueError, match="dropout"):
        PDTConfig(sidecar=sidecar).validate()


@pytest.mark.parametrize(
    "field_name",
    ("grad_accumulation", "max_steps", "save_every", "log_interval", "eval_interval"),
)
def test_training_update_counters_must_be_positive(field_name: str) -> None:
    training = replace(TrainingConfig(), **{field_name: 0})
    with pytest.raises(ValueError, match=field_name):
        PDTConfig(training=training).validate()


@pytest.mark.parametrize("learning_rate", (0.0, -1.0, float("nan")))
def test_learning_rate_must_be_finite_and_positive(learning_rate: float) -> None:
    optimizer = replace(OptimizerConfig(), learning_rate=learning_rate)
    with pytest.raises(ValueError, match="learning_rate"):
        PDTConfig(training=replace(TrainingConfig(), optimizer=optimizer)).validate()


@pytest.mark.parametrize("weight_decay", (-0.1, float("inf"), float("nan")))
def test_weight_decay_must_be_finite_and_nonnegative(weight_decay: float) -> None:
    optimizer = replace(OptimizerConfig(), weight_decay=weight_decay)
    with pytest.raises(ValueError, match="weight_decay"):
        PDTConfig(training=replace(TrainingConfig(), optimizer=optimizer)).validate()


def test_scheduler_name_must_be_canonical() -> None:
    optimizer = replace(OptimizerConfig(), lr_scheduler="plateau")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lr_scheduler"):
        PDTConfig(training=replace(TrainingConfig(), optimizer=optimizer)).validate()


@pytest.mark.parametrize("warmup_steps", (-1, 50_000, 50_001))
def test_warmup_must_precede_final_optimizer_update(warmup_steps: int) -> None:
    optimizer = replace(OptimizerConfig(), warmup_steps=warmup_steps)
    with pytest.raises(ValueError, match="warmup_steps"):
        PDTConfig(training=replace(TrainingConfig(), optimizer=optimizer)).validate()


@pytest.mark.parametrize(
    ("schedule", "message"),
    (
        ((0, 100, 100, 200), "strictly increasing"),
        ((0, 100, 200, 50_000), "reachable before max_steps"),
    ),
)
def test_stage_schedule_is_strictly_increasing_and_reachable(
    schedule: tuple[int, ...],
    message: str,
) -> None:
    curriculum = replace(CurriculumConfig(), stage_schedule=schedule)
    training = replace(TrainingConfig(), curriculum=curriculum)
    with pytest.raises(ValueError, match=message):
        PDTConfig(training=training).validate()


@pytest.mark.parametrize("field_name", tuple(LossWeights.__dataclass_fields__))
def test_every_global_loss_weight_must_be_nonnegative(field_name: str) -> None:
    weights = replace(LossWeights(), **{field_name: -0.1})
    training = replace(TrainingConfig(), loss_weights=weights)
    with pytest.raises(ValueError, match=field_name):
        PDTConfig(training=training).validate()


def test_stage_loss_overrides_are_also_validated() -> None:
    curriculum = CurriculumConfig()
    stages = dict(curriculum.stages)
    stages[0] = replace(
        stages[0],
        loss_weights=replace(LossWeights(), fact_route=-0.1),
    )
    training = replace(
        TrainingConfig(),
        curriculum=replace(curriculum, stages=stages),
    )
    with pytest.raises(ValueError, match="fact_route"):
        PDTConfig(training=training).validate()


def test_physical_stream_address_order_has_no_fixed_semantic_meaning() -> None:
    config = PDTConfig(
        runtime=replace(
            RuntimeConfig(),
            streams=("decoder_c", "decoder_a", "decoder_b"),
        )
    )
    config.training.causal_eval_mutation_producer = "decoder_c"
    config.validate()


def test_every_stage_must_exhaustively_partition_policy_controls() -> None:
    curriculum = CurriculumConfig()
    stages = dict(curriculum.stages)
    stage_zero = stages[0]
    stages[0] = replace(
        stage_zero,
        unfreeze=tuple(name for name in stage_zero.unfreeze if name != "snc_gate"),
    )
    config = PDTConfig(
        training=replace(
            TrainingConfig(),
            curriculum=replace(curriculum, stages=stages),
        )
    )
    with pytest.raises(ValueError, match="exhaustively control"):
        config.validate()


def test_frozen_trunk_is_explicit_in_every_stage() -> None:
    curriculum = CurriculumConfig()
    stages = dict(curriculum.stages)
    stage_zero = stages[0]
    stages[0] = StagePolicy(
        name=stage_zero.name,
        freeze=tuple(name for name in stage_zero.freeze if name != "trunk"),
        unfreeze=stage_zero.unfreeze + ("trunk",),
    )
    config = PDTConfig(
        training=replace(
            TrainingConfig(),
            curriculum=replace(curriculum, stages=stages),
        )
    )
    with pytest.raises(ValueError, match="keep the frozen trunk"):
        config.validate()


def test_canonical_training_rejects_unidentifiable_local_trunk() -> None:
    config = PDTConfig(trunk=replace(TrunkConfig(), local_path="/tmp/qwen"))
    with pytest.raises(ValueError, match="trunk.local_path is not allowed"):
        config.validate()
