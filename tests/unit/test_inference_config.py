"""Tests for inference configuration builders."""

from __future__ import annotations

import math

from parallel_decoder_transformer.inference.config import (
    CadencePolicyConfig,
    DecodeConfig,
    GateAnnealingConfig,
    InferenceConfig,
    SafeguardConfig,
    build_inference_config,
)


class _StubCurriculum:
    def __init__(
        self,
        *,
        B: int,
        L: int,
        delta: int,
        rho_by_stream: dict[str, float],
        cadence_min: int = 8,
        cadence_max: int = 128,
    ) -> None:
        self.B = B
        self.L = L
        self.delta = delta
        self.rho_by_stream = dict(rho_by_stream)
        self.cadence_min = cadence_min
        self.cadence_max = cadence_max


class _StubTrainingConfig:
    def __init__(
        self,
        *,
        curriculum: _StubCurriculum,
        agreement_threshold: float,
        coverage_threshold: float = 0.5,
    ) -> None:
        self.curriculum = curriculum
        self.agreement_threshold = agreement_threshold
        self.coverage_threshold = coverage_threshold


def test_build_inference_config_honours_overrides() -> None:
    training_cfg = _StubTrainingConfig(
        curriculum=_StubCurriculum(B=3, L=12, delta=2, rho_by_stream={"intro": 2.0}),
        agreement_threshold=0.2,
        coverage_threshold=0.65,
    )
    decode_cfg = DecodeConfig()

    config = build_inference_config(
        training_cfg,
        stream_to_id={"intro": 0, "core": 1},
        decode_config=decode_cfg,
        emission_cadence={"core": 5},
        gate_g=0.75,
        max_snapshots=6,
        rng_seed=123,
        logit_blend_alpha=0.6,
        gate_annealing={"enabled": True, "min_value": 0.25, "decay": 0.5},
        cadence_policy={"mode": "stochastic", "min_probability": 0.2, "max_interval": 4},
    )

    assert isinstance(config, InferenceConfig)
    assert config.streams == ("intro", "core")
    assert config.stride_B == 3
    assert config.commit_L == 12
    assert config.read_lag_delta == 2
    assert config.max_snapshots_K == 6
    assert config.topology == "all_to_all"
    assert config.cadence_for("intro") == 2
    assert config.cadence_for("core") == 5
    assert config.gate_g == 0.75
    assert config.agreement_threshold_tau == 0.2
    assert config.logit_blend_alpha == 0.6
    assert config.decode is decode_cfg
    assert config.decode.seed == 123
    assert config.rng_seed == 123
    assert isinstance(config.gate_annealing, GateAnnealingConfig)
    assert config.gate_annealing.min_value == 0.25
    assert isinstance(config.cadence_policy, CadencePolicyConfig)
    assert config.cadence_policy.mode == "stochastic"
    assert config.cadence_policy.min_probability == 0.2
    assert config.cadence_policy.max_interval == 4
    assert config.coverage_threshold == 0.65
    assert config.coverage_partial_band == 0.2


def test_build_inference_config_with_safeguards() -> None:
    training_cfg = _StubTrainingConfig(
        curriculum=_StubCurriculum(B=2, L=8, delta=1, rho_by_stream={"intro": 2.0, "core": 2.0}),
        agreement_threshold=0.2,
    )
    safeguards_payload = {
        "lipschitz": {"enabled": True, "epsilon": 5e-4, "probe_interval": 10, "threshold": 2.0},
        "flicker": {"enabled": True, "window": 6, "gate_std_threshold": 0.05},
    }
    config = build_inference_config(
        training_cfg,
        stream_to_id={"intro": 0, "core": 1},
        safeguards=safeguards_payload,
    )

    assert isinstance(config.safeguards, SafeguardConfig)
    assert config.safeguards.lipschitz.enabled
    assert config.safeguards.lipschitz.epsilon == 5e-4
    assert config.safeguards.lipschitz.probe_interval == 10
    assert config.safeguards.flicker.enabled
    assert config.safeguards.flicker.window == 6


def test_build_inference_config_log_linear_cadence_mapping() -> None:
    training_cfg = _StubTrainingConfig(
        curriculum=_StubCurriculum(
            B=2,
            L=8,
            delta=1,
            rho_by_stream={"intro": 1.0, "core": 0.0, "wrap": 0.5},
            cadence_min=4,
            cadence_max=64,
        ),
        agreement_threshold=0.15,
    )
    config = build_inference_config(
        training_cfg,
        stream_to_id={"intro": 0, "core": 1, "wrap": 2},
    )

    assert config.cadence_for("intro") == 4
    assert config.cadence_for("core") == 64
    expected_mid = math.ceil(math.exp(0.5 * math.log(4) + 0.5 * math.log(64)))
    assert config.cadence_for("wrap") == expected_mid
