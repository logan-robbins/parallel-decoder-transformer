"""Inference runtime configuration for the Parallel Decoder Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Literal

# Avoid importing training during inference module import to prevent cycles.
# Use a structural type at call sites; TrainingConfig is only needed for
# attribute access (curriculum, agreement_threshold).
from typing import Any as TrainingConfig  # type: ignore


@dataclass(slots=True)
class DecodeConfig:
    """Sampling controls used during inference-time decoding."""

    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    do_sample: bool = True
    seed: Optional[int] = None

    def as_sampling_kwargs(self) -> Dict[str, object]:
        """Return kwargs compatible with Hugging Face generation utilities."""

        payload: Dict[str, object] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass(slots=True)
class GateAnnealingConfig:
    """Runtime annealing policy for the SNC gate."""

    enabled: bool = True
    decay: float = 0.6  # multiplicative drop when volatility detected
    min_value: float = 0.1  # lower bound for the gate scaling factor
    recovery: float = 0.05  # additive recovery per stable emission
    stability_margin: float = 0.05  # margin above tau treated as stable
    cooldown: int = 1  # emissions to wait before recovering


@dataclass(slots=True)
class CadencePolicyConfig:
    """Controls how note emission cadence is decided at inference time."""

    mode: Literal["deterministic", "stochastic", "adaptive"] = "deterministic"
    min_probability: float = 1e-4
    max_interval: int = 0
    multiplier_min: float = 0.5
    multiplier_max: float = 2.0
    agreement_low: float = 0.25
    agreement_high: float = 0.6
    age_boost: float = 0.0


@dataclass(slots=True)
class CounterfactualConfig:
    """Optional interventions applied to the notes window before SNC."""

    swap_pairs: Tuple[Tuple[str, str], ...] = tuple()
    shuffle_streams: Tuple[str, ...] = tuple()
    freeze_streams: Tuple[str, ...] = tuple()
    ablate_streams: Tuple[str, ...] = tuple()
    stale_overrides: Mapping[str, int] = field(default_factory=dict)
    default_stale_extra: int = 0
    tag: Optional[str] = None

    def normalised(self, streams: Sequence[str]) -> "CounterfactualConfig":
        stream_set = {stream.lower() for stream in streams}
        swaps: list[Tuple[str, str]] = []
        for pair in self.swap_pairs:
            if len(pair) != 2:
                continue
            a, b = pair[0].lower(), pair[1].lower()
            if a == b or a not in stream_set or b not in stream_set:
                continue
            swaps.append((a, b))
        shuffle = tuple(
            stream for stream in (r.lower() for r in self.shuffle_streams) if stream in stream_set
        )
        freeze = tuple(
            stream for stream in (r.lower() for r in self.freeze_streams) if stream in stream_set
        )
        stale_map: MutableMapping[str, int] = {}
        for stream, delta in (self.stale_overrides or {}).items():
            try:
                value = int(delta)
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
            stream_norm = stream.lower()
            if stream_norm in stream_set:
                stale_map[stream_norm] = value
        default_extra = max(0, int(self.default_stale_extra))
        ablate = tuple(
            stream
            for stream in (r.lower() for r in self.ablate_streams or tuple())
            if stream in stream_set
        )
        return CounterfactualConfig(
            swap_pairs=tuple(swaps),
            shuffle_streams=shuffle,
            freeze_streams=freeze,
            ablate_streams=ablate,
            stale_overrides=dict(stale_map),
            default_stale_extra=default_extra,
            tag=self.tag.strip() if isinstance(self.tag, str) and self.tag.strip() else None,
        )

    @property
    def enabled(self) -> bool:
        return bool(
            self.swap_pairs
            or self.shuffle_streams
            or self.freeze_streams
            or self.ablate_streams
            or self.default_stale_extra > 0
            or self.stale_overrides
            or (self.tag is not None)
        )

    def swap_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for a, b in self.swap_pairs:
            mapping[a] = b
            mapping[b] = a
        return mapping

    def should_shuffle(self, stream: str) -> bool:
        return stream in self.shuffle_streams

    def should_freeze(self, stream: str) -> bool:
        return stream in self.freeze_streams

    def should_ablate(self, stream: str) -> bool:
        return stream in self.ablate_streams

    def stale_extra_for(self, stream: str) -> int:
        return int(self.stale_overrides.get(stream, self.default_stale_extra))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "swap_pairs": [list(pair) for pair in self.swap_pairs],
            "shuffle_streams": list(self.shuffle_streams),
            "freeze_streams": list(self.freeze_streams),
            "ablate_streams": list(self.ablate_streams),
            "stale_overrides": dict(self.stale_overrides),
            "default_stale_extra": int(self.default_stale_extra),
            "tag": self.tag,
        }


@dataclass(slots=True)
class LipschitzMonitorConfig:
    """Controls periodic Lipschitz monitoring on attended states."""

    enabled: bool = False
    epsilon: float = 1e-3
    probe_interval: int = 64
    max_stream_probes: int = 1
    threshold: float = 5.0
    max_history: int = 64


@dataclass(slots=True)
class FlickerMonitorConfig:
    """Detects rapid oscillations in gates or plan predictions."""

    enabled: bool = False
    window: int = 8
    gate_std_threshold: float = 0.2
    plan_switch_threshold: int = 3
    clamp_on_violation: bool = False
    clamp_value: float = 0.5


@dataclass(slots=True)
class SafeguardConfig:
    """Bundle of optional safeguards for inference stability."""

    lipschitz: LipschitzMonitorConfig = field(default_factory=LipschitzMonitorConfig)
    flicker: FlickerMonitorConfig = field(default_factory=FlickerMonitorConfig)

    def __post_init__(self) -> None:
        if isinstance(self.lipschitz, Mapping):  # type: ignore[arg-type]
            self.lipschitz = LipschitzMonitorConfig(**self.lipschitz)  # type: ignore[arg-type]
        if isinstance(self.flicker, Mapping):  # type: ignore[arg-type]
            self.flicker = FlickerMonitorConfig(**self.flicker)  # type: ignore[arg-type]


@dataclass(slots=True)
class InferenceConfig:
    """Configuration encapsulating runtime behaviour for multi-stream decoding."""

    streams: Tuple[str, ...]
    stride_B: int
    commit_L: int
    read_lag_delta: int
    max_snapshots_K: int
    topology: Literal["all_to_all"] = "all_to_all"
    gate_g: float = 1.0
    agreement_threshold_tau: float = 0.15
    emission_cadence_M_by_stream: Dict[str, int] = field(default_factory=dict)
    logit_blend_alpha: float = 1.0
    coverage_threshold: float = 0.5
    coverage_partial_band: float = 0.2
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    rng_seed: Optional[int] = None
    gate_annealing: GateAnnealingConfig = field(default_factory=GateAnnealingConfig)
    cadence_policy: CadencePolicyConfig = field(default_factory=CadencePolicyConfig)
    counterfactuals: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    memory_report: bool = False
    safeguards: SafeguardConfig = field(default_factory=SafeguardConfig)
    sectional_self_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.streams:
            raise ValueError("InferenceConfig.streams must contain at least one stream.")
        self.streams = tuple(stream.lower() for stream in self.streams)
        self._validate_integer("stride_B", self.stride_B, minimum=1)
        self._validate_integer("commit_L", self.commit_L, minimum=1)
        self._validate_integer("read_lag_delta", self.read_lag_delta, minimum=0)
        self._validate_integer("max_snapshots_K", self.max_snapshots_K, minimum=1)
        self._validate_integer("sectional_self_tokens", self.sectional_self_tokens, minimum=0)
        if self.topology != "all_to_all":
            raise ValueError("InferenceConfig.topology must be 'all_to_all'.")
        if not 0.0 <= self.gate_g <= 1.0:
            raise ValueError("InferenceConfig.gate_g must be within [0, 1].")
        if self.agreement_threshold_tau <= 0.0 or self.agreement_threshold_tau >= 1.0:
            raise ValueError("InferenceConfig.agreement_threshold_tau must lie within (0, 1).")
        if not 0.0 <= self.logit_blend_alpha <= 1.0:
            raise ValueError("InferenceConfig.logit_blend_alpha must lie within [0, 1].")
        if not 0.0 < self.coverage_threshold < 1.0:
            raise ValueError("InferenceConfig.coverage_threshold must lie within (0, 1).")
        if self.coverage_partial_band < 0.0:
            raise ValueError("InferenceConfig.coverage_partial_band must be non-negative.")
        if self.coverage_partial_band > 0.5:
            self.coverage_partial_band = 0.5
        # Normalise cadence dictionary to contain every stream.
        cadence: Dict[str, int] = {}
        for stream in self.streams:
            cadence_value = self.emission_cadence_M_by_stream.get(stream, self.stride_B)
            cadence_value = int(round(cadence_value))
            if cadence_value <= 0:
                cadence_value = self.stride_B
            cadence[stream] = cadence_value
        self.emission_cadence_M_by_stream = cadence
        self._validate_gate_policy()
        self._validate_cadence_policy()
        self.counterfactuals = self.counterfactuals.normalised(self.streams)
        self.memory_report = bool(self.memory_report)
        safeguards = self.safeguards
        if isinstance(safeguards, Mapping):  # type: ignore[arg-type]
            safeguards = SafeguardConfig(**safeguards)  # type: ignore[arg-type]
        self._validate_safeguards(safeguards)
        self.safeguards = safeguards

    def cadence_for(self, stream: str) -> int:
        """Return the note emission cadence for a stream."""

        try:
            return self.emission_cadence_M_by_stream[stream.lower()]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown stream: {stream!r}") from exc

    @staticmethod
    def _validate_integer(name: str, value: int, *, minimum: int) -> None:
        if value < minimum:
            raise ValueError(f"InferenceConfig.{name} must be >= {minimum}, received {value}.")

    def _validate_gate_policy(self) -> None:
        policy = self.gate_annealing
        if not policy.enabled:
            return
        if not 0.0 <= policy.min_value <= 1.0:
            raise ValueError("Gate annealing min_value must lie within [0,1].")
        if not 0.0 < policy.decay <= 1.0:
            raise ValueError("Gate annealing decay must lie within (0,1].")
        if not 0.0 <= policy.recovery <= 1.0:
            raise ValueError("Gate annealing recovery must lie within [0,1].")
        if policy.cooldown < 0:
            raise ValueError("Gate annealing cooldown must be non-negative.")
        if not 0.0 <= policy.stability_margin < 1.0:
            raise ValueError("Gate annealing stability_margin must lie within [0,1).")
        if policy.min_value > self.gate_g:
            raise ValueError("Gate annealing min_value cannot exceed the base gate_g.")

    def _validate_cadence_policy(self) -> None:
        policy = self.cadence_policy
        if policy.mode not in {"deterministic", "stochastic", "adaptive"}:
            raise ValueError(
                "Cadence policy mode must be one of {'deterministic','stochastic','adaptive'}."
            )
        if policy.min_probability <= 0.0 or policy.min_probability > 1.0:
            raise ValueError("Cadence min_probability must lie within (0,1].")
        if policy.max_interval < 0:
            raise ValueError("Cadence max_interval must be non-negative.")
        if policy.multiplier_min <= 0.0:
            raise ValueError("Cadence multiplier_min must be positive.")
        if policy.multiplier_max < policy.multiplier_min:
            raise ValueError("Cadence multiplier_max must be >= multiplier_min.")
        if not 0.0 <= policy.agreement_low <= 1.0:
            raise ValueError("Cadence agreement_low must lie within [0,1].")
        if not 0.0 <= policy.agreement_high <= 1.0:
            raise ValueError("Cadence agreement_high must lie within [0,1].")
        if policy.agreement_high <= policy.agreement_low:
            raise ValueError("Cadence agreement_high must exceed agreement_low.")
        if policy.age_boost < 0.0:
            raise ValueError("Cadence age_boost must be non-negative.")

    def _validate_safeguards(self, config: SafeguardConfig) -> None:
        lip = config.lipschitz
        if lip.enabled:
            if lip.epsilon <= 0.0:
                raise ValueError("Lipschitz epsilon must be positive.")
            if lip.probe_interval <= 0:
                raise ValueError("Lipschitz probe_interval must be positive.")
            if lip.threshold <= 0.0:
                raise ValueError("Lipschitz threshold must be positive.")
            if lip.max_history < 1:
                raise ValueError("Lipschitz max_history must be >=1.")
            if lip.max_stream_probes < 0:
                raise ValueError("Lipschitz max_stream_probes must be >=0.")
        flicker = config.flicker
        if flicker.enabled:
            if flicker.window < 2:
                raise ValueError("Flicker window must be >=2.")
            if flicker.gate_std_threshold <= 0.0:
                raise ValueError("Flicker gate_std_threshold must be positive.")
            if flicker.plan_switch_threshold < 1:
                raise ValueError("Flicker plan_switch_threshold must be >=1.")
            if flicker.clamp_value < 0.0 or flicker.clamp_value > 1.0:
                raise ValueError("Flicker clamp_value must be within [0,1].")


def build_inference_config(
    training_config: TrainingConfig,
    *,
    stream_to_id: Mapping[str, int],
    decode_config: Optional[DecodeConfig] = None,
    emission_cadence: Optional[Mapping[str, float]] = None,
    gate_g: Optional[float] = None,
    max_snapshots: Optional[int] = None,
    rng_seed: Optional[int] = None,
    logit_blend_alpha: Optional[float] = None,
    gate_annealing: Optional[Any] = None,
    cadence_policy: Optional[Any] = None,
    read_lag_delta: Optional[int] = None,
    counterfactuals: Optional[Any] = None,
    memory_report: Optional[bool] = None,
    safeguards: Optional[Any] = None,
) -> InferenceConfig:
    """Derive an :class:`InferenceConfig` aligned with the training curriculum."""

    if not stream_to_id:
        raise ValueError("stream_to_id mapping must not be empty.")
    ordered_streams = tuple(
        stream.lower() for stream, _ in sorted(stream_to_id.items(), key=lambda item: item[1])
    )

    curriculum = training_config.curriculum
    stride_B = int(curriculum.B)
    commit_L = int(curriculum.L)
    read_lag = int(read_lag_delta) if read_lag_delta is not None else int(curriculum.delta)
    max_snapshots_K = int(max_snapshots) if max_snapshots is not None else max(stride_B, 4)
    cadence_min = _coerce_positive_int(
        getattr(curriculum, "cadence_min", stride_B),
        fallback=stride_B,
    )
    cadence_min = max(1, cadence_min)
    cadence_max = _coerce_positive_int(
        getattr(curriculum, "cadence_max", max(stride_B, cadence_min)),
        fallback=max(stride_B, cadence_min),
    )
    cadence_max = max(cadence_min, cadence_max)
    log_cadence_min = math.log(cadence_min)
    log_cadence_max = math.log(cadence_max) if cadence_max != cadence_min else log_cadence_min

    cadence_payload: Dict[str, float] = {
        stream.lower(): float(value) for stream, value in curriculum.rho_by_stream.items()
    }
    if emission_cadence is not None:
        cadence_payload.update(
            {stream.lower(): float(value) for stream, value in emission_cadence.items()}
        )

    def _map_cadence_value(raw_value: float) -> int:
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return stride_B
        if 0.0 <= numeric <= 1.0:
            if cadence_min == cadence_max:
                return cadence_min
            blended_log = numeric * log_cadence_min + (1.0 - numeric) * log_cadence_max
            mapped = math.exp(blended_log)
            return _coerce_positive_int(math.ceil(mapped), fallback=cadence_min)
        return _coerce_positive_int(numeric, fallback=stride_B)

    def _resolve_stream_cadence(stream: str) -> int:
        if stream in cadence_payload:
            return _map_cadence_value(cadence_payload[stream])
        return stride_B

    cadence_int: Dict[str, int] = {
        stream: _resolve_stream_cadence(stream) for stream in ordered_streams
    }

    topology_value: Literal["all_to_all"] = "all_to_all"

    if isinstance(gate_annealing, Mapping):
        gate_policy = GateAnnealingConfig(**gate_annealing)  # type: ignore[arg-type]
    elif isinstance(gate_annealing, GateAnnealingConfig):
        gate_policy = gate_annealing
    elif gate_annealing is None:
        gate_policy = GateAnnealingConfig()
    else:  # pragma: no cover - defensive guard
        raise TypeError("gate_annealing must be a Mapping, GateAnnealingConfig, or None.")

    if isinstance(cadence_policy, Mapping):
        cadence_policy_cfg = CadencePolicyConfig(**cadence_policy)  # type: ignore[arg-type]
    elif isinstance(cadence_policy, CadencePolicyConfig):
        cadence_policy_cfg = cadence_policy
    elif cadence_policy is None:
        cadence_policy_cfg = CadencePolicyConfig()
    else:  # pragma: no cover - defensive guard
        raise TypeError("cadence_policy must be a Mapping, CadencePolicyConfig, or None.")

    if decode_config is None:
        decode_cfg = DecodeConfig(seed=rng_seed)
    else:
        decode_cfg = decode_config
        if rng_seed is not None and decode_cfg.seed is None:
            decode_cfg.seed = rng_seed

    if isinstance(counterfactuals, Mapping):
        counterfactual_cfg = CounterfactualConfig(**counterfactuals)  # type: ignore[arg-type]
    elif isinstance(counterfactuals, CounterfactualConfig):
        counterfactual_cfg = counterfactuals
    else:
        counterfactual_cfg = CounterfactualConfig()

    if isinstance(safeguards, Mapping):
        safeguards_cfg = SafeguardConfig(**safeguards)  # type: ignore[arg-type]
    elif isinstance(safeguards, SafeguardConfig):
        safeguards_cfg = safeguards
    else:
        safeguards_cfg = SafeguardConfig()

    config = InferenceConfig(
        streams=ordered_streams,
        stride_B=stride_B,
        commit_L=commit_L,
        read_lag_delta=read_lag,
        max_snapshots_K=max_snapshots_K,
        topology=topology_value,
        gate_g=gate_g if gate_g is not None else 1.0,
        agreement_threshold_tau=float(training_config.agreement_threshold),
        emission_cadence_M_by_stream=cadence_int,
        logit_blend_alpha=logit_blend_alpha if logit_blend_alpha is not None else 1.0,
        coverage_threshold=float(getattr(training_config, "coverage_threshold", 0.5)),
        decode=decode_cfg,
        rng_seed=rng_seed,
        gate_annealing=gate_policy,
        cadence_policy=cadence_policy_cfg,
        counterfactuals=counterfactual_cfg,
        memory_report=bool(memory_report) if memory_report is not None else False,
        safeguards=safeguards_cfg,
    )
    return config


def _coerce_positive_int(value: float, *, fallback: int) -> int:
    result = int(round(value))
    if result <= 0:
        return int(fallback)
    return result


__all__ = [
    "CadencePolicyConfig",
    "DecodeConfig",
    "GateAnnealingConfig",
    "CounterfactualConfig",
    "LipschitzMonitorConfig",
    "FlickerMonitorConfig",
    "SafeguardConfig",
    "InferenceConfig",
    "build_inference_config",
]
