"""Roofline accounting for synchronized multi-stream decode.

This module models the hardware primitive PDT ultimately needs to exploit:
apply the same frozen weight matrices to ``K`` independent frontier vectors in
one packed invocation.  It is deliberately a lower-bound model, not a latency
predictor.  Kernel launch cost, cache effects, allocator behavior, sampling,
prefill, synchronization, and host overhead can only make a real run slower.

The useful distinction is exact at the matrix level.  For a BF16 matrix
``W[m, n]`` and ``K`` frontier rows ``X[K, n]``:

* K separate calls perform K matrix-vector products and may fetch ``W`` K
  times from HBM.
* one packed call performs ``X @ W.T`` and can fetch ``W`` once while doing
  the same ``2*K*m*n`` arithmetic.

At small K this raises arithmetic intensity but does not make decode
compute-bound.  The model also accounts for the counter-pressure from KV-cache
reads, which grows with context length even when weight traffic is reused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


Channel = Literal["blind", "pdt", "full_kv"]


@dataclass(frozen=True, slots=True)
class AcceleratorRoofline:
    """Peak dense arithmetic and HBM bandwidth for one accelerator."""

    name: str
    peak_flops_per_second: float
    memory_bytes_per_second: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("accelerator name must be non-empty.")
        for name, value in (
            ("peak_flops_per_second", self.peak_flops_per_second),
            ("memory_bytes_per_second", self.memory_bytes_per_second),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")

    @property
    def ridge_flops_per_byte(self) -> float:
        """Arithmetic intensity at which the roofline becomes compute-bound."""

        return self.peak_flops_per_second / self.memory_bytes_per_second


@dataclass(frozen=True, slots=True)
class DecodeTopology:
    """Parameter and attention geometry needed by the lower-bound model.

    ``shared_recurrent_parameters`` are PDT parameters reused by every stream
    in a packed round (SNC projections and shared outer gates).
    ``per_stream_recurrent_parameters`` is the adapter parameter count used by
    one stream.  A packed K-stream round must load all K distinct adapters.
    Planner and block-writer parameters are intentionally excluded because
    they are not read at every token step.
    """

    trunk_parameters: int
    shared_recurrent_parameters: int
    per_stream_recurrent_parameters: int
    transformer_layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    instrumented_layers: int
    snc_hidden_size: int
    notes_dim: int
    bytes_per_weight: int = 2
    bytes_per_kv_element: int = 2
    bytes_per_note_element: int = 2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.query_heads < self.kv_heads:
            raise ValueError("query_heads must be greater than or equal to kv_heads.")
        if self.transformer_layers < self.instrumented_layers:
            raise ValueError("instrumented_layers cannot exceed transformer_layers.")

    @property
    def active_pdt_parameters_per_stream(self) -> int:
        return (
            self.trunk_parameters
            + self.shared_recurrent_parameters
            + self.per_stream_recurrent_parameters
        )

    def packed_weight_parameters(self, *, streams: int, channel: Channel) -> int:
        _positive_int(streams, "streams")
        if channel == "pdt":
            return (
                self.trunk_parameters
                + self.shared_recurrent_parameters
                + streams * self.per_stream_recurrent_parameters
            )
        if channel in {"blind", "full_kv"}:
            return self.trunk_parameters
        raise ValueError(f"unknown decode channel {channel!r}.")

    def active_parameters_per_stream(self, *, channel: Channel) -> int:
        if channel == "pdt":
            return self.active_pdt_parameters_per_stream
        if channel in {"blind", "full_kv"}:
            return self.trunk_parameters
        raise ValueError(f"unknown decode channel {channel!r}.")


@dataclass(frozen=True, slots=True)
class DecodeRooflineEstimate:
    """One synchronous frontier round under a specified execution strategy."""

    channel: Channel
    packed: bool
    streams: int
    context_tokens_per_stream: int
    linear_flops: int
    attention_flops: int
    weight_bytes: int
    kv_read_bytes: int
    note_read_bytes: int
    total_flops: int
    total_bytes: int
    arithmetic_intensity: float
    compute_floor_seconds: float
    memory_floor_seconds: float
    roofline_floor_seconds: float

    def to_dict(self) -> dict[str, str | bool | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkSpanEstimate:
    """Parallel-algorithm lower bound before hardware cost is applied."""

    total_work: int
    span: int
    streams: int
    work_bound_rounds: int
    lower_bound_rounds: int
    available_parallelism: float
    maximum_round_speedup: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def estimate_work_span(*, total_work: int, span: int, streams: int) -> WorkSpanEstimate:
    """Apply the work--span lower bound ``max(span, ceil(work / K))``."""

    _positive_int(total_work, "total_work")
    _positive_int(span, "span")
    _positive_int(streams, "streams")
    if span > total_work:
        raise ValueError("span cannot exceed total_work.")
    work_bound = math.ceil(total_work / streams)
    lower_bound = max(span, work_bound)
    available_parallelism = total_work / span
    return WorkSpanEstimate(
        total_work=total_work,
        span=span,
        streams=streams,
        work_bound_rounds=work_bound,
        lower_bound_rounds=lower_bound,
        available_parallelism=available_parallelism,
        maximum_round_speedup=total_work / lower_bound,
    )


def estimate_decode_round(
    *,
    accelerator: AcceleratorRoofline,
    topology: DecodeTopology,
    streams: int,
    context_tokens_per_stream: int,
    channel: Channel,
    packed: bool,
) -> DecodeRooflineEstimate:
    """Estimate the best possible time for one K-frontier decode round.

    The estimate assumes every dense weight used by a packed invocation is
    fetched once.  An unpacked round is K separate batch-1 invocations.  GQA
    cache traffic counts the K and V tensors read by every query stream.  A
    ``full_kv`` stream reads all K histories; ``blind`` and ``pdt`` read only
    the stream's private history.  PDT additionally reads its fixed ``2K``
    notes window at each instrumented layer.
    """

    _positive_int(streams, "streams")
    _positive_int(context_tokens_per_stream, "context_tokens_per_stream")
    active_parameters = topology.active_parameters_per_stream(channel=channel)
    linear_flops = 2 * streams * active_parameters

    visible_histories = streams if channel == "full_kv" else 1
    attention_flops = (
        4
        * streams
        * topology.transformer_layers
        * topology.query_heads
        * topology.head_dim
        * context_tokens_per_stream
        * visible_histories
    )
    if channel == "pdt":
        attention_flops += (
            4 * streams * topology.instrumented_layers * topology.snc_hidden_size * (2 * streams)
        )

    if packed:
        weight_parameters = topology.packed_weight_parameters(
            streams=streams,
            channel=channel,
        )
    else:
        weight_parameters = streams * active_parameters
    weight_bytes = weight_parameters * topology.bytes_per_weight

    kv_read_bytes = (
        streams
        * visible_histories
        * 2
        * topology.transformer_layers
        * topology.kv_heads
        * topology.head_dim
        * context_tokens_per_stream
        * topology.bytes_per_kv_element
    )
    note_read_bytes = 0
    if channel == "pdt":
        note_read_bytes = (
            streams
            * topology.instrumented_layers
            * (2 * streams)
            * topology.notes_dim
            * topology.bytes_per_note_element
        )

    total_flops = linear_flops + attention_flops
    total_bytes = weight_bytes + kv_read_bytes + note_read_bytes
    arithmetic_intensity = total_flops / total_bytes
    compute_floor = total_flops / accelerator.peak_flops_per_second
    memory_floor = total_bytes / accelerator.memory_bytes_per_second
    return DecodeRooflineEstimate(
        channel=channel,
        packed=packed,
        streams=streams,
        context_tokens_per_stream=context_tokens_per_stream,
        linear_flops=linear_flops,
        attention_flops=attention_flops,
        weight_bytes=weight_bytes,
        kv_read_bytes=kv_read_bytes,
        note_read_bytes=note_read_bytes,
        total_flops=total_flops,
        total_bytes=total_bytes,
        arithmetic_intensity=arithmetic_intensity,
        compute_floor_seconds=compute_floor,
        memory_floor_seconds=memory_floor,
        roofline_floor_seconds=max(compute_floor, memory_floor),
    )


def packed_round_speedup(
    *,
    accelerator: AcceleratorRoofline,
    topology: DecodeTopology,
    streams: int,
    context_tokens_per_stream: int,
    channel: Channel = "pdt",
) -> float:
    """Roofline upper bound on replacing K separate calls by one packed call."""

    unpacked = estimate_decode_round(
        accelerator=accelerator,
        topology=topology,
        streams=streams,
        context_tokens_per_stream=context_tokens_per_stream,
        channel=channel,
        packed=False,
    )
    packed = estimate_decode_round(
        accelerator=accelerator,
        topology=topology,
        streams=streams,
        context_tokens_per_stream=context_tokens_per_stream,
        channel=channel,
        packed=True,
    )
    return unpacked.roofline_floor_seconds / packed.roofline_floor_seconds


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


# Pinned July-2026 audit values.  H100 BF16 uses the dense figure (the public
# 1,979-TFLOP/s number assumes 2:4 sparsity); PDT does not prune its weights.
H100_SXM_BF16_DENSE = AcceleratorRoofline(
    name="NVIDIA H100 SXM 80GB (dense BF16)",
    peak_flops_per_second=989e12,
    memory_bytes_per_second=3.35e12,
)

QWEN3_4B_PDT = DecodeTopology(
    trunk_parameters=4_022_468_096,
    # 12 * (SNC 14,428,161 + two shared outer gates).
    shared_recurrent_parameters=173_156_388,
    # 12 * one stream's 2,624,512-parameter bottleneck adapter.
    per_stream_recurrent_parameters=31_494_144,
    transformer_layers=36,
    query_heads=32,
    kv_heads=8,
    head_dim=128,
    instrumented_layers=12,
    snc_hidden_size=2560,
    notes_dim=256,
)


__all__ = [
    "AcceleratorRoofline",
    "Channel",
    "DecodeRooflineEstimate",
    "DecodeTopology",
    "H100_SXM_BF16_DENSE",
    "QWEN3_4B_PDT",
    "estimate_decode_round",
    "estimate_work_span",
    "packed_round_speedup",
    "WorkSpanEstimate",
]
