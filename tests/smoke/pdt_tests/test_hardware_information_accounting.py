"""Exact tests for the information and hardware lower-bound diagnostics."""

from __future__ import annotations

import math

import pytest

from pdt.diagnostics.hardware import (
    H100_SXM_BF16_DENSE,
    QWEN3_4B_PDT,
    estimate_decode_round,
    estimate_work_span,
    packed_round_speedup,
)
from pdt.diagnostics.information import (
    audit_uniform_payload,
    capacity_efficiency_ceiling,
    finite_message_capacity_bits,
    nominal_storage_bits,
    uniform_source_entropy_bits,
)


def test_exact_entropy_and_finite_rate_accounting_are_separate_from_token_ce() -> None:
    assert uniform_source_entropy_bits(alphabet_size=64, symbols=3) == 18.0
    assert finite_message_capacity_bits(codebooks=4, codes_per_codebook=256) == 32.0
    assert nominal_storage_bits(elements=256, bits_per_element=16) == 4096
    assert capacity_efficiency_ceiling(source_bits=18, channel_bits=4096) == pytest.approx(
        18 / 4096
    )

    audit = audit_uniform_payload(
        alphabet_size=64,
        symbols=3,
        conditional_cross_entropy_nats=6 * math.log(2),
        channel_capacity_bits=32,
    )
    assert audit.conditional_cross_entropy_bits == pytest.approx(6.0)
    assert audit.delivered_information_lower_bound_bits == pytest.approx(12.0)
    assert audit.capacity_utilization_lower_bound == pytest.approx(12 / 32)
    assert audit.source_recovery_lower_bound == pytest.approx(12 / 18)
    assert not audit.capacity_violation


def test_information_audit_flags_a_bound_violation_and_rejects_invalid_inputs() -> None:
    impossible = audit_uniform_payload(
        alphabet_size=64,
        symbols=3,
        conditional_cross_entropy_nats=0.0,
        channel_capacity_bits=8.0,
    )
    assert impossible.delivered_information_lower_bound_bits == 18.0
    assert impossible.capacity_violation

    with pytest.raises(ValueError, match="alphabet_size"):
        uniform_source_entropy_bits(alphabet_size=1, symbols=3)
    with pytest.raises(ValueError, match="conditional_cross_entropy_nats"):
        audit_uniform_payload(
            alphabet_size=64,
            symbols=3,
            conditional_cross_entropy_nats=-1.0,
            channel_capacity_bits=32.0,
        )


def test_packed_decode_loads_shared_weights_once_but_kv_traffic_still_scales() -> None:
    separate = estimate_decode_round(
        accelerator=H100_SXM_BF16_DENSE,
        topology=QWEN3_4B_PDT,
        streams=3,
        context_tokens_per_stream=4096,
        channel="pdt",
        packed=False,
    )
    packed = estimate_decode_round(
        accelerator=H100_SXM_BF16_DENSE,
        topology=QWEN3_4B_PDT,
        streams=3,
        context_tokens_per_stream=4096,
        channel="pdt",
        packed=True,
    )
    assert separate.total_flops == packed.total_flops
    assert separate.kv_read_bytes == packed.kv_read_bytes
    assert separate.weight_bytes == (3 * QWEN3_4B_PDT.active_pdt_parameters_per_stream * 2)
    assert (
        packed.weight_bytes
        == (
            QWEN3_4B_PDT.trunk_parameters
            + QWEN3_4B_PDT.shared_recurrent_parameters
            + 3 * QWEN3_4B_PDT.per_stream_recurrent_parameters
        )
        * 2
    )
    assert packed.weight_bytes < separate.weight_bytes
    assert packed.memory_floor_seconds > packed.compute_floor_seconds
    assert packed.arithmetic_intensity < H100_SXM_BF16_DENSE.ridge_flops_per_byte
    assert (
        2.0
        < packed_round_speedup(
            accelerator=H100_SXM_BF16_DENSE,
            topology=QWEN3_4B_PDT,
            streams=3,
            context_tokens_per_stream=4096,
        )
        < 3.0
    )


def test_work_span_bound_separates_width_from_true_dependency_depth() -> None:
    wide = estimate_work_span(total_work=3000, span=1000, streams=3)
    assert wide.work_bound_rounds == 1000
    assert wide.lower_bound_rounds == 1000
    assert wide.maximum_round_speedup == 3.0

    deep = estimate_work_span(total_work=3000, span=2400, streams=3)
    assert deep.lower_bound_rounds == 2400
    assert deep.maximum_round_speedup == pytest.approx(1.25)
    with pytest.raises(ValueError, match="span cannot exceed"):
        estimate_work_span(total_work=10, span=11, streams=3)


def test_full_kv_has_quadratic_cross_stream_cache_reads() -> None:
    context = 16_384
    pdt = estimate_decode_round(
        accelerator=H100_SXM_BF16_DENSE,
        topology=QWEN3_4B_PDT,
        streams=3,
        context_tokens_per_stream=context,
        channel="pdt",
        packed=True,
    )
    full_kv = estimate_decode_round(
        accelerator=H100_SXM_BF16_DENSE,
        topology=QWEN3_4B_PDT,
        streams=3,
        context_tokens_per_stream=context,
        channel="full_kv",
        packed=True,
    )
    assert full_kv.kv_read_bytes == 3 * pdt.kv_read_bytes
    assert pdt.note_read_bytes > 0
    assert full_kv.note_read_bytes == 0
    assert pdt.note_read_bytes < pdt.kv_read_bytes
