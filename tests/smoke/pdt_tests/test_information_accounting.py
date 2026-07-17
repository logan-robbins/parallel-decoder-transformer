"""Exact tests for finite-message information accounting."""

from __future__ import annotations

import math

import pytest

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
