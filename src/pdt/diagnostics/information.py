"""Information accounting with explicit identification assumptions.

Paired model cross-entropy under two prompts is a useful causal utility metric,
but it is not automatically Shannon mutual information and is not bounded by a
message's bit width.  A bit-valid audit needs both a finite message alphabet
and an identified source entropy.

The exact register-relay corpus supplies the latter: each payload contains
``symbols`` independent draws from a known uniform alphabet.  If a variational
decoder predicts the *whole payload* from the received message and legal side
information, the Barber--Agakov lower bound is

    I(payload; message | side_info) >= H(payload | side_info) - CE(decoder).

That lower bound can be compared with the explicit finite channel capacity.
Token-level LM CE differences remain separate and should never be relabeled as
delivered bits without this construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True, slots=True)
class InformationAudit:
    source_entropy_bits: float
    channel_capacity_bits: float
    conditional_cross_entropy_bits: float
    raw_variational_lower_bound_bits: float
    delivered_information_lower_bound_bits: float
    capacity_utilization_lower_bound: float
    source_recovery_lower_bound: float
    capacity_violation: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def uniform_source_entropy_bits(*, alphabet_size: int, symbols: int) -> float:
    """Entropy of independent uniform symbols from a known alphabet."""

    _integer_at_least(alphabet_size, "alphabet_size", minimum=2)
    _integer_at_least(symbols, "symbols", minimum=1)
    return symbols * math.log2(alphabet_size)


def finite_message_capacity_bits(*, codebooks: int, codes_per_codebook: int) -> float:
    """Maximum entropy of a product code with fixed finite alphabets."""

    _integer_at_least(codebooks, "codebooks", minimum=1)
    _integer_at_least(codes_per_codebook, "codes_per_codebook", minimum=2)
    return codebooks * math.log2(codes_per_codebook)


def nominal_storage_bits(*, elements: int, bits_per_element: int) -> int:
    """Physical storage width; this is not an estimate of used information."""

    _integer_at_least(elements, "elements", minimum=1)
    _integer_at_least(bits_per_element, "bits_per_element", minimum=1)
    return elements * bits_per_element


def capacity_efficiency_ceiling(*, source_bits: float, channel_bits: float) -> float:
    """Maximum useful-source-bit fraction of a physical or finite channel."""

    _positive_finite(source_bits, "source_bits", allow_zero=True)
    _positive_finite(channel_bits, "channel_bits", allow_zero=False)
    return min(source_bits, channel_bits) / channel_bits


def audit_uniform_payload(
    *,
    alphabet_size: int,
    symbols: int,
    conditional_cross_entropy_nats: float,
    channel_capacity_bits: float,
    tolerance_bits: float = 1e-6,
) -> InformationAudit:
    """Build a bit-valid variational audit for one complete payload.

    ``conditional_cross_entropy_nats`` must be the average negative log
    likelihood of the complete discrete payload under a decoder that receives
    only the finite message and causally legal side information.  It must not be
    a token-CE delta between unrelated prompts.
    """

    source_bits = uniform_source_entropy_bits(
        alphabet_size=alphabet_size,
        symbols=symbols,
    )
    _positive_finite(
        conditional_cross_entropy_nats,
        "conditional_cross_entropy_nats",
        allow_zero=True,
    )
    _positive_finite(channel_capacity_bits, "channel_capacity_bits", allow_zero=False)
    _positive_finite(tolerance_bits, "tolerance_bits", allow_zero=True)

    conditional_bits = conditional_cross_entropy_nats / math.log(2.0)
    raw_lower_bound = source_bits - conditional_bits
    delivered_lower_bound = max(0.0, raw_lower_bound)
    identified_ceiling = min(source_bits, channel_capacity_bits)
    return InformationAudit(
        source_entropy_bits=source_bits,
        channel_capacity_bits=channel_capacity_bits,
        conditional_cross_entropy_bits=conditional_bits,
        raw_variational_lower_bound_bits=raw_lower_bound,
        delivered_information_lower_bound_bits=delivered_lower_bound,
        capacity_utilization_lower_bound=delivered_lower_bound / channel_capacity_bits,
        source_recovery_lower_bound=delivered_lower_bound / source_bits,
        capacity_violation=delivered_lower_bound > identified_ceiling + tolerance_bits,
    )


def _integer_at_least(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def _positive_finite(value: float, name: str, *, allow_zero: bool) -> None:
    lower_ok = value >= 0 if allow_zero else value > 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if not math.isfinite(float(value)) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value!r}.")


__all__ = [
    "InformationAudit",
    "audit_uniform_payload",
    "capacity_efficiency_ceiling",
    "finite_message_capacity_bits",
    "nominal_storage_bits",
    "uniform_source_entropy_bits",
]
