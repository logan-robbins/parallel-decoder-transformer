"""Generate auditable long-form documents with private cross-section constraints.

Each example is one coordinated report with three persistent prose streams and
thirty-two synchronized blocks per stream. Concatenating a stream's targets
produces 1,024 tokens after canonical retokenization; there is no question and
answer surface. Sixteen target blocks depend on fresh, uniquely used private
packets from a sibling section at lags 1, 4, 8, and 16. The other sixteen blocks
are local prose controls inside the same document.

Every private packet contains three independent symbols drawn uniformly from a
64-word alphabet, giving an exact 18-bit payload. The dynamic bus transmits four
indices into 256-entry product codebooks, or exactly 32 bits per write. ``rho=0``
constructs the surface-matched null twin by resolving the same references from
the receiving stream's own private packets.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pdt.datasets.document_contract import (
    DOCUMENT_BLOCKS,
    DOCUMENT_BLOCK_TOKENS,
    DOCUMENT_CODEWORDS,
    DOCUMENT_CONTRACT_VERSION,
    DOCUMENT_DEPENDENCY_LAGS,
    DOCUMENT_DEPENDENCY_SCHEDULE,
    DOCUMENT_HISTORY_BLOCKS,
    DOCUMENT_SECTION_ROLES,
    DOCUMENT_TOKENS_PER_STREAM,
)
from pdt.diagnostics.information import (
    capacity_efficiency_ceiling,
    finite_message_capacity_bits,
    nominal_storage_bits,
    uniform_source_entropy_bits,
)


CODEWORDS = list(DOCUMENT_CODEWORDS)
assert len(CODEWORDS) == len(set(CODEWORDS)) == 64

BITS_PER_CODEWORD = 6
NOTES_DIM = 256
NOTE_DTYPE_BITS = 16
DECODED_NOTE_STORAGE_BITS = nominal_storage_bits(
    elements=NOTES_DIM,
    bits_per_element=NOTE_DTYPE_BITS,
)
DYNAMIC_NOTE_CODEBOOKS = 4
DYNAMIC_CODES_PER_CODEBOOK = 256
TRANSMITTED_NOTE_BITS = int(
    finite_message_capacity_bits(
        codebooks=DYNAMIC_NOTE_CODEBOOKS,
        codes_per_codebook=DYNAMIC_CODES_PER_CODEBOOK,
    )
)
DEFAULT_STREAMS = 3
DEFAULT_BLOCKS = DOCUMENT_BLOCKS
DEFAULT_SLOTS = 3
DEFAULT_DELTA = 1

SUBJECTS = (
    "coastal wetlands",
    "urban tree cover",
    "regional water planning",
    "community archives",
    "public transit renewal",
    "mountain watershed recovery",
    "historic market districts",
    "agricultural soil restoration",
)
PHASES = ("foundation", "comparison", "interpretation", "synthesis")
EVIDENCE_FORMS = (
    "archival records",
    "field observations",
    "institutional reports",
    "community testimony",
    "longitudinal measurements",
    "comparative case studies",
    "implementation records",
    "policy histories",
)
LOCAL_MOVES = (
    "establishing the report's chronology",
    "clarifying the governing definitions",
    "connecting evidence across periods",
    "distinguishing causes from symptoms",
    "testing the scope of earlier claims",
    "tracking institutional consequences",
    "identifying practical tradeoffs",
    "preparing the final synthesis",
)


def bits_per_dependency(slots: int) -> int:
    """Exact conditional payload entropy for one annotated document edge."""

    if slots <= 0:
        raise ValueError(f"slots must be positive, got {slots}.")
    entropy = uniform_source_entropy_bits(alphabet_size=len(CODEWORDS), symbols=slots)
    if not entropy.is_integer():
        raise RuntimeError("The exact-entropy corpus requires an integer bit budget.")
    return int(entropy)


def bits_per_block(slots: int) -> int:
    """Compatibility name for the payload carried by one dependency-bearing write."""

    return bits_per_dependency(slots)


def attainable_eta(slots: int, note_bits: int = TRANSMITTED_NOTE_BITS) -> float:
    if note_bits <= 0:
        raise ValueError(f"note_bits must be positive, got {note_bits}.")
    return capacity_efficiency_ceiling(
        source_bits=bits_per_dependency(slots),
        channel_bits=note_bits,
    )


def _packet(rng: random.Random, slots: int) -> list[str]:
    return [rng.choice(CODEWORDS) for _ in range(slots)]


def _render_payload(words: list[str]) -> str:
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"


def _observation(
    *,
    words: list[str],
    subject: str,
    block: int,
    stream: int,
) -> str:
    phase = PHASES[min(block // 8, len(PHASES) - 1)]
    evidence = EVIDENCE_FORMS[(block + 2 * stream) % len(EVIDENCE_FORMS)]
    return (
        "private document packet: marker="
        + "|".join(words)
        + f"; subject={subject}; phase={phase}; evidence={evidence}"
    )


def _local_target(*, role: str, subject: str, block: int, stream: int) -> str:
    evidence = EVIDENCE_FORMS[(block + 2 * stream) % len(EVIDENCE_FORMS)]
    move = LOCAL_MOVES[(block + stream) % len(LOCAL_MOVES)]
    return (
        f"The {role} section develops {subject} through {evidence}, {move}, while sustaining "
        "a continuous argument for later synthesis."
    )


def _dependent_target(
    *,
    role: str,
    source_role: str,
    subject: str,
    payload: str,
    block: int,
    stream: int,
) -> str:
    evidence = EVIDENCE_FORMS[(block + 2 * stream) % len(EVIDENCE_FORMS)]
    return (
        f"The {role} section uses {payload} from {source_role} to align {subject} with "
        f"{evidence}."
    )


def _example(
    idx: int,
    rng: random.Random,
    *,
    streams: int,
    blocks: int,
    slots: int,
    delta: int,
    rho: float,
    split: str,
    example_seed: int,
) -> dict[str, Any]:
    del delta
    subject = rng.choice(SUBJECTS)
    packets: list[list[list[str]]] = [[] for _ in range(streams)]
    for _block in range(blocks):
        used: set[tuple[str, ...]] = set()
        for stream in range(streams):
            packet = _packet(rng, slots)
            while tuple(packet) in used:
                packet = _packet(rng, slots)
            used.add(tuple(packet))
            packets[stream].append(packet)
    stream_inputs: list[dict[str, Any]] = []

    for receiver in range(streams):
        observations = [
            {
                "block_index": block,
                "text": _observation(
                    words=packets[receiver][block],
                    subject=subject,
                    block=block,
                    stream=receiver,
                ),
            }
            for block in range(blocks)
        ]
        target_blocks: list[str] = []
        dependency_spans: list[dict[str, Any]] = []
        for block in range(blocks):
            scheduled = DOCUMENT_DEPENDENCY_SCHEDULE.get(block)
            if scheduled is None:
                target = _local_target(
                    role=DOCUMENT_SECTION_ROLES[receiver],
                    subject=subject,
                    block=block,
                    stream=receiver,
                )
            else:
                source_block, required_lag = scheduled
                source = (receiver + 1) % streams if rho == 1.0 else receiver
                payload = _render_payload(packets[source][source_block])
                target = _dependent_target(
                    role=DOCUMENT_SECTION_ROLES[receiver],
                    source_role=DOCUMENT_SECTION_ROLES[source],
                    subject=subject,
                    payload=payload,
                    block=block,
                    stream=receiver,
                )
                dependency_spans.append(
                    {
                        "block_index": block,
                        "token_span_text": payload,
                        "source_stream": f"stream_{source}",
                        "source_block_index": source_block,
                        "lag_blocks": required_lag,
                        "kind": (
                            "cross_section_constraint"
                            if rho == 1.0
                            else "self_section_constraint_null"
                        ),
                        "exact_bits": bits_per_dependency(slots) if rho == 1.0 else 0,
                        "payload_codewords": packets[source][source_block],
                    }
                )
            target_blocks.append(target if block == 0 else "\n" + target)

        stream_inputs.append(
            {
                "stream_id": f"stream_{receiver}",
                "section_role": DOCUMENT_SECTION_ROLES[receiver],
                "block_observations": observations,
                "target_blocks": target_blocks,
                "dependency_spans": dependency_spans,
            }
        )

    dependency_uses = len(DOCUMENT_DEPENDENCY_SCHEDULE)
    return {
        "example_id": f"longdoc_{split.strip().lower()}_{idx:06d}",
        "family": (
            "long_form_cross_section_document"
            if rho == 1.0
            else "long_form_self_section_null"
        ),
        "split": split,
        "k": streams,
        "shared_context": (
            f"Write one coordinated long-form report about {subject}. Each persistent stream "
            "owns one section and must maintain continuous prose, preserve section boundaries, "
            "and integrate delayed cross-section constraints without question-answer formatting."
        ),
        "visibility_lag_blocks": DEFAULT_DELTA,
        "stream_inputs": stream_inputs,
        "document_contract": {
            "version": DOCUMENT_CONTRACT_VERSION,
            "form": "continuous_expository_prose",
            "question_answering": False,
            "blocks_per_stream": DOCUMENT_BLOCKS,
            "tokens_per_block": DOCUMENT_BLOCK_TOKENS,
            "tokens_per_stream": DOCUMENT_TOKENS_PER_STREAM,
            "history_blocks": DOCUMENT_HISTORY_BLOCKS,
            "dependency_lags": list(DOCUMENT_DEPENDENCY_LAGS),
            "dependency_uses_per_stream": dependency_uses,
            "local_control_blocks_per_stream": DOCUMENT_BLOCKS - dependency_uses,
            "source_privacy": "one_private_document_packet_per_stream_and_block",
        },
        "eval": {
            "permutation_invariant": False,
            "dependency_span_metric": "paired_gate_zero_ce_delta",
            "nondependency_span_metric": "paired_gate_zero_ce_delta",
        },
        "entropy_accounting": {
            "codebook_size": len(CODEWORDS),
            "bits_per_codeword": BITS_PER_CODEWORD,
            "slots": slots,
            "exact_bits_per_dependency": bits_per_dependency(slots) if rho == 1.0 else 0,
            "dependency_uses_per_stream": dependency_uses,
            "total_exact_bits_per_stream": (
                dependency_uses * bits_per_dependency(slots) if rho == 1.0 else 0
            ),
            "notes_dim": NOTES_DIM,
            "note_dtype_bits": NOTE_DTYPE_BITS,
            "decoded_note_storage_bits": DECODED_NOTE_STORAGE_BITS,
            "dynamic_note_codebooks": DYNAMIC_NOTE_CODEBOOKS,
            "codes_per_codebook": DYNAMIC_CODES_PER_CODEBOOK,
            "transmitted_note_bits": TRANSMITTED_NOTE_BITS,
            "note_representation": "product_vq_indices",
            "attainable_eta_ceiling": attainable_eta(slots) if rho == 1.0 else 0.0,
            "rho": rho,
        },
        "generator": {"type": "long_form_exact_entropy_document", "seed": example_seed},
    }


def validate_generation_args(
    *,
    num_examples: int,
    streams: int,
    blocks: int,
    slots: int,
    delta: int,
    rho: float,
    split: str,
) -> None:
    if num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}.")
    if streams != DEFAULT_STREAMS:
        raise ValueError(f"long-form documents require exactly {DEFAULT_STREAMS} streams.")
    if blocks != DOCUMENT_BLOCKS:
        raise ValueError(f"long-form documents require exactly {DOCUMENT_BLOCKS} blocks.")
    if slots != DEFAULT_SLOTS:
        raise ValueError(
            f"long-form private packets require exactly {DEFAULT_SLOTS} codewords."
        )
    if delta != DEFAULT_DELTA:
        raise ValueError(f"visibility lag is locked to delta={DEFAULT_DELTA}; got {delta}.")
    if rho not in (0.0, 1.0):
        raise ValueError("rho must be exactly 0.0 (null twin) or 1.0 (dependency).")
    if not split.strip():
        raise ValueError("split must be a non-empty string.")


def generate_examples(
    *,
    num_examples: int,
    streams: int = DEFAULT_STREAMS,
    blocks: int = DEFAULT_BLOCKS,
    slots: int = DEFAULT_SLOTS,
    delta: int = DEFAULT_DELTA,
    rho: float = 1.0,
    split: str = "train",
    seed: int = 123,
) -> Iterator[dict[str, Any]]:
    validate_generation_args(
        num_examples=num_examples,
        streams=streams,
        blocks=blocks,
        slots=slots,
        delta=delta,
        rho=rho,
        split=split,
    )
    master_rng = random.Random(seed)
    for idx in range(num_examples):
        example_seed = master_rng.getrandbits(64)
        yield _example(
            idx,
            random.Random(example_seed),
            streams=streams,
            blocks=blocks,
            slots=slots,
            delta=delta,
            rho=rho,
            split=split,
            example_seed=example_seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=20000)
    parser.add_argument("--streams", type=int, default=DEFAULT_STREAMS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--delta", type=int, default=DEFAULT_DELTA)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    validate_generation_args(
        num_examples=args.num_examples,
        streams=args.streams,
        blocks=args.blocks,
        slots=args.slots,
        delta=args.delta,
        rho=args.rho,
        split=args.split,
    )
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in generate_examples(
            num_examples=args.num_examples,
            streams=args.streams,
            blocks=args.blocks,
            slots=args.slots,
            delta=args.delta,
            rho=args.rho,
            split=args.split,
            seed=args.seed,
        ):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        f"wrote {args.num_examples} long-form examples with {DOCUMENT_TOKENS_PER_STREAM} "
        f"target tokens per stream -> {out}"
    )


if __name__ == "__main__":
    main()
