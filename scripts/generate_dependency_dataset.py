"""Generate cross-stream dependency data with an EXACT, dialable entropy budget.

Why this file was rewritten from scratch
----------------------------------------
The previous generator's entire cross-stream dependency was the identity of
``top = argmax(private values)``. Exact accounting:

    H(top | own observation) = 0.864 nats

The project's pre-registered usefulness gate is eta >= 0.05.  The implemented
bus transmits dense 256-dimensional BF16 snapshots, or 4096 bits per write.
This corpus exposes the resulting ceiling honestly; it does not pretend that a
product-VQ transport already exists.

The design principle
--------------------
A coordination corpus is specified by a NATS BUDGET, not by a topic. Four
properties are load-bearing, and the old generator satisfied only two:

1. The dependency must live in per-stream PRIVATE state. If the shared prompt
   contains it, the trunk resolves it alone, D(0) collapses, and no bus is
   needed. (Old generator: OK.)
2. It must be FRESH EVERY BLOCK. A fact revealed once and reused forever drives
   eta -> 0 as block count grows. (Old generator: FAILED -- one fact, forever.)
3. It must be PROGRAMMATICALLY KNOWN, so D(0) is fixed by construction rather
   than estimated. This is what makes eta *identified* instead of guessed.
   (Old generator: FAILED -- no entropy accounting existed.)
4. Spans must be token-alignable. (Old generator: OK.)

Construction
------------
Each stream k holds a private register that REFRESHES every block:

    r_k[m] = (w_1, ..., w_S),  w_i ~ Uniform(CODEWORDS),  |CODEWORDS| = 64

At block m, stream k must report its right neighbour's register from block
m - delta (the reveal delay). Because codewords are uniform and independent of
everything stream k can observe, the information stream k needs per block is
EXACTLY:

    H = S * log2(64) = 6S bits,   independent of own register.

That is the source-entropy dial. S=3 -> 18 payload bits/block.  Through the
implemented 4096-bit dense note this gives eta <= 18/4096 ~= 0.00439, below the
0.05 gate.  Reaching the gate without compressing the bus would require at
least 35 slots (210 payload bits) per block.

The rho=0 null twin
-------------------
``--rho 0`` makes each stream report its OWN register from block m - delta.
Surface form, length, token distribution, and task framing are IDENTICAL; only
the referent changes. Cross-stream information is then exactly 0 bits, so a
correct bus MUST deliver eta = 0. Any measured eta > 0 on the null is leakage,
and localizes it immediately. This is a negative control that costs one branch.

Usage
-----
    uv run scripts/generate_dependency_dataset.py --output data/.../train.jsonl \
        --num-examples 20000 --slots 3 --blocks 8 --streams 3

    uv run scripts/generate_dependency_dataset.py --output data/.../null.jsonl \
        --num-examples 2000 --slots 3 --blocks 8 --streams 3 --rho 0
"""

from __future__ import annotations

import argparse
import json
import random
from math import log
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List

# 64 distinct, short, lowercase, non-overlapping codewords -> exactly 6 bits each.
# Chosen to be common English words so the trunk tokenizes them compactly and
# assigns them no special prior; disjoint stems so no codeword is a prefix of
# another (which would leak partial information through tokenization).
CODEWORDS: List[str] = [
    "amber",
    "anchor",
    "basalt",
    "beacon",
    "bramble",
    "cactus",
    "canyon",
    "cedar",
    "cinder",
    "cobalt",
    "copper",
    "coral",
    "dahlia",
    "delta",
    "dune",
    "ember",
    "fathom",
    "fennel",
    "flint",
    "forge",
    "gable",
    "garnet",
    "geyser",
    "glacier",
    "granite",
    "harbor",
    "hazel",
    "indigo",
    "ivory",
    "jasper",
    "juniper",
    "kelp",
    "lantern",
    "larch",
    "lichen",
    "marble",
    "meadow",
    "mesa",
    "nectar",
    "nimbus",
    "onyx",
    "opal",
    "orchid",
    "pewter",
    "pumice",
    "quarry",
    "quartz",
    "ravine",
    "saffron",
    "sable",
    "shale",
    "sienna",
    "slate",
    "spruce",
    "summit",
    "talon",
    "thicket",
    "topaz",
    "tundra",
    "umber",
    "verbena",
    "willow",
    "zenith",
    "zephyr",
]
assert len(CODEWORDS) == 64, "codeword count must be a power of two for exact bit accounting"
assert len(set(CODEWORDS)) == 64, "codewords must be distinct"

BITS_PER_CODEWORD = 6  # log2(64)
NOTES_DIM = 256
NOTE_DTYPE_BITS = 16  # BF16 transport
TRANSMITTED_NOTE_BITS = NOTES_DIM * NOTE_DTYPE_BITS
DEFAULT_STREAMS = 3
DEFAULT_BLOCKS = 8
DEFAULT_SLOTS = 3
DEFAULT_DELTA = 1


def bits_per_block(slots: int) -> int:
    """Exact cross-stream information a stream needs per block, in bits."""
    if slots <= 0:
        raise ValueError(f"slots must be positive, got {slots}.")
    return slots * BITS_PER_CODEWORD


def attainable_eta(slots: int, note_bits: int = TRANSMITTED_NOTE_BITS) -> float:
    """Ceiling on eta = delivered nats / transmitted nats for this corpus."""
    if note_bits <= 0:
        raise ValueError(f"note_bits must be positive, got {note_bits}.")
    return (bits_per_block(slots) * log(2)) / (note_bits * log(2))


def _register(rng: random.Random, slots: int) -> List[str]:
    return [rng.choice(CODEWORDS) for _ in range(slots)]


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
) -> Dict[str, Any]:
    # registers[k][m] = stream k's private register at block m, refreshed each block.
    registers = [[_register(rng, slots) for _ in range(blocks)] for _ in range(streams)]

    stream_inputs = []
    for k in range(streams):
        # Each private register is revealed only at its synchronization block.
        # Serializing the complete log into the initial prompt would let the
        # first dense note front-load all future registers and invalidate Delta.
        block_observations = [
            {
                "block_index": m,
                "text": "private register: " + " ".join(registers[k][m]),
            }
            for m in range(blocks)
        ]

        target_blocks: List[str] = []
        dependency_spans: List[Dict[str, Any]] = []

        for m in range(blocks):
            if m < delta:
                # Runway: nothing is visible yet. No dependency span here by
                # construction -- these blocks are the nondependency control.
                target_blocks.append(f"stream_{k} block {m}: register not yet visible, holding.")
                continue

            src_block = m - delta
            if rho > 0:
                # rho > 0: report the RIGHT NEIGHBOUR's register. Unobservable
                # from stream k's private state => exactly 6*slots bits needed.
                source = (k + 1) % streams
            else:
                # rho = 0 null: report OWN register. Identical surface form,
                # zero cross-stream information.
                source = k

            payload = " ".join(registers[source][src_block])
            span = payload
            target_blocks.append(f"\nstream_{k} block {m}: relayed register is {span}.")
            dependency_spans.append(
                {
                    "block_index": m,
                    "token_span_text": span,
                    "source_stream": f"stream_{source}",
                    "source_block_index": src_block,
                    "kind": "sibling_register" if rho > 0 else "self_register_null",
                    "exact_bits": bits_per_block(slots) if rho > 0 else 0,
                }
            )

        stream_inputs.append(
            {
                "stream_id": f"stream_{k}",
                "block_observations": block_observations,
                "target_blocks": target_blocks,
                "dependency_spans": dependency_spans,
            }
        )

    referent = "your right neighbour's" if rho > 0 else "your own"
    return {
        "example_id": f"xdep_{split.strip().lower()}_{idx:06d}",
        "family": "cross_stream_register_relay" if rho > 0 else "self_register_null",
        "split": split,
        "k": streams,
        "shared_context": (
            f"Coordinate {streams} streams. Each stream holds a private register that "
            f"refreshes every block. At each block, relay {referent} register as it "
            f"stood {delta} block(s) ago."
        ),
        "visibility_lag_blocks": delta,
        "stream_inputs": stream_inputs,
        "eval": {
            "permutation_invariant": False,
            "dependency_span_metric": "target_model_ce_delta",
            "nondependency_span_metric": "target_model_ce_delta",
        },
        "entropy_accounting": {
            "codebook_size": len(CODEWORDS),
            "bits_per_codeword": BITS_PER_CODEWORD,
            "slots": slots,
            "exact_bits_per_block": bits_per_block(slots) if rho > 0 else 0,
            "notes_dim": NOTES_DIM,
            "note_dtype_bits": NOTE_DTYPE_BITS,
            "transmitted_note_bits": TRANSMITTED_NOTE_BITS,
            "note_representation": "dense_bfloat16",
            "attainable_eta_ceiling": attainable_eta(slots) if rho > 0 else 0.0,
            "rho": rho,
        },
        "generator": {"type": "exact_entropy_budget", "seed": example_seed},
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
    """Fail before opening the output when a corpus contract is impossible."""
    if num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}.")
    if streams < 2:
        raise ValueError(f"streams must be at least 2, got {streams}.")
    if blocks <= 0:
        raise ValueError(f"blocks must be positive, got {blocks}.")
    bits_per_block(slots)
    if delta != DEFAULT_DELTA:
        raise ValueError(f"visibility lag is locked to delta={DEFAULT_DELTA}; got {delta}.")
    if blocks <= delta:
        raise ValueError(
            f"blocks ({blocks}) must exceed delta ({delta}), else every block is runway."
        )
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
) -> Iterator[Dict[str, Any]]:
    """Yield reproducible examples whose recorded seed regenerates that row."""
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
    parser.add_argument("--streams", type=int, default=DEFAULT_STREAMS, help="K")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS, help="blocks per stream")
    parser.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
        help="codewords per register; cross-stream bits/block = 6 * slots",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=DEFAULT_DELTA,
        help="reveal delay in blocks (locked to 1)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="1.0 = real cross-stream dependency; 0.0 = the null twin (self-relay)",
    )
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

    bits = bits_per_block(args.slots)
    ceiling = attainable_eta(args.slots)
    dep_blocks = args.blocks - args.delta

    print(f"codebook           : {len(CODEWORDS)} codewords = {BITS_PER_CODEWORD} bits each")
    print(f"slots per register : {args.slots}")
    print(f"EXACT bits/block   : {bits if args.rho > 0 else 0}")
    print(f"dependency blocks  : {dep_blocks}/{args.blocks} (first {args.delta} are runway)")
    print(f"eta ceiling @{TRANSMITTED_NOTE_BITS}bit: {ceiling if args.rho > 0 else 0.0:.6f}")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for rec in generate_examples(
            num_examples=args.num_examples,
            streams=args.streams,
            blocks=args.blocks,
            slots=args.slots,
            delta=args.delta,
            rho=args.rho,
            split=args.split,
            seed=args.seed,
        ):
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {args.num_examples} examples -> {out}")


if __name__ == "__main__":
    main()
