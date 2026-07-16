"""Print the lower-bound hardware case for packed K-stream decoding."""

from __future__ import annotations

import argparse
import json

from pdt.diagnostics.hardware import (
    Channel,
    DecodeRooflineEstimate,
    H100_SXM_BF16_DENSE,
    QWEN3_4B_PDT,
    estimate_decode_round,
    packed_round_speedup,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=3)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=(1024, 4096, 16384, 65536),
        help="Per-stream cached context lengths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    estimates: list[DecodeRooflineEstimate] = []
    paths: tuple[tuple[Channel, bool], ...] = (
        ("blind", True),
        ("pdt", False),
        ("pdt", True),
        ("full_kv", True),
    )
    for context in args.contexts:
        for channel, packed in paths:
            estimate = estimate_decode_round(
                accelerator=H100_SXM_BF16_DENSE,
                topology=QWEN3_4B_PDT,
                streams=args.streams,
                context_tokens_per_stream=context,
                channel=channel,
                packed=packed,
            )
            estimates.append(estimate)

    if args.json:
        print(
            json.dumps(
                {
                    "accelerator": H100_SXM_BF16_DENSE.name,
                    "ridge_flops_per_byte": H100_SXM_BF16_DENSE.ridge_flops_per_byte,
                    "rows": [
                        {
                            **estimate.to_dict(),
                            "roofline_floor_ms": estimate.roofline_floor_seconds * 1000,
                            "memory_bound": (
                                estimate.memory_floor_seconds >= estimate.compute_floor_seconds
                            ),
                        }
                        for estimate in estimates
                    ],
                },
                indent=2,
            )
        )
        return

    print(f"accelerator : {H100_SXM_BF16_DENSE.name}")
    print(f"ridge point: {H100_SXM_BF16_DENSE.ridge_flops_per_byte:.1f} FLOP/byte")
    print(
        f"{'context':>8}  {'path':>13}  {'AI':>8}  {'weights GiB':>11}  "
        f"{'KV GiB':>8}  {'floor ms':>9}  {'bound':>7}"
    )
    for estimate in estimates:
        path = f"{estimate.channel}-{'packed' if estimate.packed else 'separate'}"
        bound = (
            "memory"
            if estimate.memory_floor_seconds >= estimate.compute_floor_seconds
            else "compute"
        )
        print(
            f"{estimate.context_tokens_per_stream:8d}  {path:>13}  "
            f"{estimate.arithmetic_intensity:8.2f}  "
            f"{estimate.weight_bytes / 2**30:11.3f}  "
            f"{estimate.kv_read_bytes / 2**30:8.3f}  "
            f"{estimate.roofline_floor_seconds * 1000:9.3f}  {bound:>7}"
        )
    print()
    for context in args.contexts:
        speedup = packed_round_speedup(
            accelerator=H100_SXM_BF16_DENSE,
            topology=QWEN3_4B_PDT,
            streams=args.streams,
            context_tokens_per_stream=context,
        )
        print(
            f"context={context:6d}: packed PDT roofline advantage over the current "
            f"K-call sequential baseline = {speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
