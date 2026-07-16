"""Fail-fast Torch/CUDA launch preflight for the canonical PDT run."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import torch


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-memory-gb", type=float, default=75.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_memory_gb) or args.min_memory_gb <= 0:
        parser.error("--min-memory-gb must be finite and positive")

    print("torch:", torch.__version__)
    print("cuda.is_available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA preflight failed: torch.cuda.is_available() is False.")

    eligible = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)} {memory_gb:.1f} GB")
        if memory_gb >= args.min_memory_gb:
            eligible.append(i)
    if not eligible:
        raise SystemExit(
            f"CUDA preflight failed: no visible device has at least {args.min_memory_gb:.1f} GB."
        )
    print("eligible_device_indices:", eligible)


if __name__ == "__main__":
    main()
