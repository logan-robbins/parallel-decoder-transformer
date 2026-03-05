#!/usr/bin/env python
"""Migrate scalar gate checkpoints to per-head format.

Usage::

    uv run python scripts/migrate_gate_checkpoint.py \\
        --input adapters_step_50000.pt \\
        --output adapters_step_50000_per_head.pt \\
        --num-heads 32 \\
        --gate-mode per_head_dynamic

The script scans the checkpoint for gate parameters with shape ``(1,)`` and
broadcasts them to ``(num_heads,)``.  New parameters required by the
``per_head_dynamic`` mode (``gate_pool_norm``, ``gate_dyn_proj``) are *not*
injected -- they will be initialized from scratch when the model is
constructed with the new ``gate_mode``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from parallel_decoder_transformer.inference.gate_utils import (
    migrate_scalar_gate_checkpoint,
)

LOGGER = logging.getLogger("migrate_gate_checkpoint")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Migrate scalar gate checkpoints to per-head format."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the existing adapter checkpoint (.pt)."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path for the migrated checkpoint (.pt)."
    )
    parser.add_argument(
        "--num-heads", type=int, required=True,
        help="Number of attention heads for the target gate shape."
    )
    parser.add_argument(
        "--gate-mode", choices=["per_head", "per_head_dynamic"], default="per_head_dynamic",
        help="Target gate mode (default: per_head_dynamic)."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input.exists():
        LOGGER.error("Input checkpoint not found: %s", args.input)
        sys.exit(1)

    state_dict: dict = torch.load(args.input, map_location="cpu", weights_only=True)

    migrated_count = 0
    keys = list(state_dict.keys())
    for key in keys:
        if key.endswith(".gate") and state_dict[key].shape == (1,):
            LOGGER.info("Migrating %s from shape (1,) to (%d,)", key, args.num_heads)
            migrate_scalar_gate_checkpoint(state_dict, key, num_heads=args.num_heads)
            migrated_count += 1

    if migrated_count == 0:
        LOGGER.warning("No scalar gate parameters found in checkpoint.")
    else:
        LOGGER.info("Migrated %d gate parameter(s).", migrated_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    LOGGER.info("Saved migrated checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
