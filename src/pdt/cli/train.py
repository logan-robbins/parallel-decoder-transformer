"""Training entry point.

Usage:
    uv run scripts/train.py --config configs/pdt_qwen3_4b.yaml

Or via torchrun for DDP:
    uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pdt.config import load_config
from pdt.model import PDTModel
from pdt.training.trainer import PDTTrainer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Strictly resume model, optimizer, scheduler, and step state from this checkpoint.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        parser.error(f"--config must name an existing file: {args.config}")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"--resume must name an existing checkpoint file: {args.resume}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(args.config)
    model = PDTModel(config)
    trainer = PDTTrainer(model, config)
    if args.resume is not None:
        trainer.resume_from_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
