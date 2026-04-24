"""Training entry point.

Usage:
    uv run scripts/train.py --config configs/pdt_qwen3_4b.yaml

Or via torchrun for DDP:
    uv run torchrun --nproc_per_node=N -m pdt.cli.train --config configs/pdt_qwen3_4b.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pdt.config import load_config
from pdt.model import PDTModel
from pdt.training.trainer import PDTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(args.config)
    model = PDTModel(config)
    trainer = PDTTrainer(model, config)
    trainer.train()


if __name__ == "__main__":
    main()
