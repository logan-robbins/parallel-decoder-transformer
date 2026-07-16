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

from pdt.config import TRUNK_PROFILES, apply_trunk_profile, load_config
from pdt.model import PDTModel
from pdt.training.trainer import PDTTrainer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=None,
        help="Select a pinned dense-Qwen3 scale while preserving one sidecar implementation.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Strictly resume model, optimizer, scheduler, and step state from this checkpoint.",
    )
    parser.add_argument(
        "--coordination-source",
        choices=("bus", "self_only"),
        default=None,
        help="Override the checkpoint-identified scientific condition before model construction.",
    )
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=None,
        help="Override the run/checkpoint directory; required with --coordination-source.",
    )
    parser.add_argument(
        "--optimizer-probe",
        action="store_true",
        help="Run the canonical two-update CUDA gradient/optimizer probe, then exit.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --resume and write causal telemetry for --eval-dataset-path without updates.",
    )
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--eval-dataset-path", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accumulation", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--stage-schedule", type=int, nargs=4, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
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
    for name, path in (
        ("--dataset-path", args.dataset_path),
        ("--eval-dataset-path", args.eval_dataset_path),
    ):
        if path is not None and not path.is_file():
            parser.error(f"{name} must name an existing file: {path}")
    if args.coordination_source is not None and args.telemetry_dir is None:
        parser.error("--telemetry-dir is required with --coordination-source to isolate conditions")
    if args.optimizer_probe and args.resume is not None:
        parser.error("--optimizer-probe must start from fresh step-0 weights and cannot resume")
    if args.optimizer_probe and args.telemetry_dir is None:
        parser.error("--telemetry-dir is required with --optimizer-probe")
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if args.eval_only and args.telemetry_dir is None:
        parser.error("--eval-only requires --telemetry-dir to isolate its telemetry")
    if args.eval_only and args.optimizer_probe:
        parser.error("--eval-only and --optimizer-probe are mutually exclusive")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = load_config(args.config)
    if args.trunk_profile is not None:
        apply_trunk_profile(config, args.trunk_profile)
    if args.coordination_source is not None:
        config.instrumentation.coordination_source = args.coordination_source
    if args.telemetry_dir is not None:
        config.training.telemetry_dir = str(args.telemetry_dir)
    if args.dataset_path is not None:
        config.training.dataset_path = str(args.dataset_path)
    if args.eval_dataset_path is not None:
        config.training.eval_dataset_path = str(args.eval_dataset_path)
    for argument, field in (
        (args.max_steps, "max_steps"),
        (args.grad_accumulation, "grad_accumulation"),
        (args.save_every, "save_every"),
        (args.eval_interval, "eval_interval"),
        (args.log_interval, "log_interval"),
    ):
        if argument is not None:
            setattr(config.training, field, argument)
    if args.warmup_steps is not None:
        config.training.optimizer.warmup_steps = args.warmup_steps
    if args.stage_schedule is not None:
        config.training.curriculum.stage_schedule = tuple(args.stage_schedule)
    if args.optimizer_probe:
        config.training.grad_accumulation = 1
    config.validate()
    model = PDTModel(config)
    trainer = PDTTrainer(model, config)
    if args.resume is not None:
        trainer.resume_from_checkpoint(args.resume)
    if args.optimizer_probe:
        trainer.optimizer_probe()
    elif args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
