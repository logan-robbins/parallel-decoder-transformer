# ruff: noqa: E402
"""WandB-enabled training entry point for remote 8xH100 runs.

Wraps the standard training loop with WandB integration for real-time remote monitoring.

Usage:
    uv run wandb login  # one-time setup
    tmux new -s training
    uv run python scripts/train_wandb.py --config configs/gpt_oss_transfer.yaml
"""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import json
import logging
from pathlib import Path
from typing import Dict

import torch
import wandb

from parallel_decoder_transformer.models import ParallelDecoderTransformer
from parallel_decoder_transformer.training.trainer import Trainer
from parallel_decoder_transformer.data.extraction import NOTES_SCHEMA_VERSION
from parallel_decoder_transformer.utils import configure_logging, seed_everything, get_git_metadata
from parallel_decoder_transformer.utils.plan_catalog import resolve_plan_hash_params

# Import the existing train.py components to avoid duplication
from train import (
    parse_args,
    load_config,
    maybe_create_dataset,
    _coerce_model_config,
    _coerce_training_config,
)


class WandBTrainer(Trainer):
    """Trainer subclass that logs all metrics to WandB in addition to local stdout."""

    def _log_metrics(self, prefix: str, metrics: Dict[str, float]) -> None:
        # Only log on rank 0
        if self.rank != 0:
            return

        # First, call the parent implementation for stdout logging and local persistence
        super()._log_metrics(prefix, metrics)

        # Then, push to WandB with namespaced keys
        wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb_metrics["global_step"] = self.state.global_step
        wandb_metrics["epoch"] = self.state.epoch

        # Log the current stage for monitoring stage transitions
        wandb_metrics["stage"] = self.state.stage_index

        wandb.log(wandb_metrics, step=self.state.global_step)


def main() -> None:
    """Main training loop with WandB integration."""
    # DDP Initialization
    local_rank = int(_os.environ.get("LOCAL_RANK", -1))
    world_size = int(_os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    args = parse_args()
    # Configure logging only on rank 0 to avoid spam, or use different levels
    log_level = logging.INFO if local_rank <= 0 else logging.WARNING
    logger = configure_logging(log_level, name="parallel decoder transformer.cli.train_wandb")

    # Load config files
    raw_cfg = load_config(args.config)
    model_cfg = _coerce_model_config(raw_cfg.get("model", {}))
    training_cfg = _coerce_training_config(raw_cfg.get("training", {}))

    # Initialize WandB with metadata (Rank 0 only)
    git_meta = get_git_metadata()
    if local_rank <= 0:
        wandb.init(
            project="parallel-decoder-transformer",
            name=f"gpt-oss-8xH100-{training_cfg.max_steps}steps",
            config={
                "config_path": str(args.config.resolve()),
                "dataset": training_cfg.dataset_path,
                "eval_dataset": training_cfg.eval_dataset_path,
                "batch_size": training_cfg.batch_size,
                "learning_rate": training_cfg.learning_rate,
                "max_steps": training_cfg.max_steps,
                "notes_dim": model_cfg.notes_dim,
                "curriculum_B": training_cfg.curriculum.B,
                "curriculum_L": training_cfg.curriculum.L,
                "curriculum_delta": training_cfg.curriculum.delta,
                "kd_temperature_planner": training_cfg.kd_temperature_planner,
                "kd_temperature_lm": training_cfg.kd_temperature_lm,
                "git_sha": git_meta.sha,
                "git_dirty": git_meta.dirty,
                "notes_schema_version": NOTES_SCHEMA_VERSION,
                "world_size": world_size,
            },
            tags=["8xH100", "lambda-labs", "gpt-oss-transfer", "ddp"],
        )

    # Augment telemetry_dir with WandB run name for isolation
    if training_cfg.telemetry_dir and wandb.run is not None:
        training_cfg.telemetry_dir = f"{training_cfg.telemetry_dir}/{wandb.run.name}"
        logger.info("telemetry_dir_augmented | dir=%s", training_cfg.telemetry_dir)

    # Set seed
    seed_everything(training_cfg.seed)
    if training_cfg.seed is not None:
        logger.info("seed_configured | seed=%d", int(training_cfg.seed))

    logger.info(
        "train_start_wandb | config=%s | dataset=%s | eval=%s | run=%s",
        str(args.config),
        training_cfg.dataset_path,
        training_cfg.eval_dataset_path,
        wandb.run.name if wandb.run else "unknown",
    )

    # Load datasets
    dataset = maybe_create_dataset(training_cfg.dataset_path)
    eval_dataset = maybe_create_dataset(training_cfg.eval_dataset_path)

    # Initialize model
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.load_model()
    model.to_trunk_device_and_dtype()

    # Use WandBTrainer instead of standard Trainer
    trainer = WandBTrainer(
        model,
        training_cfg,
        collator_config=model_cfg.collator,
        dataset=dataset,
        eval_dataset=eval_dataset,
    )

    # Train and evaluate
    trainer.fit()
    trainer.evaluate()

    # Write local telemetry artifacts (Rank 0 only)
    if local_rank <= 0:
        telemetry_dir = Path(training_cfg.telemetry_dir or "experiments/gpt_oss")
        telemetry_dir.mkdir(parents=True, exist_ok=True)

        stages_path = trainer.write_stage_history(telemetry_dir)
    thresholds_path = trainer.write_agreement_threshold(telemetry_dir)

    # Build training manifest
    manifest = {
        "config_path": str(args.config.resolve()),
        "dataset": training_cfg.dataset_path,
        "eval_dataset": training_cfg.eval_dataset_path,
        "global_step": trainer.state.global_step,
        "best_eval_loss": trainer.state.best_eval_loss,
        "agreement_threshold": float(trainer.config.agreement_threshold),
        "coverage_threshold": float(training_cfg.coverage_threshold),
        "notes_schema_version": NOTES_SCHEMA_VERSION,
        "git_sha": git_meta.sha,
        "git_dirty": git_meta.dirty,
        "wandb_run_name": wandb.run.name if wandb.run else None,
        "wandb_run_url": wandb.run.url if wandb.run else None,
    }

    if stages_path is not None:
        if stages_path.parent == telemetry_dir:
            manifest["stages_file"] = stages_path.name
        else:
            manifest["stages_file"] = str(stages_path)

    if thresholds_path is not None:
        if thresholds_path.parent == telemetry_dir:
            manifest["agreement_thresholds_file"] = thresholds_path.name
        else:
            manifest["agreement_thresholds_file"] = str(thresholds_path)

    plan_params = resolve_plan_hash_params(model.config)
    manifest.update(plan_params.as_dict())

    manifest_path = telemetry_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("train_complete | manifest=%s", str(manifest_path))

    # Generate training report
    report = trainer.generate_training_report()
    report_path = telemetry_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("train_report_written | %s", str(report_path))

    # Save adapter checkpoint (Rank 0 only)
    if local_rank <= 0:
        try:
            # Unwrap DDP if needed
            model_to_save = model.module if hasattr(model, "module") else model
            adapter_state = model_to_save.adapter_state_dict()
            ckpt_path = telemetry_dir / "adapters.pt"
            torch.save(adapter_state, ckpt_path)
            logger.info("adapters_checkpoint_written | %s", str(ckpt_path))

            # Log checkpoint as WandB artifact for easy retrieval
            artifact = wandb.Artifact(
                name=f"adapters-{wandb.run.name}" if wandb.run else "adapters",
                type="model",
                description="PDT adapter weights (SNC, heads, stream adapters)",
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)
            logger.info("wandb_artifact_logged | checkpoint=%s", str(ckpt_path))

        except Exception as err:  # pragma: no cover - defensive path
            logger.warning("adapters_checkpoint_write_failed | err=%s", str(err))

    # Finalize WandB run
    if local_rank <= 0:
        wandb.finish()
        logger.info("wandb_run_finalized | url=%s", manifest.get("wandb_run_url", "unknown"))

    if is_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
