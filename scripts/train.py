# ruff: noqa: E402
"""Training entry point for the GPT-OSS powered Parallel Decoder Transformer."""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import json
from pathlib import Path

import logging

from parallel_decoder_transformer.config.loader import (
    parse_args,
    load_config,
    maybe_create_dataset,
    _coerce_model_config,
    _coerce_training_config,
)
from parallel_decoder_transformer.models import ParallelDecoderTransformer
from parallel_decoder_transformer.training.trainer import Trainer
from parallel_decoder_transformer.data.extraction import NOTES_SCHEMA_VERSION
from parallel_decoder_transformer.utils import configure_logging, seed_everything, get_git_metadata
from parallel_decoder_transformer.utils.plan_catalog import resolve_plan_hash_params


def main() -> None:
    args = parse_args()
    logger = configure_logging(logging.INFO, name="parallel decoder transformer.cli.train")
    raw_cfg = load_config(args.config)
    model_cfg = _coerce_model_config(raw_cfg.get("model", {}))
    training_cfg = _coerce_training_config(raw_cfg.get("training", {}))

    seed_everything(training_cfg.seed)
    if training_cfg.seed is not None:
        logger.info("seed_configured | seed=%d", int(training_cfg.seed))

    logger.info(
        "train_start | config=%s | dataset=%s | eval=%s",
        str(args.config),
        training_cfg.dataset_path,
        training_cfg.eval_dataset_path,
    )

    dataset = maybe_create_dataset(training_cfg.dataset_path)
    eval_dataset = maybe_create_dataset(training_cfg.eval_dataset_path)

    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.load_model()
    model.to_trunk_device_and_dtype()
    trainer = Trainer(
        model,
        training_cfg,
        collator_config=model_cfg.collator,
        dataset=dataset,
        eval_dataset=eval_dataset,
    )
    trainer.fit()
    trainer.evaluate()

    telemetry_dir = Path(training_cfg.telemetry_dir or "experiments/gpt_oss")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stages_path = trainer.write_stage_history(telemetry_dir)
    thresholds_path = trainer.write_agreement_threshold(telemetry_dir)
    trainer.write_coverage_threshold(telemetry_dir)

    git_meta = get_git_metadata()
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

    report = trainer.generate_training_report()
    report_path = telemetry_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("train_report_written | %s", str(report_path))

    # Save lightweight adapter checkpoint for evaluation/inference reuse
    try:
        adapter_state = model.adapter_state_dict()
        ckpt_path = telemetry_dir / "adapters.pt"
        import torch

        torch.save(adapter_state, ckpt_path)
        logger.info("adapters_checkpoint_written | %s", str(ckpt_path))
    except Exception as err:  # pragma: no cover - defensive path
        logger.warning("adapters_checkpoint_write_failed | err=%s", str(err))


if __name__ == "__main__":
    main()
