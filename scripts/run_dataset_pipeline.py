#!/usr/bin/env python3
"""End-to-end dataset pipeline driver (plans -> notes -> Arrow splits)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

try:  # pragma: no cover - optional dependency
    from omegaconf import OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    OmegaConf = None  # type: ignore

from parallel_decoder_transformer.datasets import (
    CollateConfig,
    DatasetBuildConfig,
    DatasetCollator,
    KDExportConfig,
    KDExporter,
    NotesGenerationConfig,
    NotesGenerator,
    PlanGenerationConfig,
    PlanGenerator,
    SpeculativeNotesNoiseConfig,
    resolve_notes_llm_config,
)


def _load_config(path: Path | None) -> DatasetBuildConfig:
    if path is None:
        return DatasetBuildConfig()
    if OmegaConf is None:  # pragma: no cover
        raise RuntimeError("OmegaConf is required to load configs. Install it or omit --config.")
    base = OmegaConf.structured(DatasetBuildConfig())
    overrides = OmegaConf.load(path)
    merged = OmegaConf.merge(base, overrides)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _domain_targets(
    total: int, qa: int | None, math: int | None, survey: int | None
) -> Dict[str, int]:
    if any(value is not None for value in (qa, math, survey)):
        return {
            "qa": int(qa or 0),
            "math": int(math or 0),
            "survey": int(survey or 0),
        }
    per_domain = max(1, total // 3)
    remaining = total - 2 * per_domain
    return {"qa": per_domain, "math": per_domain, "survey": max(0, remaining)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to DatasetBuildConfig YAML/JSON")
    parser.add_argument("--total", type=int, help="Total plan count override (defaults to config)")
    parser.add_argument("--qa", type=int, help="QA plan count override")
    parser.add_argument("--math", type=int, help="Math plan count override")
    parser.add_argument("--survey", type=int, help="Survey plan count override")
    parser.add_argument(
        "--reasoning-gym-dataset",
        type=str,
        default="simple_equations",
        help="Reasoning Gym dataset name for math domain (default: %(default)s)",
    )
    parser.add_argument(
        "--preflight-manifest", type=Path, help="Optional preflight_accepted.jsonl path"
    )
    parser.add_argument("--plan-dir", type=Path, default=Path("data/prep/plans"))
    parser.add_argument("--notes-dir", type=Path, default=Path("data/prep/notes"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/pdt_corpus"))
    parser.add_argument(
        "--notes-limit", type=int, help="Optional cap on plans to convert into notes"
    )
    parser.add_argument("--notes-batch-size", type=int, default=24)
    parser.add_argument(
        "--notes-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent notes requests (default: %(default)s)",
    )
    parser.add_argument("--spec-variants", type=int, default=3)
    parser.add_argument(
        "--augment", type=int, default=2, help="Augmented copies per sample during collation"
    )
    parser.add_argument("--max-len", type=int, default=2048, help="Tokenizer truncation length")
    parser.add_argument(
        "--plan-batch-size",
        type=int,
        default=32,
        help="Planner batch size when using async mode (default: %(default)s)",
    )
    parser.add_argument(
        "--plan-concurrency",
        type=int,
        default=24,
        help="Maximum number of concurrent planner requests (default: %(default)s)",
    )
    parser.add_argument(
        "--plan-sync",
        action="store_true",
        help="Force sequential planner calls instead of the default async batches",
    )
    parser.add_argument(
        "--plan-sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between sequential planner calls (default: %(default)s)",
    )
    parser.add_argument(
        "--no-plan-resume",
        action="store_true",
        help="Disable resume behavior (regenerate even if plan JSON already exists)",
    )
    parser.add_argument(
        "--skip-plans", action="store_true", help="Reuse existing plans in plan-dir"
    )
    parser.add_argument("--skip-notes", action="store_true", help="Reuse notes in notes-dir")
    parser.add_argument("--skip-collate", action="store_true", help="Skip Arrow export")
    parser.add_argument("--skip-kd-export", action="store_true", help="Skip KD JSONL export")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="KD JSONL output directory (default: data/processed/<dataset-name>)",
    )
    parser.add_argument(
        "--notes-dim",
        type=int,
        default=2048,
        help="Embedding dimension for notes vectors (default: %(default)s)",
    )
    parser.add_argument(
        "--kd-splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to export to KD JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--no-notes-resume",
        action="store_true",
        help="Disable resume behavior for notes generation (regenerate existing notes files)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve arguments and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg = _load_config(args.config)
    total = args.total or cfg.total_target_examples
    targets = _domain_targets(total, args.qa, args.math, args.survey)

    summary: dict[str, object] = {
        "plan_dir": str(args.plan_dir),
        "notes_dir": str(args.notes_dir),
        "dataset_dir": str(args.dataset_dir),
        "targets": targets,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    # ------------------------------------------------------------------ #
    # Plan generation                                                    #
    # ------------------------------------------------------------------ #

    if not args.skip_plans:
        plan_cfg = PlanGenerationConfig(
            total_per_domain=targets,
            output_root=args.plan_dir,
            batch_size=max(1, args.plan_batch_size),
            concurrency=max(1, args.plan_concurrency),
            use_async_client=not args.plan_sync,
            sequential_sleep_seconds=max(0.0, args.plan_sleep),
            resume_existing=not args.no_plan_resume,
            preflight_manifest=args.preflight_manifest,
            reasoning_gym_dataset=args.reasoning_gym_dataset,
        )
        plan_generator = PlanGenerator(
            llm_config=cfg.llm,
            generation_cfg=cfg.generation_plan,
            plan_cfg=plan_cfg,
        )
        tasks = plan_generator.build_tasks()
        logging.info("Generating %d plans -> %s", len(tasks), args.plan_dir)
        logging.info(
            "Planner running in %s mode (batch size=%d)",
            "sequential" if args.plan_sync else "async",
            max(1, args.plan_batch_size),
        )
        plan_generator.generate(tasks)
    else:
        logging.info("Skipping plan generation (reusing %s)", args.plan_dir)

    # ------------------------------------------------------------------ #
    # Notes generation                                                   #
    # ------------------------------------------------------------------ #

    notes_generator = NotesGenerator(
        llm_config=resolve_notes_llm_config(cfg),
        true_cfg=cfg.generation_true_notes,
        speculative_cfg=cfg.generation_speculative_notes,
        noise_cfg=(
            cfg.speculative_noise
            if isinstance(cfg.speculative_noise, SpeculativeNotesNoiseConfig)
            else SpeculativeNotesNoiseConfig()
        ),
        notes_cfg=NotesGenerationConfig(
            batch_size=args.notes_batch_size,
            concurrency=args.notes_concurrency,
            output_root=args.notes_dir,
            variants_per_sample=args.spec_variants,
            resume_existing=not args.no_notes_resume,
        ),
    )
    plan_docs = notes_generator.load_plan_documents(args.plan_dir, limit=args.notes_limit)
    summary["plans_detected"] = len(plan_docs)
    if not args.skip_notes:
        logging.info("Generating notes for %d plans -> %s", len(plan_docs), args.notes_dir)
        notes_generator.generate(plan_docs)
    else:
        logging.info("Skipping notes generation (reusing %s)", args.notes_dir)

    # ------------------------------------------------------------------ #
    # Collation                                                         #
    # ------------------------------------------------------------------ #

    if not args.skip_collate:
        logging.info("Collating notes from %s -> %s", args.notes_dir, args.dataset_dir)
        collator = DatasetCollator(
            CollateConfig(
                notes_dir=args.notes_dir,
                output_dir=args.dataset_dir,
                tokenizer_path=Path("gpt-oss-20b/tokenizer"),
                augment_per_sample=args.augment,
                max_seq_len=args.max_len,
            )
        )
        export_paths = collator.collate()
        summary["export_paths"] = {split: str(path) for split, path in export_paths.items()}
    else:
        logging.info("Skipping collation (export dir %s)", args.dataset_dir)

    # ------------------------------------------------------------------ #
    # KD JSONL export                                                    #
    # ------------------------------------------------------------------ #

    if not args.skip_kd_export:
        dataset_name = (
            args.dataset_dir.name
            if isinstance(args.dataset_dir, Path)
            else Path(args.dataset_dir).name
        )
        processed_dir = args.processed_dir or (Path("data/processed") / dataset_name)
        logging.info("Exporting KD JSONL from %s -> %s", args.dataset_dir, processed_dir)
        kd_config = KDExportConfig(
            dataset_dir=args.dataset_dir,
            output_dir=processed_dir,
            splits=args.kd_splits,
            notes_dim=args.notes_dim,
        )
        exporter = KDExporter(kd_config)
        output_path = exporter.export()
        summary["kd_jsonl_path"] = str(output_path)
        logging.info("KD JSONL exported to %s", output_path)
    else:
        logging.info("Skipping KD export")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
