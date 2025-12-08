#!/usr/bin/env python3
"""Run deterministic preflight checks for plan-generation inputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists():  # pragma: no cover - bootstrap import path
    sys.path.insert(0, str(_SRC))

from parallel_decoder_transformer.datasets.plan_generation import PlanGenerationConfig
from parallel_decoder_transformer.datasets.preflight import (
    PreflightRunner,
    PreflightSettings,
    ReasoningGymPreflightConfig,
    SquadPreflightConfig,
    WikipediaClassifierConfig,
    WikipediaPreflightConfig,
)


def _domain_targets(args: argparse.Namespace, total: int) -> Dict[str, int]:
    if args.qa is not None or args.math is not None or args.survey is not None:
        return {
            "qa": int(args.qa or 0),
            "math": int(args.math or 0),
            "survey": int(args.survey or 0),
        }
    per_domain = max(1, total // 3)
    return {"qa": per_domain, "math": per_domain, "survey": max(0, total - 2 * per_domain)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--total", type=int, default=9, help="Total candidate count (balanced per domain)"
    )
    parser.add_argument("--qa", type=int, help="Override QA candidate count")
    parser.add_argument("--math", type=int, help="Override Reasoning Gym candidate count")
    parser.add_argument("--survey", type=int, help="Override Wikipedia candidate count")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prep/preflight"),
        help="Directory for preflight manifests",
    )

    # Dataset/source overrides
    parser.add_argument("--squad-split", default="train", help="SQuAD split to scan")
    parser.add_argument("--reasoning-gym-dataset", default="simple_equations")
    parser.add_argument("--reasoning-gym-offset", type=int, default=0)
    parser.add_argument(
        "--wiki-manifest",
        type=Path,
        default=Path("data/manifests/wikipedia_20231101_en_train.json"),
        help="Wikipedia manifest path",
    )
    parser.add_argument(
        "--wiki-shard-limit", type=int, help="Optional cap on Wikipedia shards to scan"
    )
    parser.add_argument(
        "--wiki-offset",
        type=int,
        default=0,
        help="Skip first N Wikipedia articles (for resuming from previous runs)",
    )

    # Threshold customisation
    parser.add_argument("--squad-max-question-tokens", type=int, default=512)
    parser.add_argument("--squad-max-context-tokens", type=int, default=3_000)
    parser.add_argument("--squad-max-total-tokens", type=int, default=8_192)
    parser.add_argument("--squad-min-context-tokens", type=int, default=20)

    parser.add_argument("--wiki-max-article-tokens", type=int, default=25_000)
    parser.add_argument("--wiki-min-article-tokens", type=int, default=200)
    parser.add_argument("--wiki-max-total-tokens", type=int, default=100_000)
    parser.add_argument("--wiki-min-article-chars", type=int, default=10_000)
    parser.add_argument("--wiki-max-article-chars", type=int, default=30_000)
    parser.add_argument("--no-wiki-disambiguation-filter", action="store_true")
    parser.add_argument("--wiki-classifier-model", default="gpt-5.1")
    parser.add_argument("--wiki-classifier-top-n-chars", type=int, default=8_000)
    parser.add_argument("--wiki-classifier-batch-size", type=int, default=10)
    parser.add_argument("--wiki-classifier-concurrency", type=int, default=5)
    parser.add_argument("--wiki-classifier-max-output-tokens", type=int, default=2_048)
    parser.add_argument("--wiki-classifier-service-tier", default="flex")
    parser.add_argument(
        "--wiki-classifier-reasoning-effort",
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for classifier (default: low)",
    )

    parser.add_argument("--rg-max-question-tokens", type=int, default=1_500)
    parser.add_argument("--rg-max-answer-tokens", type=int, default=256)
    parser.add_argument("--rg-max-total-tokens", type=int, default=8_192)

    parser.add_argument("--per-message-overhead", type=int, default=8)
    parser.add_argument("--max-json-mb", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument(
        "--plan-max-tokens", type=int, default=256, help="Planner max_new_tokens setting"
    )
    parser.add_argument("--seed", type=int, default=41, help="Seed used for sample IDs")
    parser.add_argument(
        "--target-model", default="gpt-5.1", help="Model id recorded in the preflight report"
    )
    parser.add_argument(
        "--language-ratio-threshold",
        type=float,
        default=0.3,
        help="Maximum allowed fraction of non-Latin characters before rejection",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    targets = _domain_targets(args, args.total)

    settings = PreflightSettings(
        squad=SquadPreflightConfig(
            max_question_tokens=args.squad_max_question_tokens,
            max_context_tokens=args.squad_max_context_tokens,
            max_total_tokens=args.squad_max_total_tokens,
            min_context_tokens=args.squad_min_context_tokens,
            max_non_latin_ratio=args.language_ratio_threshold,
        ),
        wikipedia=WikipediaPreflightConfig(
            max_article_tokens=args.wiki_max_article_tokens,
            min_article_tokens=args.wiki_min_article_tokens,
            max_total_tokens=args.wiki_max_total_tokens,
            disambiguation_filter=not args.no_wiki_disambiguation_filter,
            max_non_latin_ratio=args.language_ratio_threshold,
            min_article_chars=args.wiki_min_article_chars,
            max_article_chars=args.wiki_max_article_chars,
        ),
        reasoning_gym=ReasoningGymPreflightConfig(
            max_question_tokens=args.rg_max_question_tokens,
            max_answer_tokens=args.rg_max_answer_tokens,
            max_total_tokens=args.rg_max_total_tokens,
            max_non_latin_ratio=args.language_ratio_threshold,
        ),
        per_message_overhead=args.per_message_overhead,
        max_json_bytes=args.max_json_mb * 1024 * 1024,
        target_model=args.target_model,
        language_ratio_threshold=args.language_ratio_threshold,
        wikipedia_classifier=WikipediaClassifierConfig(
            enabled=True,  # Classification is now mandatory
            model=args.wiki_classifier_model,
            top_n_chars=args.wiki_classifier_top_n_chars,
            batch_size=args.wiki_classifier_batch_size,
            concurrency=args.wiki_classifier_concurrency,
            max_output_tokens=args.wiki_classifier_max_output_tokens,
            service_tier=args.wiki_classifier_service_tier,
            reasoning_effort=args.wiki_classifier_reasoning_effort,
        ),
    )

    plan_cfg = PlanGenerationConfig(
        total_per_domain=targets,
        squad_split=args.squad_split,
        reasoning_gym_dataset=args.reasoning_gym_dataset,
        reasoning_gym_offset=args.reasoning_gym_offset,
        wiki_manifest=args.wiki_manifest,
        wiki_shard_limit=args.wiki_shard_limit,
        wiki_offset=args.wiki_offset,
        seed=args.seed,
    )

    if args.dry_run:
        preview = {
            "targets": targets,
            "output_dir": str(args.output_dir),
            "settings": {
                "squad": asdict(settings.squad),
                "wikipedia": asdict(settings.wikipedia),
                "reasoning_gym": asdict(settings.reasoning_gym),
                "per_message_overhead": settings.per_message_overhead,
                "max_json_bytes": settings.max_json_bytes,
            },
        }
        print(json.dumps(preview, indent=2))
        return

    runner = PreflightRunner(
        plan_cfg=plan_cfg,
        settings=settings,
        max_output_tokens=args.max_output_tokens,
        plan_max_tokens=args.plan_max_tokens,
        output_dir=args.output_dir,
    )
    record = runner.run()

    # Only write the report JSON since accepted/rejected were written incrementally
    report_path = args.output_dir / "preflight_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(record.report, handle, indent=2, sort_keys=True)

    accepted_path = args.output_dir / "preflight_accepted.jsonl"
    rejected_path = args.output_dir / "preflight_rejected.jsonl"
    print(
        json.dumps(
            {
                "accepted_path": str(accepted_path),
                "rejected_path": str(rejected_path),
                "report_path": str(report_path),
                "accepted_total": record.report["accepted_total"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
