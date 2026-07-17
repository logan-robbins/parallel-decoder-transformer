"""Run the held-out four-condition long-form PDT evidence evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdt.evaluation.real_plan_generation import (
    GenerationEvaluationConfig,
    run_generation_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-raw", type=Path, required=True)
    parser.add_argument("--calibration-tokenized", type=Path, required=True)
    parser.add_argument("--evaluation-raw", type=Path, required=True)
    parser.add_argument("--evaluation-tokenized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--coordination-source",
        choices=("bus", "self_only"),
        default=None,
    )
    parser.add_argument("--entailment-device", default="cuda")
    parser.add_argument("--entailment-batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run_generation_evaluation(
        GenerationEvaluationConfig(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            calibration_raw_path=args.calibration_raw,
            calibration_tokenized_path=args.calibration_tokenized,
            evaluation_raw_path=args.evaluation_raw,
            evaluation_tokenized_path=args.evaluation_tokenized,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            coordination_source=args.coordination_source,
            entailment_device=args.entailment_device,
            entailment_batch_size=args.entailment_batch_size,
        )
    )
    checks = result["automatic_configured_checks"]
    print(
        f"wrote {result['held_out_examples']} held-out documents to {args.output}; "
        f"automatic_screen_only={checks}; human_fact_audit_required=true"
    )


if __name__ == "__main__":
    main()
