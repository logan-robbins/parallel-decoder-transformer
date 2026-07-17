"""Retokenize and embed validated real-plan JSONL with pinned model identities."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES
from pdt.datasets.real_plan_retokenize import (
    RealPlanRetokenizeConfig,
    run_real_plan_retokenize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=DEFAULT_TRUNK_PROFILE,
    )
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args()
    count = run_real_plan_retokenize(
        RealPlanRetokenizeConfig(
            input_path=args.input,
            output_path=args.output,
            trunk_profile=args.trunk_profile,
            embedding_device=args.embedding_device,
        )
    )
    print(f"retokenized {count} real-plan examples -> {args.output}")


if __name__ == "__main__":
    main()
