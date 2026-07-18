"""Retokenize exact dependency JSONL with the locked Instruct chat schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES
from pdt.datasets.retokenize import RetokenizeConfig, run_retokenize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=DEFAULT_TRUNK_PROFILE,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    count = run_retokenize(
        RetokenizeConfig(
            input_path=args.input,
            output_path=args.output,
            trunk_profile=args.trunk_profile,
            force=args.force,
        )
    )
    print(f"retokenized {count} examples -> {args.output}")


if __name__ == "__main__":
    main()
