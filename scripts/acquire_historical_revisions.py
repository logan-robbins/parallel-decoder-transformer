"""Acquire immutable Wikimedia revision bundles from a reviewed candidate catalog."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pdt.datasets.immutable_io import write_jsonl_new
from pdt.datasets.wikimedia_ingest import acquire_candidate_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--user-agent",
        required=True,
        help="Wikimedia-compliant project identifier containing a contact email.",
    )
    parser.add_argument(
        "--acquisition-date",
        type=date.fromisoformat,
        required=True,
        help="Pinned UTC snapshot date in YYYY-MM-DD form.",
    )
    args = parser.parse_args()
    bundles = acquire_candidate_file(
        args.candidates,
        user_agent=args.user_agent,
        acquisition_date=args.acquisition_date,
    )
    count = write_jsonl_new(
        args.output,
        (bundle.model_dump(mode="json") for bundle in bundles),
    )
    print(f"wrote {count} immutable Wikimedia revision bundles to {args.output}")


if __name__ == "__main__":
    main()
