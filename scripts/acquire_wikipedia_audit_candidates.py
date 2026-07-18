from __future__ import annotations

import argparse
from pathlib import Path

from model_intrinsic_parallel.wikimedia_acquisition import (
    acquire_revisions,
    load_titles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire immutable current raw revisions for manually audited examples."
    )
    parser.add_argument("--titles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-agent", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = acquire_revisions(
        load_titles(args.titles),
        output_path=args.output,
        user_agent=args.user_agent,
    )
    total_bytes = sum(int(record["transfer"]["bytes"]) for record in records)
    total_seconds = sum(float(record["transfer"]["seconds"]) for record in records)
    print(
        f"complete records={len(records)} bytes={total_bytes} "
        f"aggregate_MiB/s={total_bytes / (1024**2) / total_seconds:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
