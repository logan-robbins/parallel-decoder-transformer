"""Compile a strict architecture-screen spec into validated PDT run configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdt.experiments.sweep import compile_architecture_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_architecture_sweep(args.spec, args.output_dir)
    print(
        f"compiled {manifest.variants} scientific variants and "
        f"{len(manifest.runs)} seeded runs into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
