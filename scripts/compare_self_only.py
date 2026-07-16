"""Apply the preregistered self-only recovery test to two eval telemetry files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdt.evaluation.control_comparison import compare_self_only_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus", type=Path, required=True)
    parser.add_argument("--self-only", type=Path, required=True)
    parser.add_argument("--maximum-recovery-fraction", type=float, default=0.5)
    args = parser.parse_args()
    for name, path in (("--bus", args.bus), ("--self-only", args.self_only)):
        if not path.is_file():
            parser.error(f"{name} must name an existing telemetry file: {path}")
    bus = json.loads(args.bus.read_text())
    self_only = json.loads(args.self_only.read_text())
    if not isinstance(bus, dict) or not isinstance(self_only, dict):
        parser.error("both telemetry roots must be JSON objects")
    result = compare_self_only_recovery(
        bus,
        self_only,
        maximum_recovery_fraction=args.maximum_recovery_fraction,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.passes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
