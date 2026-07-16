"""Apply the strict document-paired PDT quality-bounds comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdt.evaluation.quality_comparison import compare_quality_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--pdt-telemetry", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--minimum-documents", type=int, default=32)
    args = parser.parse_args()
    for name, path in (
        ("--baseline-report", args.baseline_report),
        ("--pdt-telemetry", args.pdt_telemetry),
    ):
        if not path.is_file():
            parser.error(f"{name} must name an existing JSON file: {path}")
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    telemetry = json.loads(args.pdt_telemetry.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(telemetry, dict):
        parser.error("both report roots must be JSON objects")
    result = compare_quality_bounds(
        baseline,
        telemetry,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        minimum_documents=args.minimum_documents,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    args.output_report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result.passes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
