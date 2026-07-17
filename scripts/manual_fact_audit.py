"""Export and adjudicate the blinded lane-exact human fact audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdt.evaluation.manual_fact_audit import (
    adjudicate_fact_audit,
    export_blinded_fact_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--generation-evaluation", type=Path, required=True)
    export.add_argument("--raw-examples", type=Path, required=True)
    export.add_argument("--queue", type=Path, required=True)
    export.add_argument("--key", type=Path, required=True)
    export.add_argument("--randomization-seed", type=int, required=True)

    adjudicate = commands.add_parser("adjudicate")
    adjudicate.add_argument("--queue", type=Path, required=True)
    adjudicate.add_argument("--key", type=Path, required=True)
    adjudicate.add_argument("--annotator-a", type=Path, required=True)
    adjudicate.add_argument("--annotator-b", type=Path, required=True)
    adjudicate.add_argument("--adjudicator", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    adjudicate.add_argument("--bootstrap-samples", type=int, default=10_000)
    adjudicate.add_argument("--confidence-level", type=float, default=0.95)
    adjudicate.add_argument("--minimum-documents", type=int, default=32)

    args = parser.parse_args()
    if args.command == "export":
        count = export_blinded_fact_audit(
            generation_evaluation_path=args.generation_evaluation,
            raw_examples_path=args.raw_examples,
            queue_path=args.queue,
            key_path=args.key,
            randomization_seed=args.randomization_seed,
        )
        print(f"exported {count} blinded human fact decisions to {args.queue}")
    elif args.command == "adjudicate":
        result = adjudicate_fact_audit(
            queue_path=args.queue,
            key_path=args.key,
            annotator_a_path=args.annotator_a,
            annotator_b_path=args.annotator_b,
            adjudicator_path=args.adjudicator,
            output_path=args.output,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            minimum_documents=args.minimum_documents,
        )
        print(json.dumps(result["manual_evidence_gate"], sort_keys=True))
    else:
        raise AssertionError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    main()
