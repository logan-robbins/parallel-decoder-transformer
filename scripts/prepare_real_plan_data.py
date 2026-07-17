"""Build, submit, retrieve, and parse the two-stage real-plan Batch pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdt.datasets.real_plan_batch import (
    build_fact_requests,
    build_joint_requests,
    download_batch_results,
    parse_fact_results,
    parse_joint_results,
    retrieve_batch,
    split_real_plan_examples,
    submit_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_facts = subparsers.add_parser("build-fact-requests")
    build_facts.add_argument("--sources", type=Path, required=True)
    build_facts.add_argument("--accepted-manifest", type=Path, required=True)
    build_facts.add_argument("--accepted-manifest-sha256", required=True)
    build_facts.add_argument("--output", type=Path, required=True)

    parse_facts = subparsers.add_parser("parse-fact-results")
    parse_facts.add_argument("--sources", type=Path, required=True)
    parse_facts.add_argument("--accepted-manifest", type=Path, required=True)
    parse_facts.add_argument("--accepted-manifest-sha256", required=True)
    parse_facts.add_argument("--results", type=Path, required=True)
    parse_facts.add_argument("--output", type=Path, required=True)

    build_joint = subparsers.add_parser("build-joint-requests")
    build_joint.add_argument("--sources", type=Path, required=True)
    build_joint.add_argument("--accepted-manifest", type=Path, required=True)
    build_joint.add_argument("--accepted-manifest-sha256", required=True)
    build_joint.add_argument("--facts", type=Path, required=True)
    build_joint.add_argument("--output", type=Path, required=True)

    parse_joint = subparsers.add_parser("parse-joint-results")
    parse_joint.add_argument("--sources", type=Path, required=True)
    parse_joint.add_argument("--accepted-manifest", type=Path, required=True)
    parse_joint.add_argument("--accepted-manifest-sha256", required=True)
    parse_joint.add_argument("--facts", type=Path, required=True)
    parse_joint.add_argument("--results", type=Path, required=True)
    parse_joint.add_argument("--output", type=Path, required=True)

    split_examples = subparsers.add_parser("split-examples")
    split_examples.add_argument("--input", type=Path, required=True)
    split_examples.add_argument("--output-dir", type=Path, required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--requests", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--batch-id", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--batch-id", required=True)
    download.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build-fact-requests":
        count = build_fact_requests(
            args.sources,
            args.accepted_manifest,
            args.accepted_manifest_sha256,
            args.output,
        )
        print(f"wrote {count} fact requests to {args.output}")
    elif args.command == "parse-fact-results":
        count = parse_fact_results(
            args.sources,
            args.accepted_manifest,
            args.accepted_manifest_sha256,
            args.results,
            args.output,
        )
        print(f"validated {count} fact results into {args.output}")
    elif args.command == "build-joint-requests":
        count = build_joint_requests(
            args.sources,
            args.accepted_manifest,
            args.accepted_manifest_sha256,
            args.facts,
            args.output,
        )
        print(f"wrote {count} joint requests to {args.output}")
    elif args.command == "parse-joint-results":
        count = parse_joint_results(
            args.sources,
            args.accepted_manifest,
            args.accepted_manifest_sha256,
            args.facts,
            args.results,
            args.output,
        )
        print(f"validated {count} real-plan examples into {args.output}")
    elif args.command == "split-examples":
        count = split_real_plan_examples(args.input, args.output_dir)
        print(f"partitioned {count} real-plan examples into {args.output_dir}")
    elif args.command == "submit":
        print(json.dumps(submit_batch(args.requests), sort_keys=True))
    elif args.command == "status":
        print(json.dumps(retrieve_batch(args.batch_id), sort_keys=True))
    elif args.command == "download":
        download_batch_results(args.batch_id, args.output)
        print(f"downloaded Batch output to {args.output}")
    else:
        raise AssertionError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    main()
