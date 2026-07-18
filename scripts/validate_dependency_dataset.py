"""Score blind and causal full-information controls on long-form targets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES, TrunkConfig
from pdt.evaluation.quality_controls import score_quality_control_records
from pdt.trunk.qwen3_adapter import Qwen3TrunkAdapter


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=DEFAULT_TRUNK_PROFILE,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--minimum-documents", type=int, default=32)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"--input must name an existing JSONL file: {args.input}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must lie in (0, 1)")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.minimum_documents <= 1:
        parser.error("--minimum-documents must be greater than one")
    if not torch.cuda.is_available():
        raise RuntimeError("Long-form quality-control scoring requires a CUDA GPU.")
    profile = TRUNK_PROFILES[args.trunk_profile]
    records = _load_records(args.input)
    if len(records) < args.minimum_documents:
        raise ValueError(
            f"Quality-control input has {len(records)} documents, fewer than "
            f"--minimum-documents={args.minimum_documents}."
        )
    print(
        f"loading {profile.base_model}@{profile.revision} on cuda:0; "
        f"documents={len(records)} batch_size={args.batch_size}",
        flush=True,
    )
    trunk = Qwen3TrunkAdapter(
        TrunkConfig(
            profile=profile.name,
            base_model=profile.base_model,
            revision=profile.revision,
            torch_dtype="bfloat16",
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
    )
    pad_token_id = trunk.tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("Pinned trunk tokenizer did not resolve a pad token ID.")
    evaluation = score_quality_control_records(
        records,
        trunk.model,
        device=torch.device("cuda:0"),
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        expected_tokenizer=profile.base_model,
        expected_tokenizer_revision=profile.revision,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        minimum_documents=args.minimum_documents,
    )
    report = {
        "trunk_profile": profile.name,
        "base_model": profile.base_model,
        "revision": profile.revision,
        "input": str(args.input.resolve()),
        "evaluation": evaluation.to_dict(),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not evaluation.expected_outcome_passes:
        raise SystemExit(1)


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: each JSONL row must be an object.")
            records.append(record)
    if not records:
        raise ValueError(f"input JSONL contains no records: {path}")
    return records


if __name__ == "__main__":
    main()
