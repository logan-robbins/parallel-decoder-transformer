"""Build canonical v2 and tokenized-v3 records from audited Wikipedia examples."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast


os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES  # noqa: E402
from pdt.datasets.audited_real_plan import (  # noqa: E402
    build_audited_real_plan_example,
    load_audited_catalog,
    load_catalog_raw_revision,
    load_training_example,
)
from pdt.datasets.immutable_io import write_jsonl_new  # noqa: E402
from pdt.datasets.real_plan_retokenize import (  # noqa: E402
    SEMANTIC_EMBEDDING_DIM,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_EMBEDDING_REVISION,
    retokenize_real_plan_example,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "data/model_intrinsic_parallel/candidates/trainer_record_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("data/model_intrinsic_parallel/examples"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path(
            "data/processed/model_intrinsic_parallel/"
            "qwen3_4b_instruct_2507/audited_real_plan_v2.jsonl"
        ),
    )
    parser.add_argument(
        "--tokenized-output",
        type=Path,
        default=Path(
            "data/processed/model_intrinsic_parallel/"
            "qwen3_4b_instruct_2507/audited_real_plan_tokenized_v3.jsonl"
        ),
    )
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=DEFAULT_TRUNK_PROFILE,
    )
    parser.add_argument(
        "--embedding-device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    args = parser.parse_args(argv)

    repository_root = Path.cwd().resolve()
    catalog_path = (repository_root / args.catalog).resolve()
    examples_dir = (repository_root / args.examples_dir).resolve()
    raw_output = (repository_root / args.raw_output).resolve()
    tokenized_output = (repository_root / args.tokenized_output).resolve()
    if raw_output == tokenized_output:
        parser.error("--raw-output and --tokenized-output must differ")
    for output in (raw_output, tokenized_output):
        if output.exists():
            raise FileExistsError(f"Refusing to replace immutable output: {output}")
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"Audited examples directory does not exist: {examples_dir}")

    profile = TRUNK_PROFILES[args.trunk_profile]
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        profile.base_model,
        revision=profile.revision,
        use_fast=True,
        local_files_only=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Audited record construction requires a fast tokenizer.")
    embedder = SentenceTransformer(
        SEMANTIC_EMBEDDING_MODEL,
        revision=SEMANTIC_EMBEDDING_REVISION,
        device=args.embedding_device,
        local_files_only=True,
    )
    dimension = embedder.get_sentence_embedding_dimension()
    if dimension != SEMANTIC_EMBEDDING_DIM:
        raise RuntimeError(
            f"Semantic embedder dimension must be {SEMANTIC_EMBEDDING_DIM}, "
            f"got {dimension}."
        )

    raw_records = []
    tokenized_records = []
    for entry in load_audited_catalog(catalog_path):
        example = load_training_example(examples_dir / entry.example_file)
        raw_revision = load_catalog_raw_revision(repository_root, entry)
        converted = build_audited_real_plan_example(
            example=example,
            catalog=entry,
            raw_revision=raw_revision,
            tokenizer=tokenizer,
            trunk_profile=profile,
        )
        raw_records.append(converted)
        tokenized_record = retokenize_real_plan_example(
            converted,
            tokenizer=tokenizer,
            embedder=embedder,
            tokenizer_name=profile.base_model,
            tokenizer_revision=profile.revision,
        )
        tokenized_records.append(tokenized_record)
        lane_counts = [
            cast(int, lane["target_token_count"])
            for lane in cast(list[dict[str, object]], tokenized_record["lanes"])
        ]
        print(
            f"validated {converted.source.source_id}: "
            f"source_tokens={converted.source.qwen_token_count} "
            f"target_tokens={lane_counts}",
            flush=True,
        )

    raw_count = write_jsonl_new(
        raw_output,
        (record.model_dump(mode="json") for record in raw_records),
    )
    tokenized_count = write_jsonl_new(tokenized_output, tokenized_records)
    if raw_count != tokenized_count:
        raise RuntimeError(
            f"Raw/tokenized record count mismatch: {raw_count} != {tokenized_count}."
        )
    print(f"raw_output={raw_output}", flush=True)
    print(f"tokenized_output={tokenized_output}", flush=True)
    print(f"record_count={raw_count}", flush=True)


if __name__ == "__main__":
    main()
