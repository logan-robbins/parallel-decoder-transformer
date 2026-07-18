from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from model_intrinsic_parallel.training_example import (
    compile_training_example,
    load_curation,
    load_source_document,
    summarize_example,
    write_training_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and strictly validate one source-grounded three-lane training example."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        revision=args.tokenizer_revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    source = load_source_document(args.source)
    curation = load_curation(args.curation)
    example = compile_training_example(
        source,
        curation,
        tokenizer=tokenizer,
        tokenizer_model=args.tokenizer_model,
        tokenizer_revision=args.tokenizer_revision,
    )
    write_training_example(example, args.output)
    print(summarize_example(example))


if __name__ == "__main__":
    main()
