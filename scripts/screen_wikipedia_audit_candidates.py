from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from model_intrinsic_parallel.wikipedia_source import (
    extract_wikipedia_source,
    render_model_visible_text,
    write_source_document,
)


MIN_TOKENS = 3_000
MAX_TOKENS = 7_000
MIN_HEADINGS = 6
MIN_PARAGRAPHS = 12
MIN_REFERENCE_OCCURRENCES = 30
MIN_DISTINCT_REFERENCES = 15
MIN_CITED_PARAGRAPH_FRACTION = 0.70
_SLUG = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and screen pinned raw Wikipedia candidates for manual teaching."
    )
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-output", type=Path, required=True)
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
    records = _load_jsonl(args.raw_jsonl)
    screen_rows: list[dict[str, Any]] = []
    for raw_record in records:
        source = extract_wikipedia_source(raw_record)
        output_path = args.output_dir / f"{_slug(source.source.title)}.json"
        write_source_document(source, output_path)
        visible_text = render_model_visible_text(source.headings, source.paragraphs)
        token_count = len(tokenizer.encode(visible_text, add_special_tokens=False))
        reference_occurrences = sum(
            len(paragraph.citation_ids) for paragraph in source.paragraphs
        )
        distinct_references = len(
            {
                reference_id
                for paragraph in source.paragraphs
                for reference_id in paragraph.citation_ids
            }
        )
        cited_paragraphs = sum(
            bool(paragraph.citation_ids) for paragraph in source.paragraphs
        )
        cited_fraction = cited_paragraphs / len(source.paragraphs)
        failures: list[str] = []
        if not MIN_TOKENS <= token_count <= MAX_TOKENS:
            failures.append("token_length")
        if len(source.headings) < MIN_HEADINGS:
            failures.append("insufficient_headings")
        if len(source.paragraphs) < MIN_PARAGRAPHS:
            failures.append("insufficient_paragraphs")
        if reference_occurrences < MIN_REFERENCE_OCCURRENCES:
            failures.append("insufficient_reference_occurrences")
        if distinct_references < MIN_DISTINCT_REFERENCES:
            failures.append("insufficient_distinct_references")
        if cited_fraction < MIN_CITED_PARAGRAPH_FRACTION:
            failures.append("insufficient_cited_paragraph_fraction")
        row = {
            "source_id": source.source.source_id,
            "title": source.source.title,
            "revision_id": source.source.revision_id,
            "source_path": str(output_path),
            "qwen_token_count": token_count,
            "headings": len(source.headings),
            "paragraphs": len(source.paragraphs),
            "cited_paragraphs": cited_paragraphs,
            "cited_paragraph_fraction": cited_fraction,
            "reference_occurrences": reference_occurrences,
            "distinct_references": distinct_references,
            "automatic_failures": failures,
            "requires_manual_reference_quality_audit": not failures,
        }
        screen_rows.append(row)
        print(
            f"screened title={source.source.title!r} tokens={token_count} "
            f"headings={len(source.headings)} paragraphs={len(source.paragraphs)} "
            f"cited_fraction={cited_fraction:.3f} references={distinct_references} "
            f"failures={failures}",
            flush=True,
        )

    args.screen_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.screen_output.with_suffix(f"{args.screen_output.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "schema_version": "model-intrinsic-parallel-source-screen-v1",
                "tokenizer_model": args.tokenizer_model,
                "tokenizer_revision": args.tokenizer_revision,
                "records": screen_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(args.screen_output)


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(payload)
    if not records:
        raise ValueError(f"raw candidate file is empty: {path}")
    return tuple(records)


def _slug(title: str) -> str:
    slug = _SLUG.sub("_", title.casefold()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive source filename from title={title!r}")
    return slug


if __name__ == "__main__":
    main()
