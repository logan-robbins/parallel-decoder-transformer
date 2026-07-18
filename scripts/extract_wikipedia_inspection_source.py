from __future__ import annotations

import argparse
from pathlib import Path

from model_intrinsic_parallel.wikipedia_source import (
    extract_wikipedia_source,
    load_pinned_revision,
    write_source_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract content headings and ordinary prose from one pinned raw Wikipedia revision."
    )
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--revision-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_record = load_pinned_revision(
        args.raw_jsonl,
        title=args.title,
        revision_id=args.revision_id,
    )
    document = extract_wikipedia_source(raw_record)
    write_source_document(document, args.output)
    cited_paragraphs = sum(
        bool(paragraph.citation_ids) for paragraph in document.paragraphs
    )
    print(
        "extracted "
        f"title={document.source.title!r} "
        f"revision={document.source.revision_id} "
        f"headings={len(document.headings)} "
        f"paragraphs={len(document.paragraphs)} "
        f"cited_paragraphs={cited_paragraphs} "
        f"citations={len(document.citations)} "
        f"sha256={document.model_visible_text_sha256}"
    )


if __name__ == "__main__":
    main()
