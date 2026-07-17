"""Sample immutable long-form English Wikipedia packets for real-plan annotation."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Mapping

import tiktoken
from datasets import load_dataset  # type: ignore[import-untyped]

from pdt.datasets.real_plan_batch import MAX_SOURCE_TOKENS, MIN_SOURCE_TOKENS
from pdt.datasets.immutable_io import write_jsonl_new


WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKIPEDIA_LICENSE = "CC BY-SA 3.0 and GFDL"
_SPLIT_RANGES = {
    "train": (0, 90),
    "validation": (90, 95),
    "test": (95, 100),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(_SPLIT_RANGES), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace immutable source data: {args.output}")

    dataset = load_dataset(
        WIKIPEDIA_DATASET,
        WIKIPEDIA_CONFIG,
        split="train",
        revision=WIKIPEDIA_REVISION,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=50_000)
    encoding = tiktoken.get_encoding("o200k_base")
    def packets() -> Iterator[Mapping[str, object]]:
        written = 0
        for raw in dataset:
            if not isinstance(raw, Mapping):
                raise TypeError("Wikipedia streaming rows must be mappings.")
            packet = _packet_from_article(raw, encoding=encoding, split=args.split)
            if packet is None:
                continue
            yield packet
            written += 1
            if written == args.count:
                return
        raise RuntimeError(
            f"Wikipedia stream yielded only {written} valid packets; requested {args.count}."
        )
    written = write_jsonl_new(args.output, packets())
    print(f"wrote {written} immutable {args.split} source packets -> {args.output}")


def _packet_from_article(
    raw: Mapping[str, object],
    *,
    encoding: object,
    split: str,
) -> dict[str, object] | None:
    source_id = str(raw.get("id", "")).strip()
    title = str(raw.get("title", "")).strip()
    url = str(raw.get("url", "")).strip()
    text = str(raw.get("text", "")).strip()
    if not source_id or not title or not url or not text:
        return None
    if not _belongs_to_split(source_id, split):
        return None
    title_lower = title.lower()
    if (
        "disambiguation" in title_lower
        or title_lower.startswith(("list of ", "outline of ", "index of "))
    ):
        return None
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n{2,}", text)
    ]
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= 80 and not paragraph.startswith(("Category:", "File:"))
    ]
    if len(paragraphs) < 6:
        return None
    selected: list[str] = []
    token_count = 0
    for paragraph in paragraphs:
        if len(selected) == 80:
            break
        paragraph_tokens = len(encoding.encode(paragraph))  # type: ignore[attr-defined]
        separator_tokens = 2 if selected else 0
        if token_count + separator_tokens + paragraph_tokens > MAX_SOURCE_TOKENS:
            break
        selected.append(paragraph)
        token_count += separator_tokens + paragraph_tokens
    if token_count < MIN_SOURCE_TOKENS or len(selected) < 6:
        return None
    return {
        "source_id": f"wikipedia-{source_id}",
        "title": title,
        "source_url": url,
        "license": WIKIPEDIA_LICENSE,
        "paragraphs": [
            {"paragraph_id": f"p{index:03d}", "text": paragraph}
            for index, paragraph in enumerate(selected)
        ],
    }


def _belongs_to_split(source_id: str, split: str) -> bool:
    bucket = int.from_bytes(
        hashlib.sha256(source_id.encode("utf-8")).digest()[:8],
        "big",
    ) % 100
    lower, upper = _SPLIT_RANGES[split]
    return lower <= bucket < upper


if __name__ == "__main__":
    main()
