"""Atomic exclusive publication for immutable dataset artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import tempfile


__all__ = ["write_bytes_new", "write_jsonl_new"]


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Stream JSONL through a same-filesystem temporary, then publish once."""

    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            if count == 0:
                raise ValueError("Refusing to create an empty JSONL output.")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def write_bytes_new(path: Path, content: bytes) -> None:
    """Publish non-empty bytes exactly once without exposing a partial file."""

    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable output: {path}")
    if not content:
        raise ValueError("Refusing to write an empty immutable output.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_new_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_new_file(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to replace immutable output: {destination}"
        ) from exc
