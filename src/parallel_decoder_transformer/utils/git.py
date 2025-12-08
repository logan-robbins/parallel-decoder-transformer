"""Utilities for capturing repository metadata for telemetry manifests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GitMetadata:
    """Lightweight metadata describing the current git checkout."""

    sha: str
    dirty: Optional[bool]


def get_git_metadata(repo_root: Optional[Path | str] = None) -> GitMetadata:
    """Return the current commit hash and dirty status if git is available."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    sha = "unknown"
    dirty: Optional[bool] = None
    try:
        rev_parse = subprocess.run(  # noqa: S603,S607 - git invocation is intentional
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        candidate = rev_parse.stdout.decode().strip()
        if candidate:
            sha = candidate
            status = subprocess.run(  # noqa: S603,S607 - git invocation is intentional
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
        dirty = None
    return GitMetadata(sha=sha, dirty=dirty)


__all__ = ["GitMetadata", "get_git_metadata"]
