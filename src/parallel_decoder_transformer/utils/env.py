"""Environment helpers for resolving repository-local .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_repo_dotenv() -> bool:
    """Load the .env file at the repository root once."""

    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return False
    load_dotenv(env_path, override=False)
    return True


__all__ = ["load_repo_dotenv"]
