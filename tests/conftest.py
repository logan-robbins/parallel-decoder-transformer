"""Pytest fixtures and path configuration for Parallel Decoder Transformer tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the editable ``src/`` layout is importable even if the editable
# install's .pth file has not been loaded by the pytest-invocation Python.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
