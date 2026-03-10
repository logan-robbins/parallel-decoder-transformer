"""Smoke tests verifying that CLI entry points load without errors."""

import subprocess
import sys


def test_infer_help():
    result = subprocess.run(
        [sys.executable, "scripts/infer.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"infer.py --help failed:\n{result.stderr}"


def test_train_help():
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"train.py --help failed:\n{result.stderr}"
