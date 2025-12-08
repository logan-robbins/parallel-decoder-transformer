from __future__ import annotations

from types import SimpleNamespace

import pytest

from parallel_decoder_transformer.utils.git import get_git_metadata


def test_get_git_metadata_handles_missing_git(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def raise_error(*_: object, **__: object) -> None:
        raise OSError("git unavailable")

    monkeypatch.setattr("parallel_decoder_transformer.utils.git.subprocess.run", raise_error)
    meta = get_git_metadata(tmp_path)
    assert meta.sha == "unknown"
    assert meta.dirty is None


def test_get_git_metadata_reports_dirty_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def fake_run(cmd, **_):  # type: ignore[no-untyped-def]
        if tuple(cmd) == ("git", "rev-parse", "HEAD"):
            return SimpleNamespace(stdout=b"abc123\n")
        if tuple(cmd) == ("git", "status", "--porcelain"):
            return SimpleNamespace(stdout=b" M file.txt\n")
        raise AssertionError(f"unexpected git call: {cmd}")

    monkeypatch.setattr("parallel_decoder_transformer.utils.git.subprocess.run", fake_run)
    meta = get_git_metadata(tmp_path)
    assert meta.sha == "abc123"
    assert meta.dirty is True
