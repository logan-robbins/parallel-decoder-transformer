"""Negative tests for removed hash-era supervision fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdt.training.dataset import PDTDependencyDataset


def test_hash_era_fields_are_rejected(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    rec = {
        "example_id": "bad",
        "family": "latent_dependency_control",
        "visibility_lag_blocks": 1,
        "planner_ids": [1, 2, 3],
        "stream_inputs": [
            {
                "stream_id": f"stream_{idx}",
                "target_blocks": ["a", "b"],
                "target_block_ids": [[1], [2]],
            }
            for idx in range(3)
        ],
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="removed hash-era fields"):
        PDTDependencyDataset(path, num_streams=3)


def test_delta_one_requires_two_blocks(tmp_path: Path):
    path = tmp_path / "one_block.jsonl"
    rec = {
        "example_id": "one_block",
        "family": "latent_dependency_control",
        "visibility_lag_blocks": 1,
        "stream_inputs": [
            {
                "stream_id": f"stream_{idx}",
                "target_blocks": ["a"],
                "target_block_ids": [[1]],
            }
            for idx in range(3)
        ],
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least two target blocks"):
        PDTDependencyDataset(path, num_streams=3)
