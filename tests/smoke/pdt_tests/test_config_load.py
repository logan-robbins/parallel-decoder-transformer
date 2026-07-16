"""Config loader smoke tests."""

from __future__ import annotations

import pytest

from pdt.config import load_config
from pdt.config.loader import _materialize_stage_policy


def test_canonical_config_loads_after_hash_scrub():
    config = load_config("configs/pdt_qwen3_4b.yaml")

    assert config.sidecar.planner_head.vocab_size == config.sidecar.plan_vocab_size
    assert config.training.loss_weights.vq_commit == 0.25
    assert config.training.dataset_path.endswith("latent_dependency_control/train.jsonl")


def test_stage_policy_unknown_keys_fail_fast():
    with pytest.raises(ValueError, match=r"Unknown keys for StagePolicy: \['freze'\]"):
        _materialize_stage_policy({"name": "broken", "freze": ["trunk"]})
