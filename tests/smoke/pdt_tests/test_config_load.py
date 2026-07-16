"""Config loader smoke tests."""

from __future__ import annotations

import pytest

from pdt.config import load_config
from pdt.config.loader import _materialize_stage_policy


def test_canonical_config_loads_after_hash_scrub():
    config = load_config("configs/pdt_qwen3_4b.yaml")

    assert config.sidecar.planner_head.vocab_size == config.sidecar.plan_vocab_size
    assert config.training.loss_weights.planner_vq_commit == 0.25
    assert config.training.loss_weights.dynamic_vq_commit == 0.25
    assert config.sidecar.speculation_head.num_codebooks == 4
    assert config.sidecar.speculation_head.codes_per_codebook == 256
    assert config.training.dataset_path.endswith(
        "long_form_dependency/qwen3_4b_instruct_2507/train.jsonl"
    )
    assert config.training.max_blocks == 32
    assert config.runtime.notes_bus.history_blocks == 16
    assert config.instrumentation.coordination_source == "bus"


def test_stage_policy_unknown_keys_fail_fast():
    with pytest.raises(ValueError, match=r"Unknown keys for StagePolicy: \['freze'\]"):
        _materialize_stage_policy({"name": "broken", "freze": ["trunk"]})
