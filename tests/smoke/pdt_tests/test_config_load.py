"""Config loader smoke tests."""

from __future__ import annotations

import pytest

from pdt.config import load_config
from pdt.config.loader import _materialize_stage_policy


def test_canonical_config_loads_after_hash_scrub():
    config = load_config("configs/pdt_qwen3_4b.yaml")

    assert config.sidecar.planner_head.num_streams == 3
    assert config.sidecar.planner_head.max_nodes_per_stream == 8
    assert config.sidecar.planner_head.planner_width == 512
    assert config.sidecar.plan_memory_proj.planner_width == 512
    assert config.sidecar.semantic_supervision.max_facts == 128
    assert config.training.loss_weights.plan_semantic == 1.0
    assert config.training.loss_weights.dynamic_vq_commit == 0.25
    assert config.sidecar.speculation_head.num_codebooks == 4
    assert config.sidecar.speculation_head.codes_per_codebook == 256
    assert config.training.dataset_path.endswith(
        "real_plan/qwen3_4b_instruct_2507/train.jsonl"
    )
    assert config.training.max_blocks == 32
    assert config.runtime.notes_bus.history_blocks == 16
    assert config.instrumentation.coordination_source == "bus"
    assert config.instrumentation.fork_layer == 24
    assert config.instrumentation.target_layers == tuple(range(24, 36))


def test_stage_policy_unknown_keys_fail_fast():
    with pytest.raises(ValueError, match=r"Unknown keys for StagePolicy: \['freze'\]"):
        _materialize_stage_policy({"name": "broken", "freze": ["trunk"]})
