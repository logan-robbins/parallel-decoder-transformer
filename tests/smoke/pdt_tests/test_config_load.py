"""Config loader smoke tests."""

from __future__ import annotations

from pdt.config import load_config


def test_canonical_config_loads_after_hash_scrub():
    config = load_config("configs/pdt_qwen3_4b.yaml")

    assert config.sidecar.planner_head.vocab_size == config.sidecar.plan_vocab_size
    assert config.training.loss_weights.vq_commit == 0.25
    assert config.training.dataset_path.endswith("latent_dependency_control/train.jsonl")
