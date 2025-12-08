from __future__ import annotations

from types import SimpleNamespace

import pytest

from parallel_decoder_transformer.utils.plan_catalog import (
    PlanHashParams,
    hash_plan_text,
    plan_hash_fingerprint,
    plan_hash_params_from_manifest,
    resolve_plan_hash_params,
)


def test_hash_plan_text_uses_salt():
    bucket_count = 1024
    text = "Outline scope"
    baseline = hash_plan_text(text, bucket_count, salt="baseline")
    alternate = hash_plan_text(text, bucket_count, salt="alternate")
    assert baseline != alternate
    assert 0 <= baseline < bucket_count
    assert 0 <= alternate < bucket_count


def test_plan_hash_params_from_manifest_roundtrip():
    manifest = {
        "plan_vocab_size": 4096,
        "plan_hash_buckets": 2048,
        "plan_hash_salt": "demo",
    }
    params = plan_hash_params_from_manifest(manifest)
    assert params == PlanHashParams(vocab_size=4096, hash_buckets=2048, salt="demo")
    assert plan_hash_fingerprint(params) == "4096:2048:demo"


def test_resolve_plan_hash_params_prefers_collator_settings():
    collator = SimpleNamespace(plan_hash_buckets=128, plan_hash_salt="custom")
    config = SimpleNamespace(plan_vocab_size=256, plan_hash_salt="fallback", collator=collator)
    params = resolve_plan_hash_params(config)
    assert params.vocab_size == 256
    assert params.hash_buckets == 128
    assert params.salt == "custom"


def test_plan_hash_params_from_manifest_requires_metadata():
    with pytest.raises(ValueError):
        plan_hash_params_from_manifest({"plan_vocab_size": 0})
