from __future__ import annotations

from types import SimpleNamespace

import pytest

from parallel_decoder_transformer.utils.plan_catalog import (
    PlanHashParams,
    canonical_plan_catalog_entries,
    hash_plan_entry,
    hash_plan_text,
    pad_plan_ids,
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


def test_canonical_plan_catalog_entries_preserve_stream_order() -> None:
    payload = {
        "plan": [
            {
                "stream_id": "intro",
                "summary": "Set up context",
                "section_contract": {"type": "section", "index": 0},
                "notes_contract": ["Introduce the topic"],
            },
            {
                "stream_id": "answer",
                "summary": "Deliver answer",
                "notes_contract": ["State the final answer"],
            },
        ]
    }
    entries = canonical_plan_catalog_entries(payload)
    assert [entry["stream"] for entry in entries] == [
        "stream_intro",
        "stream_intro",
        "stream_intro",
        "stream_answer",
        "stream_answer",
    ]
    assert entries[0]["text"] == "Introduce the topic"
    assert entries[3]["text"] == "State the final answer"
    assert entries[-1]["text"] == "Deliver answer"


def test_hash_plan_entry_is_stream_aware() -> None:
    bucket_count = 4096
    intro = hash_plan_entry("stream_intro", "Same text", bucket_count, salt="demo")
    answer = hash_plan_entry("stream_answer", "Same text", bucket_count, salt="demo")
    assert intro != answer


def test_pad_plan_ids_uses_zero_as_null_slot() -> None:
    ids, mask = pad_plan_ids([11, 17], 4)
    assert ids == [11, 17, 0, 0]
    assert mask == [1, 1, 0, 0]
    with pytest.raises(ValueError):
        pad_plan_ids([1, 2, 3], 2)
