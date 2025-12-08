from __future__ import annotations

import torch

from parallel_decoder_transformer.inference.config import InferenceConfig
from parallel_decoder_transformer.inference.replay import (
    LogitReplayArtifact,
    ReplayArtifactWriter,
)
from parallel_decoder_transformer.data.tokenizer import TokenizerConfig, TokenizerManifest
from parallel_decoder_transformer.utils.git import GitMetadata


def _build_inference_config() -> InferenceConfig:
    return InferenceConfig(
        streams=("stream_1", "stream_2"),
        stride_B=2,
        commit_L=4,
        read_lag_delta=1,
        max_snapshots_K=3,
    )


def _tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        source="custom",
        identifier="stub-tokenizer",
        tokenizer_class="AutoTokenizer",
        is_fast=True,
        vocab_size=128,
        added_tokens=("foo",),
        special_tokens=("<plan>",),
        padding_side="right",
        truncation_side="right",
    )


def test_replay_writer_roundtrip(tmp_path):
    config = _build_inference_config()
    tokenizer_cfg = TokenizerConfig(pretrained_name="stub-model")
    manifest = _tokenizer_manifest()
    writer = ReplayArtifactWriter(
        tmp_path / "artifact",
        prompt="hello world",
        tokenizer_config=tokenizer_cfg,
        tokenizer_manifest=manifest,
        inference_config=config,
        notes_dim=4,
        hidden_size=8,
        plan_vocab_size=16,
        lm_vocab_size=32,
        chunk_size=2,
        git_metadata=GitMetadata(sha="deadbeef", dirty=False),
    )
    plan_ids = torch.tensor([[1, 2, 3, 4]])
    plan_mask = torch.ones_like(plan_ids)
    plan_logits = torch.randn(1, 4, 16)
    writer.record_plan(
        plan_token_ids=plan_ids,
        plan_mask=plan_mask,
        plan_logits=plan_logits,
        source="model",
        catalog=[{"plan_item_id": "p0", "stream": "stream_1", "text": "foo"}],
    )
    writer.record_bootstrap("stream_1", torch.ones(1, 1, 4))
    writer.record_bootstrap("stream_2", torch.zeros(1, 1, 4))
    writer.record_step(
        "stream_1",
        token_id=10,
        agreement=0.7,
        coverage_logits=[0.1, 0.2],
        note_emitted=True,
        note_vector=torch.randn(1, 1, 4),
        delta_norm=0.5,
    )
    writer.record_step(
        "stream_2",
        token_id=11,
        agreement=0.6,
        coverage_logits=None,
        note_emitted=False,
        note_vector=None,
        delta_norm=0.4,
    )
    artifact_dir = writer.finalize()

    artifact = LogitReplayArtifact.load(artifact_dir)
    assert artifact.streams == ("stream_1", "stream_2")
    assert artifact.token_ids["stream_1"] == [10]
    assert artifact.token_ids["stream_2"] == [11]
    assert torch.allclose(artifact.planner_token_ids, plan_ids)
    assert artifact.planner_logits.shape == plan_logits.shape
    assert artifact.plan_hash_buckets == 16
    assert artifact.plan_hash_salt == ""
    assert "stream_1" in artifact.agreement
    assert artifact.agreement["stream_1"].shape[0] == 1
    assert artifact.note_snapshots["stream_1"].shape[-1] == 4
    assert artifact.bootstrap_notes["stream_1"].numel() == 4
