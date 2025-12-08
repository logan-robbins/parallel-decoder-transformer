"""Integration smoke test for local GPT-OSS 20B inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not Path("gpt-oss-20b/original/model.safetensors").exists(),
    reason="Local GPT-OSS 20B weights not available.",
)
def test_local_gpt_oss_inference(tmp_path: Path) -> None:
    """Run a short inference pass against the local GPT-OSS 20B weights.

    This test is intentionally skipped unless the environment variable
    ``PDT_RUN_GPTOSS_TESTS`` is set, because loading a 20B parameter model
    is time- and memory-intensive. When enabled, it exercises the full
    inference stack – tokenizer resolution, trunk loading, per-stream prefixes,
    and the decode loop – on the prompt ``"Tell me some facts about the US."``.
    """

    if not os.environ.get("PDT_RUN_GPTOSS_TESTS"):
        pytest.skip("Set PDT_RUN_GPTOSS_TESTS=1 to run the GPT-OSS 20B smoke test.")

    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    # Local imports deferred until heavy deps confirmed available.
    from parallel_decoder_transformer.config import ModelConfig, TrainingConfig, TrunkAdapterConfig
    from parallel_decoder_transformer.data.tokenizer import TokenizerConfig, resolve_tokenizer
    from parallel_decoder_transformer.inference import (
        MultiStreamOrchestrator,
        build_inference_config,
    )
    from parallel_decoder_transformer.models import ParallelDecoderTransformer

    weights_dir = Path("gpt-oss-20b/original").resolve()
    tokenizer_dir = Path("gpt-oss-20b/tokenizer").resolve()

    model_cfg = ModelConfig(
        trunk=TrunkAdapterConfig(
            base_model=str(weights_dir),
            device_map="auto",
            torch_dtype="bfloat16",
            trust_remote_code=True,
        )
    )
    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.load_model()

    tokenizer_cfg = TokenizerConfig(custom_path=tokenizer_dir)
    tokenizer, _ = resolve_tokenizer(tokenizer_cfg)

    training_cfg = TrainingConfig()
    training_cfg.curriculum.B = 1
    training_cfg.curriculum.L = 16
    training_cfg.curriculum.delta = 0

    inference_cfg = build_inference_config(
        training_cfg,
        stream_to_id=dict(model_cfg.collator.stream_to_id),
        rng_seed=13,
    )
    inference_cfg.decode.max_new_tokens = 6
    inference_cfg.decode.do_sample = False

    # Provide distinct per-stream instructions via the new prefix mapping.
    prefix_by_stream = {
        stream: f"You are {stream}. Produce your portion succinctly: "
        for stream in inference_cfg.streams
    }

    orchestrator = MultiStreamOrchestrator(model, tokenizer, inference_cfg)
    prompt = "Tell me some facts about the US."
    orchestrator.start(prompt, prefix_by_stream=prefix_by_stream)

    events = 0
    while True:
        outcome = orchestrator.step()
        if outcome is None:
            break
        events += 1
        # Guard against runaway decoding if runtime config changes.
        if events > 64:
            pytest.fail("Inference did not terminate within 64 steps.")

    manifest = orchestrator.finalize()
    assert manifest["steps"] == events
    stream_payloads = manifest["streams"]
    assert set(stream_payloads.keys()) == set(inference_cfg.streams)
    for stream, payload in stream_payloads.items():
        text = payload["text"].strip()
        assert text, f"Stream {stream} produced empty output."

    # Persist manifest for inspection when the test runs locally.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
