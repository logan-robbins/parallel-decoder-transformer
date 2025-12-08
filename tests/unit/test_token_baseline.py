from __future__ import annotations

from types import SimpleNamespace

import torch

from parallel_decoder_transformer.baselines import (
    TokenBaselineConfig,
    build_token_baseline_config,
    run_token_baseline,
)
from parallel_decoder_transformer.inference import DecodeConfig


class DummyTokenizer:
    eos_token_id = 4

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        del prompt, return_tensors
        input_ids = torch.tensor([[0]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, clean_up_tokenization_spaces: bool = False, skip_special_tokens: bool = False) -> str:  # type: ignore[override]
        del clean_up_tokenization_spaces, skip_special_tokens
        mapping = {3: "X", 4: "</s>"}
        return "".join(mapping.get(int(token), str(int(token))) for token in ids)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_stub", torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):  # type: ignore[override]
        del attention_mask
        step = input_ids.size(1) - 1
        vocab = torch.zeros((input_ids.size(0), 1, 5))
        if step == 0:
            vocab[..., 3] = 5.0
        else:
            vocab[..., 4] = 5.0
        return SimpleNamespace(logits=vocab)


def test_build_token_baseline_config_known_name() -> None:
    cfg = build_token_baseline_config("medusa")
    assert cfg.name == "medusa"
    assert cfg.chunk_size > 0


def test_run_token_baseline_generates_manifest() -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    decode_cfg = DecodeConfig(max_new_tokens=3, do_sample=False, temperature=1.0)
    baseline_cfg = TokenBaselineConfig(name="medusa", chunk_size=2, branch_factor=3)
    manifest, events = run_token_baseline(
        model,
        tokenizer,
        prompt="demo",
        decode_config=decode_cfg,
        baseline_config=baseline_cfg,
        max_new_tokens=3,
    )
    assert manifest["baseline"] == "medusa"
    assert manifest["streams"]["medusa"]["token_ids"]
    assert manifest["cadence_events"]
    assert events[-1]["stride_index"] >= 0
    assert events[-1]["stride_completed"] is True
