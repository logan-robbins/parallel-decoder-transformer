from __future__ import annotations

import math

import pytest
import torch
from torch.nn.utils import parametrize

from parallel_decoder_transformer.models.stream_adapters import StreamAdapterConfig, StreamAdapters
from parallel_decoder_transformer.inference.snc_cross_attn import (
    SharedNotesCrossAttention,
    SharedNotesCrossAttentionConfig,
)
from parallel_decoder_transformer.inference import MultiStreamOrchestrator


def test_stream_adapters_apply_spectral_norm() -> None:
    config = StreamAdapterConfig(
        hidden_size=8,
        bottleneck_size=4,
        streams=("intro",),
        spectral_norm=True,
    )
    adapters = StreamAdapters(config)
    block = adapters.adapters["intro"]
    assert parametrize.is_parametrized(block.down, "weight")
    assert parametrize.is_parametrized(block.up, "weight")


def test_shared_notes_cross_attention_spectral_norm() -> None:
    cfg = SharedNotesCrossAttentionConfig(
        hidden_size=8,
        notes_dim=4,
        num_heads=2,
        spectral_norm=True,
    )
    attn = SharedNotesCrossAttention(cfg)
    assert parametrize.is_parametrized(attn.q_proj, "weight")
    assert parametrize.is_parametrized(attn.k_proj, "weight")
    assert parametrize.is_parametrized(attn.v_proj, "weight")
    assert parametrize.is_parametrized(attn.o_proj, "weight")


def test_safeguard_helper_functions() -> None:
    stddev = MultiStreamOrchestrator._stddev([0.0, 1.0])
    assert stddev == pytest.approx(0.5, rel=1e-6)

    switches = MultiStreamOrchestrator._count_plan_switches([None, 1, 1, 2, None, 3, 2])
    assert switches == 3

    ratio = MultiStreamOrchestrator._lipschitz_ratio(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
    )
    assert math.isclose(ratio, 2.0, rel_tol=1e-6)
