"""Unit tests for gate diagnostic and migration utilities."""

from __future__ import annotations

import math

import torch

from parallel_decoder_transformer.inference.gate_utils import (
    gate_entropy,
    log_gate_stats,
    migrate_scalar_gate_checkpoint,
)
from parallel_decoder_transformer.inference.snc_cross_attn import (
    SharedNotesCrossAttention,
    SharedNotesCrossAttentionConfig,
)


def test_gate_entropy_uniform() -> None:
    """Uniform gate values produce maximum entropy."""
    g = torch.full((8,), 0.5)
    e = gate_entropy(g)
    assert abs(e - math.log(8)) < 1e-4


def test_gate_entropy_degenerate() -> None:
    """One-hot gate values produce zero entropy.

    gate_entropy applies softmax internally, so pass extreme logits to
    approximate a degenerate distribution.
    """
    g = torch.tensor([100.0, 0.0, 0.0, 0.0])
    e = gate_entropy(g)
    assert e < 1e-4


def test_log_gate_stats_returns_per_head() -> None:
    """log_gate_stats returns H keys plus entropy."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=4, gate_mode="per_head"
    )
    layer = SharedNotesCrossAttention(config)
    stats = log_gate_stats(layer, prefix="snc")
    assert "snc/gate_head_0_mean" in stats
    assert "snc/gate_head_3_mean" in stats
    assert "snc/gate_entropy" in stats
    assert len([k for k in stats if k.startswith("snc/gate_head_")]) == 4
