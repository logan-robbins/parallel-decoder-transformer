"""Tests for retention scoring, eviction policy, and compressive merge."""

from __future__ import annotations

import torch

from parallel_decoder_transformer.inference.dnb_bus import Snapshot
from parallel_decoder_transformer.inference.retention import (
    RetentionConfig,
    RetentionPolicy,
    SnapshotScoreBank,
    compress_snapshots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(version: int, *, dim: int = 4, stride: int = 1, pin: bool = False) -> Snapshot:
    """Build a minimal Snapshot for testing."""
    meta: dict = {}
    if pin:
        meta["pin"] = True
    return Snapshot(
        version=version,
        stride=stride,
        notes=torch.randn(1, 2, dim),
        metadata=meta,
    )


def _attn_weights(K: int, *, per_slot: list[float] | None = None) -> torch.Tensor:
    """Build a fake attention weight tensor [1, 1, 1, K].

    If *per_slot* is given, sets each slot's weight to the specified value.
    """
    w = torch.zeros(1, 1, 1, K)
    if per_slot is not None:
        for i, val in enumerate(per_slot):
            if i < K:
                w[0, 0, 0, i] = val
    return w


# ---------------------------------------------------------------------------
# SnapshotScoreBank tests
# ---------------------------------------------------------------------------

def test_score_bank_initial_scores_at_floor() -> None:
    """Untracked snapshots should return score_floor."""
    cfg = RetentionConfig(mode="scored", score_floor=1e-6)
    bank = SnapshotScoreBank(cfg)
    snaps = [_snapshot(v) for v in [1, 2, 3]]
    scores = bank.scores_for(snaps)
    assert scores.shape == (3,)
    assert torch.allclose(scores, torch.tensor([1e-6, 1e-6, 1e-6]))


def test_score_bank_ema_update() -> None:
    """A single update should blend floor with the observed weight."""
    cfg = RetentionConfig(mode="scored", ema_decay=0.9, score_floor=0.0)
    bank = SnapshotScoreBank(cfg)
    # Slot 0 gets weight 1.0, slot 1 gets 0.0.
    bank.update(_attn_weights(2, per_slot=[1.0, 0.0]), [10, 20])
    scores = bank.scores_for([_snapshot(10), _snapshot(20)])
    # omega_10 = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
    # omega_20 = 0.9 * 0.0 + 0.1 * 0.0 = 0.0
    assert abs(scores[0].item() - 0.1) < 1e-6
    assert abs(scores[1].item() - 0.0) < 1e-6


def test_score_bank_ema_accumulates() -> None:
    """Repeated updates should accumulate via EMA."""
    cfg = RetentionConfig(mode="scored", ema_decay=0.5, score_floor=0.0)
    bank = SnapshotScoreBank(cfg)
    bank.update(_attn_weights(1, per_slot=[1.0]), [1])
    # omega = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    bank.update(_attn_weights(1, per_slot=[1.0]), [1])
    # omega = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    scores = bank.scores_for([_snapshot(1)])
    assert abs(scores[0].item() - 0.75) < 1e-6


def test_score_bank_reset_slot() -> None:
    """After reset, a version should return to floor."""
    cfg = RetentionConfig(mode="scored", score_floor=1e-6)
    bank = SnapshotScoreBank(cfg)
    bank.update(_attn_weights(1, per_slot=[1.0]), [5])
    bank.reset_slot(5)
    scores = bank.scores_for([_snapshot(5)])
    assert abs(scores[0].item() - 1e-6) < 1e-8


def test_score_bank_empty_input_noop() -> None:
    """Updating with empty tensors or empty versions should not crash."""
    cfg = RetentionConfig(mode="scored")
    bank = SnapshotScoreBank(cfg)
    bank.update(torch.zeros(0), [])
    bank.update(torch.zeros(1, 1, 1, 0), [])
    scores = bank.scores_for([])
    assert scores.numel() == 0


# ---------------------------------------------------------------------------
# RetentionPolicy tests
# ---------------------------------------------------------------------------

def test_fifo_mode_evicts_first() -> None:
    """FIFO mode should always select index 0 with no merge."""
    cfg = RetentionConfig(mode="fifo")
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4, 5]]
    scores = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.2])
    evict, merge = policy.select_eviction(buffer, scores, lag=1)
    assert evict == 0
    assert merge is None


def test_scored_evicts_lowest() -> None:
    """Scored mode should evict the slot with the lowest importance."""
    cfg = RetentionConfig(
        mode="scored",
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=False,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4, 5]]
    # Slot 2 (version 3) has the lowest score.
    scores = torch.tensor([0.5, 0.4, 0.1, 0.8, 0.9])
    evict, merge = policy.select_eviction(buffer, scores, lag=0)
    assert evict == 2
    assert merge is None


def test_recency_pinned_fraction_protects_tail() -> None:
    """The most-recent K_recent slots must be protected."""
    cfg = RetentionConfig(
        mode="scored",
        recency_pinned_fraction=0.5,
        min_score_spread=0.0,
        compression_enabled=False,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4]]
    # The tail 2 slots (indices 2, 3) are pinned.  Index 1 is lowest among candidates.
    scores = torch.tensor([0.5, 0.1, 0.01, 0.01])
    evict, merge = policy.select_eviction(buffer, scores, lag=0)
    assert evict == 1  # cannot evict 2 or 3 (recency-pinned)
    assert merge is None


def test_lag_slots_protected() -> None:
    """Slots inside the lag window must not be evicted."""
    cfg = RetentionConfig(
        mode="scored",
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=False,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4, 5]]
    # With lag=2, slots 3 and 4 are protected.
    # Slot 4 has the lowest score but is protected.
    scores = torch.tensor([0.5, 0.3, 0.4, 0.1, 0.05])
    evict, merge = policy.select_eviction(buffer, scores, lag=2)
    assert evict == 1  # lowest among unprotected
    assert merge is None


def test_pinned_metadata_protects_slot() -> None:
    """A snapshot with metadata["pin"]=True must be excluded from candidates."""
    cfg = RetentionConfig(
        mode="scored",
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=False,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    # With 5 slots and recency_pinned_fraction=0.0, K_recent = max(1, ceil(0))=1
    # so index 4 is recency-pinned.  Index 0 is pin-protected.
    # Candidates: [1, 2, 3]. Lowest among those is index 2 (score=0.3).
    buffer = [
        _snapshot(1, pin=True),   # pinned, lowest score
        _snapshot(2),
        _snapshot(3),
        _snapshot(4),
        _snapshot(5),
    ]
    scores = torch.tensor([0.01, 0.5, 0.3, 0.4, 0.2])
    evict, merge = policy.select_eviction(buffer, scores, lag=0)
    assert evict != 0  # pinned slot must not be evicted
    assert evict == 2  # index 2 has score 0.3, lowest non-pinned non-recency candidate


def test_all_protected_falls_back_to_fifo() -> None:
    """When all slots are protected, should fall back to FIFO (evict index 0)."""
    cfg = RetentionConfig(
        mode="scored",
        recency_pinned_fraction=1.0,  # all recency-pinned
        min_score_spread=0.0,
        compression_enabled=False,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4]]
    scores = torch.tensor([0.5, 0.1, 0.3, 0.4])
    evict, merge = policy.select_eviction(buffer, scores, lag=0)
    assert evict == 0
    assert merge is None


def test_score_spread_below_threshold_falls_back() -> None:
    """When score spread < min_score_spread, fall back to FIFO of candidates."""
    cfg = RetentionConfig(
        mode="scored",
        recency_pinned_fraction=0.0,
        min_score_spread=0.1,
        compression_enabled=False,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4, 5]]
    # All scores very close -- spread = 0.02 < 0.1
    scores = torch.tensor([0.50, 0.51, 0.50, 0.52, 0.51])
    evict, merge = policy.select_eviction(buffer, scores, lag=0)
    assert evict == 0  # first candidate (FIFO fallback)
    assert merge is None


# ---------------------------------------------------------------------------
# Compression tests (Phase 3)
# ---------------------------------------------------------------------------

def test_hybrid_selects_merge_pair() -> None:
    """Hybrid mode with compression should return both evict and merge indices."""
    cfg = RetentionConfig(
        mode="hybrid",
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=True,
    )
    policy = RetentionPolicy(cfg, max_snapshots=4)
    buffer = [_snapshot(v) for v in [1, 2, 3, 4, 5]]
    scores = torch.tensor([0.5, 0.1, 0.2, 0.8, 0.9])
    evict, merge = policy.select_eviction(buffer, scores, lag=0)
    # Lowest is index 1 (0.1), second lowest is index 2 (0.2)
    assert evict == 1
    assert merge == 2


def test_compress_snapshots_weighted_average() -> None:
    """Merged notes should be the importance-weighted average."""
    a = Snapshot(version=3, stride=2, notes=torch.tensor([[[1.0, 0.0]]]), metadata={})
    b = Snapshot(version=5, stride=3, notes=torch.tensor([[[0.0, 1.0]]]), metadata={})
    merged = compress_snapshots(a, b, score_a=0.25, score_b=0.75)
    expected = 0.25 * torch.tensor([[[1.0, 0.0]]]) + 0.75 * torch.tensor([[[0.0, 1.0]]])
    assert torch.allclose(merged.notes, expected, atol=1e-5)


def test_compression_version_max() -> None:
    """Merged snapshot version should be max of the two inputs."""
    a = Snapshot(version=3, stride=1, notes=torch.zeros(1, 1, 2), metadata={})
    b = Snapshot(version=7, stride=1, notes=torch.zeros(1, 1, 2), metadata={})
    merged = compress_snapshots(a, b, 0.5, 0.5)
    assert merged.version == 7


def test_compression_stride_sum() -> None:
    """Merged stride should be the sum of both strides."""
    a = Snapshot(version=1, stride=3, notes=torch.zeros(1, 1, 2), metadata={})
    b = Snapshot(version=2, stride=5, notes=torch.zeros(1, 1, 2), metadata={})
    merged = compress_snapshots(a, b, 0.5, 0.5)
    assert merged.stride == 8


def test_compression_detach_default() -> None:
    """By default, merged notes should be detached from the graph."""
    a = Snapshot(version=1, stride=1, notes=torch.randn(1, 1, 4, requires_grad=True), metadata={})
    b = Snapshot(version=2, stride=1, notes=torch.randn(1, 1, 4, requires_grad=True), metadata={})
    merged = compress_snapshots(a, b, 0.5, 0.5, detach=True)
    assert not merged.notes.requires_grad


def test_compression_gradient_flow() -> None:
    """With detach=False, gradients should flow through the merged tensor."""
    notes_a = torch.randn(1, 1, 4, requires_grad=True)
    notes_b = torch.randn(1, 1, 4, requires_grad=True)
    a = Snapshot(version=1, stride=1, notes=notes_a, metadata={})
    b = Snapshot(version=2, stride=1, notes=notes_b, metadata={})
    merged = compress_snapshots(a, b, 0.3, 0.7, detach=False)
    loss = merged.notes.sum()
    loss.backward()
    assert notes_a.grad is not None
    assert notes_b.grad is not None


def test_compression_zero_scores() -> None:
    """When both scores are zero, should use equal weighting (0.5/0.5)."""
    a = Snapshot(version=1, stride=1, notes=torch.tensor([[[2.0, 0.0]]]), metadata={})
    b = Snapshot(version=2, stride=1, notes=torch.tensor([[[0.0, 4.0]]]), metadata={})
    merged = compress_snapshots(a, b, 0.0, 0.0)
    expected = torch.tensor([[[1.0, 2.0]]])
    assert torch.allclose(merged.notes, expected, atol=1e-5)


def test_compression_metadata_cleared() -> None:
    """Merged snapshot should have empty metadata."""
    a = Snapshot(version=1, stride=1, notes=torch.zeros(1, 1, 2), metadata={"pin": True})
    b = Snapshot(version=2, stride=1, notes=torch.zeros(1, 1, 2), metadata={"pin": True})
    merged = compress_snapshots(a, b, 0.5, 0.5)
    assert merged.metadata == {}


# ---------------------------------------------------------------------------
# Version monotonicity
# ---------------------------------------------------------------------------

def test_version_monotonicity_after_merge() -> None:
    """After a compressive merge, the merged version >= all evicted versions."""
    a = _snapshot(10)
    b = _snapshot(15)
    merged = compress_snapshots(a, b, 0.5, 0.5)
    assert merged.version >= a.version
    assert merged.version >= b.version
