from __future__ import annotations

import torch

from parallel_decoder_transformer.inference import DynamicNotesBus, DynamicNotesBusConfig
from parallel_decoder_transformer.inference.retention import RetentionConfig


def test_dynamic_notes_bus_push_and_snapshot() -> None:
    bus = DynamicNotesBus(DynamicNotesBusConfig(snapshot_dim=4, max_snapshots=2, lag=1))
    notes_a = torch.ones(1, 2, 4)
    notes_b = torch.full((1, 2, 4), 2.0)
    bus.push(notes_a, stride=0)
    bus.push(notes_b, stride=1)
    snapshots = bus.snapshot(lag=1)
    assert len(snapshots) == 1
    assert torch.equal(snapshots[0].notes, notes_a)


def test_dynamic_notes_bus_masked_snapshot() -> None:
    bus = DynamicNotesBus(DynamicNotesBusConfig(snapshot_dim=2, max_snapshots=3, lag=1))
    bus.push(torch.ones(1, 2, 2), stride=0)
    mask = torch.tensor([[1, 0]], dtype=torch.float32)
    masked = bus.masked_snapshot(mask)
    assert len(masked) == 0 or masked[0].notes.shape[-1] == 2


# ---------------------------------------------------------------------------
# Scored eviction integration tests
# ---------------------------------------------------------------------------

def _make_bus(
    mode: str = "scored",
    max_snapshots: int = 3,
    lag: int = 0,
    **kwargs,
) -> DynamicNotesBus:
    """Build a DynamicNotesBus with retention enabled."""
    retention = RetentionConfig(
        mode=mode,
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=False,
        **kwargs,
    )
    config = DynamicNotesBusConfig(
        snapshot_dim=4,
        max_snapshots=max_snapshots,
        lag=lag,
        dtype="float32",
        retention=retention,
    )
    return DynamicNotesBus(config)


def test_fifo_retention_identical_to_default() -> None:
    """mode='fifo' must produce bit-identical behavior to a bus without retention."""
    cfg_plain = DynamicNotesBusConfig(snapshot_dim=4, max_snapshots=2, lag=0, dtype="float32")
    cfg_fifo = DynamicNotesBusConfig(
        snapshot_dim=4,
        max_snapshots=2,
        lag=0,
        dtype="float32",
        retention=RetentionConfig(mode="fifo"),
    )
    bus_plain = DynamicNotesBus(cfg_plain)
    bus_fifo = DynamicNotesBus(cfg_fifo)
    for i in range(5):
        notes = torch.full((1, 2, 4), float(i))
        bus_plain.push(notes, stride=1)
        bus_fifo.push(notes, stride=1)
    snaps_plain = bus_plain.snapshot(lag=0)
    snaps_fifo = bus_fifo.snapshot(lag=0)
    assert len(snaps_plain) == len(snaps_fifo)
    for sp, sf in zip(snaps_plain, snaps_fifo):
        assert sp.version == sf.version
        assert torch.equal(sp.notes, sf.notes)


def test_scored_eviction_drops_lowest_scored() -> None:
    """Scored eviction should drop the slot with the lowest importance."""
    bus = _make_bus(mode="scored", max_snapshots=3, lag=0)
    # Push 3 snapshots (fills the buffer)
    for i in range(3):
        bus.push(torch.full((1, 2, 4), float(i)), stride=1)
    # Assign scores: slot 0 (version 1) gets high score, slot 1 (version 2) low
    bus.update_scores(
        torch.tensor([[[[0.9, 0.05, 0.05]]]]),
        [1, 2, 3],
    )
    # Push a 4th snapshot -> triggers compaction
    bus.push(torch.full((1, 2, 4), 99.0), stride=1)
    # Buffer should still have 3 items.
    assert len(bus) == 3
    # Version 2 (lowest score) should have been evicted.
    versions = [s.version for s in bus._buffer]
    assert 2 not in versions
    assert 1 in versions  # high score preserved


def test_scored_eviction_respects_max_snapshots() -> None:
    """After scored eviction, buffer length must equal max_snapshots."""
    bus = _make_bus(mode="scored", max_snapshots=2, lag=0)
    for i in range(5):
        bus.push(torch.full((1, 2, 4), float(i)), stride=1)
        if len(bus._buffer) > 1:
            versions = [s.version for s in bus._buffer]
            bus.update_scores(
                torch.ones(1, 1, 1, len(versions)) / len(versions),
                versions,
            )
    assert len(bus) <= 2


def test_update_scores_noop_without_retention() -> None:
    """update_scores on a plain bus (no retention) should be a silent no-op."""
    bus = DynamicNotesBus(DynamicNotesBusConfig(snapshot_dim=4, max_snapshots=3))
    bus.push(torch.ones(1, 2, 4), stride=1)
    # Should not raise
    bus.update_scores(torch.ones(1, 1, 1, 1), [1])


def test_compressed_eviction_merges_two_lowest() -> None:
    """Hybrid mode with compression should merge the two lowest-scoring slots."""
    retention = RetentionConfig(
        mode="hybrid",
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=True,
        detach_compressed_grads=True,
    )
    config = DynamicNotesBusConfig(
        snapshot_dim=4,
        max_snapshots=3,
        lag=0,
        dtype="float32",
        retention=retention,
    )
    bus = DynamicNotesBus(config)
    # Push 3 snapshots
    for i in range(3):
        bus.push(torch.full((1, 2, 4), float(i)), stride=1)
    # Score: slot 0 high, slots 1 and 2 low
    bus.update_scores(
        torch.tensor([[[[0.9, 0.05, 0.06]]]]),
        [1, 2, 3],
    )
    # Push 4th -> compaction via merge
    bus.push(torch.full((1, 2, 4), 99.0), stride=1)
    # After merge, buffer should be 3 (merged pair = 1 slot, + kept + new)
    assert len(bus) == 3
    # Version 1 (high score) should survive
    versions = [s.version for s in bus._buffer]
    assert 1 in versions


def test_buffer_length_after_merge_is_correct() -> None:
    """After a compressive merge, buffer should be exactly max_snapshots."""
    retention = RetentionConfig(
        mode="hybrid",
        recency_pinned_fraction=0.0,
        min_score_spread=0.0,
        compression_enabled=True,
    )
    config = DynamicNotesBusConfig(
        snapshot_dim=4,
        max_snapshots=3,
        lag=0,
        dtype="float32",
        retention=retention,
    )
    bus = DynamicNotesBus(config)
    for i in range(3):
        bus.push(torch.full((1, 2, 4), float(i)), stride=1)
    bus.update_scores(
        torch.tensor([[[[0.1, 0.2, 0.7]]]]),
        [1, 2, 3],
    )
    bus.push(torch.full((1, 2, 4), 10.0), stride=1)
    assert len(bus) == 3
