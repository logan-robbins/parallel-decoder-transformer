from __future__ import annotations

import torch

from parallel_decoder_transformer.inference import DynamicNotesBus, DynamicNotesBusConfig


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
