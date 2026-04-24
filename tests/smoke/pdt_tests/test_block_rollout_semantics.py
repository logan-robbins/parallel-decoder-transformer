"""Training rollout helper semantics."""

from __future__ import annotations

import torch

from pdt.training.trainer import _visible_notes


def test_training_lag_reveals_block_zero_write_at_block_one():
    snapshots = [
        [torch.full((1, 4), 10.0), torch.full((1, 4), 11.0)],
        [torch.full((1, 4), 20.0), torch.full((1, 4), 21.0)],
        [torch.full((1, 4), 30.0), torch.full((1, 4), 31.0)],
    ]

    block0 = _visible_notes(snapshots, consumer=0, block_idx=0, lag=1, max_snapshots=8)
    block1 = _visible_notes(snapshots, consumer=0, block_idx=1, lag=1, max_snapshots=8)

    assert block0.shape == (1, 3, 4)
    assert block1.shape == (1, 6, 4)
    assert torch.equal(block0[0, :, 0], torch.tensor([10.0, 20.0, 30.0]))
    assert torch.equal(block1[0, :, 0], torch.tensor([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]))
