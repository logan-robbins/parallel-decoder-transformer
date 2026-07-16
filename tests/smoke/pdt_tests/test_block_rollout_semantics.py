"""Training rollout helper semantics."""

from __future__ import annotations

import torch

from pdt.training.trainer import _visible_notes


def test_training_lag_reveals_block_zero_write_at_block_one():
    snapshots = [
        [torch.full((1, 4), 10.0), torch.full((1, 4), 11.0), torch.full((1, 4), 12.0)],
        [torch.full((1, 4), 20.0), torch.full((1, 4), 21.0), torch.full((1, 4), 22.0)],
        [torch.full((1, 4), 30.0), torch.full((1, 4), 31.0), torch.full((1, 4), 32.0)],
    ]

    block0, mask0 = _visible_notes(
        snapshots, consumer=0, block_idx=0, lag=1, history_blocks=2
    )
    block1, mask1 = _visible_notes(
        snapshots, consumer=0, block_idx=1, lag=1, history_blocks=2
    )
    block2, mask2 = _visible_notes(
        snapshots, consumer=0, block_idx=2, lag=1, history_blocks=2
    )

    assert block0.shape == (1, 9, 4)
    assert block1.shape == (1, 9, 4)
    assert block2.shape == (1, 9, 4)
    assert mask0.tolist() == [[True, True, True, False, False, False, False, False, False]]
    assert mask1.tolist() == [[True, True, True, True, True, True, False, False, False]]
    assert torch.equal(
        block0[0, :, 0],
        torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert torch.equal(
        block1[0, :, 0],
        torch.tensor([10.0, 20.0, 30.0, 11.0, 21.0, 31.0, 0.0, 0.0, 0.0]),
    )
    assert torch.equal(
        block2[0, :, 0],
        torch.tensor([10.0, 20.0, 30.0, 12.0, 22.0, 32.0, 11.0, 21.0, 31.0]),
    )
    assert mask2.all()
