"""Dynamic-only bus, packed windows, and finite-message contracts."""

from __future__ import annotations

import pytest
import torch

from pdt.config.schemas import NotesBusConfig
from pdt.runtime.dnb_bus import DynamicNotesBus
from pdt.runtime.counterfactuals import apply_plan_intervention
from pdt.runtime.window import read_notes_history
from pdt.sidecar.product_vq import ProductVectorQuantizer
from pdt.training.trainer import _dynamic_window


def _bus() -> DynamicNotesBus:
    return DynamicNotesBus(
        NotesBusConfig(
            snapshot_dim=8,
            lag=1,
            dtype="float32",
            num_codebooks=4,
            codes_per_codebook=256,
            history_blocks=16,
        ),
        producers=("stream_0", "stream_1", "stream_2"),
        device=torch.device("cpu"),
        codec=ProductVectorQuantizer(
            width=8,
            num_codebooks=4,
            codes_per_codebook=256,
        ),
    )


def test_bus_accepts_only_finite_dynamic_writes_and_enforces_delay() -> None:
    bus = _bus()
    assert len(bus) == 0
    assert not hasattr(bus, "seed_anchor")
    with pytest.raises(ValueError, match="tuple length"):
        bus.publish(
            "stream_0",
            published_block=0,
            stride=32,
            code_indices=(1, 2, 3),
        )
    snapshot = bus.publish(
        "stream_0",
        published_block=0,
        stride=32,
        code_indices=(1, 2, 3, 4),
    )
    assert snapshot.kind == "dynamic"
    assert bus.delivered_updates(consumer_block=0) == ()
    assert bus.delivered_updates(consumer_block=1) == (snapshot,)


def test_runtime_window_is_age_major_dynamic_only() -> None:
    bus = _bus()
    first = bus.publish(
        "stream_0",
        published_block=0,
        stride=32,
        code_indices=(1, 2, 3, 4),
    )
    second = bus.publish(
        "stream_2",
        published_block=1,
        stride=64,
        code_indices=(5, 6, 7, 8),
    )
    window = read_notes_history(
        bus.delivered_updates(consumer_block=2),
        producers=bus.producers,
        consumer_block=2,
        notes_dim=8,
        history_blocks=2,
    )
    assert window.producers == (
        "stream_0",
        "stream_1",
        "stream_2",
        "stream_0",
        "stream_1",
        "stream_2",
    )
    assert window.mask.tolist() == [[False, False, True, True, False, False]]
    torch.testing.assert_close(window.notes[0, 2], second.notes)
    torch.testing.assert_close(window.notes[0, 3], first.notes)
    assert window.lags.tolist() == [1, 1, 1, 2, 2, 2]


def test_training_window_matches_three_receiver_physical_frontier() -> None:
    notes = [
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        torch.arange(12, 24, dtype=torch.float32).reshape(1, 3, 4),
    ]
    validity = [
        torch.tensor([[True, True, True]]),
        torch.tensor([[True, False, True]]),
    ]
    packed, mask, producers, lags = _dynamic_window(
        notes,
        validity,
        consumer_block=2,
        producers=3,
        history_blocks=2,
        notes_dim=4,
        device=torch.device("cpu"),
    )
    assert packed.shape == (3, 6, 4)
    assert mask.shape == (3, 6)
    assert producers.tolist() == [[0, 1, 2, 0, 1, 2]] * 3
    assert lags.tolist() == [[1, 1, 1, 2, 2, 2]] * 3
    torch.testing.assert_close(packed[0], packed[2])


def test_physical_lane_plan_swap_moves_nodes_and_validity_together() -> None:
    nodes = torch.tensor(
        [[[[0.0]], [[1.0]], [[2.0]]]],
    )
    mask = torch.tensor([[[True], [False], [True]]])
    swapped_nodes = apply_plan_intervention(
        nodes,
        mode="lane_swap",
        lane_pair=(0, 1),
    )
    swapped_mask = apply_plan_intervention(
        mask,
        mode="lane_swap",
        lane_pair=(0, 1),
    )
    assert swapped_nodes.flatten().tolist() == [1.0, 0.0, 2.0]
    assert swapped_mask.flatten().tolist() == [False, True, True]
