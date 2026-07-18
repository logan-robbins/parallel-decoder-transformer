"""Parameter-matched self-only dynamic-memory control."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pdt.baselines.self_only import (
    ParameterMatchedSelfOnlyAttention,
    SelfOnlyMemory,
    build_self_only_memory,
)
from pdt.config.schemas import SNCConfig
from pdt.sidecar.snc import SharedNotesCrossAttention


CONFIG = SNCConfig(
    hidden_size=8,
    notes_dim=4,
    attention_width=8,
    num_heads=2,
    dropout=0.0,
)


def _inputs() -> dict[str, object]:
    return {
        "receiver_hidden_states": torch.randn(2, 3, 8),
        "memory": SelfOnlyMemory(
            hidden_states=torch.randn(2, 3, 8),
            mask=torch.tensor([[True, True, True], [True, True, False]]),
            positions=torch.tensor([[0, 1, 2], [0, 1, -1]]),
            slot_ids=torch.tensor([[0, 0, 0], [1, 1, 1]]),
            kind_ids=torch.ones(2, 3, dtype=torch.long),
            lags=torch.tensor([[3, 2, 1], [3, 2, 1]]),
            owner_streams=("stream_0", "stream_1"),
        ),
        "query_positions": torch.tensor([[4, 5, 6], [3, 4, 5]]),
        "receiver_streams": ("stream_0", "stream_1"),
    }


def test_parameterization_exactly_matches_bus_snc() -> None:
    bus = SharedNotesCrossAttention(CONFIG, num_producers=2)
    control = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    assert {
        name: tuple(parameter.shape)
        for name, parameter in bus.named_parameters()
    } == {
        name: tuple(parameter.shape)
        for name, parameter in control.named_parameters()
    }


def test_control_reads_only_receiver_owned_history() -> None:
    torch.manual_seed(7)
    bus = SharedNotesCrossAttention(CONFIG, num_producers=2)
    control = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    with torch.no_grad():
        bus.o_proj.weight.normal_(std=0.1)
        control.load_state_dict(bus.state_dict(), strict=True)
    inputs = _inputs()
    memory = inputs["memory"]
    selected = control.select_own_history_features(memory.hidden_states)
    expected = bus(
        inputs["receiver_hidden_states"],
        selected,
        notes_mask=memory.mask,
        producer_ids=memory.slot_ids,
        kind_ids=memory.kind_ids,
        lags=memory.lags,
        force_gate=True,
    )
    actual = control(**inputs, force_gate=True)
    torch.testing.assert_close(actual, expected)

    sibling = _inputs()
    sibling["memory"] = replace(
        sibling["memory"],
        owner_streams=("stream_1", "stream_0"),
    )
    with pytest.raises(ValueError, match="rejects sibling history"):
        control(**sibling)


def test_empty_dynamic_history_returns_zero_delta() -> None:
    module = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    with torch.no_grad():
        module.o_proj.weight.normal_(std=0.1)
    inputs = _inputs()
    inputs["memory"] = replace(
        inputs["memory"],
        hidden_states=torch.empty(2, 0, 8),
        mask=torch.empty(2, 0, dtype=torch.bool),
        positions=torch.empty(2, 0, dtype=torch.long),
        slot_ids=torch.empty(2, 0, dtype=torch.long),
        kind_ids=torch.empty(2, 0, dtype=torch.long),
        lags=torch.empty(2, 0, dtype=torch.long),
    )
    delta, weights = module(**inputs, force_gate=True, return_attn_weights=True)
    assert delta.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 0)
    assert torch.count_nonzero(delta) == 0


def test_self_only_training_window_is_causal_and_horizon_bounded() -> None:
    states = [torch.randn(1, 3, 8) for _ in range(4)]
    validity = [torch.ones(1, 3, dtype=torch.bool) for _ in range(4)]
    positions = [
        torch.tensor([[31 + 32 * block] * 3], dtype=torch.long)
        for block in range(4)
    ]
    memory = build_self_only_memory(
        states,
        validity,
        positions,
        consumer_block=4,
        lanes=3,
        history_blocks=2,
        hidden_size=8,
        streams=("stream_0", "stream_1", "stream_2"),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert memory.hidden_states.shape == (3, 2, 8)
    assert memory.positions.tolist() == [[95, 127]] * 3
    assert memory.lags.tolist() == [[2, 1]] * 3
    assert memory.owner_streams == ("stream_0", "stream_1", "stream_2")


def test_current_or_future_self_history_fails_fast() -> None:
    inputs = _inputs()
    inputs["memory"].positions[1, 1] = 3
    with pytest.raises(ValueError, match="current/future leakage"):
        ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)(**inputs)
