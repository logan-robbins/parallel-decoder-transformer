"""Contracts for the parameter-matched self-only SNC replacement."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from pdt.baselines.self_only import ParameterMatchedSelfOnlyAttention, SelfOnlyMemory
from pdt.config.schemas import SNCConfig
from pdt.sidecar.snc import SharedNotesCrossAttention


CONFIG = SNCConfig(hidden_size=8, notes_dim=4, num_heads=2, dropout=0.0)


def _inputs() -> dict[str, object]:
    memory = SelfOnlyMemory(
        hidden_states=torch.randn(2, 4, 8),
        mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
        positions=torch.tensor([[0, 1, 2, -1], [0, 1, -1, -1]]),
        slot_ids=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
        kind_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        lags=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        owner_streams=("stream_0", "stream_1"),
    )
    return {
        "receiver_hidden_states": torch.randn(2, 3, 8),
        "memory": memory,
        "query_positions": torch.tensor([[4, 5, 6], [3, 4, 5]]),
        "receiver_streams": ("stream_0", "stream_1"),
    }


def _trainable_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad
    }


def test_trainable_parameter_count_and_attention_parameters_exactly_match_snc() -> None:
    snc = SharedNotesCrossAttention(CONFIG, num_producers=2)
    self_only = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    snc_parameters = _trainable_parameters(snc)
    self_parameters = _trainable_parameters(self_only)

    assert snc_parameters.keys() == self_parameters.keys()
    assert {name: tuple(parameter.shape) for name, parameter in snc_parameters.items()} == {
        name: tuple(parameter.shape) for name, parameter in self_parameters.items()
    }
    assert sum(parameter.numel() for parameter in snc_parameters.values()) == sum(
        parameter.numel() for parameter in self_parameters.values()
    )
    assert "_self_feature_indices" not in self_parameters


def test_self_only_uses_exact_snc_attention_after_parameter_free_selection() -> None:
    torch.manual_seed(7)
    snc = SharedNotesCrossAttention(CONFIG, num_producers=2)
    self_only = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    with torch.no_grad():
        snc.o_proj.weight.normal_(std=0.1)
        self_only.load_state_dict(snc.state_dict(), strict=True)

    inputs = _inputs()
    memory = inputs["memory"]
    selected = self_only.select_own_history_features(memory.hidden_states)
    expected = snc(
        inputs["receiver_hidden_states"],
        selected,
        notes_mask=memory.mask,
        producer_ids=memory.slot_ids,
        kind_ids=memory.kind_ids,
        lags=memory.lags,
        force_gate=True,
    )
    actual = self_only(**inputs, force_gate=True)
    torch.testing.assert_close(actual, expected)


def test_api_cannot_accept_bus_notes_and_rejects_sibling_ownership() -> None:
    parameters = inspect.signature(ParameterMatchedSelfOnlyAttention.forward).parameters
    assert "notes" not in parameters
    assert "notes_mask" not in parameters
    assert "sibling" not in " ".join(parameters)

    module = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    with pytest.raises(TypeError, match="unexpected keyword argument 'notes'"):
        module(**_inputs(), notes=torch.randn(2, 2, 4))

    sibling = _inputs()
    sibling["memory"] = replace(
        sibling["memory"],
        owner_streams=("stream_1", "stream_0"),
    )
    with pytest.raises(ValueError, match="rejects sibling history"):
        module(**sibling)


def test_output_attention_mask_shapes_and_empty_history() -> None:
    module = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    with torch.no_grad():
        module.o_proj.weight.normal_(std=0.1)
    inputs = _inputs()
    delta, weights = module(**inputs, force_gate=True, return_attn_weights=True)

    assert delta.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 4)
    expanded_mask = inputs["memory"].mask[:, None, None, :].expand_as(weights)
    assert torch.equal(weights[~expanded_mask], torch.zeros_like(weights[~expanded_mask]))

    empty = {
        **inputs,
        "memory": SelfOnlyMemory(
            hidden_states=torch.empty(2, 0, 8),
            mask=torch.empty(2, 0, dtype=torch.bool),
            positions=torch.empty(2, 0, dtype=torch.long),
            slot_ids=torch.empty(2, 0, dtype=torch.long),
            kind_ids=torch.empty(2, 0, dtype=torch.long),
            lags=torch.empty(2, 0, dtype=torch.long),
            owner_streams=("stream_0", "stream_1"),
        ),
    }
    empty_delta, empty_weights = module(**empty, return_attn_weights=True)
    assert torch.equal(empty_delta, torch.zeros_like(empty_delta))
    assert empty_weights.shape == (2, 2, 3, 0)


def test_gradients_reach_identical_attention_parameters_and_receiver_history() -> None:
    module = ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)
    with torch.no_grad():
        module.o_proj.weight.normal_(std=0.1)
    inputs = _inputs()
    receiver = inputs["receiver_hidden_states"].requires_grad_()
    memory = inputs["memory"]
    own_prior = memory.hidden_states.requires_grad_()
    inputs["receiver_hidden_states"] = receiver
    inputs["memory"] = replace(memory, hidden_states=own_prior)

    loss = module(**inputs).square().mean()
    loss.backward()

    assert receiver.grad is not None and bool(torch.isfinite(receiver.grad).all())
    assert own_prior.grad is not None and bool(torch.isfinite(own_prior.grad).all())
    assert float(receiver.grad.abs().sum()) > 0
    assert float(own_prior.grad.abs().sum()) > 0
    for name, parameter in _trainable_parameters(module).items():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert bool(torch.isfinite(parameter.grad).all()), f"non-finite gradient for {name}"


@pytest.mark.parametrize("leaking_position", [3, 4])
def test_current_or_future_history_fails_fast(leaking_position: int) -> None:
    inputs = _inputs()
    inputs["memory"].positions[1, 1] = leaking_position
    with pytest.raises(ValueError, match="current/future leakage"):
        ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)(**inputs)


def test_invalid_history_masks_fail_fast() -> None:
    non_boolean = _inputs()
    memory = non_boolean["memory"]
    non_boolean["memory"] = replace(memory, mask=memory.mask.long())
    with pytest.raises(TypeError, match="torch.bool"):
        ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)(**non_boolean)

    non_contiguous = _inputs()
    non_contiguous["memory"].mask[0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="prefix-contiguous"):
        ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)(**non_contiguous)

    bad_padding = _inputs()
    bad_padding["memory"].positions[0, -1] = 0
    with pytest.raises(ValueError, match="sentinel value -1"):
        ParameterMatchedSelfOnlyAttention(CONFIG, num_producers=2)(**bad_padding)
