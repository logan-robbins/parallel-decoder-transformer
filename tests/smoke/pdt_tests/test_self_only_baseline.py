"""Contracts for the parameter-matched self-only SNC replacement."""

from __future__ import annotations

import inspect

import pytest
import torch

from pdt.baselines.self_only import ParameterMatchedSelfOnlyAttention
from pdt.config.schemas import SNCConfig
from pdt.sidecar.snc import SharedNotesCrossAttention


CONFIG = SNCConfig(hidden_size=8, notes_dim=4, num_heads=2, dropout=0.0)


def _inputs() -> dict[str, object]:
    return {
        "receiver_hidden_states": torch.randn(2, 3, 8),
        "own_prior_hidden_states": torch.randn(2, 4, 8),
        "own_prior_mask": torch.tensor([[True, True, True, False], [True, True, False, False]]),
        "query_positions": torch.tensor([[4, 5, 6], [3, 4, 5]]),
        "prior_positions": torch.tensor([[0, 1, 2, -1], [0, 1, -1, -1]]),
        "receiver_stream": "stream_0",
        "prior_stream": "stream_0",
    }


def _trainable_parameters(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad
    }


def test_trainable_parameter_count_and_attention_parameters_exactly_match_snc() -> None:
    snc = SharedNotesCrossAttention(CONFIG)
    self_only = ParameterMatchedSelfOnlyAttention(CONFIG)
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
    snc = SharedNotesCrossAttention(CONFIG)
    self_only = ParameterMatchedSelfOnlyAttention(CONFIG)
    with torch.no_grad():
        snc.o_proj.weight.normal_(std=0.1)
        self_only.load_state_dict(snc.state_dict(), strict=True)

    inputs = _inputs()
    selected = self_only.select_own_history_features(inputs["own_prior_hidden_states"])
    expected = snc(
        inputs["receiver_hidden_states"],
        selected,
        notes_mask=inputs["own_prior_mask"],
        force_gate=True,
    )
    actual = self_only(**inputs, force_gate=True)
    torch.testing.assert_close(actual, expected)


def test_api_cannot_accept_bus_notes_and_rejects_sibling_ownership() -> None:
    parameters = inspect.signature(ParameterMatchedSelfOnlyAttention.forward).parameters
    assert "notes" not in parameters
    assert "notes_mask" not in parameters
    assert "sibling" not in " ".join(parameters)

    module = ParameterMatchedSelfOnlyAttention(CONFIG)
    with pytest.raises(TypeError, match="unexpected keyword argument 'notes'"):
        module(**_inputs(), notes=torch.randn(2, 2, 4))

    sibling = _inputs()
    sibling["prior_stream"] = "stream_1"
    with pytest.raises(ValueError, match="rejects sibling history"):
        module(**sibling)


def test_output_attention_mask_shapes_and_empty_history() -> None:
    module = ParameterMatchedSelfOnlyAttention(CONFIG)
    with torch.no_grad():
        module.o_proj.weight.normal_(std=0.1)
    inputs = _inputs()
    delta, weights = module(**inputs, force_gate=True, return_attn_weights=True)

    assert delta.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 4)
    expanded_mask = inputs["own_prior_mask"][:, None, None, :].expand_as(weights)
    assert torch.equal(weights[~expanded_mask], torch.zeros_like(weights[~expanded_mask]))

    empty = {
        **inputs,
        "own_prior_hidden_states": torch.empty(2, 0, 8),
        "own_prior_mask": torch.empty(2, 0, dtype=torch.bool),
        "prior_positions": torch.empty(2, 0, dtype=torch.long),
    }
    empty_delta, empty_weights = module(**empty, return_attn_weights=True)
    assert torch.equal(empty_delta, torch.zeros_like(empty_delta))
    assert empty_weights.shape == (2, 2, 3, 0)


def test_gradients_reach_identical_attention_parameters_and_receiver_history() -> None:
    module = ParameterMatchedSelfOnlyAttention(CONFIG)
    with torch.no_grad():
        module.o_proj.weight.normal_(std=0.1)
    inputs = _inputs()
    receiver = inputs["receiver_hidden_states"].requires_grad_()
    own_prior = inputs["own_prior_hidden_states"].requires_grad_()
    inputs["receiver_hidden_states"] = receiver
    inputs["own_prior_hidden_states"] = own_prior

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
    inputs["prior_positions"][1, 1] = leaking_position
    with pytest.raises(ValueError, match="current/future leakage"):
        ParameterMatchedSelfOnlyAttention(CONFIG)(**inputs)


def test_invalid_history_masks_fail_fast() -> None:
    non_boolean = _inputs()
    non_boolean["own_prior_mask"] = non_boolean["own_prior_mask"].long()
    with pytest.raises(TypeError, match="torch.bool"):
        ParameterMatchedSelfOnlyAttention(CONFIG)(**non_boolean)

    non_contiguous = _inputs()
    non_contiguous["own_prior_mask"][0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="prefix-contiguous"):
        ParameterMatchedSelfOnlyAttention(CONFIG)(**non_contiguous)

    bad_padding = _inputs()
    bad_padding["prior_positions"][0, -1] = 0
    with pytest.raises(ValueError, match="sentinel value -1"):
        ParameterMatchedSelfOnlyAttention(CONFIG)(**bad_padding)
