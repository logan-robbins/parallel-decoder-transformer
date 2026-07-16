"""Parameter-matched self-only replacement for SNC.

The module inherits the complete :class:`SharedNotesCrossAttention` attention
path, so its trainable parameters and q/k/v/o computation are identical to
SNC.  Its public forward API accepts only receiver hidden state and explicitly
owned receiver history; there is no notes-bus or sibling-memory argument.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from pdt.config.schemas import SNCConfig
from pdt.sidecar.snc import SharedNotesCrossAttention


__all__ = ["ParameterMatchedSelfOnlyAttention"]


class ParameterMatchedSelfOnlyAttention(SharedNotesCrossAttention):
    """Run the exact SNC attention path over receiver-only causal history.

    ``own_prior_hidden_states`` must contain states produced by the same stream
    as ``receiver_hidden_states`` and strictly before every query position.
    A deterministic feature selection maps ``hidden_size`` to ``notes_dim``;
    it has no parameters and is excluded from the state dict.
    """

    def __init__(self, config: SNCConfig, *, gating_init: float = -4.0) -> None:
        if config.notes_dim > config.hidden_size:
            raise ValueError(
                "The parameter-free self-only projection requires notes_dim <= hidden_size; "
                f"got notes_dim={config.notes_dim}, hidden_size={config.hidden_size}."
            )
        super().__init__(config, gating_init=gating_init)
        feature_indices = torch.div(
            torch.arange(config.notes_dim, dtype=torch.long) * config.hidden_size,
            config.notes_dim,
            rounding_mode="floor",
        )
        self.register_buffer("_self_feature_indices", feature_indices, persistent=False)

    def select_own_history_features(self, own_prior_hidden_states: torch.Tensor) -> torch.Tensor:
        """Map receiver history to note width with deterministic feature selection."""

        if own_prior_hidden_states.ndim != 3:
            raise ValueError(
                "own_prior_hidden_states must have shape [batch, prior_tokens, hidden_size]."
            )
        if own_prior_hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(
                "own_prior_hidden_states width must equal SNC hidden_size: "
                f"expected {self.config.hidden_size}, got {own_prior_hidden_states.size(-1)}."
            )
        feature_indices = self.get_buffer("_self_feature_indices")
        return own_prior_hidden_states.index_select(-1, feature_indices)

    def forward(  # type: ignore[override]
        self,
        receiver_hidden_states: torch.Tensor,
        own_prior_hidden_states: torch.Tensor,
        *,
        own_prior_mask: torch.Tensor,
        query_positions: torch.Tensor,
        prior_positions: torch.Tensor,
        receiver_stream: str,
        prior_stream: str,
        force_gate: Optional[Union[torch.Tensor, bool]] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return an SNC-shaped delta using receiver-owned prior states only."""

        _validate_self_only_inputs(
            receiver_hidden_states=receiver_hidden_states,
            own_prior_hidden_states=own_prior_hidden_states,
            own_prior_mask=own_prior_mask,
            query_positions=query_positions,
            prior_positions=prior_positions,
            receiver_stream=receiver_stream,
            prior_stream=prior_stream,
            hidden_size=self.config.hidden_size,
        )
        self_memory = self.select_own_history_features(own_prior_hidden_states)
        return super().forward(
            receiver_hidden_states,
            self_memory,
            notes_mask=own_prior_mask,
            force_gate=force_gate,
            return_attn_weights=return_attn_weights,
        )


def _validate_self_only_inputs(
    *,
    receiver_hidden_states: torch.Tensor,
    own_prior_hidden_states: torch.Tensor,
    own_prior_mask: torch.Tensor,
    query_positions: torch.Tensor,
    prior_positions: torch.Tensor,
    receiver_stream: str,
    prior_stream: str,
    hidden_size: int,
) -> None:
    if receiver_hidden_states.ndim != 3:
        raise ValueError(
            "receiver_hidden_states must have shape [batch, query_tokens, hidden_size]."
        )
    if own_prior_hidden_states.ndim != 3:
        raise ValueError(
            "own_prior_hidden_states must have shape [batch, prior_tokens, hidden_size]."
        )
    batch, queries, receiver_width = receiver_hidden_states.shape
    prior_batch, prior_tokens, prior_width = own_prior_hidden_states.shape
    if batch == 0 or queries == 0:
        raise ValueError("receiver_hidden_states must contain a non-empty batch and query axis.")
    if prior_batch != batch:
        raise ValueError("receiver and prior history batch sizes must match.")
    if receiver_width != hidden_size or prior_width != hidden_size:
        raise ValueError(
            f"receiver and prior hidden widths must both equal hidden_size={hidden_size}."
        )
    if receiver_hidden_states.device != own_prior_hidden_states.device:
        raise ValueError("receiver and prior hidden states must be on the same device.")
    if receiver_hidden_states.dtype != own_prior_hidden_states.dtype:
        raise TypeError("receiver and prior hidden states must have the same dtype.")
    if not receiver_hidden_states.is_floating_point():
        raise TypeError("receiver and prior hidden states must use a floating-point dtype.")

    expected_prior_shape = (batch, prior_tokens)
    expected_query_shape = (batch, queries)
    if own_prior_mask.shape != expected_prior_shape:
        raise ValueError(
            f"own_prior_mask must have shape {expected_prior_shape}, "
            f"got {tuple(own_prior_mask.shape)}."
        )
    if prior_positions.shape != expected_prior_shape:
        raise ValueError(
            f"prior_positions must have shape {expected_prior_shape}, "
            f"got {tuple(prior_positions.shape)}."
        )
    if query_positions.shape != expected_query_shape:
        raise ValueError(
            f"query_positions must have shape {expected_query_shape}, "
            f"got {tuple(query_positions.shape)}."
        )
    for name, tensor in (
        ("own_prior_mask", own_prior_mask),
        ("prior_positions", prior_positions),
        ("query_positions", query_positions),
    ):
        if tensor.device != receiver_hidden_states.device:
            raise ValueError(
                f"{name} must be on {receiver_hidden_states.device}, got {tensor.device}."
            )
    if own_prior_mask.dtype != torch.bool:
        raise TypeError("own_prior_mask must have dtype torch.bool.")
    for name, positions in (
        ("query_positions", query_positions),
        ("prior_positions", prior_positions),
    ):
        if positions.dtype == torch.bool or positions.is_floating_point() or positions.is_complex():
            raise TypeError(f"{name} must have an integer dtype other than bool.")

    receiver = _stream_id(receiver_stream, "receiver_stream")
    prior = _stream_id(prior_stream, "prior_stream")
    if receiver != prior:
        raise ValueError(
            "self-only attention rejects sibling history: "
            f"receiver_stream={receiver!r}, prior_stream={prior!r}."
        )

    if bool((query_positions < 0).any()):
        raise ValueError("query_positions must be non-negative.")
    if queries > 1 and bool((query_positions[:, 1:] <= query_positions[:, :-1]).any()):
        raise ValueError("query_positions must be strictly increasing within every batch row.")

    if prior_tokens == 0:
        return
    if bool((own_prior_mask[:, 1:] & ~own_prior_mask[:, :-1]).any()):
        raise ValueError("own_prior_mask must be prefix-contiguous within every batch row.")
    if bool((~own_prior_mask).all(dim=1).any()):
        raise ValueError(
            "Non-empty own_prior_hidden_states require at least one active prior token per row."
        )
    if bool((prior_positions[~own_prior_mask] != -1).any()):
        raise ValueError("Masked prior_positions must use the sentinel value -1.")
    active_prior_positions = prior_positions.masked_fill(~own_prior_mask, -1)
    if bool((active_prior_positions[own_prior_mask] < 0).any()):
        raise ValueError("Active prior_positions must be non-negative.")
    if prior_tokens > 1:
        adjacent_active = own_prior_mask[:, 1:] & own_prior_mask[:, :-1]
        non_increasing = prior_positions[:, 1:] <= prior_positions[:, :-1]
        if bool((adjacent_active & non_increasing).any()):
            raise ValueError(
                "Active prior_positions must be strictly increasing within every batch row."
            )
    current_or_future = own_prior_mask & (prior_positions >= query_positions[:, :1])
    if bool(current_or_future.any()):
        raise ValueError(
            "Self-only history contains current/future leakage: every active prior position "
            "must be strictly earlier than the first query position."
        )


def _stream_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip().lower()
