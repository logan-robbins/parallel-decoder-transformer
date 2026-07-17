"""Parameter-matched self-only replacement for SNC.

The module inherits the complete :class:`SharedNotesCrossAttention` attention
path, so its trainable parameters and q/k/v/o computation are identical to
SNC.  Its public forward API accepts only receiver hidden state and explicitly
owned receiver history; there is no notes-bus or sibling-memory argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from pdt.config.schemas import SNCConfig
from pdt.sidecar.snc import SharedNotesCrossAttention


__all__ = [
    "ParameterMatchedSelfOnlyAttention",
    "SelfOnlyMemory",
    "build_self_only_memory",
]


@dataclass(frozen=True, slots=True)
class SelfOnlyMemory:
    """One addressed window containing only receiver-owned prior states.

    The slot/kind/lag tensors are parameter-free headers matching the bus SNC
    interface. ``owner_streams`` is row-addressed provenance and must equal the
    receiver rows at every call; no sibling-memory field exists.
    """

    hidden_states: torch.Tensor  # (B, S, hidden_size)
    mask: torch.Tensor  # (B, S) bool
    positions: torch.Tensor  # (B, S) long; -1 in masked slots
    slot_ids: torch.Tensor  # (B, S) long in [0, K)
    kind_ids: torch.Tensor  # (B, S) long; 0 prompt, 1 dynamic
    lags: torch.Tensor  # (B, S) long
    owner_streams: Tuple[str, ...]


def build_self_only_memory(
    states_by_block: list[torch.Tensor],
    validity_by_block: list[torch.Tensor],
    positions_by_block: list[torch.Tensor],
    *,
    consumer_block: int,
    lanes: int,
    history_blocks: int,
    hidden_size: int,
    streams: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> SelfOnlyMemory:
    """Build the one canonical bounded receiver-owned block history.

    Training and free generation both retain exactly one final hidden state
    for each completed block and physical lane. A consumer in block ``m`` may
    read only completed blocks strictly before ``m``.
    """

    if not (
        len(states_by_block)
        == len(validity_by_block)
        == len(positions_by_block)
        == consumer_block
    ):
        raise ValueError(
            "Self-only block states, validity, positions, and consumer index must align."
        )
    if len(streams) != lanes:
        raise ValueError("Self-only stream addresses must match the physical lane count.")
    start = max(0, consumer_block - history_blocks)
    states = states_by_block[start:consumer_block]
    validity = validity_by_block[start:consumer_block]
    positions = positions_by_block[start:consumer_block]
    if not states:
        return SelfOnlyMemory(
            hidden_states=torch.empty(
                lanes,
                0,
                hidden_size,
                device=device,
                dtype=dtype,
            ),
            mask=torch.empty(lanes, 0, device=device, dtype=torch.bool),
            positions=torch.empty(lanes, 0, device=device, dtype=torch.long),
            slot_ids=torch.empty(lanes, 0, device=device, dtype=torch.long),
            kind_ids=torch.empty(lanes, 0, device=device, dtype=torch.long),
            lags=torch.empty(lanes, 0, device=device, dtype=torch.long),
            owner_streams=streams,
        )
    for block_state, block_validity, block_positions in zip(
        states,
        validity,
        positions,
        strict=True,
    ):
        if block_state.shape != (1, lanes, hidden_size):
            raise ValueError("Self-only block hidden states have an invalid shape.")
        if block_validity.shape != (1, lanes) or block_validity.dtype != torch.bool:
            raise ValueError("Self-only block validity has an invalid shape or dtype.")
        if block_positions.shape != (1, lanes):
            raise ValueError("Self-only block positions have an invalid shape.")
    memory_states = torch.stack(states, dim=2).reshape(
        lanes,
        len(states),
        hidden_size,
    )
    memory_mask = torch.stack(validity, dim=2).reshape(lanes, len(states))
    memory_positions = torch.stack(positions, dim=2).reshape(lanes, len(states))
    lane_ids = torch.arange(lanes, device=device, dtype=torch.long).unsqueeze(1)
    slot_ids = lane_ids.expand(lanes, len(states))
    lag_values = torch.arange(
        len(states),
        0,
        -1,
        device=device,
        dtype=torch.long,
    )
    return SelfOnlyMemory(
        hidden_states=memory_states,
        mask=memory_mask,
        positions=memory_positions,
        slot_ids=slot_ids,
        kind_ids=torch.ones_like(slot_ids),
        lags=lag_values.unsqueeze(0).expand(lanes, -1),
        owner_streams=streams,
    )


class ParameterMatchedSelfOnlyAttention(SharedNotesCrossAttention):
    """Run the exact SNC attention path over receiver-only causal history.

    ``memory.hidden_states`` must contain states produced by the same stream as
    ``receiver_hidden_states`` and strictly before every query position.
    Persistent plan memory is handled by the physical decoder's dedicated
    Plan-KV cross-attention in both the bus and self-only conditions. A
    deterministic feature selection maps ``hidden_size`` to ``notes_dim``;
    it has no parameters and is excluded from the state dict.
    """

    def __init__(
        self,
        config: SNCConfig,
        *,
        num_producers: int,
        gating_init: float = -4.0,
    ) -> None:
        if config.notes_dim > config.hidden_size:
            raise ValueError(
                "The parameter-free self-only projection requires notes_dim <= hidden_size; "
                f"got notes_dim={config.notes_dim}, hidden_size={config.hidden_size}."
            )
        super().__init__(
            config,
            num_producers=num_producers,
            gating_init=gating_init,
        )
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
        memory: SelfOnlyMemory,
        *,
        query_positions: torch.Tensor,
        receiver_streams: Sequence[str],
        force_gate: Optional[Union[torch.Tensor, bool]] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return an SNC-shaped delta using receiver-owned prior states only."""

        _validate_self_only_inputs(
            receiver_hidden_states=receiver_hidden_states,
            memory=memory,
            query_positions=query_positions,
            receiver_streams=receiver_streams,
            hidden_size=self.config.hidden_size,
            num_producers=self.num_producers,
        )
        return super().forward(
            receiver_hidden_states,
            self.select_own_history_features(memory.hidden_states),
            notes_mask=memory.mask,
            producer_ids=memory.slot_ids,
            kind_ids=memory.kind_ids,
            lags=memory.lags,
            force_gate=force_gate,
            return_attn_weights=return_attn_weights,
        )


def _validate_self_only_inputs(
    *,
    receiver_hidden_states: torch.Tensor,
    memory: SelfOnlyMemory,
    query_positions: torch.Tensor,
    receiver_streams: Sequence[str],
    hidden_size: int,
    num_producers: int,
) -> None:
    if receiver_hidden_states.ndim != 3:
        raise ValueError(
            "receiver_hidden_states must have shape [batch, query_tokens, hidden_size]."
        )
    if memory.hidden_states.ndim != 3:
        raise ValueError(
            "Self-only memory hidden_states must have shape [batch, prior_tokens, hidden_size]."
        )
    batch, queries, receiver_width = receiver_hidden_states.shape
    prior_batch, prior_tokens, prior_width = memory.hidden_states.shape
    if batch == 0 or queries == 0:
        raise ValueError("receiver_hidden_states must contain a non-empty batch and query axis.")
    if prior_batch != batch:
        raise ValueError("receiver and prior history batch sizes must match.")
    if receiver_width != hidden_size or prior_width != hidden_size:
        raise ValueError(
            f"receiver and prior hidden widths must both equal hidden_size={hidden_size}."
        )
    if receiver_hidden_states.device != memory.hidden_states.device:
        raise ValueError("receiver and prior hidden states must be on the same device.")
    if receiver_hidden_states.dtype != memory.hidden_states.dtype:
        raise TypeError("receiver and prior hidden states must have the same dtype.")
    if not receiver_hidden_states.is_floating_point():
        raise TypeError("receiver and prior hidden states must use a floating-point dtype.")

    expected_prior_shape = (batch, prior_tokens)
    expected_query_shape = (batch, queries)
    for name, tensor in (
        ("mask", memory.mask),
        ("positions", memory.positions),
        ("slot_ids", memory.slot_ids),
        ("kind_ids", memory.kind_ids),
        ("lags", memory.lags),
    ):
        if tensor.shape != expected_prior_shape:
            raise ValueError(
                f"Self-only memory {name} must have shape {expected_prior_shape}, "
                f"got {tuple(tensor.shape)}."
            )
    if query_positions.shape != expected_query_shape:
        raise ValueError(
            f"query_positions must have shape {expected_query_shape}, "
            f"got {tuple(query_positions.shape)}."
        )
    for name, tensor in (
        ("memory.mask", memory.mask),
        ("memory.positions", memory.positions),
        ("memory.slot_ids", memory.slot_ids),
        ("memory.kind_ids", memory.kind_ids),
        ("memory.lags", memory.lags),
        ("query_positions", query_positions),
    ):
        if tensor.device != receiver_hidden_states.device:
            raise ValueError(
                f"{name} must be on {receiver_hidden_states.device}, got {tensor.device}."
            )
    if memory.mask.dtype != torch.bool:
        raise TypeError("Self-only memory mask must have dtype torch.bool.")
    for name, positions in (
        ("query_positions", query_positions),
        ("memory.positions", memory.positions),
        ("memory.slot_ids", memory.slot_ids),
        ("memory.kind_ids", memory.kind_ids),
        ("memory.lags", memory.lags),
    ):
        if positions.dtype == torch.bool or positions.is_floating_point() or positions.is_complex():
            raise TypeError(f"{name} must have an integer dtype other than bool.")

    receivers = tuple(_stream_id(value, "receiver_streams") for value in receiver_streams)
    owners = tuple(_stream_id(value, "memory.owner_streams") for value in memory.owner_streams)
    if len(receivers) != batch or len(owners) != batch:
        raise ValueError("Self-only receiver and owner IDs must address every batch row.")
    if receivers != owners:
        raise ValueError(
            "self-only attention rejects sibling history: "
            f"receiver_streams={receivers!r}, owner_streams={owners!r}."
        )

    if bool((query_positions < 0).any()):
        raise ValueError("query_positions must be non-negative.")
    if queries > 1 and bool((query_positions[:, 1:] <= query_positions[:, :-1]).any()):
        raise ValueError("query_positions must be strictly increasing within every batch row.")

    if prior_tokens == 0:
        return
    if bool((memory.mask[:, 1:] & ~memory.mask[:, :-1]).any()):
        raise ValueError("Self-only memory mask must be prefix-contiguous within every row.")
    if bool((~memory.mask).all(dim=1).any()):
        raise ValueError(
            "Non-empty self-only memory requires at least one active prior token per row."
        )
    if bool((memory.positions[~memory.mask] != -1).any()):
        raise ValueError("Masked self-only positions must use the sentinel value -1.")
    active_prior_positions = memory.positions.masked_fill(~memory.mask, -1)
    if bool((active_prior_positions[memory.mask] < 0).any()):
        raise ValueError("Active prior_positions must be non-negative.")
    if bool(((memory.slot_ids < 0) | (memory.slot_ids >= num_producers)).any()):
        raise ValueError(f"Self-only slot_ids must lie in [0, {num_producers}).")
    if bool(((memory.kind_ids < 0) | (memory.kind_ids > 1)).any()):
        raise ValueError("Self-only kind_ids must contain only 0 or 1.")
    if bool((memory.lags < 0).any()):
        raise ValueError("Self-only lags must be non-negative.")
    if prior_tokens > 1:
        adjacent_active = memory.mask[:, 1:] & memory.mask[:, :-1]
        non_increasing = memory.positions[:, 1:] <= memory.positions[:, :-1]
        if bool((adjacent_active & non_increasing).any()):
            raise ValueError(
                "Active prior_positions must be strictly increasing within every batch row."
            )
    current_or_future = memory.mask & (memory.positions >= query_positions[:, :1])
    if bool(current_or_future.any()):
        raise ValueError(
            "Self-only history contains current/future leakage: every active prior position "
            "must be strictly earlier than the first query position."
        )


def _stream_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip().lower()
