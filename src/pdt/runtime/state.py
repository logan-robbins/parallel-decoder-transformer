"""Logical stream state and the one physical packed-frontier KV state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, List, Mapping, Optional, Sequence

import torch


__all__ = ["PackedAppend", "PackedFrontierState", "PackedTokenRows", "StreamState"]


@dataclass(frozen=True, slots=True)
class PackedTokenRows:
    """Left-padded rows for one physical trunk invocation.

    ``position_ids`` are logical per row; ``cache_position`` is the shared
    physical cache axis. Keeping those two coordinates distinct is what lets
    unequal prompt or transition lengths share one Qwen3 KV cache without
    introducing RoPE gaps.
    """

    input_ids: torch.Tensor  # (K, T)
    valid_mask: torch.Tensor  # (K, T) bool
    position_ids: torch.Tensor  # (K, T) long
    cache_position: torch.Tensor  # (T,) long


@dataclass(frozen=True, slots=True)
class PackedAppend:
    """Prepared append plus the full mask expected by cached attention."""

    rows: PackedTokenRows
    attention_mask: torch.Tensor  # (K, P + T)


@dataclass(slots=True)
class PackedFrontierState:
    """The sole owner of the synchronized K-stream cache.

    A per-stream cache representation cannot prove physical stream packing:
    it permits K separate trunk calls by construction. This state instead
    owns one batch-shaped cache and advances it only with a complete frontier.
    """

    streams: tuple[str, ...]
    attention_mask: torch.Tensor  # (K, P) bool
    past_key_values: Any

    def __post_init__(self) -> None:
        normalized = tuple(stream.lower() for stream in self.streams)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("Packed frontier streams must be non-empty and unique.")
        if self.attention_mask.dim() != 2 or self.attention_mask.size(0) != len(normalized):
            raise ValueError("Packed frontier attention_mask must have shape (K, P).")
        if self.attention_mask.size(1) == 0:
            raise ValueError("Packed frontier cache cannot be empty.")
        self.streams = normalized
        self.attention_mask = self.attention_mask.to(dtype=torch.bool)
        self._validate_cache_length(self.past_key_values, self.physical_length)

    @property
    def physical_length(self) -> int:
        return self.attention_mask.size(1)

    @property
    def logical_lengths(self) -> torch.Tensor:
        return self.attention_mask.sum(dim=1, dtype=torch.long)

    def prepare_append(
        self,
        token_ids_by_stream: Mapping[str, torch.Tensor],
        *,
        pad_token_id: int,
        active_streams: Optional[Collection[str]] = None,
    ) -> PackedAppend:
        """Pack one physical row per stream while masking completed lanes."""

        rows = pack_token_rows(
            self.streams,
            token_ids_by_stream,
            pad_token_id=pad_token_id,
            prior_logical_lengths=self.logical_lengths,
            cache_position_start=self.physical_length,
            active_streams=active_streams,
        )
        return PackedAppend(
            rows=rows,
            attention_mask=torch.cat((self.attention_mask, rows.valid_mask), dim=1),
        )

    def commit(self, append: PackedAppend, *, past_key_values: Any) -> None:
        """Commit exactly the cache returned by the prepared packed call."""

        if append.attention_mask.size(0) != len(self.streams):
            raise ValueError("Packed append batch does not match frontier streams.")
        expected_prefix = append.attention_mask[:, : self.physical_length]
        if not torch.equal(expected_prefix, self.attention_mask):
            raise ValueError("Packed append does not extend the current frontier mask.")
        self._validate_cache_length(past_key_values, append.attention_mask.size(1))
        self.attention_mask = append.attention_mask
        self.past_key_values = past_key_values

    @staticmethod
    def _validate_cache_length(cache: Any, expected: int) -> None:
        get_seq_length = getattr(cache, "get_seq_length", None)
        if not callable(get_seq_length):
            raise TypeError(
                "Packed PDT requires a Hugging Face Cache with get_seq_length(); "
                f"got {type(cache).__name__}."
            )
        actual = int(get_seq_length())
        if actual != expected:
            raise RuntimeError(
                f"Packed KV cache length mismatch: expected {expected}, got {actual}."
            )


def pack_token_rows(
    streams: Sequence[str],
    token_ids_by_stream: Mapping[str, torch.Tensor],
    *,
    pad_token_id: int,
    prior_logical_lengths: Optional[torch.Tensor] = None,
    cache_position_start: int = 0,
    active_streams: Optional[Collection[str]] = None,
) -> PackedTokenRows:
    """Left-pad one row per stream and mask physically present inactive lanes."""

    ordered = tuple(stream.lower() for stream in streams)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("Packed token streams must be non-empty and unique.")
    if set(token_ids_by_stream) != set(ordered):
        raise ValueError("Packed token rows must exactly match the ordered streams.")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer.")
    if type(cache_position_start) is not int or cache_position_start < 0:
        raise ValueError("cache_position_start must be a non-negative integer.")
    active = (
        set(ordered)
        if active_streams is None
        else {stream.lower() for stream in active_streams}
    )
    if not active or not active <= set(ordered):
        raise ValueError(
            "active_streams must name a non-empty subset of the packed streams."
        )

    rows: list[torch.Tensor] = []
    reference: Optional[torch.Tensor] = None
    for stream in ordered:
        row = token_ids_by_stream[stream]
        if row.dim() != 2 or row.size(0) != 1 or row.size(1) == 0:
            raise ValueError(f"Packed row for {stream!r} must have shape (1, T>0).")
        if row.dtype != torch.long:
            raise ValueError(f"Packed row for {stream!r} must use torch.long token IDs.")
        if reference is not None and row.device != reference.device:
            raise ValueError("All packed token rows must use the same device.")
        reference = row
        rows.append(row)
    assert reference is not None

    batch = len(ordered)
    width = max(row.size(1) for row in rows)
    input_ids = torch.full(
        (batch, width),
        pad_token_id,
        dtype=torch.long,
        device=reference.device,
    )
    valid_mask = torch.zeros((batch, width), dtype=torch.bool, device=reference.device)
    for index, row in enumerate(rows):
        length = row.size(1)
        input_ids[index, width - length :] = row[0]
        if ordered[index] in active:
            valid_mask[index, width - length :] = True

    if prior_logical_lengths is None:
        prior = torch.zeros(batch, dtype=torch.long, device=reference.device)
    else:
        prior = prior_logical_lengths.to(device=reference.device, dtype=torch.long)
        if prior.shape != (batch,) or bool((prior < 0).any()):
            raise ValueError("prior_logical_lengths must be non-negative with shape (K,).")
    positions = prior[:, None] + valid_mask.cumsum(dim=1, dtype=torch.long) - 1
    positions = positions.masked_fill(~valid_mask, 0)
    cache_position = torch.arange(
        cache_position_start,
        cache_position_start + width,
        dtype=torch.long,
        device=reference.device,
    )
    return PackedTokenRows(
        input_ids=input_ids,
        valid_mask=valid_mask,
        position_ids=positions,
        cache_position=cache_position,
    )


@dataclass(slots=True)
class StreamState:
    stream: str
    input_ids: torch.Tensor  # (1, T) long
    attention_mask: torch.Tensor  # (1, T)
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    generated_pieces: List[str] = field(default_factory=list)
    tokens_since_snapshot: int = 0
    current_notes: Optional[torch.Tensor] = None
    current_notes_mask: Optional[torch.Tensor] = None
    latest_snapshot_version: int = 0

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2 or self.input_ids.size(0) != 1:
            raise ValueError("StreamState requires (1, T) input_ids.")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids shape.")
        self.stream = self.stream.lower()

    @property
    def device(self) -> torch.device:
        return self.input_ids.device

    @property
    def total_tokens(self) -> int:
        return self.input_ids.size(1)

    @property
    def generated_count(self) -> int:
        return len(self.generated_tokens)

    def update_notes_window(self, notes: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        if notes.dim() != 3 or notes.size(0) != 1:
            raise ValueError("notes must be shaped (1, S, notes_dim).")
        self.current_notes = notes
        self.current_notes_mask = mask

    def append_token(
        self,
        token_id: int,
        *,
        token_text: Optional[str] = None,
    ) -> None:
        token_tensor = torch.tensor([[token_id]], dtype=self.input_ids.dtype, device=self.device)
        mask_tensor = torch.ones_like(token_tensor, dtype=self.attention_mask.dtype)
        self.input_ids = torch.cat([self.input_ids, token_tensor], dim=1)
        self.attention_mask = torch.cat([self.attention_mask, mask_tensor], dim=1)
        self.generated_tokens.append(int(token_id))
        piece = token_text or ""
        self.generated_pieces.append(piece)
        if piece:
            self.generated_text += piece
        self.tokens_since_snapshot += 1

    def append_context_tokens(
        self,
        token_ids: torch.Tensor,
    ) -> None:
        """Append a non-generated chat transition already consumed by the trunk."""

        if token_ids.dim() != 2 or token_ids.size(0) != 1 or token_ids.size(1) == 0:
            raise ValueError("Context token_ids must be non-empty with shape (1, T).")
        if token_ids.device != self.device or token_ids.dtype != self.input_ids.dtype:
            raise ValueError(
                "Context token_ids must use the stream state's device and integer dtype."
            )
        mask = torch.ones_like(token_ids, dtype=self.attention_mask.dtype)
        self.input_ids = torch.cat((self.input_ids, token_ids), dim=1)
        self.attention_mask = torch.cat((self.attention_mask, mask), dim=1)

    def mark_snapshot_version(self, version: int) -> None:
        if version <= self.latest_snapshot_version:
            raise ValueError(
                f"Expected snapshot version > {self.latest_snapshot_version}, got {version}."
            )
        self.latest_snapshot_version = version

    def reset_snapshot_counter(self) -> None:
        self.tokens_since_snapshot = 0
