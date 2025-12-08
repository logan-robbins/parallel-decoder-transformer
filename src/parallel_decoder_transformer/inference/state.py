"""Stream state management for the Parallel Decoder Transformer inference runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch


PastKeyValueLayer = Tuple[Optional[torch.Tensor], ...]
PastKeyValues = Tuple[PastKeyValueLayer, ...]


@dataclass(slots=True)
class KVCheckpoint:
    """Cached key/value tensors anchored at a commit boundary."""

    token_count: int
    past_key_values: Optional[PastKeyValues]


@dataclass(slots=True)
class StreamState:
    """Tracks decode-time tensors and metadata for a single stream lane."""

    stream: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    commit_stride: int
    commit_horizon: int
    past_key_values: Optional[PastKeyValues] = None
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    generated_pieces: List[str] = field(default_factory=list)
    tokens_since_snapshot: int = 0
    tokens_since_commit: int = 0
    commit_pointer: int = 0
    last_seen_version: Dict[str, int] = field(default_factory=dict)
    current_notes: Optional[torch.Tensor] = None
    current_notes_mask: Optional[torch.Tensor] = None
    latest_snapshot_version: int = 0
    _rollback_buffer: Deque[int] = field(init=False, repr=False)
    _kv_checkpoints: Deque[KVCheckpoint] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2:
            raise ValueError("StreamState.input_ids must be rank-2 (batch, sequence).")
        if self.attention_mask.dim() != 2:
            raise ValueError("StreamState.attention_mask must be rank-2 (batch, sequence).")
        if self.input_ids.size() != self.attention_mask.size():
            raise ValueError("StreamState.input_ids and attention_mask must have identical shapes.")
        if self.input_ids.size(0) != 1:
            raise ValueError("StreamState expects batch size 1 during inference.")
        if self.commit_stride <= 0:
            raise ValueError("StreamState.commit_stride must be positive.")
        if self.commit_horizon <= 0:
            raise ValueError("StreamState.commit_horizon must be positive.")
        self.stream = self.stream.lower()
        self.commit_pointer = self.input_ids.size(1)
        self._rollback_buffer: Deque[int] = deque(maxlen=self.commit_horizon)
        self._kv_checkpoints: Deque[KVCheckpoint] = deque()
        if self.commit_pointer:
            # Seed with prompt boundary so initial rollback targets the prompt state.
            self._kv_checkpoints.append(
                KVCheckpoint(
                    token_count=self.commit_pointer,
                    past_key_values=_clone_cache(self.past_key_values),
                )
            )

    @property
    def device(self) -> torch.device:
        return self.input_ids.device

    @property
    def dtype(self) -> torch.dtype:
        return self.input_ids.dtype

    @property
    def total_tokens(self) -> int:
        return self.input_ids.size(1)

    @property
    def generated_count(self) -> int:
        return len(self.generated_tokens)

    @property
    def rollback_buffer(self) -> Tuple[int, ...]:
        return tuple(self._rollback_buffer)

    def cadence_reached(self, cadence: int) -> bool:
        return self.tokens_since_snapshot >= cadence

    def update_notes_window(self, notes: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        if notes.dim() != 3:
            raise ValueError("notes tensor must be shaped [batch, window, dim].")
        if notes.size(0) != 1:
            raise ValueError("StreamState only supports batch size 1 for notes windows.")
        if mask is not None and mask.dim() != 2:
            raise ValueError("notes_mask must be shaped [batch, window].")
        self.current_notes = notes
        self.current_notes_mask = mask

    def append_token(
        self,
        token_id: int,
        *,
        past_key_values: Optional[PastKeyValues],
        token_text: Optional[str] = None,
    ) -> None:
        """Record a newly sampled token and advance bookkeeping."""

        token_tensor = torch.tensor([[token_id]], dtype=self.dtype, device=self.device)
        mask_tensor = torch.ones_like(token_tensor, dtype=self.attention_mask.dtype)
        self.input_ids = torch.cat([self.input_ids, token_tensor], dim=1)
        self.attention_mask = torch.cat([self.attention_mask, mask_tensor], dim=1)
        self.generated_tokens.append(int(token_id))
        piece = token_text or ""
        self.generated_pieces.append(piece)
        if piece:
            self.generated_text += piece
        self.tokens_since_snapshot += 1
        self.tokens_since_commit += 1
        self._rollback_buffer.append(int(token_id))
        self.past_key_values = past_key_values

    def reset_snapshot_counter(self) -> None:
        self.tokens_since_snapshot = 0

    def mark_snapshot_version(self, version: int) -> None:
        if version <= self.latest_snapshot_version:
            raise ValueError(
                f"Expected snapshot version > {self.latest_snapshot_version}, received {version}."
            )
        self.latest_snapshot_version = version

    def update_last_seen_version(self, producer: str, version: int) -> None:
        cached = self.last_seen_version.get(producer)
        if cached is not None and version < cached:
            raise ValueError(
                f"Monotonic violation for producer {producer!r}: existing={cached}, new={version}."
            )
        self.last_seen_version[producer] = version

    def register_commit(self) -> None:
        """Store a checkpoint at the current commit boundary."""

        if self.past_key_values is None:
            raise RuntimeError("Cannot register commit without past_key_values.")
        committed = False
        while self.tokens_since_commit >= self.commit_stride:
            checkpoint = KVCheckpoint(
                token_count=self.total_tokens,
                past_key_values=_clone_cache(self.past_key_values),
            )
            self._kv_checkpoints.append(checkpoint)
            self.commit_pointer = checkpoint.token_count
            self.tokens_since_commit -= self.commit_stride
            committed = True
        max_checkpoints = max(1, (self.commit_horizon // self.commit_stride) + 2)
        while len(self._kv_checkpoints) > max_checkpoints:
            self._kv_checkpoints.popleft()
        # Ensure rollback buffer never exceeds the commitment horizon.
        while len(self._rollback_buffer) > self.commit_horizon:
            self._rollback_buffer.popleft()
        if not committed:
            return

    def can_rollback(self) -> bool:
        return len(self._kv_checkpoints) > 0

    def rollback(self) -> Tuple[List[int], Optional[PastKeyValues]]:
        """Rollback to the most recent commit boundary and return removed tokens."""

        if not self.can_rollback():
            raise RuntimeError("No KV checkpoint available for rollback.")
        checkpoint = self._kv_checkpoints[-1]
        tokens_to_remove = self.total_tokens - checkpoint.token_count
        if tokens_to_remove <= 0:
            return [], checkpoint.past_key_values
        if tokens_to_remove > self.commit_horizon:
            raise RuntimeError(
                f"Cannot rollback {tokens_to_remove} tokens; horizon is {self.commit_horizon}."
            )
        removed: List[int] = []
        for _ in range(tokens_to_remove):
            if not self.generated_tokens:
                break
            removed_token = self.generated_tokens.pop()
            removed.insert(0, removed_token)
            if self.generated_pieces:
                self.generated_pieces.pop()
        if removed:
            self.input_ids = self.input_ids[:, : -len(removed)]
            self.attention_mask = self.attention_mask[:, : -len(removed)]
            self.generated_text = "".join(self.generated_pieces)
        self.tokens_since_snapshot = max(0, self.tokens_since_snapshot - len(removed))
        self.tokens_since_commit = 0
        # Trim rollback buffer entries for the removed region.
        for _ in range(min(len(removed), len(self._rollback_buffer))):
            self._rollback_buffer.pop()
        self.commit_pointer = checkpoint.token_count
        self.past_key_values = _clone_cache(checkpoint.past_key_values)
        return removed, self.past_key_values


def _clone_cache(cache: Optional[PastKeyValues]) -> Optional[PastKeyValues]:
    if cache is None:
        return None
    return tuple(tuple(_clone_entry(entry) for entry in layer) for layer in cache)


def _clone_entry(entry: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if entry is None:
        return None
    return entry.clone()


__all__ = ["PastKeyValues", "StreamState", "KVCheckpoint"]
