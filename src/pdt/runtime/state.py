"""Per-stream inference state.

Holds the input_ids / attention_mask / KV cache for a single stream plus
bookkeeping for commit checkpoints and rollback. The KV cache slot is
polymorphic: it accepts either HF's legacy tuple-of-tuples layout or a
modern ``transformers.cache_utils.Cache`` object. The orchestrator converts
between them at forward boundaries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch


# Type alias: either legacy tuple or Cache object.
PastKeyValues = Any


__all__ = ["KVCheckpoint", "PastKeyValues", "StreamState"]


@dataclass(slots=True)
class KVCheckpoint:
    """Cached KV state at a commit boundary."""

    token_count: int
    past_key_values: Optional[PastKeyValues]


@dataclass(slots=True)
class StreamState:
    stream: str
    input_ids: torch.Tensor  # (1, T) long
    attention_mask: torch.Tensor  # (1, T)
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
        if self.input_ids.dim() != 2 or self.input_ids.size(0) != 1:
            raise ValueError("StreamState requires (1, T) input_ids.")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids shape.")
        if self.commit_stride <= 0:
            raise ValueError("commit_stride must be positive.")
        if self.commit_horizon <= 0:
            raise ValueError("commit_horizon must be positive.")
        self.stream = self.stream.lower()
        self.commit_pointer = self.input_ids.size(1)
        self._rollback_buffer = deque(maxlen=self.commit_horizon)
        self._kv_checkpoints = deque()
        if self.commit_pointer:
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
    def total_tokens(self) -> int:
        return self.input_ids.size(1)

    @property
    def generated_count(self) -> int:
        return len(self.generated_tokens)

    def update_last_seen_version(self, producer: str, version: int) -> None:
        cached = self.last_seen_version.get(producer)
        if cached is not None and version < cached:
            raise ValueError(
                f"Monotonic violation for producer {producer!r}: "
                f"existing={cached}, new={version}."
            )
        self.last_seen_version[producer] = version

    def update_notes_window(
        self, notes: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> None:
        if notes.dim() != 3 or notes.size(0) != 1:
            raise ValueError("notes must be shaped (1, S, notes_dim).")
        self.current_notes = notes
        self.current_notes_mask = mask

    def append_token(
        self,
        token_id: int,
        *,
        past_key_values: Optional[PastKeyValues],
        token_text: Optional[str] = None,
    ) -> None:
        token_tensor = torch.tensor(
            [[token_id]], dtype=self.input_ids.dtype, device=self.device
        )
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

    def mark_snapshot_version(self, version: int) -> None:
        if version <= self.latest_snapshot_version:
            raise ValueError(
                f"Expected snapshot version > {self.latest_snapshot_version}, "
                f"got {version}."
            )
        self.latest_snapshot_version = version

    def reset_snapshot_counter(self) -> None:
        self.tokens_since_snapshot = 0

    def register_commit(self) -> None:
        if self.past_key_values is None:
            raise RuntimeError("Cannot register commit without past_key_values.")
        while self.tokens_since_commit >= self.commit_stride:
            self._kv_checkpoints.append(
                KVCheckpoint(
                    token_count=self.total_tokens,
                    past_key_values=_clone_cache(self.past_key_values),
                )
            )
            self.commit_pointer = self.total_tokens
            self.tokens_since_commit -= self.commit_stride
        max_checkpoints = max(1, (self.commit_horizon // self.commit_stride) + 2)
        while len(self._kv_checkpoints) > max_checkpoints:
            self._kv_checkpoints.popleft()

    def can_rollback(self) -> bool:
        return len(self._kv_checkpoints) > 0

    def rollback(self) -> Tuple[List[int], Optional[PastKeyValues]]:
        if not self.can_rollback():
            raise RuntimeError("No checkpoint available for rollback.")
        checkpoint = self._kv_checkpoints[-1]
        tokens_to_remove = self.total_tokens - checkpoint.token_count
        if tokens_to_remove <= 0:
            return [], checkpoint.past_key_values
        if tokens_to_remove > self.commit_horizon:
            raise RuntimeError(
                f"Rollback would remove {tokens_to_remove} tokens "
                f"(horizon={self.commit_horizon})."
            )
        removed: List[int] = []
        for _ in range(tokens_to_remove):
            if not self.generated_tokens:
                break
            tok = self.generated_tokens.pop()
            removed.insert(0, tok)
            if self.generated_pieces:
                self.generated_pieces.pop()
        if removed:
            self.input_ids = self.input_ids[:, : -len(removed)]
            self.attention_mask = self.attention_mask[:, : -len(removed)]
            self.generated_text = "".join(self.generated_pieces)
        self.tokens_since_snapshot = max(0, self.tokens_since_snapshot - len(removed))
        self.tokens_since_commit = 0
        for _ in range(min(len(removed), len(self._rollback_buffer))):
            self._rollback_buffer.pop()
        self.commit_pointer = checkpoint.token_count
        self.past_key_values = _clone_cache(checkpoint.past_key_values)
        return removed, self.past_key_values


def _clone_cache(cache: Optional[PastKeyValues]) -> Optional[PastKeyValues]:
    """Deep-clone a KV cache. Accepts legacy tuples or a DynamicCache object."""
    if cache is None:
        return None
    # transformers Cache object -> try .to_legacy_cache() / .__deepcopy__
    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            cloned = tuple(
                tuple(entry.clone() if entry is not None else None for entry in layer)
                for layer in legacy
            )
            # Prefer to return the same type the caller gave us -- but since
            # we need a deep copy and DynamicCache.from_legacy_cache exists,
            # we just return the tuple form which HF still accepts.
            return cloned
        except Exception:
            pass
    if isinstance(cache, tuple):
        return tuple(
            tuple(entry.clone() if entry is not None else None for entry in layer)
            for layer in cache
        )
    # Fallback: assume it's a cloneable object.
    return cache
