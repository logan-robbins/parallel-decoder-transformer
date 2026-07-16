"""Per-stream inference state for canonical synchronized generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


# Type alias: either legacy tuple or Cache object.
PastKeyValues = Any


__all__ = ["PastKeyValues", "StreamState"]


@dataclass(slots=True)
class StreamState:
    stream: str
    input_ids: torch.Tensor  # (1, T) long
    attention_mask: torch.Tensor  # (1, T)
    past_key_values: Optional[PastKeyValues] = None
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
        past_key_values: Optional[PastKeyValues],
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
        self.past_key_values = past_key_values

    def append_context_tokens(
        self,
        token_ids: torch.Tensor,
        *,
        past_key_values: Optional[PastKeyValues],
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
        self.past_key_values = past_key_values

    def mark_snapshot_version(self, version: int) -> None:
        if version <= self.latest_snapshot_version:
            raise ValueError(
                f"Expected snapshot version > {self.latest_snapshot_version}, got {version}."
            )
        self.latest_snapshot_version = version

    def reset_snapshot_counter(self) -> None:
        self.tokens_since_snapshot = 0
