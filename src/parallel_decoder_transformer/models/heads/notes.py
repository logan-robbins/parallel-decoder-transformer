"""Notes head generating structured note embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class NotesHeadConfig:
    hidden_size: int
    notes_dim: int
    dropout: float = 0.0
    gated: bool = True


class NotesHead(nn.Module):
    def __init__(self, config: NotesHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.notes_dim)
        self.gate = nn.Parameter(torch.zeros(1)) if config.gated else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        states = self.dropout(hidden_states)
        notes = self.projector(states)
        if self.gate is not None:
            return torch.sigmoid(self.gate) * notes
        return notes


__all__ = ["NotesHead", "NotesHeadConfig"]
