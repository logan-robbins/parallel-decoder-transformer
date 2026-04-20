"""NotesHead: projects trunk hidden states to d_notes summaries."""

from __future__ import annotations

import torch
from torch import nn

from pdt.config.schemas import NotesHeadConfig


__all__ = ["NotesHead"]


class NotesHead(nn.Module):
    def __init__(self, config: NotesHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.notes_dim)
        self.gate = nn.Parameter(torch.zeros(1)) if config.gated else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        states = self.dropout(hidden_states)
        notes = self.projector(states)
        if self.gate is not None:
            return torch.sigmoid(self.gate) * notes
        return notes
