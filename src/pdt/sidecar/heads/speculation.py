"""SpeculationHead: the learned block-end writer to the Dynamic Notes Bus."""

from __future__ import annotations

import torch
from torch import nn

from pdt.config.schemas import SpeculationHeadConfig


__all__ = ["SpeculationHead"]


class SpeculationHead(nn.Module):
    def __init__(self, config: SpeculationHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.notes_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.projector.weight.dtype)
        states = self.dropout(hidden_states)
        return self.projector(states)
