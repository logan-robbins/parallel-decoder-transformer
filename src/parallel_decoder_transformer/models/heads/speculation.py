"""Speculation head for predicting provisional notes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class SpeculationHeadConfig:
    hidden_size: int
    notes_dim: int
    dropout: float = 0.0
    teacher_scale: float = 1.0


class SpeculationHead(nn.Module):
    def __init__(self, config: SpeculationHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.notes_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        states = self.dropout(hidden_states)
        notes = self.projector(states)
        return notes * self.config.teacher_scale


__all__ = ["SpeculationHead", "SpeculationHeadConfig"]
