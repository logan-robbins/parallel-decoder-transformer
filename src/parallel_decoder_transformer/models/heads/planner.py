"""Planner head projecting shared trunk states into plan token logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class PlannerHeadConfig:
    hidden_size: int
    vocab_size: int
    dropout: float = 0.0


class PlannerHead(nn.Module):
    def __init__(self, config: PlannerHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        states = self.dropout(hidden_states)
        return self.projector(states)


__all__ = ["PlannerHead", "PlannerHeadConfig"]
