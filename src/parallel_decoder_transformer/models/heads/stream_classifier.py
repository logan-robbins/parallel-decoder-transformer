"""Stream adherence head supervising stream-specific representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class StreamClassifierConfig:
    hidden_size: int
    num_streams: int
    dropout: float = 0.0


class StreamClassifierHead(nn.Module):
    def __init__(self, config: StreamClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_streams)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        features = torch.tanh(self.proj(self.dropout(pooled)))
        return self.classifier(features)


__all__ = ["StreamClassifierConfig", "StreamClassifierHead"]
