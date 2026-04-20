"""Stream-adherence classifier: predicts which stream produced a hidden state."""

from __future__ import annotations

import torch
from torch import nn

from pdt.config.schemas import StreamClassifierConfig


__all__ = ["StreamClassifierHead"]


class StreamClassifierHead(nn.Module):
    def __init__(self, config: StreamClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_streams)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        features = torch.tanh(self.proj(self.dropout(pooled)))
        return self.classifier(features)
