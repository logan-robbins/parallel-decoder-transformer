"""Coverage head estimating plan item fulfilment probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class CoverageHeadConfig:
    hidden_size: int
    dropout: float = 0.0


class CoverageHead(nn.Module):
    def __init__(self, config: CoverageHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_embeddings: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(self.dropout(plan_embeddings))
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        scale = 1.0 / math.sqrt(query.size(-1))
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, value)
        logits = self.proj(context).squeeze(-1)
        logits = logits.masked_fill(~plan_mask, 0.0)
        return logits


__all__ = ["CoverageHead", "CoverageHeadConfig"]
