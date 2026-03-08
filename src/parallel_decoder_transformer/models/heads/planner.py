"""Planner head projecting pooled trunk states into latent plan-slot logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class PlannerHeadConfig:
    hidden_size: int
    vocab_size: int
    num_slots: int = 16
    dropout: float = 0.0


class PlannerHead(nn.Module):
    def __init__(self, config: PlannerHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(
            config.hidden_size,
            config.vocab_size * config.num_slots,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        if hidden_states.dim() != 3:
            raise ValueError(
                f"PlannerHead expects hidden states shaped [batch, seq, hidden], got {tuple(hidden_states.shape)}."
            )
        states = self.dropout(hidden_states)
        if attention_mask is None:
            pooled = states.mean(dim=1)
        else:
            mask = attention_mask.to(device=states.device, dtype=states.dtype)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() != 2 or mask.shape[:2] != states.shape[:2]:
                raise ValueError(
                    "PlannerHead attention_mask must match hidden state batch/sequence dimensions."
                )
            weights = mask.unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            pooled = (states * weights).sum(dim=1) / denom
        logits = self.projector(pooled)
        return logits.view(hidden_states.size(0), self.config.num_slots, self.config.vocab_size)


__all__ = ["PlannerHead", "PlannerHeadConfig"]
