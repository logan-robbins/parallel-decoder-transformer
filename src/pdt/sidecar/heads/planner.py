"""Prompt-time latent planner head.

Given prompt hidden states ``H_x in R^{T x hidden_size}``, the planner:

1. Masked-mean-pools across the T dimension using the prompt attention mask.
2. Projects the pooled ``(B, hidden_size)`` vector into ``(B, S, V_p)`` logits
   -- one distribution over the latent plan vocabulary per slot.

At V_p=8192 and hidden_size=2560 the projector is ~336M params. Full
utilization diagnostics are emitted from ``pdt.diagnostics.codebook``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from pdt.config.schemas import PlannerHeadConfig


__all__ = ["PlannerHead"]


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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(B, S, V_p)`` planner logits."""
        if hidden_states.dim() != 3:
            raise ValueError(
                f"PlannerHead expected hidden_states shape [B, T, H], got "
                f"{tuple(hidden_states.shape)}"
            )

        states = self.dropout(hidden_states)
        batch = states.size(0)

        if attention_mask is None:
            pooled = states.mean(dim=1)
        else:
            mask = attention_mask.to(device=states.device, dtype=states.dtype)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != states.shape[:2]:
                raise ValueError(
                    f"attention_mask shape {tuple(mask.shape)} must match "
                    f"(batch, seq) == {tuple(states.shape[:2])}"
                )
            weights = mask.unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            pooled = (states * weights).sum(dim=1) / denom

        logits = self.projector(pooled)
        return logits.view(batch, self.config.num_slots, self.config.vocab_size)
