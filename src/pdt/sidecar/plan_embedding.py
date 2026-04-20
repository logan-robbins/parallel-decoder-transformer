"""Shared plan-embedding matrix.

E_plan: Embedding(V_p, hidden_size). Indexed by planner-slot argmax to
re-embed selected latent plan items.
"""

from __future__ import annotations

import torch
from torch import nn


__all__ = ["PlanEmbedding"]


class PlanEmbedding(nn.Module):
    """Thin wrapper around ``nn.Embedding(V_p, hidden_size)``.

    Kept as its own module so the curriculum name resolver has a stable name
    to freeze/unfreeze without reaching into ``nn.Module._modules``.
    """

    def __init__(self, plan_vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.plan_vocab_size = plan_vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(plan_vocab_size, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: ``(..., S)`` long tensor of slot indices. Returns
        ``(..., S, hidden_size)``."""
        return self.embedding(ids)
