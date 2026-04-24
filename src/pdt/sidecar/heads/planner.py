"""Prompt-time VQ planner head.

The planner pools the shared prompt, emits per-slot continuous vectors,
quantizes them through a learned codebook with a straight-through estimator,
and exposes those quantized vectors to ``plan_notes_proj``. There are no
external planner-id targets in this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from pdt.config.schemas import PlannerHeadConfig


__all__ = ["PlannerHead", "PlannerOutput"]


@dataclass(slots=True)
class PlannerOutput:
    logits: torch.Tensor
    indices: torch.Tensor
    pre_quantized: torch.Tensor
    quantized: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class PlannerHead(nn.Module):
    def __init__(self, config: PlannerHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.slot_projector = nn.Linear(
            config.hidden_size,
            config.hidden_size * config.num_slots,
            bias=False,
        )
        self.codebook = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PlannerOutput:
        """Return VQ planner outputs for ``plan_notes_proj`` and diagnostics."""
        if hidden_states.dim() != 3:
            raise ValueError(
                f"PlannerHead expected hidden_states shape [B, T, H], got "
                f"{tuple(hidden_states.shape)}"
            )

        states = self.dropout(hidden_states)
        batch = states.size(0)
        pooled = self._masked_mean(states, attention_mask)
        pre_q = self.slot_projector(pooled).view(
            batch,
            self.config.num_slots,
            self.config.hidden_size,
        )

        codebook = self.codebook.weight
        distances = (
            pre_q.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(pre_q, codebook.t())
            + codebook.pow(2).sum(dim=-1).view(1, 1, -1)
        )
        logits = -distances
        indices = logits.argmax(dim=-1)
        embedded = self.codebook(indices)

        quantized = pre_q + (embedded - pre_q).detach()
        commitment_loss = F.mse_loss(pre_q, embedded.detach())
        codebook_loss = F.mse_loss(embedded, pre_q.detach())
        return PlannerOutput(
            logits=logits,
            indices=indices,
            pre_quantized=pre_q,
            quantized=quantized,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def _masked_mean(
        self,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return states.mean(dim=1)
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
        return (states * weights).sum(dim=1) / denom
