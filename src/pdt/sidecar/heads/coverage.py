"""Coverage head: multi-head cross-attention with multi-scale keys.

Given attended hidden states ``(B, T, H)`` and plan-item embeddings
``(B, P, H)``, produces per-plan-item logits ``(B, P)`` indicating the
probability that each plan item has been covered by the generated text.

Token-level keys are augmented with sentence-level keys obtained by
non-overlapping mean-pooling (window=sentence_window). A learned
temperature parameter replaces the fixed 1/sqrt(d) scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pdt.config.schemas import CoverageHeadConfig


__all__ = ["CoverageHead"]


class CoverageHead(nn.Module):
    def __init__(self, config: CoverageHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.score = nn.Linear(config.hidden_size, 1)
        if config.learn_temperature:
            init_val = 0.5 * math.log(self.head_dim)
            self.log_temperature = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(0.5 * math.log(self.head_dim)),
                persistent=False,
            )

    def _multiscale_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        W = self.config.sentence_window
        if W <= 0:
            return hidden_states
        batch, seq, hidden = hidden_states.shape
        remainder = seq % W
        if remainder != 0:
            padded = F.pad(hidden_states, (0, 0, 0, W - remainder))
        else:
            padded = hidden_states
        windows = padded.size(1) // W
        sentence = padded.view(batch, windows, W, hidden).mean(dim=2)
        return torch.cat([hidden_states, sentence], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_embeddings: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        if plan_embeddings.dim() != 3:
            raise ValueError("plan_embeddings must be rank 3 (B, P, H).")
        if plan_mask.dim() != 2:
            raise ValueError("plan_mask must be rank 2 (B, P).")
        if plan_mask.dtype != torch.bool:
            plan_mask = plan_mask != 0

        batch, num_plan, _ = plan_embeddings.shape
        keys_source = self._multiscale_keys(hidden_states)
        kv_len = keys_source.size(1)

        q = self.q_proj(self.dropout(plan_embeddings))
        k = self.k_proj(keys_source)
        v = self.v_proj(keys_source)

        heads = self.config.num_heads
        d = self.head_dim
        q = q.view(batch, num_plan, heads, d).transpose(1, 2)
        k = k.view(batch, kv_len, heads, d).transpose(1, 2)
        v = v.view(batch, kv_len, heads, d).transpose(1, 2)

        temperature = self.log_temperature.exp()
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, num_plan, heads * d)

        out = self.o_proj(context)
        logits = self.score(out).squeeze(-1)
        logits = logits.masked_fill(~plan_mask, 0.0)
        return logits
