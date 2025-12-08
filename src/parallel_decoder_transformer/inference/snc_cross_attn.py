"""Shared Notes Cross-Attention layers used in the GPT-OSS integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


@dataclass(slots=True)
class SharedNotesCrossAttentionConfig:
    hidden_size: int
    notes_dim: int
    num_heads: int
    gating_init: float = -5.0
    spectral_norm: bool = False
    spectral_norm_n_power_iterations: int = 1
    spectral_norm_eps: float = 1e-12


class SharedNotesCrossAttention(nn.Module):
    """Cross-attention that queries Dynamic Notes Bus snapshots."""

    def __init__(self, config: SharedNotesCrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_heads
        if head_dim * config.num_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.v_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        if config.spectral_norm:
            self.q_proj = spectral_norm(
                self.q_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.k_proj = spectral_norm(
                self.k_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.v_proj = spectral_norm(
                self.v_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.o_proj = spectral_norm(
                self.o_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
        self.gate = nn.Parameter(torch.full((1,), config.gating_init))

    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_gate: Optional[torch.Tensor | bool] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        batch, sequence, _ = hidden_states.size()
        _, notes_len, _ = notes.size()
        if notes_len == 0:
            return hidden_states
        q = self.q_proj(hidden_states)
        k = self.k_proj(notes)
        v = self.v_proj(notes)
        q = q.view(batch, sequence, self.config.num_heads, -1).transpose(1, 2)
        k = k.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)
        v = v.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if notes_mask is not None:
            mask = notes_mask
            if mask.dtype != torch.bool:
                mask = mask != 0
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError("notes_mask must be rank 2 or 3.")
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)
        projected = self.o_proj(context)
        gating = torch.sigmoid(self.gate).to(dtype=projected.dtype, device=projected.device)
        gating = gating.view(1, *([1] * (projected.dim() - 1))).expand(
            projected.size(0), *([1] * (projected.dim() - 1))
        )
        override_mask: Optional[torch.Tensor] = None
        if force_gate is not None:
            if isinstance(force_gate, bool):
                if force_gate:
                    override_mask = torch.ones_like(gating, dtype=torch.bool)
            else:
                override_tensor = torch.as_tensor(force_gate, device=projected.device)
                if override_tensor.numel() == 1:
                    if bool(override_tensor.item()):
                        override_mask = torch.ones_like(gating, dtype=torch.bool)
                else:
                    override_tensor = override_tensor.to(dtype=torch.bool)
                    if override_tensor.dim() == 1 and override_tensor.size(0) == projected.size(0):
                        reshape_dims = (override_tensor.size(0),) + (1,) * (projected.dim() - 1)
                        override_mask = override_tensor.view(*reshape_dims)
                    elif override_tensor.shape == gating.shape:
                        override_mask = override_tensor
                    else:
                        raise ValueError("force_gate tensor must broadcast to the batch dimension.")
        if override_mask is not None:
            gating = torch.where(override_mask, torch.ones_like(gating), gating)
        return hidden_states + gating * projected


__all__ = ["SharedNotesCrossAttention", "SharedNotesCrossAttentionConfig"]
