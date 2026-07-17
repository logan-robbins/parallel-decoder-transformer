"""Shared plan-conditioned bottleneck adapter."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from pdt.config.schemas import PlanAdapterConfig


__all__ = ["PlanConditionedAdapter"]


class PlanConditionedAdapter(nn.Module):
    """One parameter-shared adapter whose computation is bound by plan memory."""

    def __init__(self, config: PlanAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Linear(config.hidden_size, config.bottleneck_size)
        self.film = nn.Linear(config.plan_width, 2 * config.bottleneck_size)
        self.up = nn.Linear(config.bottleneck_size, config.hidden_size)
        if config.activation == "relu":
            self.activation: nn.Module = nn.ReLU()
        elif config.activation == "tanh":
            self.activation = nn.Tanh()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(
                "PlanAdapterConfig.activation must be one of gelu, relu, tanh; "
                f"got {config.activation!r}."
            )
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_nodes: torch.Tensor,
        plan_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must have shape [B, T, H].")
        if plan_nodes.dim() != 3:
            raise ValueError("plan_nodes must have shape [B, N, planner_width].")
        if hidden_states.size(0) != plan_nodes.size(0):
            raise ValueError("hidden_states and plan_nodes batch axes must match.")
        if hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.config.hidden_size}, got {hidden_states.size(-1)}."
            )
        if plan_nodes.size(-1) != self.config.plan_width:
            raise ValueError(
                f"Expected plan width {self.config.plan_width}, got {plan_nodes.size(-1)}."
            )
        plan = plan_nodes.to(dtype=self.down.weight.dtype)
        if plan_mask is None:
            pooled = plan.mean(dim=1)
        else:
            if plan_mask.shape != plan_nodes.shape[:2]:
                raise ValueError(
                    f"plan_mask must have shape {tuple(plan_nodes.shape[:2])}, "
                    f"got {tuple(plan_mask.shape)}."
                )
            mask = plan_mask.to(device=plan.device, dtype=plan.dtype)
            if bool((mask.sum(dim=1) == 0).any()):
                raise ValueError("Every lane must contain at least one valid plan node.")
            pooled = (plan * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=1,
                keepdim=True,
            )
        gamma, beta = self.film(pooled).chunk(2, dim=-1)
        hidden = self.down(hidden_states.to(dtype=self.down.weight.dtype))
        hidden = hidden * (1.0 + torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        hidden = self.dropout(self.activation(hidden))
        delta = self.dropout(self.up(hidden))
        return delta.to(dtype=hidden_states.dtype)
