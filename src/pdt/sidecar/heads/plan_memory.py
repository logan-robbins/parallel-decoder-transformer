"""Persistent read-only projection for structured planner nodes."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from pdt.config.schemas import PlanMemoryProjectionConfig


__all__ = ["PlanMemoryProjection"]


class PlanMemoryProjection(nn.Module):
    """Project every outline node without pooling away its structure."""

    def __init__(self, config: PlanMemoryProjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.planner_width, config.notes_dim)
        self.norm = nn.LayerNorm(config.notes_dim)

    def forward(
        self,
        plan_nodes: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if plan_nodes.dim() != 4:
            raise ValueError(
                "plan_nodes must have shape [B, K, N, planner_width], "
                f"got {tuple(plan_nodes.shape)}."
            )
        if plan_nodes.size(-1) != self.config.planner_width:
            raise ValueError(
                f"Expected planner width {self.config.planner_width}, "
                f"got {plan_nodes.size(-1)}."
            )
        projected = self.norm(
            self.projection(plan_nodes.to(dtype=self.projection.weight.dtype))
        )
        if node_mask is None:
            return projected
        if node_mask.shape != plan_nodes.shape[:-1]:
            raise ValueError(
                f"node_mask must have shape {tuple(plan_nodes.shape[:-1])}, "
                f"got {tuple(node_mask.shape)}."
            )
        return projected * node_mask.to(
            device=projected.device,
            dtype=projected.dtype,
        ).unsqueeze(-1)
