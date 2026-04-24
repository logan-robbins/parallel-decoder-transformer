"""Per-stream snapshot-0 projection (fix for the papered-ghost module).

For each stream, ``plan_notes_proj`` pools the embeddings of the planner
slots assigned to that stream and projects into ``notes_dim``. The output is
published on the Dynamic Notes Bus as snapshot 0 *per stream*, implementing
the paper's disjoint-ownership invariant and the thesis's per-stream
symmetry-breaking.

Input:
    plan_embeddings: ``(B, S, hidden_size)`` straight-through quantized
                     planner slot vectors.
    ownership:     ``(B, K, S)`` bool tensor indicating which slots each
                   stream owns. Mutually disjoint in columns (each slot is
                   owned by exactly one stream).

Output:
    ``(B, K, notes_dim)`` -- snapshot-0 vector per stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pdt.config.schemas import PlanNotesProjectionConfig


__all__ = ["PlanNotesProjection"]


class PlanNotesProjection(nn.Module):
    def __init__(self, config: PlanNotesProjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.hidden_size, config.notes_dim)
        self.norm = nn.LayerNorm(config.notes_dim)

    def forward(
        self,
        plan_embeddings: torch.Tensor,
        ownership: torch.Tensor,
    ) -> torch.Tensor:
        """Produce per-stream snapshot-0 vectors.

        Args:
            plan_embeddings: ``(B, S, hidden_size)`` quantized planner vectors.
            ownership: ``(B, K, S)`` bool -- True where stream k owns slot s.

        Returns:
            ``(B, K, notes_dim)`` unit-snapshot vectors per stream.
        """
        if plan_embeddings.dim() != 3:
            raise ValueError(
                f"plan_embeddings must be rank 3 (B, S, H), got "
                f"{tuple(plan_embeddings.shape)}"
            )
        if ownership.dim() != 3:
            raise ValueError(
                f"ownership must be rank 3 (B, K, S), got {tuple(ownership.shape)}"
            )
        if ownership.size(-1) != plan_embeddings.size(1):
            raise ValueError(
                "ownership slot dimension must equal plan_embeddings slot dimension"
            )

        ownership_f = ownership.to(dtype=plan_embeddings.dtype)
        # (B, K, S, 1) * (B, 1, S, H) -> sum over S -> (B, K, H)
        weighted = ownership_f.unsqueeze(-1) * plan_embeddings.unsqueeze(1)
        summed = weighted.sum(dim=2)  # (B, K, H)
        counts = ownership_f.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, K, 1)
        pooled = summed / counts

        projected = self.proj(pooled)  # (B, K, notes_dim)
        projected = self.norm(projected)
        return projected
