"""Semantic supervision heads for exact per-lane decomposition."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pdt.config.schemas import SemanticSupervisionConfig


__all__ = ["SemanticSupervisionHeads"]


class SemanticSupervisionHeads(nn.Module):
    """Score plan meaning, fact routing, outline progress, and fact writing."""

    def __init__(self, config: SemanticSupervisionConfig) -> None:
        super().__init__()
        self.config = config
        width = config.attention_width
        self.route_node_projection = nn.Linear(config.planner_width, width)
        self.route_fact_projection = nn.Linear(config.fact_embedding_dim, width)
        self.progress_block_projection = nn.Linear(config.hidden_size, width)
        self.progress_node_projection = nn.Linear(config.planner_width, width)
        self.write_block_projection = nn.Linear(config.hidden_size, width)
        self.write_fact_projection = nn.Linear(config.fact_embedding_dim, width)
        self.write_classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(width, config.num_fact_roles),
        )
        self.scale = math.sqrt(width)

    def project_plan_semantics(self, plan_nodes: torch.Tensor) -> torch.Tensor:
        self._validate_plan_nodes(plan_nodes)
        return F.normalize(plan_nodes.float(), dim=-1)

    def fact_route_logits(
        self,
        plan_nodes: torch.Tensor,
        fact_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[B, K, N, F]` plan-node/fact ownership scores."""

        self._validate_plan_nodes(plan_nodes)
        self._validate_fact_embeddings(fact_embeddings, batch=plan_nodes.size(0))
        nodes = self.route_node_projection(
            plan_nodes.to(dtype=self.route_node_projection.weight.dtype)
        )
        facts = self.route_fact_projection(
            fact_embeddings.to(dtype=self.route_fact_projection.weight.dtype)
        )
        return torch.einsum("bkna,bfa->bknf", nodes, facts) / self.scale

    def outline_progress_logits(
        self,
        block_hidden: torch.Tensor,
        plan_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[B, K, M, N]` active-outline-node logits."""

        self._validate_block_hidden(block_hidden)
        self._validate_plan_nodes(plan_nodes)
        if block_hidden.shape[:2] != plan_nodes.shape[:2]:
            raise ValueError("block_hidden and plan_nodes must share batch and lane axes.")
        blocks = self.progress_block_projection(
            block_hidden.to(dtype=self.progress_block_projection.weight.dtype)
        )
        nodes = self.progress_node_projection(
            plan_nodes.to(dtype=self.progress_node_projection.weight.dtype)
        )
        return torch.einsum("bkma,bkna->bkmn", blocks, nodes) / self.scale

    def fact_write_logits(
        self,
        block_hidden: torch.Tensor,
        fact_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[B, K, M, F, 3]` OWNER/REFERENCE/ABSENT logits."""

        self._validate_block_hidden(block_hidden)
        self._validate_fact_embeddings(fact_embeddings, batch=block_hidden.size(0))
        blocks = self.write_block_projection(
            block_hidden.to(dtype=self.write_block_projection.weight.dtype)
        )
        facts = self.write_fact_projection(
            fact_embeddings.to(dtype=self.write_fact_projection.weight.dtype)
        )
        joint = blocks.unsqueeze(3) * facts[:, None, None, :, :]
        return self.write_classifier(joint)

    def _validate_plan_nodes(self, plan_nodes: torch.Tensor) -> None:
        expected_tail = (
            self.config.max_nodes_per_stream,
            self.config.planner_width,
        )
        if plan_nodes.dim() != 4 or plan_nodes.shape[-2:] != expected_tail:
            raise ValueError(
                "plan_nodes must have shape [B, K, "
                f"{expected_tail[0]}, {expected_tail[1]}], got {tuple(plan_nodes.shape)}."
            )

    def _validate_fact_embeddings(
        self,
        fact_embeddings: torch.Tensor,
        *,
        batch: int,
    ) -> None:
        if (
            fact_embeddings.dim() != 3
            or fact_embeddings.size(0) != batch
            or fact_embeddings.size(-1) != self.config.fact_embedding_dim
        ):
            raise ValueError(
                "fact_embeddings must have shape [B, F, "
                f"{self.config.fact_embedding_dim}], got {tuple(fact_embeddings.shape)}."
            )
        if fact_embeddings.size(1) > self.config.max_facts:
            raise ValueError(
                f"Fact count {fact_embeddings.size(1)} exceeds max_facts="
                f"{self.config.max_facts}."
            )

    def _validate_block_hidden(self, block_hidden: torch.Tensor) -> None:
        if block_hidden.dim() != 4 or block_hidden.size(-1) != self.config.hidden_size:
            raise ValueError(
                "block_hidden must have shape [B, K, M, "
                f"{self.config.hidden_size}], got {tuple(block_hidden.shape)}."
            )
