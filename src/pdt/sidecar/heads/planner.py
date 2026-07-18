"""Prompt-time continuous structured-outline planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from pdt.config.schemas import PlannerHeadConfig


__all__ = ["PlannerHead", "PlannerOutput"]


@dataclass(slots=True)
class PlannerOutput:
    """Static planner state produced exactly once for a document."""

    nodes: torch.Tensor
    node_validity_logits: torch.Tensor
    presentation_order_logits: torch.Tensor


class PlannerHead(nn.Module):
    """Decode three unordered eight-node outlines from frozen prompt states."""

    def __init__(self, config: PlannerHeadConfig) -> None:
        super().__init__()
        self.config = config
        query_count = config.num_streams * config.max_nodes_per_stream
        self.prompt_projection = nn.Linear(config.hidden_size, config.planner_width)
        self.queries = nn.Parameter(torch.empty(query_count, config.planner_width))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.planner_width,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.planner_width)
        self.validity_head = nn.Linear(config.planner_width, 1)
        self.presentation_head = nn.Linear(config.planner_width, 1)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PlannerOutput:
        if hidden_states.dim() != 3:
            raise ValueError(
                "PlannerHead requires hidden_states with shape [B, T, H], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch, sequence, hidden = hidden_states.shape
        if hidden != self.config.hidden_size:
            raise ValueError(
                f"PlannerHead expected hidden size {self.config.hidden_size}, got {hidden}."
            )
        padding_mask = self._padding_mask(
            attention_mask,
            batch=batch,
            sequence=sequence,
            device=hidden_states.device,
        )
        prompt = self.prompt_projection(
            hidden_states.to(dtype=self.prompt_projection.weight.dtype)
        )
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(
            tgt=queries,
            memory=prompt,
            memory_key_padding_mask=padding_mask,
        )
        nodes = self.output_norm(decoded).reshape(
            batch,
            self.config.num_streams,
            self.config.max_nodes_per_stream,
            self.config.planner_width,
        )
        validity = self.validity_head(nodes).squeeze(-1)
        presentation = self.presentation_head(nodes.mean(dim=2)).squeeze(-1)
        return PlannerOutput(
            nodes=nodes,
            node_validity_logits=validity,
            presentation_order_logits=presentation,
        )

    @staticmethod
    def _padding_mask(
        attention_mask: Optional[torch.Tensor],
        *,
        batch: int,
        sequence: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape != (batch, sequence):
            raise ValueError(
                f"attention_mask must have shape {(batch, sequence)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        valid = attention_mask.to(device=device, dtype=torch.bool)
        if bool((~valid.any(dim=1)).any()):
            raise ValueError("Every planner prompt must contain at least one unmasked token.")
        return ~valid
