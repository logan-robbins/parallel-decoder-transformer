"""SpeculationHead: project and finitely quantize each dynamic bus write."""

from __future__ import annotations

import torch
from torch import nn

from pdt.config.schemas import SpeculationHeadConfig
from pdt.sidecar.product_vq import ProductVQOutput, ProductVectorQuantizer


__all__ = ["ProductVQOutput", "SpeculationHead"]


class SpeculationHead(nn.Module):
    def __init__(self, config: SpeculationHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.notes_dim)
        self.quantizer = ProductVectorQuantizer(
            width=config.notes_dim,
            num_codebooks=config.num_codebooks,
            codes_per_codebook=config.codes_per_codebook,
        )

    @property
    def capacity_bits(self) -> int:
        return self.quantizer.capacity_bits

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.projector.weight.dtype)
        states = self.dropout(hidden_states)
        return self.projector(states)

    def quantize(self, projected: torch.Tensor) -> ProductVQOutput:
        return self.quantizer(projected)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.decode(indices)

    @property
    def width(self) -> int:
        return self.quantizer.width

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    @property
    def codes_per_codebook(self) -> int:
        return self.quantizer.codes_per_codebook

    def forward(self, hidden_states: torch.Tensor) -> ProductVQOutput:
        return self.quantize(self.project(hidden_states))
