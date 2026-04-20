"""SpeculationHead: provisional note writer.

Distinct from NotesHead: trained against the speculative-teacher-note targets
(noisier views) rather than the true-teacher-note targets.
"""

from __future__ import annotations

import torch
from torch import nn

from pdt.config.schemas import SpeculationHeadConfig


__all__ = ["SpeculationHead"]


class SpeculationHead(nn.Module):
    def __init__(self, config: SpeculationHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.projector = nn.Linear(config.hidden_size, config.notes_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        states = self.dropout(hidden_states)
        notes = self.projector(states)
        return notes * self.config.teacher_scale
