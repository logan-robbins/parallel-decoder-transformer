"""Runtime memory addressed to the three physical decoder frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from pdt.baselines.self_only import SelfOnlyMemory


__all__ = ["LayerRuntimeContext"]


@dataclass(slots=True)
class LayerRuntimeContext:
    """Persistent plan state and causally visible dynamic memory for one call."""

    stream_ids: Optional[Tuple[str, ...]] = None
    plan_nodes: Optional[torch.Tensor] = None
    plan_mask: Optional[torch.Tensor] = None
    plan_memory: Optional[torch.Tensor] = None
    plan_producer_ids: Optional[torch.Tensor] = None
    notes: Optional[torch.Tensor] = None
    notes_mask: Optional[torch.Tensor] = None
    note_producer_ids: Optional[torch.Tensor] = None
    note_kind_ids: Optional[torch.Tensor] = None
    note_lags: Optional[torch.Tensor] = None
    self_only_memory: Optional[SelfOnlyMemory] = None
    self_only_query_positions: Optional[torch.Tensor] = None
    snc_force_gate: Optional[object] = None
