"""Shared Notes Cross-Attention backend abstraction.

This isolates the post-trunk cross-attention implementation behind a simple
interface so we can swap in mid-stack instrumented variants without touching
call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from ..inference.snc_cross_attn import SharedNotesCrossAttention


class SNCBackend(Protocol):
    """Protocol describing the operations required by the model runtime."""

    def apply(
        self,
        hidden: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_open: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply shared-notes conditioning to the hidden states."""


@dataclass(slots=True)
class PostTrunkSNC:
    """Thin wrapper over the existing post-trunk SharedNotesCrossAttention."""

    cross_attention: SharedNotesCrossAttention

    def apply(
        self,
        hidden: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_open: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.cross_attention(
            hidden,
            notes,
            notes_mask=notes_mask,
            force_gate=force_open,
        )


@dataclass(slots=True)
class MidStackSNC:
    """Placeholder backend for mid-stack instrumentation.

    The trunk will own cross-attention when this path is active, so the backend
    devolves to a no-op until instrumentation is introduced.
    """

    def apply(
        self,
        hidden: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_open: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return hidden


__all__ = ["SNCBackend", "PostTrunkSNC", "MidStackSNC"]
