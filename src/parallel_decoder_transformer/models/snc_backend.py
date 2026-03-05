"""Shared Notes Cross-Attention backend abstraction.

This isolates the post-trunk cross-attention implementation behind a simple
interface so we can swap in mid-stack instrumented variants without touching
call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Thin wrapper over the existing post-trunk SharedNotesCrossAttention.

    After each :meth:`apply` call, :attr:`last_attn_weights` holds the
    detached cross-attention weight tensor ``[B, H, T, K]`` from the most
    recent forward pass.
    """

    cross_attention: SharedNotesCrossAttention
    _last_attn_weights: Optional[torch.Tensor] = field(
        default=None, init=False, repr=False
    )

    @property
    def last_attn_weights(self) -> Optional[torch.Tensor]:
        """Detached attention weights from the last :meth:`apply` call."""
        return self._last_attn_weights

    def apply(
        self,
        hidden: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_open: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, attn_weights = self.cross_attention(
            hidden,
            notes,
            notes_mask=notes_mask,
            force_gate=force_open,
            return_attn_weights=True,
        )
        self._last_attn_weights = attn_weights
        return output


@dataclass(slots=True)
class MidStackSNC:
    """Placeholder backend for mid-stack instrumentation.

    The trunk will own cross-attention when this path is active, so the backend
    devolves to a no-op until instrumentation is introduced.
    """

    _last_attn_weights: Optional[torch.Tensor] = field(
        default=None, init=False, repr=False
    )

    @property
    def last_attn_weights(self) -> Optional[torch.Tensor]:
        """Always ``None`` for mid-stack SNC (no post-trunk attention)."""
        return self._last_attn_weights

    def apply(
        self,
        hidden: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_open: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._last_attn_weights = None
        return hidden


__all__ = ["SNCBackend", "PostTrunkSNC", "MidStackSNC"]
