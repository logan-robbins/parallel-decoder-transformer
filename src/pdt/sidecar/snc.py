"""Shared Notes Cross-Attention (SNC).

SNC is the read mechanism over the visible notes window. It uses its own
independent ``W_Q, W_K, W_V, W_O`` projections -- no weight sharing with the
trunk's self-attention -- which is what lets it ride cleanly on top of
Qwen3's GQA without any collision. The attention computation has a fixed
communication width, so moving from a 4B to a 14B trunk scales SNC as
``O(hidden_size * attention_width)`` rather than ``O(hidden_size**2)``.

Gate: a single scalar pre-sigmoid parameter initialized to ``gating_init``.
With ``gating_init=-4.0`` the initial contribution is sigmoid(-4) \u2248 0.018,
so instrumentation preserves trunk magnitude statistics at step 0 and
training opens the gate as the auxiliary path becomes reliable.

The gate implemented here is the *internal* gate baked into the SNC context
projection. The instrumented decoder layer also applies an *outer* gated
residual (``notes_gate``) when it adds SNC's output to the trunk hidden
state; either gate closing collapses SNC's contribution to zero.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from pdt.config.schemas import SNCConfig

__all__ = ["SharedNotesCrossAttention"]


# Alias for backward compatibility / discoverability.
SharedNotesCrossAttentionConfig = SNCConfig


class SharedNotesCrossAttention(nn.Module):
    """Cross-attend over the Dynamic Notes Bus window.

    Inputs:
        hidden_states: ``(B, T, hidden_size)`` trunk hidden states.
        notes:         ``(B, S, notes_dim)`` visible workspace window.
        notes_mask:    ``(B, S)`` bool mask over the S notes; True means
                        the slot is valid.
        producer_ids:  ``(B, S)`` producer addresses in ``[0, K)``.
        kind_ids:      ``(B, S)`` persistent-plan/dynamic IDs in ``{0, 1}``.
        lags:          ``(B, S)`` non-negative delivery ages in blocks.
        force_gate:    Optional override. When True, gate is forced to 1.0.
                        When False or None, the learned gate is used.
                        When a (B,) bool tensor, per-batch override.

    Output:
        SNC context delta shaped ``(B, T, hidden_size)`` ready to be added as
        a gated residual by the caller. Note: this module returns the **delta**
        -- the caller is responsible for the residual add and the outer gate.
    """

    def __init__(
        self,
        config: SNCConfig,
        *,
        num_producers: int,
        gating_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.config = config
        if type(num_producers) is not int or num_producers <= 0:
            raise ValueError("num_producers must be a positive integer.")
        self.num_producers = num_producers
        head_dim = config.attention_width // config.num_heads
        if head_dim * config.num_heads != config.attention_width:
            raise ValueError(
                f"attention_width ({config.attention_width}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.attention_width)
        self.k_proj = nn.Linear(config.notes_dim, config.attention_width)
        self.v_proj = nn.Linear(config.notes_dim, config.attention_width)
        self.o_proj = nn.Linear(config.attention_width, config.hidden_size)
        # These embeddings encode the known bus address/header. They are not
        # transmitted payload and therefore do not increase message capacity.
        self.producer_embedding = nn.Embedding(num_producers, config.notes_dim)
        self.kind_embedding = nn.Embedding(2, config.notes_dim)
        self.lag_projection = nn.Linear(1, config.notes_dim, bias=False)
        # Zero-init output projection so SNC contribution is exactly zero at
        # step 0 in addition to the gate being closed.
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

        self.gate = nn.Parameter(torch.full((1,), gating_init))
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def _compute_gate(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        force_gate: Optional[Union[torch.Tensor, bool]],
        batch: int,
    ) -> torch.Tensor:
        """Compute the scalar gate, honoring an optional override.

        Returns a tensor broadcastable to ``(B, 1, 1, 1)``.
        """
        if isinstance(force_gate, bool):
            if force_gate:
                return torch.ones((1, 1, 1, 1), dtype=dtype, device=device)
            # force_gate=False means "force gate closed" (Intervention A).
            return torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)

        if force_gate is None:
            return torch.sigmoid(self.gate.to(dtype=dtype, device=device)).view(1, 1, 1, 1)

        override = torch.as_tensor(force_gate, device=device)
        if override.numel() == 1:
            if bool(override.item()):
                return torch.ones((1, 1, 1, 1), dtype=dtype, device=device)
            return torch.zeros((1, 1, 1, 1), dtype=dtype, device=device)
        if override.dim() == 1 and override.size(0) == batch:
            per_batch = override.to(dtype=dtype).view(batch, 1, 1, 1)
            # Interpret non-zero as "open"; bool-friendly.
            return per_batch
        raise ValueError("force_gate must be None, bool, a 1-element tensor, or a (B,) tensor.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: torch.Tensor,
        producer_ids: torch.Tensor,
        kind_ids: torch.Tensor,
        lags: torch.Tensor,
        force_gate: Optional[Union[torch.Tensor, bool]] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return the gated SNC delta (pre-residual).

        Shape conventions:
            hidden_states: (B, T, H)
            notes:         (B, S, notes_dim)
            notes_mask:    (B, S), bool
            producer_ids:  (B, S), integer
            kind_ids:      (B, S), integer in {0, 1}
            lags:          (B, S), non-negative integer
            returns:       (B, T, H) delta to be added as a residual by the
                           caller.
        """
        batch, sequence, hidden = hidden_states.size()
        input_dtype = hidden_states.dtype
        if hidden != self.config.hidden_size:
            raise ValueError(f"SNC expected hidden_size={self.config.hidden_size}, got {hidden}.")
        if notes.dim() != 3 or notes.size(0) != batch:
            raise ValueError("notes must have shape (B, S, notes_dim) matching hidden batch.")
        if notes.size(-1) != self.config.notes_dim:
            raise ValueError(
                f"SNC expected notes_dim={self.config.notes_dim}, got {notes.size(-1)}."
            )
        notes_len = notes.size(1) if notes.numel() > 0 else 0

        expected_metadata_shape = (batch, notes_len)
        for name, tensor in (
            ("notes_mask", notes_mask),
            ("producer_ids", producer_ids),
            ("kind_ids", kind_ids),
            ("lags", lags),
        ):
            if tensor.shape != expected_metadata_shape:
                raise ValueError(
                    f"{name} must have shape {expected_metadata_shape}, got {tuple(tensor.shape)}."
                )
            if tensor.device != hidden_states.device:
                raise ValueError(f"{name} must be on {hidden_states.device}, got {tensor.device}.")
        if producer_ids.dtype == torch.bool or producer_ids.is_floating_point():
            raise TypeError("producer_ids must use a non-bool integer dtype.")
        if kind_ids.dtype == torch.bool or kind_ids.is_floating_point():
            raise TypeError("kind_ids must use a non-bool integer dtype.")
        if lags.dtype == torch.bool or lags.is_floating_point():
            raise TypeError("lags must use a non-bool integer dtype.")
        if bool(((producer_ids < 0) | (producer_ids >= self.num_producers)).any()):
            raise ValueError(f"producer_ids must lie in [0, {self.num_producers}).")
        if bool(((kind_ids < 0) | (kind_ids > 1)).any()):
            raise ValueError(
                "kind_ids must contain only 0 (persistent plan) or 1 (dynamic)."
            )
        if bool((lags < 0).any()):
            raise ValueError("lags must be non-negative.")

        if notes_len == 0:
            zeros = torch.zeros_like(hidden_states)
            if return_attn_weights:
                empty_weights = torch.zeros(
                    batch,
                    self.config.num_heads,
                    sequence,
                    0,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                return zeros, empty_weights
            return zeros

        q = self.q_proj(hidden_states.to(dtype=self.q_proj.weight.dtype))
        metadata_dtype = self.producer_embedding.weight.dtype
        addressed_notes = notes.to(dtype=metadata_dtype)
        addressed_notes = addressed_notes + self.producer_embedding(producer_ids)
        addressed_notes = addressed_notes + self.kind_embedding(kind_ids)
        log_lags = torch.log1p(lags.to(dtype=metadata_dtype)).unsqueeze(-1)
        addressed_notes = addressed_notes + self.lag_projection(log_lags)
        k = self.k_proj(addressed_notes.to(dtype=self.k_proj.weight.dtype))
        v = self.v_proj(addressed_notes.to(dtype=self.v_proj.weight.dtype))

        q = q.view(batch, sequence, self.config.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, notes_len, self.config.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, notes_len, self.config.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = notes_mask if notes_mask.dtype == torch.bool else notes_mask != 0
        mask = mask[:, None, None, :]  # (B, 1, 1, S)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)

        gate = self._compute_gate(
            dtype=context.dtype,
            device=context.device,
            force_gate=force_gate,
            batch=batch,
        )
        gated_context = gate * context

        gated_context = (
            gated_context.transpose(1, 2)
            .contiguous()
            .view(batch, sequence, self.config.attention_width)
        )
        delta = self.o_proj(gated_context).to(dtype=input_dtype)

        if return_attn_weights:
            return delta, attn_weights.detach()
        return delta
