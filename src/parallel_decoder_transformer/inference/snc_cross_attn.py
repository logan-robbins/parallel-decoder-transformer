"""Shared Notes Cross-Attention layers used in the GPT-OSS integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


@dataclass(slots=True)
class SharedNotesCrossAttentionConfig:
    hidden_size: int
    notes_dim: int
    num_heads: int
    gating_init: float = -5.0
    spectral_norm: bool = False
    spectral_norm_n_power_iterations: int = 1
    spectral_norm_eps: float = 1e-12
    gate_mode: Literal["scalar", "per_head", "per_head_dynamic"] = "scalar"


class SharedNotesCrossAttention(nn.Module):
    """Cross-attention that queries Dynamic Notes Bus snapshots.

    Supports three gating modes controlled by ``config.gate_mode``:

    * ``"scalar"`` -- A single learned gate parameter broadcast over all heads.
      This is the original behavior and is fully backward-compatible.
    * ``"per_head"`` -- One learned gate parameter per attention head, allowing
      heads to specialize their openness to bus content independently.
    * ``"per_head_dynamic"`` -- Per-head static bias plus an input-dependent
      dynamic offset computed from mean-pooled hidden states.  The dynamic
      projection weight is zero-initialized so that at step zero the gate
      equals the static per-head bias exactly, preserving the "start closed"
      initialization property.

    In all modes the gate is applied to the per-head context vectors *before*
    the output projection (``o_proj``), so that ``o_proj`` learns to mix
    already-gated head contributions.
    """

    def __init__(self, config: SharedNotesCrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_heads
        if head_dim * config.num_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.v_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        if config.spectral_norm:
            self.q_proj = spectral_norm(
                self.q_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.k_proj = spectral_norm(
                self.k_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.v_proj = spectral_norm(
                self.v_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.o_proj = spectral_norm(
                self.o_proj,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )

        # Gate parameter shape depends on gate_mode.
        if config.gate_mode == "scalar":
            self.gate = nn.Parameter(torch.full((1,), config.gating_init))
            self.gate_pool_norm: Optional[nn.LayerNorm] = None
            self.gate_dyn_proj: Optional[nn.Linear] = None
        elif config.gate_mode == "per_head":
            self.gate = nn.Parameter(torch.full((config.num_heads,), config.gating_init))
            self.gate_pool_norm = None
            self.gate_dyn_proj = None
        else:  # per_head_dynamic
            self.gate = nn.Parameter(torch.full((config.num_heads,), config.gating_init))
            self.gate_pool_norm = nn.LayerNorm(config.hidden_size)
            self.gate_dyn_proj = nn.Linear(config.hidden_size, config.num_heads, bias=True)
            nn.init.zeros_(self.gate_dyn_proj.weight)
            nn.init.zeros_(self.gate_dyn_proj.bias)
            if config.spectral_norm:
                self.gate_dyn_proj = spectral_norm(
                    self.gate_dyn_proj,
                    n_power_iterations=config.spectral_norm_n_power_iterations,
                    eps=config.spectral_norm_eps,
                )

    # ------------------------------------------------------------------
    # Gate computation
    # ------------------------------------------------------------------

    def _compute_gate(
        self,
        hidden_states: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the per-head gate tensor.

        Args:
            hidden_states: Input hidden states, shape ``(B, T, d)``.
            dtype: Target dtype for the gate tensor.
            device: Target device for the gate tensor.

        Returns:
            Gate tensor of shape ``(1, 1, 1, 1)`` for scalar mode,
            ``(1, H, 1, 1)`` for per_head mode, or ``(B, H, 1, 1)`` for
            per_head_dynamic mode.
        """
        gate_param = self.gate.to(dtype=dtype, device=device)

        if self.config.gate_mode == "scalar":
            return torch.sigmoid(gate_param).view(1, 1, 1, 1)

        if self.config.gate_mode == "per_head":
            return torch.sigmoid(gate_param).view(1, self.config.num_heads, 1, 1)

        # per_head_dynamic
        batch_size = hidden_states.size(0)
        h_pool = hidden_states.mean(dim=1)  # (B, d)
        h_norm = self.gate_pool_norm(h_pool)  # type: ignore[misc]  # (B, d)
        delta = self.gate_dyn_proj(h_norm.to(dtype=dtype))  # type: ignore[misc]  # (B, H)
        logit = gate_param.unsqueeze(0) + delta  # (B, H)
        return torch.sigmoid(logit).view(batch_size, self.config.num_heads, 1, 1)

    def _apply_force_gate(
        self,
        gating: torch.Tensor,
        force_gate: Optional[torch.Tensor | bool],
        batch: int,
    ) -> torch.Tensor:
        """Override *gating* according to *force_gate* semantics.

        Args:
            gating: Current gate tensor, shape ``(B|1, H|1, 1, 1)``.
            force_gate: ``True`` to force all gates to 1.0, a 1-D bool tensor
                of shape ``(B,)`` for per-batch overrides, or ``None``/``False``
                for no override.
            batch: Batch size (used for validation of per-batch tensors).

        Returns:
            Possibly-overridden gate tensor with the same shape as *gating*.
        """
        if force_gate is None:
            return gating
        if isinstance(force_gate, bool):
            if force_gate:
                return torch.ones_like(gating)
            return gating

        override_tensor = torch.as_tensor(force_gate, device=gating.device)
        if override_tensor.numel() == 1:
            if bool(override_tensor.item()):
                return torch.ones_like(gating)
            return gating

        override_tensor = override_tensor.to(dtype=torch.bool)
        if override_tensor.dim() == 1 and override_tensor.size(0) == batch:
            mask = override_tensor.view(batch, 1, 1, 1)
            return torch.where(mask, torch.ones_like(gating), gating)

        raise ValueError("force_gate tensor must broadcast to the batch dimension.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        *,
        notes_mask: Optional[torch.Tensor] = None,
        force_gate: Optional[torch.Tensor | bool] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        batch, sequence, _ = hidden_states.size()
        _, notes_len, _ = notes.size()
        if notes_len == 0:
            return hidden_states

        q = self.q_proj(hidden_states)
        k = self.k_proj(notes)
        v = self.v_proj(notes)
        q = q.view(batch, sequence, self.config.num_heads, -1).transpose(1, 2)
        k = k.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)
        v = v.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if notes_mask is not None:
            mask = notes_mask
            if mask.dtype != torch.bool:
                mask = mask != 0
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError("notes_mask must be rank 2 or 3.")
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # (B, H, T, d_h)

        # Compute per-head gate and apply force_gate overrides.
        gating = self._compute_gate(hidden_states, dtype=context.dtype, device=context.device)
        gating = self._apply_force_gate(gating, force_gate, batch)

        # Gate applied to context per-head BEFORE o_proj.
        context = gating * context  # (B, H, T, d_h)
        context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)
        projected = self.o_proj(context)  # (B, T, d)

        return hidden_states + projected


__all__ = ["SharedNotesCrossAttention", "SharedNotesCrossAttentionConfig"]
