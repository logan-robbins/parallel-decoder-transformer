"""Coverage head estimating plan item fulfilment probabilities.

Upgrade 03: Multi-Head Cross-Attention with Multi-Scale Keys.

Architecture replaces the single 4096-dim dot-product with an 8-head
decomposition (512-dim per head) and augments token-level keys with
sentence-level summaries constructed by non-overlapping mean-pooling.
A learned temperature scalar replaces the fixed 1/sqrt(d) scale.

The output contract (B, P) logits is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class MultiHeadCoverageHeadConfig:
    """Configuration for the multi-head coverage head.

    Attributes:
        hidden_size: Dimensionality of hidden states and plan embeddings.
        num_heads: Number of attention heads.  Must divide hidden_size evenly.
        dropout: Pre-query dropout rate applied to plan embeddings.
        sentence_window: Tokens per sentence-level window for multi-scale keys.
            Set to 0 to disable the multi-scale path (token-level only).
        learn_temperature: When True the attention temperature is a trainable
            ``nn.Parameter``; when False it is a frozen buffer initialised to
            ``sqrt(head_dim)`` (the standard scale).
    """

    hidden_size: int
    num_heads: int = 8
    dropout: float = 0.0
    sentence_window: int = 32
    learn_temperature: bool = True


class MultiHeadCoverageHead(nn.Module):
    """Multi-head cross-attention coverage head with multi-scale keys.

    Given attended hidden states ``(B, T, H)`` and plan embeddings
    ``(B, P, H)``, produces per-plan-item logits ``(B, P)`` indicating
    the probability that each plan item has been covered by the generated
    text so far.

    The module supports three independently toggleable innovations:

    1. **Multi-head decomposition** -- ``num_heads`` parallel attention
       patterns over ``head_dim = hidden_size // num_heads`` subspaces.
    2. **Multi-scale keys** -- token-level keys concatenated with
       sentence-level keys obtained by non-overlapping mean-pooling of
       width ``sentence_window``.
    3. **Learned temperature** -- a scalar ``log_temperature`` parameter
       that replaces the fixed ``1 / sqrt(d)`` attention scale.
    """

    def __init__(self, config: MultiHeadCoverageHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.score = nn.Linear(config.hidden_size, 1)
        if config.learn_temperature:
            init_val = 0.5 * math.log(self.head_dim)
            self.log_temperature = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(0.5 * math.log(self.head_dim)),
                persistent=False,
            )

    def _build_multiscale_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Concatenate token-level and sentence-level (window-pooled) hidden states.

        Args:
            hidden_states: Tensor of shape ``(B, T, H)``.

        Returns:
            Tensor of shape ``(B, T + S, H)`` where ``S = ceil(T / W)`` and
            ``W = sentence_window``.  When ``sentence_window <= 0`` returns
            *hidden_states* unchanged (no copy).
        """
        W = self.config.sentence_window
        if W <= 0:
            return hidden_states
        B, T, H = hidden_states.shape
        # Pad T to a multiple of W so we can reshape cleanly.
        remainder = T % W
        if remainder != 0:
            pad_len = W - remainder
            hidden_padded = F.pad(hidden_states, (0, 0, 0, pad_len))  # right-pad on T dim
        else:
            hidden_padded = hidden_states
        S = hidden_padded.size(1) // W
        # (B, S, W, H) -> mean over W -> (B, S, H)
        sentence_states = hidden_padded.view(B, S, W, H).mean(dim=2)
        # Concatenate along sequence dimension: token-level first, sentence-level second
        return torch.cat([hidden_states, sentence_states], dim=1)  # (B, T+S, H)

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_embeddings: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-plan-item coverage logits.

        Args:
            hidden_states: Attended trunk output, shape ``(B, T, H)``.
            plan_embeddings: Plan item embeddings, shape ``(B, P, H)``.
            plan_mask: Boolean mask, shape ``(B, P)``.  True where the plan
                item is valid, False for padding positions.

        Returns:
            Coverage logits of shape ``(B, P)``.  Padding positions are
            filled with ``0.0``.
        """
        B, P, _ = plan_embeddings.shape

        # Build multi-scale key/value source
        multiscale = self._build_multiscale_keys(hidden_states)  # (B, T+S, H)
        KV_len = multiscale.size(1)

        # Project queries, keys, values
        q = self.q_proj(self.dropout(plan_embeddings))  # (B, P, H)
        k = self.k_proj(multiscale)                     # (B, T+S, H)
        v = self.v_proj(multiscale)                     # (B, T+S, H)

        # Reshape for multi-head: (B, h, seq, d)
        h = self.config.num_heads
        d = self.head_dim
        q = q.view(B, P,      h, d).transpose(1, 2)  # (B, h, P,    d)
        k = k.view(B, KV_len, h, d).transpose(1, 2)  # (B, h, T+S,  d)
        v = v.view(B, KV_len, h, d).transpose(1, 2)  # (B, h, T+S,  d)

        # Scaled dot-product with learned temperature
        temperature = self.log_temperature.exp()
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature  # (B, h, P, T+S)
        attn_weights = torch.softmax(scores, dim=-1)

        # Aggregate values
        context = torch.matmul(attn_weights, v)             # (B, h, P, d)
        context = context.transpose(1, 2).contiguous()      # (B, P, h, d)
        context = context.view(B, P, h * d)                 # (B, P, H)

        # Output projection + scalar scoring
        out = self.o_proj(context)                           # (B, P, H)
        logits = self.score(out).squeeze(-1)                 # (B, P)
        logits = logits.masked_fill(~plan_mask, 0.0)
        return logits


# Backward-compatibility aliases -- all existing import sites work unchanged
CoverageHead = MultiHeadCoverageHead
CoverageHeadConfig = MultiHeadCoverageHeadConfig

__all__ = [
    "MultiHeadCoverageHead",
    "MultiHeadCoverageHeadConfig",
    "CoverageHead",
    "CoverageHeadConfig",
]
