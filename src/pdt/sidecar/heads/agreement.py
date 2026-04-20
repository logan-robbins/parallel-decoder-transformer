"""AgreementHead: readiness score for the paper-specified signature.

Per paper \u00a72, the readiness score ``r^{(k)}_v`` is a function of:

    r^{(k)}_v = AgreeHead(\u0303H^{(k)}_{v,\u03c4}, W^{(k)}_v, c^{(k)}_v, \u00f1^{(k)}_v)

where:
    \u0303H: the attended trunk hidden at block end
    W:   the visible notes window (what siblings wrote)
    c:   the coverage logits (what plan items are owned / covered)
    \u00f1:  the stream's own speculative note summary

The previous implementation was a 1-liner over hidden states only. This
rewrite pools the window, compresses coverage, concatenates all four
sources, and emits one scalar readiness score per (batch, block).

The threshold \u03b3 is tuned offline from an ROC sweep; it is stored on the
head as a buffer for observability and is not a trainable parameter.
"""

from __future__ import annotations

import torch
from torch import nn

from pdt.config.schemas import AgreementHeadConfig


__all__ = ["AgreementHead"]


class AgreementHead(nn.Module):
    def __init__(self, config: AgreementHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        # Project the visible window into a fixed-dim summary via attention-
        # pooled query over the hidden block-end state.
        self.window_proj = nn.Linear(config.notes_dim, config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.notes_self_proj = nn.Linear(config.notes_dim, config.hidden_size)
        # Coverage logits are variable-length (B, P). Compress to a fixed-dim
        # statistic via (mean, std, min, max, proportion-above-threshold).
        # That gives ``coverage_features`` = 5 core stats that we then project
        # into a dense embedding.
        self.coverage_encoder = nn.Sequential(
            nn.Linear(5, config.coverage_features),
            nn.GELU(),
            nn.Linear(config.coverage_features, config.hidden_size),
        )
        # Final readiness scorer fuses all four sources.
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_size, 1),
        )
        # gamma is stored as a buffer -- tuned offline via ROC sweeps rather
        # than learned. Consumers read it via ``.gamma``.
        self.register_buffer(
            "gamma", torch.tensor(float(config.gamma_init)), persistent=False
        )

    def _attn_pool_window(
        self,
        hidden: torch.Tensor,  # (B, H)
        window: torch.Tensor,  # (B, S, notes_dim)
        window_mask: torch.Tensor | None,  # (B, S) bool
    ) -> torch.Tensor:
        """Single-query cross-attention pooling of the visible window."""
        if window.numel() == 0:
            return torch.zeros_like(hidden)
        q = self.query_proj(hidden).unsqueeze(1)  # (B, 1, H)
        k = self.window_proj(window)  # (B, S, H)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, 1, S)
        scores = scores / (k.size(-1) ** 0.5)
        if window_mask is not None:
            mask = window_mask if window_mask.dtype == torch.bool else window_mask != 0
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            # Handle all-masked rows: fall back to zeros.
            all_masked = (~mask).all(dim=-1)
            if all_masked.any():
                scores = scores.masked_fill(all_masked[:, None, None], 0.0)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, k).squeeze(1)  # (B, H)
        return pooled

    def _encode_coverage(self, coverage_logits: torch.Tensor) -> torch.Tensor:
        """Produce a ``(B, hidden_size)`` summary of coverage state.

        ``coverage_logits`` can be (B, P) or (B, K, P) -- we reduce over
        trailing dims first.
        """
        probs = torch.sigmoid(coverage_logits)
        if probs.dim() == 3:
            probs = probs.mean(dim=1)  # (B, P)
        if probs.dim() != 2:
            raise ValueError(
                f"coverage_logits must be rank 2 or 3, got rank {probs.dim()}"
            )
        mean = probs.mean(dim=-1, keepdim=True)
        std = probs.std(dim=-1, keepdim=True)
        mn = probs.min(dim=-1, keepdim=True).values
        mx = probs.max(dim=-1, keepdim=True).values
        above = (probs > 0.5).to(probs.dtype).mean(dim=-1, keepdim=True)
        stats = torch.cat([mean, std, mn, mx, above], dim=-1)  # (B, 5)
        return self.coverage_encoder(stats)

    def forward(
        self,
        hidden_block_end: torch.Tensor,  # (B, H)
        notes_window: torch.Tensor,  # (B, S, notes_dim)
        coverage_logits: torch.Tensor,  # (B, P)
        own_notes: torch.Tensor,  # (B, notes_dim)
        *,
        window_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return readiness probability ``(B, 1)``."""
        if hidden_block_end.dim() == 3:
            # Accept (B, T, H): take the last token.
            hidden_block_end = hidden_block_end[:, -1, :]
        if hidden_block_end.dim() != 2:
            raise ValueError("hidden_block_end must be rank 2 or 3.")

        h = self.dropout(hidden_block_end)

        window_feat = self._attn_pool_window(h, notes_window, window_mask)
        coverage_feat = self._encode_coverage(coverage_logits)
        if own_notes.dim() == 3:
            own_notes = own_notes.mean(dim=1)
        own_feat = self.notes_self_proj(own_notes)

        fused = torch.cat([h, window_feat, coverage_feat, own_feat], dim=-1)
        logits = self.scorer(fused)  # (B, 1)
        return torch.sigmoid(logits)
