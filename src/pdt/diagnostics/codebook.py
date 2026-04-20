"""Codebook-utilization diagnostics for the planner's V_p latent vocabulary.

These metrics gate Stage 0 -> Stage 1 transition. Per the plan:

- **unique_entries_used**: fraction of V_p selected at least once over the epoch.
- **per_slot_entropy**: Shannon entropy of each of the S slots' selection
  distribution (bits). Collapse = near-zero entropy on most slots.
- **pairwise_anchor_cosine**: mean cosine similarity between per-stream
  snapshot-0 anchors on the same prompt. Near-identical \u21d2 streams will
  not differentiate regardless of SNC.
- **usage_histogram**: top-k most-used entries. Heavy concentration in < 50
  entries is collapse.

Stage 0 gate: unique_entries_used * V_p >= 1000 AND max(per_slot_entropy) >= 2 bits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


__all__ = ["CodebookDiagnostics", "CodebookStats"]


@dataclass(slots=True)
class CodebookStats:
    vocab_size: int
    num_slots: int
    total_selections: int = 0
    unique_entries: int = 0
    unique_fraction: float = 0.0
    per_slot_entropy_bits: List[float] = field(default_factory=list)
    top_k_entries: List[int] = field(default_factory=list)
    top_k_counts: List[int] = field(default_factory=list)
    pairwise_anchor_cosine_mean: float = 0.0
    pairwise_anchor_cosine_max: float = 0.0

    def passes_stage0_gate(
        self,
        *,
        min_unique_entries: int = 1000,
        min_max_slot_entropy_bits: float = 2.0,
    ) -> bool:
        if self.unique_entries < min_unique_entries:
            return False
        if not self.per_slot_entropy_bits:
            return False
        return max(self.per_slot_entropy_bits) >= min_max_slot_entropy_bits

    def to_dict(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "num_slots": self.num_slots,
            "total_selections": self.total_selections,
            "unique_entries": self.unique_entries,
            "unique_fraction": self.unique_fraction,
            "per_slot_entropy_bits": list(self.per_slot_entropy_bits),
            "top_k_entries": list(self.top_k_entries),
            "top_k_counts": list(self.top_k_counts),
            "pairwise_anchor_cosine_mean": self.pairwise_anchor_cosine_mean,
            "pairwise_anchor_cosine_max": self.pairwise_anchor_cosine_max,
        }


class CodebookDiagnostics:
    """Streaming accumulator. Reset at the start of each eval pass."""

    def __init__(self, vocab_size: int, num_slots: int, top_k: int = 20) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        self.vocab_size = vocab_size
        self.num_slots = num_slots
        self.top_k = top_k
        self._global_counter: Counter[int] = Counter()
        self._per_slot_counters: List[Counter[int]] = [
            Counter() for _ in range(num_slots)
        ]
        self._anchor_cosine_sum: float = 0.0
        self._anchor_cosine_max: float = 0.0
        self._anchor_cosine_count: int = 0

    def reset(self) -> None:
        self._global_counter.clear()
        for c in self._per_slot_counters:
            c.clear()
        self._anchor_cosine_sum = 0.0
        self._anchor_cosine_max = 0.0
        self._anchor_cosine_count = 0

    def observe_selections(self, slot_ids: torch.Tensor) -> None:
        """``slot_ids`` shape: ``(B, S)`` long tensor of chosen entry ids."""
        if slot_ids.dim() != 2:
            raise ValueError(
                f"slot_ids must be rank 2 (B, S), got rank {slot_ids.dim()}"
            )
        if slot_ids.size(1) != self.num_slots:
            raise ValueError(
                f"slot_ids slot dim {slot_ids.size(1)} != num_slots {self.num_slots}"
            )
        flat = slot_ids.detach().cpu().flatten().tolist()
        self._global_counter.update(flat)
        for s in range(self.num_slots):
            self._per_slot_counters[s].update(
                slot_ids[:, s].detach().cpu().tolist()
            )

    def observe_anchors(self, anchors: torch.Tensor) -> None:
        """``anchors`` shape: ``(B, K, d_notes)`` snapshot-0 per stream."""
        if anchors.dim() != 3:
            raise ValueError(
                f"anchors must be rank 3 (B, K, d_notes), got rank {anchors.dim()}"
            )
        if anchors.size(1) < 2:
            return  # need >=2 streams to compute pairwise
        a = torch.nn.functional.normalize(anchors, dim=-1)
        # (B, K, K) pairwise cosine matrix.
        sim = torch.matmul(a, a.transpose(-2, -1))
        # Keep upper triangle, excluding diagonal.
        k = anchors.size(1)
        mask = torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)
        pair_sims = sim[:, mask]  # (B, n_pairs)
        self._anchor_cosine_sum += float(pair_sims.mean().item()) * pair_sims.numel()
        self._anchor_cosine_max = max(
            self._anchor_cosine_max, float(pair_sims.max().item())
        )
        self._anchor_cosine_count += pair_sims.numel()

    def compute(self) -> CodebookStats:
        total = sum(self._global_counter.values())
        unique = len(self._global_counter)
        unique_fraction = unique / self.vocab_size if self.vocab_size > 0 else 0.0

        per_slot_entropy: List[float] = []
        for counter in self._per_slot_counters:
            total_slot = sum(counter.values())
            if total_slot == 0:
                per_slot_entropy.append(0.0)
                continue
            entropy = 0.0
            log2 = torch.log2
            for count in counter.values():
                p = count / total_slot
                if p > 0:
                    entropy -= p * float(log2(torch.tensor(p)).item())
            per_slot_entropy.append(entropy)

        top_k_pairs = self._global_counter.most_common(self.top_k)
        top_k_entries = [e for e, _ in top_k_pairs]
        top_k_counts = [c for _, c in top_k_pairs]

        cosine_mean = (
            self._anchor_cosine_sum / self._anchor_cosine_count
            if self._anchor_cosine_count > 0
            else 0.0
        )
        return CodebookStats(
            vocab_size=self.vocab_size,
            num_slots=self.num_slots,
            total_selections=total,
            unique_entries=unique,
            unique_fraction=unique_fraction,
            per_slot_entropy_bits=per_slot_entropy,
            top_k_entries=top_k_entries,
            top_k_counts=top_k_counts,
            pairwise_anchor_cosine_mean=cosine_mean,
            pairwise_anchor_cosine_max=self._anchor_cosine_max,
        )
