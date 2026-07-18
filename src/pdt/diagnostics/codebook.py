"""Codebook-utilization diagnostics for finite dynamic notes.

Per the plan:

- **unique_entries_used**: fraction of V_p selected at least once over the epoch.
- **per_codebook_entropy**: Shannon entropy of each product codebook's selection
  distribution (bits). Collapse = near-zero entropy on most slots.
- **usage_histogram**: top-k most-used entries. Heavy concentration in < 50
  entries is collapse.

Utilization is descriptive.  The observable ceiling is reported explicitly so
a small mechanism run is not judged against entries it could not possibly have
selected.  Causal acceptance is handled by document-paired intervention
metrics, not by an arbitrary codebook threshold.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import torch


__all__ = ["CodebookDiagnostics", "CodebookStats"]


@dataclass(slots=True)
class CodebookStats:
    vocab_size: int
    num_codebooks: int
    total_selections: int = 0
    unique_entries: int = 0
    unique_fraction: float = 0.0
    per_codebook_entropy_bits: List[float] = field(default_factory=list)
    top_k_entries: List[int] = field(default_factory=list)
    top_k_counts: List[int] = field(default_factory=list)

    @property
    def selection_rows(self) -> int:
        if (
            self.num_codebooks <= 0
            or self.total_selections % self.num_codebooks != 0
        ):
            raise RuntimeError(
                "Codebook selections do not form complete product-code rows: "
                f"total={self.total_selections}, codebooks={self.num_codebooks}."
            )
        return self.total_selections // self.num_codebooks

    @property
    def observable_unique_ceiling(self) -> int:
        return min(self.vocab_size, self.total_selections)

    @property
    def unique_fraction_of_observable_ceiling(self) -> float:
        ceiling = self.observable_unique_ceiling
        return self.unique_entries / ceiling if ceiling else 0.0

    @property
    def effective_entries_per_codebook(self) -> List[float]:
        return [2.0**entropy for entropy in self.per_codebook_entropy_bits]

    @property
    def exactly_collapsed(self) -> bool:
        return self.total_selections > 0 and (
            self.unique_entries <= 1
            or not self.per_codebook_entropy_bits
            or max(self.per_codebook_entropy_bits) == 0.0
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "num_codebooks": self.num_codebooks,
            "total_selections": self.total_selections,
            "selection_rows": self.selection_rows,
            "unique_entries": self.unique_entries,
            "unique_fraction": self.unique_fraction,
            "observable_unique_ceiling": self.observable_unique_ceiling,
            "unique_fraction_of_observable_ceiling": (
                self.unique_fraction_of_observable_ceiling
            ),
            "per_codebook_entropy_bits": list(self.per_codebook_entropy_bits),
            "effective_entries_per_codebook": self.effective_entries_per_codebook,
            "exactly_collapsed": self.exactly_collapsed,
            "top_k_entries": list(self.top_k_entries),
            "top_k_counts": list(self.top_k_counts),
        }


class CodebookDiagnostics:
    """Streaming finite-code accumulator, reset at each evaluation pass."""

    def __init__(
        self,
        vocab_size: int,
        num_codebooks: int,
        top_k: int = 20,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive.")
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.top_k = top_k
        self._global_counter: Counter[int] = Counter()
        self._per_codebook_counters: List[Counter[int]] = [
            Counter() for _ in range(num_codebooks)
        ]

    def reset(self) -> None:
        self._global_counter.clear()
        for counter in self._per_codebook_counters:
            counter.clear()

    def observe_selections(self, slot_ids: torch.Tensor) -> None:
        """``slot_ids`` shape: ``(B, S)`` long tensor of chosen entry ids."""
        if slot_ids.dim() != 2:
            raise ValueError(f"slot_ids must be rank 2 (B, S), got rank {slot_ids.dim()}")
        if slot_ids.size(1) != self.num_codebooks:
            raise ValueError(
                "slot_ids codebook axis "
                f"{slot_ids.size(1)} != num_codebooks {self.num_codebooks}"
            )
        flat = slot_ids.detach().cpu().flatten().tolist()
        self._global_counter.update(flat)
        for codebook in range(self.num_codebooks):
            self._per_codebook_counters[codebook].update(
                slot_ids[:, codebook].detach().cpu().tolist()
            )

    def compute(self) -> CodebookStats:
        total = sum(self._global_counter.values())
        unique = len(self._global_counter)
        unique_fraction = unique / self.vocab_size if self.vocab_size > 0 else 0.0

        per_codebook_entropy: List[float] = []
        for counter in self._per_codebook_counters:
            total_slot = sum(counter.values())
            if total_slot == 0:
                per_codebook_entropy.append(0.0)
                continue
            entropy = 0.0
            log2 = torch.log2
            for count in counter.values():
                p = count / total_slot
                if p > 0:
                    entropy -= p * float(log2(torch.tensor(p)).item())
            per_codebook_entropy.append(entropy)

        top_k_pairs = self._global_counter.most_common(self.top_k)
        top_k_entries = [e for e, _ in top_k_pairs]
        top_k_counts = [c for _, c in top_k_pairs]

        return CodebookStats(
            vocab_size=self.vocab_size,
            num_codebooks=self.num_codebooks,
            total_selections=total,
            unique_entries=unique,
            unique_fraction=unique_fraction,
            per_codebook_entropy_bits=per_codebook_entropy,
            top_k_entries=top_k_entries,
            top_k_counts=top_k_counts,
        )
