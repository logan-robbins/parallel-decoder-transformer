"""Observation-bounded finite-codebook diagnostics."""

from __future__ import annotations

import pytest
import torch

from pdt.diagnostics.codebook import CodebookDiagnostics


def test_codebook_reports_observable_ceiling_and_effective_entries() -> None:
    diagnostics = CodebookDiagnostics(vocab_size=8, num_codebooks=2)
    diagnostics.observe_selections(torch.tensor([[0, 1], [2, 1]]))
    stats = diagnostics.compute()

    assert stats.total_selections == 4
    assert stats.selection_rows == 2
    assert stats.unique_entries == 3
    assert stats.observable_unique_ceiling == 4
    assert stats.unique_fraction_of_observable_ceiling == pytest.approx(0.75)
    assert stats.effective_entries_per_codebook == pytest.approx([2.0, 1.0])
    assert stats.exactly_collapsed is False


def test_codebook_marks_only_exact_single_entry_collapse() -> None:
    diagnostics = CodebookDiagnostics(vocab_size=8192, num_codebooks=3)
    diagnostics.observe_selections(torch.zeros((5, 3), dtype=torch.long))
    stats = diagnostics.compute()

    assert stats.selection_rows == 5
    assert stats.observable_unique_ceiling == 15
    assert stats.unique_entries == 1
    assert stats.exactly_collapsed is True
