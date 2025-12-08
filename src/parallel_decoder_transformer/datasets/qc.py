"""Quality control utilities for filtering dataset examples."""

from __future__ import annotations

import logging
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np  # type: ignore

from parallel_decoder_transformer.data.extraction import StreamNotes

from .config import DatasetBuildConfig
from .example import DatasetExample
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QualityFilterResult:
    retained: list[DatasetExample]
    rejected: list[tuple[DatasetExample, list[str]]]
    audit_sample: list[str]


class QualityFilter:
    """Applies quantitative checks to ensure dataset quality."""

    def __init__(self, cfg: DatasetBuildConfig, llm: LLMClient) -> None:
        self._cfg = cfg
        self._llm = llm
        self._rng = random.Random(cfg.quality.random_seed)
        self._embedder = self._load_embedder() if cfg.quality.use_embedding_filter else None
        self._nli_pipeline = self._load_nli_pipeline() if cfg.quality.use_nli_filter else None

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - heavy dependency guard
            raise RuntimeError(
                "sentence-transformers is required for embedding-based QC. "
                "Install with `pip install parallel-decoder-transformer[data]` "
                "or disable with quality.use_embedding_filter=false"
            ) from exc

        logger.info("Loading embedding model for quality filtering...")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _load_nli_pipeline(self):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - heavy dependency guard
            raise RuntimeError(
                "transformers is required for NLI-based contradiction filtering. "
                "Install with `pip install transformers` "
                "or disable with quality.use_nli_filter=false"
            ) from exc

        logger.info("Loading NLI model for contradiction detection...")
        return pipeline("text-classification", model="typeform/distilbert-base-uncased-mnli")

    def filter(self, examples: Iterable[DatasetExample]) -> QualityFilterResult:
        retained: list[DatasetExample] = []
        rejected: list[tuple[DatasetExample, list[str]]] = []
        audit: list[str] = []
        for example in examples:
            reasons = self._evaluate(example)
            if reasons:
                rejected.append((example, reasons))
                continue
            retained.append(example)
            if self._rng.random() < self._cfg.quality.sample_for_manual_audit:
                audit.append(example.example_id)
        return QualityFilterResult(retained=retained, rejected=rejected, audit_sample=audit)

    # ------------------------------------------------------------------ #
    # Individual checks                                                 #
    # ------------------------------------------------------------------ #

    def _evaluate(self, example: DatasetExample) -> list[str]:
        reasons: list[str] = []
        if not self._check_length_balance(example.sections.texts):
            reasons.append("length_balance")
        if not self._check_max_length_ratio(example.sections.texts):
            reasons.append("excessive_length_imbalance")
        alignment_score = self._notes_alignment(
            example.notes.true_notes, example.notes.speculative_notes
        )
        if alignment_score > self._cfg.quality.notes_kl_threshold:
            reasons.append(f"notes_alignment>{alignment_score:.3f}")
        contradiction = self._contradiction_rate(example)
        if contradiction > self._cfg.quality.contradiction_threshold:
            reasons.append(f"contradiction>{contradiction:.3f}")
        coverage_issues = self._coverage_alignment(example)
        reasons.extend(coverage_issues)
        return reasons

    def _check_length_balance(self, texts: Sequence[str]) -> bool:
        token_lengths = [self._llm.token_length(text) for text in texts]
        mean_len = statistics.mean(token_lengths)
        variance = statistics.pvariance(token_lengths) / (mean_len**2)
        return variance <= self._cfg.quality.length_balance_variance

    def _check_max_length_ratio(self, texts: Sequence[str]) -> bool:
        """Ensure no stream dominates (e.g., reject if one stream is 3x longer than another)."""
        token_lengths = [self._llm.token_length(text) for text in texts]
        if not token_lengths:
            return False
        max_len = max(token_lengths)
        min_len = min(token_lengths)
        if min_len == 0:
            return False
        ratio = max_len / min_len
        return ratio <= self._cfg.quality.max_length_ratio

    def _notes_alignment(
        self, true_notes: Sequence[StreamNotes], speculative_notes: Sequence[StreamNotes]
    ) -> float:
        # Skip if embedder not loaded
        if self._embedder is None:
            return 0.0  # Return 0 distance (perfect alignment) to skip this check

        distances: list[float] = []
        for true_payload, spec_payload in zip(true_notes, speculative_notes):
            true_text = self._notes_to_text(true_payload)
            spec_text = self._notes_to_text(spec_payload)
            embeddings = self._embedder.encode([true_text, spec_text])
            true_vec, spec_vec = embeddings[0], embeddings[1]
            cosine_sim = float(
                np.dot(true_vec, spec_vec)
                / (np.linalg.norm(true_vec) * np.linalg.norm(spec_vec) + 1e-8)
            )
            distances.append(1.0 - cosine_sim)
        return float(np.mean(distances)) if distances else 1.0

    def _notes_to_text(self, payload: StreamNotes) -> str:
        parts: list[str] = []
        for entity in payload.entities:
            if entity.name:
                parts.append(entity.name)
            parts.extend(alias for alias in entity.aliases if alias)
        for fact in payload.facts:
            snippet = " ".join(part for part in (fact.subj_id, fact.predicate, fact.object) if part)
            if snippet:
                parts.append(snippet)
            if fact.evidence_span.text:
                parts.append(fact.evidence_span.text)
        for coverage in payload.coverage:
            parts.append(f"{coverage.plan_item_id}:{coverage.status.value}")
        return " ".join(parts)

    def _contradiction_rate(self, example: DatasetExample) -> float:
        # Skip if NLI pipeline not loaded
        if self._nli_pipeline is None:
            return 0.0  # Return 0 contradiction rate to skip this check

        scores: list[float] = []
        for true_payload, spec_payload in zip(
            example.notes.true_notes, example.notes.speculative_notes
        ):
            premise = self._notes_to_text(true_payload)
            hypothesis = self._notes_to_text(spec_payload)
            result = self._nli_pipeline({"text": premise, "text_pair": hypothesis}, truncation=True)
            if not result:
                continue
            contradiction_score = 0.0
            for item in result:
                if item["label"].lower().startswith("contradiction"):
                    contradiction_score = float(item["score"])
                    break
            scores.append(contradiction_score)
        return float(np.mean(scores)) if scores else 0.0

    def _coverage_alignment(self, example: DatasetExample) -> list[str]:
        metadata = example.metadata or {}
        plan_items = metadata.get("plan_items")
        if not isinstance(plan_items, Mapping):
            return []
        catalog: Dict[str, set[str]] = {}
        for stream_id, entries in plan_items.items():
            stream_key = str(stream_id).strip().lower()
            if not stream_key:
                continue
            if isinstance(entries, Sequence):
                normalized_entries = {
                    str(item).strip().lower() for item in entries if str(item).strip()
                }
            else:
                normalized_entries = set()
            catalog[stream_key] = normalized_entries
        issues: list[str] = []
        for payload in example.notes.true_notes:
            stream_key = (payload.stream_id or "").strip().lower()
            expected = catalog.get(stream_key, set())
            coverage_items = {
                signal.plan_item_id.strip().lower()
                for signal in payload.coverage
                if signal.plan_item_id.strip()
            }
            missing = expected - coverage_items
            extra = coverage_items - expected
            if missing:
                issues.append(f"coverage_missing::{stream_key or 'unknown'}::{len(missing)}")
            if extra:
                issues.append(f"coverage_misaligned::{stream_key or 'unknown'}::{len(extra)}")
        return issues


__all__ = ["QualityFilter", "QualityFilterResult"]
