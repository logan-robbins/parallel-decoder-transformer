"""Utility wrapper around sequence classification NLI models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required to use NliScorer. Install it via `pip install transformers`."
    ) from exc


@dataclass(slots=True)
class NliScorerConfig:
    model_name: str
    max_length: int = 384


class NliScorer:
    def __init__(self, config: NliScorerConfig, *, device: torch.device) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.label_index = self._resolve_label_indices()

    def _resolve_label_indices(self) -> Dict[str, int]:
        mapping = self.model.config.label2id
        if mapping:
            return {key.lower(): value for key, value in mapping.items()}
        # Fallback for common ordering (contradiction, neutral, entailment)
        return {"contradiction": 0, "neutral": 1, "entailment": 2}

    def score(self, pairs: Iterable[Tuple[str, str]]) -> torch.Tensor:
        premises: List[str] = []
        hypotheses: List[str] = []
        for premise, hypothesis in pairs:
            premises.append(premise)
            hypotheses.append(hypothesis)
        if not premises:
            return torch.zeros((0, len(self.label_index)), device=self.device)
        encoded = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
        return torch.softmax(logits, dim=-1)


__all__ = ["NliScorer", "NliScorerConfig"]
