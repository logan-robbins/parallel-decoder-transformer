"""Raw data sourcing utilities for the dataset pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .config import WikipediaSourceConfig
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WikipediaArticle:
    """Structured representation of a Wikipedia article."""

    article_id: str
    title: str
    abstract: str | None
    text: str
    sections: list[dict[str, Any]]
    metadata: dict[str, Any]


def _flatten_sections(sections: Sequence[Mapping[str, Any]] | None) -> str:
    if not sections:
        return ""
    return "\n\n".join(
        str(section.get("text") or section.get("content") or "") for section in sections
    )


class WikipediaSource:
    """Loads and filters Wikipedia articles based on length and section constraints."""

    def __init__(self, config: WikipediaSourceConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    def stream(self) -> Iterator[WikipediaArticle]:
        """Yield articles that pass filtering constraints."""

        logger.info(
            "Loading Wikipedia dataset %s config=%s split=%s",
            self._config.dataset,
            self._config.config_name,
            self._config.split,
        )
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "The 'datasets' package is required for Wikipedia sourcing. Install it via `pip install datasets`."
            ) from exc
        # Load dataset - use local files if provided
        load_kwargs = {
            "path": self._config.dataset,
            "split": self._config.split,
            "streaming": self._config.max_articles is None,
        }

        # Add config name if not using local files
        if self._config.data_files:
            load_kwargs["data_files"] = self._config.data_files
        else:
            load_kwargs["name"] = self._config.config_name

        dataset = load_dataset(**load_kwargs)
        if hasattr(dataset, "shuffle"):
            # buffer_size is only for streaming mode
            if self._config.max_articles is None:  # streaming mode
                dataset = dataset.shuffle(seed=self._config.random_seed, buffer_size=self._config.shuffle_buffer)  # type: ignore[attr-defined]
            else:  # regular mode
                dataset = dataset.shuffle(seed=self._config.random_seed)  # type: ignore[attr-defined]

        yielded = 0
        for record in dataset:
            article = self._convert_record(record)
            if article is None:
                continue
            yield article
            yielded += 1
            if self._config.max_articles is not None and yielded >= self._config.max_articles:
                break

    def sample(self, target: int) -> list[WikipediaArticle]:
        """Materialise a finite list of filtered Wikipedia articles."""

        results: list[WikipediaArticle] = []
        for article in self.stream():
            results.append(article)
            if len(results) >= target:
                break
        return results

    def _convert_record(self, record: Mapping[str, Any]) -> WikipediaArticle | None:
        text = str(record.get(self._config.text_field, "")).strip()
        if not text:
            return None
        token_length = self._llm.token_length(text)
        if token_length < self._config.min_article_tokens:
            return None

        sections_raw = (
            record.get(self._config.section_field) if self._config.section_field else None
        )
        sections: list[dict[str, Any]]
        if isinstance(sections_raw, Mapping):
            sections = [
                {
                    "title": key,
                    "text": value if isinstance(value, str) else _flatten_sections(value),
                }
                for key, value in sections_raw.items()
            ]
        elif isinstance(sections_raw, Sequence):
            sections = [
                {
                    "title": sec.get("title", ""),
                    "text": sec.get("text") or sec.get("content") or "",
                }
                for sec in sections_raw
                if isinstance(sec, Mapping)
            ]
        else:
            sections = []

        if self._config.min_sections and len(sections) < self._config.min_sections:
            return None

        article_id = str(
            record.get(self._config.article_id_field) or record.get("id") or record.get("pageid")
        )
        title = str(record.get(self._config.title_field) or record.get("title") or "").strip()
        abstract: str | None = None
        if self._config.abstract_field:
            abstract_value = record.get(self._config.abstract_field)
            if isinstance(abstract_value, str):
                abstract = abstract_value.strip()

        metadata = {
            "source": "wikipedia",
            "snapshot": self._config.config_name,
            "token_length": token_length,
            "section_count": len(sections),
        }
        return WikipediaArticle(
            article_id=article_id,
            title=title or article_id,
            abstract=abstract,
            text=text,
            sections=sections,
            metadata=metadata,
        )


__all__ = ["WikipediaArticle", "WikipediaSource"]
