"""Preflight validators for plan generation inputs.

The preflight stage scans raw SQuAD/Wikipedia/Reasoning Gym records before any
LLM calls. It enforces token budgets, encoding checks, duplicate suppression,
and lightweight heuristics so that the downstream plan-generation runner only
sees clean candidates.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency guard
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
try:  # pragma: no cover
    from reasoning_gym.factory import create_dataset as rg_create_dataset  # type: ignore
except ImportError:  # pragma: no cover
    rg_create_dataset = None  # type: ignore[assignment]
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

import asyncio
from .async_llm import AsyncStructuredLLMClient, StructuredOutputRequest
from .plan_generation import (
    RG_PROMPT,
    SQUAD_PROMPT,
    WIKI_PROMPT,
    PlanGenerationConfig,
    _digest,
    _load_wiki_manifest,
    _stable_suffix,
)
from ..utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class WikiArticleClassifier:
    """LLM-based classifier that filters Wikipedia articles by structure."""

    _SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "keep": {"type": "boolean"},
            "tier": {
                "type": "string",
                "description": "tier1=multi-aspect technical, tier2=biography, tier3=place, tier4=event/problem-solution, avoid=reject",
            },
            "article_type": {"type": "string"},
            "rationale": {"type": "string"},
            "rejection_reason": {"type": "string"},
        },
        "required": ["keep", "tier", "article_type", "rationale", "rejection_reason"],
        "additionalProperties": False,
    }

    _SYSTEM_PROMPT = (
        "You are a senior curriculum designer and corpus curator for a Parallel Decoder Transformer (PDT). "
        "The PDT decodes three streams in parallel and is supervised with structured ENT/FACT/COVERAGE notes. "
        "Your job is to inspect a single Wikipedia article and decide whether it is a good candidate for PDT training.\n\n"
        "Core goals:\n"
        "- Approve only articles that can be decomposed into approximately three major, partially independent sections that could become the three PDT streams.\n"
        "- Prefer multi-aspect technical or conceptual topics, biographies with distinct life phases, geographic profiles, and historical events or problem–solution narratives.\n"
        "- Reject disambiguation or redirect pages, lists or catalogs, timelines, tables, short stubs, heavy-quotation or dialogue-only pages, pure how-to/procedure content, puzzle/quiz pages, and plot-only summaries.\n\n"
        "Use the following tier labels:\n"
        "- tier1: multi-aspect technical or conceptual topics with rich structure and multiple independent subtopics.\n"
        "- tier2: biographies with well-separated periods (for example, early life, career, impact).\n"
        "- tier3: places or institutions with clearly separated sections (for example, history, geography, culture).\n"
        "- tier4: historical events or problem–solution narratives that can be broken into setup, development, and outcome.\n"
        "- avoid: any article that does not work well for three partially independent streams or has severe quality or formatting issues.\n\n"
        "Output requirements:\n"
        "- You MUST return a single JSON object that strictly follows the provided schema.\n"
        "- Do NOT include any free-form text outside the JSON object.\n"
        "- Set `keep` to true only when the article is a strong candidate for a three-stream PDT plan; otherwise set `keep` to false.\n"
        "- Set `tier` to exactly one of: tier1, tier2, tier3, tier4, avoid.\n"
        '- Use `article_type` to briefly summarize the article category (for example, "technical_concept", "biography", "place", "event", "list", "plot_summary", "how_to", "other").\n'
        "- Use `rationale` to concisely explain the decision, referencing the article's structure (headings and sections) and why it is or is not suitable for three partially independent streams.\n"
        '- When `keep` is false, set `rejection_reason` to a short, stable label such as "list_like", "disambiguation", "plot_summary", "how_to", "stub", "unstructured", or "off_topic".\n\n'
        "Assume downstream systems will build detailed three-stream plans and ENT/FACT/COVERAGE notes. "
        "Your responsibility is only to filter for structural suitability and record a clear justification using the JSON fields."
    )

    def __init__(self, config: WikipediaClassifierConfig) -> None:
        self._config = config
        # Use the async Responses client for classifier calls to match
        # the rest of the pipeline and enable batch processing
        self._client = AsyncStructuredLLMClient(
            OpenAIClient(
                model=config.model,
                service_tier=config.service_tier,
                reasoning_effort=config.reasoning_effort,
            )
        )
        # Keep the retry budget low so we do not burn many calls on a single
        # problematic article.
        self._max_retries = 1

    async def classify(
        self, candidates: Sequence[_WikiCandidate]
    ) -> list[WikiClassificationResult]:
        if not candidates:
            return []
        requests: list[StructuredOutputRequest] = []
        for candidate in candidates:
            snippet = candidate.article_text[: self._config.top_n_chars]
            headings = candidate.headings[:15]
            heading_text = ", ".join(headings) if headings else "None"
            user_prompt = (
                f"Title: {candidate.title}\n"
                f"Headings: {heading_text}\n"
                f"Article excerpt (first {len(snippet)} chars):\n{snippet.strip() or '[empty]'}\n\n"
                "Decide if this article should be kept for PDT training."
            )
            requests.append(
                StructuredOutputRequest(
                    request_id=candidate.candidate_id,
                    messages=[
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema_name="wiki_article_classifier",
                    schema=self._SCHEMA,
                    max_output_tokens=self._config.max_output_tokens,
                    temperature=self._config.temperature,
                    top_p=0.95,
                    metadata={"candidate_id": candidate.candidate_id},
                )
            )

        # Submit batch of requests asynchronously
        results = await self._client.submit_batch(
            requests,
            concurrency=self._config.concurrency,
            max_retries=self._max_retries,
            retry_backoff=1.5,
            stop_on_error=False,  # Continue processing even if some requests fail
        )

        result_map: dict[str, WikiClassificationResult] = {}
        for result in results:
            if result.error is not None:
                outcome = result.error
                status = getattr(outcome, "status_code", None)
                error_type = getattr(outcome, "error_type", "")
                message = str(outcome)
                logger.warning(
                    "Wiki classifier request %s failed (status=%s, type=%s, attempts=%d): %s",
                    result.request.request_id,
                    status,
                    error_type,
                    getattr(outcome, "attempts", 0),
                    message,
                )
                # Treat all classifier errors as soft rejects so that a single bad
                # response does not abort the run. Rate limiting (429) is recorded
                # separately to make it easy to diagnose operational issues.
                reason = (
                    "classifier_rate_limited"
                    if status == 429 or error_type == "http_429"
                    else "classifier_error"
                )
                result_map[result.request.request_id] = WikiClassificationResult(
                    keep=False,
                    rejection_reason=reason,
                    rationale=message,
                    error=f"{error_type or 'unknown_error'} (status={status})",
                )
                continue
            response = result.response
            if response is None:
                result_map[result.request.request_id] = WikiClassificationResult(
                    keep=False,
                    rejection_reason="classifier_error",
                    rationale="No response received",
                    error="no_response",
                )
                continue
            payload = response.parsed_json or {}
            keep = bool(payload.get("keep"))
            tier = payload.get("tier")
            article_type = payload.get("article_type")
            rationale = payload.get("rationale")
            rejection_reason = payload.get("rejection_reason")
            result_map[result.request.request_id] = WikiClassificationResult(
                keep=keep,
                tier=str(tier) if tier else None,
                article_type=str(article_type) if article_type else None,
                rationale=str(rationale) if rationale else None,
                rejection_reason=str(rejection_reason) if rejection_reason else None,
            )
        ordered_results: list[WikiClassificationResult] = []
        for request in requests:
            classification = result_map.get(request.request_id)
            if classification is None:
                classification = WikiClassificationResult(
                    keep=False,
                    rejection_reason="classifier_missing_result",
                    rationale="No classifier result returned for this article.",
                    error="missing_result",
                )
            ordered_results.append(classification)
        return ordered_results

    # The _submit_with_retries method is no longer needed as
    # AsyncStructuredLLMClient.submit_batch handles retries internally


# --------------------------------------------------------------------------- #
# Configuration dataclasses                                                   #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SquadPreflightConfig:
    max_question_tokens: int = 512
    max_context_tokens: int = 3_000
    max_total_tokens: int = 8_192
    min_question_tokens: int = 1
    min_context_tokens: int = 20
    max_control_ratio: float = 0.01
    max_non_latin_ratio: float = 0.3


@dataclass(slots=True)
class WikipediaPreflightConfig:
    max_article_tokens: int = 25_000
    min_article_tokens: int = 200
    max_total_tokens: int = 100_000
    max_control_ratio: float = 0.01
    max_non_latin_ratio: float = 0.3
    disambiguation_filter: bool = True
    min_article_chars: int = 10_000
    max_article_chars: int = 30_000


@dataclass(slots=True)
class WikipediaClassifierConfig:
    enabled: bool = True
    model: str = "gpt-5.1"
    top_n_chars: int = 8_000
    batch_size: int = 100
    concurrency: int = 8
    max_output_tokens: int = 2_048
    temperature: float = 0.0
    service_tier: str | None = "flex"
    reasoning_effort: str = "low"


@dataclass(slots=True)
class ReasoningGymPreflightConfig:
    max_question_tokens: int = 1_500
    max_answer_tokens: int = 256
    max_total_tokens: int = 8_192
    max_control_ratio: float = 0.01
    max_non_latin_ratio: float = 0.3


@dataclass(slots=True)
class PreflightSettings:
    squad: SquadPreflightConfig = field(default_factory=SquadPreflightConfig)
    wikipedia: WikipediaPreflightConfig = field(default_factory=WikipediaPreflightConfig)
    reasoning_gym: ReasoningGymPreflightConfig = field(default_factory=ReasoningGymPreflightConfig)
    wikipedia_classifier: WikipediaClassifierConfig = field(
        default_factory=WikipediaClassifierConfig
    )
    per_message_overhead: int = 8
    max_json_bytes: int = 5 * 1024 * 1024
    dedupe: bool = True
    target_model: str = "gpt-5.1"
    language_ratio_threshold: float = 0.3


@dataclass(slots=True)
class PreflightRecord:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    report: dict[str, Any]


@dataclass(slots=True)
class _WikiCandidate:
    idx: int
    record_id: str
    candidate_id: str
    title: str
    article_text: str
    headings: list[str]
    system_prompt: str
    prompt: str
    token_counts: Mapping[str, int]
    source_metadata: dict[str, Any]
    digest_value: str


@dataclass(slots=True)
class WikiClassificationResult:
    keep: bool
    tier: str | None = None
    article_type: str | None = None
    rationale: str | None = None
    rejection_reason: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #


def _require_dependencies() -> None:
    if load_dataset is None or rg_create_dataset is None:
        raise RuntimeError(
            "datasets>=2.20 and reasoning_gym are required for preflight validation. "
            "Install them inside dataset_test_venv."
        )


def _tokenizer() -> tiktoken.Encoding:
    if tiktoken is None:  # pragma: no cover - guard for optional dependency
        raise RuntimeError(
            "tiktoken is required for preflight validation. Install it inside dataset_test_venv."
        )
    try:
        return tiktoken.get_encoding("cl100k_base")
    except ValueError:  # pragma: no cover - encoding lookup fallback
        return tiktoken.encoding_for_model("gpt-4o-mini")


def _token_len(encoding: tiktoken.Encoding, text: str) -> int:
    return len(encoding.encode(text))


def _chat_tokens(
    encoding: tiktoken.Encoding,
    system_prompt: str,
    user_prompt: str,
    *,
    overhead: int,
) -> dict[str, int]:
    system_tokens = _token_len(encoding, system_prompt)
    user_tokens = _token_len(encoding, user_prompt)
    total = system_tokens + user_tokens + 2 * overhead
    return {"system": system_tokens, "user": user_tokens, "total": total}


def _control_ratio(text: str) -> float:
    if not text:
        return 0.0
    controls = 0
    for char in text:
        code_point = ord(char)
        if code_point < 32 and char not in ("\n", "\r", "\t"):
            controls += 1
    return controls / max(1, len(text))


def _non_latin_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for char in letters if not char.isascii())
    return non_latin / max(1, len(letters))


def _has_nul(text: str) -> bool:
    return "\x00" in text


def _normalize_for_digest(text: str) -> str:
    return " ".join(text.lower().split())


def _json_size_bytes(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _make_accept_record(
    *,
    sample_id: str,
    domain: str,
    source: str,
    dataset_index: int,
    messages: list[dict[str, str]],
    prompt: str,
    token_counts: Mapping[str, int],
    source_metadata: Mapping[str, Any],
    digest_value: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "source": source,
        "dataset_index": dataset_index,
        "messages": messages,
        "prompt": prompt,
        "token_counts": dict(token_counts),
        "source_metadata": dict(source_metadata),
        "dedupe_key": digest_value,
    }


def _make_reject_record(
    *,
    domain: str,
    candidate_id: str,
    reason: str,
    token_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "domain": domain,
        "reason": reason,
    }
    if token_counts:
        record["token_counts"] = dict(token_counts)
    if metadata:
        record["metadata"] = dict(metadata)
    return record


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #


class PreflightRunner:
    """Validates dataset records and writes accepted/rejected manifests."""

    def __init__(
        self,
        *,
        plan_cfg: PlanGenerationConfig,
        settings: PreflightSettings | None = None,
        max_output_tokens: int = 16_384,
        plan_max_tokens: int = 256,
        output_dir: Path | None = None,
    ) -> None:
        _require_dependencies()
        self._plan_cfg = plan_cfg
        self._settings = settings or PreflightSettings()
        self._encoding = _tokenizer()
        self._tokenizer_name = getattr(self._encoding, "name", "cl100k_base")
        self._max_output_tokens = max_output_tokens
        self._plan_max_tokens = plan_max_tokens
        self._output_dir = output_dir
        classifier_cfg = self._settings.wikipedia_classifier
        self._wiki_classifier = (
            WikiArticleClassifier(classifier_cfg) if classifier_cfg.enabled else None
        )

    def run(self) -> PreflightRecord:
        start = time.time()
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        accepted_counts: MutableMapping[str, int] = defaultdict(int)
        rejects_by_reason: Counter[str] = Counter()

        # Open output files if output_dir is provided for incremental writing
        accepted_file = None
        rejected_file = None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            accepted_file = (self._output_dir / "preflight_accepted.jsonl").open(
                "w", encoding="utf-8"
            )
            rejected_file = (self._output_dir / "preflight_rejected.jsonl").open(
                "w", encoding="utf-8"
            )
            logger.info("Incremental output enabled: writing to %s", self._output_dir)

        try:
            self._run_squad_stage(
                accepted, rejected, accepted_counts, rejects_by_reason, accepted_file, rejected_file
            )
            self._run_reasoning_gym_stage(
                accepted, rejected, accepted_counts, rejects_by_reason, accepted_file, rejected_file
            )
            # Run the async wikipedia stage
            asyncio.run(
                self._run_wikipedia_stage(
                    accepted,
                    rejected,
                    accepted_counts,
                    rejects_by_reason,
                    accepted_file,
                    rejected_file,
                )
            )
        finally:
            if accepted_file:
                accepted_file.close()
            if rejected_file:
                rejected_file.close()

        elapsed = time.time() - start
        token_ceilings = {
            "qa": self._settings.squad.max_total_tokens,
            "math": self._settings.reasoning_gym.max_total_tokens,
            "survey": self._settings.wikipedia.max_total_tokens,
        }
        report = {
            "accepted_total": (
                len(accepted)
                if not self._output_dir
                else accepted_counts.get("qa", 0)
                + accepted_counts.get("math", 0)
                + accepted_counts.get("survey", 0)
            ),
            "accepted_by_domain": dict(accepted_counts),
            "rejected_total": (
                len(rejected) if not self._output_dir else sum(rejects_by_reason.values())
            ),
            "rejected_by_reason": dict(rejects_by_reason.most_common()),
            "targets": dict(self._plan_cfg.total_per_domain),
            "max_output_tokens": self._max_output_tokens,
            "plan_max_tokens": self._plan_max_tokens,
            "plan_within_output_budget": self._plan_max_tokens <= self._max_output_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "model_id": self._settings.target_model,
            "tokenizer": self._tokenizer_name,
            "token_ceilings": token_ceilings,
            "language_ratio_threshold": self._settings.language_ratio_threshold,
        }
        return PreflightRecord(accepted=accepted, rejected=rejected, report=report)

    # ------------------------------------------------------------------ #
    # Domain runners                                                     #
    # ------------------------------------------------------------------ #

    def _run_squad_stage(
        self,
        accepted: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        counts: MutableMapping[str, int],
        rejects_by_reason: Counter[str],
        accepted_file: Any = None,
        rejected_file: Any = None,
    ) -> None:
        target = int(self._plan_cfg.total_per_domain.get("qa", 0))
        if target <= 0:
            return
        dataset = load_dataset("squad", split=self._plan_cfg.squad_split)
        indices = list(range(len(dataset)))
        rng = json.dumps(self._plan_cfg.total_per_domain)  # deterministic seed proxy
        # Deterministic shuffle via hash of seed + dataset len
        indices.sort(key=lambda idx: _digest(f"qa:{idx}:{self._plan_cfg.seed}:{rng}"))
        dedupe_keys: set[str] = set()
        for idx in indices:
            record = dataset[int(idx)]
            outcome = self._validate_squad_record(idx, record, dedupe_keys)
            if outcome is None:
                continue
            if outcome["accepted"]:
                accepted.append(outcome["record"])
                if accepted_file:
                    accepted_file.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")
                    accepted_file.flush()
                counts["qa"] += 1
                if counts["qa"] >= target:
                    break
            else:
                rejected.append(outcome["record"])
                if rejected_file:
                    rejected_file.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")
                    rejected_file.flush()
                rejects_by_reason.update([outcome["record"]["reason"]])

    def _run_reasoning_gym_stage(
        self,
        accepted: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        counts: MutableMapping[str, int],
        rejects_by_reason: Counter[str],
        accepted_file: Any = None,
        rejected_file: Any = None,
    ) -> None:
        target = int(self._plan_cfg.total_per_domain.get("math", 0))
        if target <= 0:
            return
        dataset = rg_create_dataset(self._plan_cfg.reasoning_gym_dataset)
        total = len(dataset)
        if total == 0:
            raise RuntimeError(
                f"Reasoning Gym dataset '{self._plan_cfg.reasoning_gym_dataset}' is empty."
            )
        dedupe_keys: set[str] = set()
        idx = 0
        while counts["math"] < target and idx < total:
            dataset_idx = (self._plan_cfg.reasoning_gym_offset + idx) % total
            record = dataset[int(dataset_idx)]
            outcome = self._validate_reasoning_gym_record(dataset_idx, record, dedupe_keys)
            if outcome is None:
                idx += 1
                continue
            if outcome["accepted"]:
                accepted.append(outcome["record"])
                if accepted_file:
                    accepted_file.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")
                    accepted_file.flush()
                counts["math"] += 1
            else:
                rejected.append(outcome["record"])
                if rejected_file:
                    rejected_file.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")
                    rejected_file.flush()
                rejects_by_reason.update([outcome["record"]["reason"]])
            idx += 1

    async def _run_wikipedia_stage(
        self,
        accepted: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        counts: MutableMapping[str, int],
        rejects_by_reason: Counter[str],
        accepted_file: Any = None,
        rejected_file: Any = None,
    ) -> None:
        target = int(self._plan_cfg.total_per_domain.get("survey", 0))
        if target <= 0:
            return
        shard_paths = _load_wiki_manifest(self._plan_cfg.wiki_manifest)
        if self._plan_cfg.wiki_shard_limit is not None:
            shard_paths = shard_paths[: self._plan_cfg.wiki_shard_limit]
        dedupe_keys: set[str] = set()
        record_index = 0
        wiki_offset = self._plan_cfg.wiki_offset
        if wiki_offset > 0:
            logger.info("Wikipedia offset enabled: skipping first %d articles", wiki_offset)
        batch: list[_WikiCandidate] = []

        async def _flush_batch() -> bool:
            if not batch:
                return False
            if self._wiki_classifier is None:
                for candidate in batch:
                    self._accept_wiki_candidate(candidate, accepted, classification_result=None)
                    if accepted_file:
                        accepted_file.write(json.dumps(accepted[-1], ensure_ascii=False) + "\n")
                        accepted_file.flush()
                    counts["survey"] += 1
                    if counts["survey"] >= target:
                        batch.clear()
                        return True
                batch.clear()
                return False
            logger.info("Classifying batch of %d Wikipedia articles", len(batch))
            results = await self._wiki_classifier.classify(batch)
            for candidate, result in zip(batch, results):
                if result.keep:
                    self._accept_wiki_candidate(candidate, accepted, classification_result=result)
                    if accepted_file:
                        accepted_file.write(json.dumps(accepted[-1], ensure_ascii=False) + "\n")
                        accepted_file.flush()
                    counts["survey"] += 1
                    if counts["survey"] >= target:
                        batch.clear()
                        return True
                else:
                    reason = result.rejection_reason or (
                        "classifier_error" if result.error else "wiki_classifier_reject"
                    )
                    metadata = {
                        "classifier_tier": result.tier,
                        "classifier_article_type": result.article_type,
                        "classifier_rationale": result.rationale,
                    }
                    reject_record = _make_reject_record(
                        domain="survey",
                        candidate_id=candidate.candidate_id,
                        reason=reason,
                        token_counts=candidate.token_counts,
                        metadata=metadata,
                    )
                    rejected.append(reject_record)
                    if rejected_file:
                        rejected_file.write(json.dumps(reject_record, ensure_ascii=False) + "\n")
                        rejected_file.flush()
                    rejects_by_reason.update([reason])
            logger.info(
                "Batch complete: %d accepted, %d total survey accepted",
                sum(1 for c, r in zip(batch, results) if r.keep),
                counts["survey"],
            )
            batch.clear()
            return False

        for record in self._iter_wiki_records(shard_paths):
            # Skip records before the offset
            if record_index < wiki_offset:
                record_index += 1
                continue
            outcome = self._screen_wikipedia_record(record_index, record, dedupe_keys)
            record_index += 1
            if outcome is None:
                continue
            if not outcome["accepted"]:
                reject_record = outcome["record"]
                rejected.append(reject_record)
                if rejected_file:
                    rejected_file.write(json.dumps(reject_record, ensure_ascii=False) + "\n")
                    rejected_file.flush()
                rejects_by_reason.update([reject_record["reason"]])
                continue
            candidate: _WikiCandidate = outcome["candidate"]
            if self._wiki_classifier is None:
                self._accept_wiki_candidate(candidate, accepted, classification_result=None)
                if accepted_file:
                    accepted_file.write(json.dumps(accepted[-1], ensure_ascii=False) + "\n")
                    accepted_file.flush()
                counts["survey"] += 1
                if counts["survey"] >= target:
                    break
                continue
            batch.append(candidate)
            if len(batch) >= self._settings.wikipedia_classifier.batch_size:
                if await _flush_batch():
                    break
            if counts["survey"] >= target:
                break
        if counts["survey"] < target:
            await _flush_batch()

    def _screen_wikipedia_record(
        self,
        idx: int,
        record: Mapping[str, Any],
        dedupe_keys: set[str],
    ) -> dict[str, Any] | None:
        cfg = self._settings.wikipedia
        candidate_id = f"survey_{idx}"
        title = (record.get("title") or record.get("id") or "Untitled").strip()
        article_text = (record.get("text") or "").strip()
        if not article_text:
            return _reject("survey", candidate_id, "missing_article_text")
        article_chars = len(article_text)
        if article_chars > cfg.max_article_chars:
            return _reject(
                "survey",
                candidate_id,
                "article_chars_above_max",
                metadata={"article_chars": article_chars},
            )
        if article_chars < cfg.min_article_chars:
            return _reject(
                "survey",
                candidate_id,
                "article_chars_below_min",
                metadata={"article_chars": article_chars},
            )
        article_tokens = _token_len(self._encoding, article_text)
        if article_tokens > cfg.max_article_tokens:
            return _reject(
                "survey",
                candidate_id,
                f"article_tokens>{cfg.max_article_tokens}",
                token_counts={"article": article_tokens},
            )
        if article_tokens < cfg.min_article_tokens:
            return _reject("survey", candidate_id, "article_too_short")
        if _has_nul(article_text):
            return _reject("survey", candidate_id, "contains_nul")
        if _control_ratio(article_text) > cfg.max_control_ratio:
            return _reject("survey", candidate_id, "control_ratio_exceeded")
        language_ratio = _non_latin_ratio(article_text)
        if language_ratio > cfg.max_non_latin_ratio:
            return _reject(
                "survey",
                candidate_id,
                "non_latin_ratio_exceeded",
                metadata={"language_ratio": language_ratio},
            )
        lowered_title = title.lower()
        if cfg.disambiguation_filter and (
            lowered_title.startswith("list of")
            or lowered_title.endswith("(disambiguation)")
            or "may refer to" in article_text[:512].lower()
        ):
            return _reject("survey", candidate_id, "disambiguation_filter")
        system_prompt = "Return valid JSON following the provided schema."
        prompt = WIKI_PROMPT.format(title=title, body=article_text)
        tokens = _chat_tokens(
            self._encoding,
            system_prompt,
            prompt,
            overhead=self._settings.per_message_overhead,
        )
        if tokens["total"] > cfg.max_total_tokens:
            return _reject(
                "survey",
                candidate_id,
                f"total_tokens>{cfg.max_total_tokens}",
                token_counts=tokens,
            )
        payload = {
            "title": title,
            "article_text": article_text,
        }
        if _json_size_bytes(payload) > self._settings.max_json_bytes:
            return _reject("survey", candidate_id, "json_bytes_exceeded")
        digest_value = _digest(_normalize_for_digest(article_text))
        if self._settings.dedupe and digest_value in dedupe_keys:
            return _reject("survey", candidate_id, "duplicate_article")
        dedupe_keys.add(digest_value)
        headings = _extract_wiki_headings(article_text)
        record_id = str(record.get("id") or idx)
        source_metadata = {
            "source": "wikipedia",
            "dataset_index": int(idx),
            "title": title,
            "article_text": article_text,
            "input_text": article_text,
            "language_ratio": language_ratio,
            "article_chars": article_chars,
            "headings": headings[:32],
        }
        candidate = _WikiCandidate(
            idx=idx,
            record_id=record_id,
            candidate_id=candidate_id,
            title=title,
            article_text=article_text,
            headings=headings,
            system_prompt=system_prompt,
            prompt=prompt,
            token_counts=tokens,
            source_metadata=source_metadata,
            digest_value=digest_value,
        )
        return {"accepted": True, "candidate": candidate}

    def _accept_wiki_candidate(
        self,
        candidate: _WikiCandidate,
        accepted: list[dict[str, Any]],
        classification_result: WikiClassificationResult | None = None,
    ) -> None:
        suffix = _stable_suffix(
            "survey", f"{candidate.record_id}_{candidate.idx}", self._plan_cfg.seed
        )
        sample_id = f"survey_{candidate.record_id}_{suffix}"

        # Include classification results in source_metadata if available
        source_metadata = dict(candidate.source_metadata)
        if classification_result:
            source_metadata["classifier_tier"] = classification_result.tier
            source_metadata["classifier_article_type"] = classification_result.article_type
            source_metadata["classifier_rationale"] = classification_result.rationale

        # Don't include plan generation prompts in preflight accept records
        # The plan generation stage will build its own prompts
        accept_record = _make_accept_record(
            sample_id=sample_id,
            domain="survey",
            source="wikipedia",
            dataset_index=int(candidate.idx),
            messages=[],  # Empty messages for preflight - plan generation will build its own
            prompt="",  # Empty prompt for preflight - plan generation will build its own
            token_counts=candidate.token_counts,
            source_metadata=source_metadata,
            digest_value=candidate.digest_value,
        )
        accepted.append(accept_record)

    # ------------------------------------------------------------------ #
    # Record validators                                                  #
    # ------------------------------------------------------------------ #

    def _validate_squad_record(
        self,
        idx: int,
        record: Mapping[str, Any],
        dedupe_keys: set[str],
    ) -> dict[str, Any] | None:
        cfg = self._settings.squad
        question = (record.get("question") or "").strip()
        context = (record.get("context") or "").strip()
        candidate_id = f"qa_{idx}"
        if not question:
            return _reject("qa", candidate_id, "missing_question")
        if not context:
            return _reject("qa", candidate_id, "missing_context")
        if _has_nul(question) or _has_nul(context):
            return _reject("qa", candidate_id, "contains_nul")
        q_tokens = _token_len(self._encoding, question)
        c_tokens = _token_len(self._encoding, context)
        if q_tokens > cfg.max_question_tokens:
            return _reject(
                "qa",
                candidate_id,
                f"question_tokens>{cfg.max_question_tokens}",
                token_counts={"question": q_tokens},
            )
        if c_tokens > cfg.max_context_tokens:
            return _reject(
                "qa",
                candidate_id,
                f"context_tokens>{cfg.max_context_tokens}",
                token_counts={"context": c_tokens},
            )
        if c_tokens < cfg.min_context_tokens:
            return _reject("qa", candidate_id, "context_too_short")
        if q_tokens < cfg.min_question_tokens:
            return _reject("qa", candidate_id, "question_too_short")
        if _control_ratio(question + context) > cfg.max_control_ratio:
            return _reject("qa", candidate_id, "control_ratio_exceeded")
        language_ratio = _non_latin_ratio(context)
        if language_ratio > cfg.max_non_latin_ratio:
            return _reject(
                "qa",
                candidate_id,
                "non_latin_ratio_exceeded",
                metadata={"language_ratio": language_ratio},
            )
        prompt = SQUAD_PROMPT.format(question=question, context=context)
        system_prompt = "You always return valid JSON for the provided schema."
        tokens = _chat_tokens(
            self._encoding, system_prompt, prompt, overhead=self._settings.per_message_overhead
        )
        if tokens["total"] > cfg.max_total_tokens:
            return _reject(
                "qa",
                candidate_id,
                f"total_tokens>{cfg.max_total_tokens}",
                token_counts=tokens,
            )
        payload = {
            "question": question,
            "context": context,
            "system_prompt": system_prompt,
            "prompt": prompt,
        }
        json_size = _json_size_bytes(payload)
        if json_size > self._settings.max_json_bytes:
            return _reject("qa", candidate_id, "json_bytes_exceeded")
        digest_value = _digest(_normalize_for_digest(context))
        if self._settings.dedupe and digest_value in dedupe_keys:
            return _reject("qa", candidate_id, "duplicate_context")
        dedupe_keys.add(digest_value)
        record_id = str(record.get("id") or idx)
        suffix = _stable_suffix("qa", f"{record_id}_{idx}", self._plan_cfg.seed)
        sample_id = f"qa_{record_id}_{suffix}"
        source_metadata = {
            "source": "squad",
            "dataset_index": int(idx),
            "question": question,
            "context": context,
            "input_text": context,
            "language_ratio": language_ratio,
        }
        # Don't include plan generation prompts in preflight accept records
        # The plan generation stage will build its own prompts
        accept_record = _make_accept_record(
            sample_id=sample_id,
            domain="qa",
            source="squad",
            dataset_index=int(idx),
            messages=[],  # Empty messages for preflight
            prompt="",  # Empty prompt for preflight
            token_counts=tokens,
            source_metadata=source_metadata,
            digest_value=digest_value,
        )
        return {"accepted": True, "record": accept_record}

    def _validate_reasoning_gym_record(
        self,
        idx: int,
        record: Mapping[str, Any],
        dedupe_keys: set[str],
    ) -> dict[str, Any] | None:
        cfg = self._settings.reasoning_gym
        question = str(record.get("question", "")).strip()
        answer_json = json.dumps(record.get("answer"), ensure_ascii=False)
        if not question:
            return _reject("math", f"math_{idx}", "missing_question")
        if _has_nul(question) or _has_nul(answer_json):
            return _reject("math", f"math_{idx}", "contains_nul")
        q_tokens = _token_len(self._encoding, question)
        a_tokens = _token_len(self._encoding, answer_json)
        if q_tokens > cfg.max_question_tokens:
            return _reject(
                "math",
                f"math_{idx}",
                f"question_tokens>{cfg.max_question_tokens}",
                token_counts={"question": q_tokens},
            )
        if a_tokens > cfg.max_answer_tokens:
            return _reject(
                "math",
                f"math_{idx}",
                f"answer_tokens>{cfg.max_answer_tokens}",
                token_counts={"answer": a_tokens},
            )
        if _control_ratio(question + answer_json) > cfg.max_control_ratio:
            return _reject("math", f"math_{idx}", "control_ratio_exceeded")
        language_ratio = _non_latin_ratio(question + " " + answer_json)
        if language_ratio > cfg.max_non_latin_ratio:
            return _reject(
                "math",
                f"math_{idx}",
                "non_latin_ratio_exceeded",
                metadata={"language_ratio": language_ratio},
            )
        prompt = RG_PROMPT.format(
            dataset_name=self._plan_cfg.reasoning_gym_dataset,
            question=question,
            answer=answer_json,
        )
        system_prompt = "Return valid JSON following the provided schema."
        tokens = _chat_tokens(
            self._encoding, system_prompt, prompt, overhead=self._settings.per_message_overhead
        )
        if tokens["total"] > cfg.max_total_tokens:
            return _reject(
                "math",
                f"math_{idx}",
                f"total_tokens>{cfg.max_total_tokens}",
                token_counts=tokens,
            )
        digest_value = _digest(_normalize_for_digest(question))
        if self._settings.dedupe and digest_value in dedupe_keys:
            return _reject("math", f"math_{idx}", "duplicate_question")
        dedupe_keys.add(digest_value)
        identifier = f"{self._plan_cfg.reasoning_gym_dataset}_{idx}"
        suffix = _stable_suffix("math", identifier, self._plan_cfg.seed)
        sample_id = f"math_{identifier}_{suffix}"
        source_metadata = {
            "source": self._plan_cfg.reasoning_gym_dataset,
            "dataset_index": int(idx),
            "question": question,
            "answer": answer_json,
            "input_text": question,
            "language_ratio": language_ratio,
        }
        # Don't include plan generation prompts in preflight accept records
        # The plan generation stage will build its own prompts
        accept_record = _make_accept_record(
            sample_id=sample_id,
            domain="math",
            source=self._plan_cfg.reasoning_gym_dataset,
            dataset_index=int(idx),
            messages=[],  # Empty messages for preflight
            prompt="",  # Empty prompt for preflight
            token_counts=tokens,
            source_metadata=source_metadata,
            digest_value=digest_value,
        )
        return {"accepted": True, "record": accept_record}

    def _validate_wikipedia_record(
        self,
        idx: int,
        record: Mapping[str, Any],
        dedupe_keys: set[str],
    ) -> dict[str, Any] | None:
        cfg = self._settings.wikipedia
        title = (record.get("title") or record.get("id") or "Untitled").strip()
        article_text = (record.get("text") or "").strip()
        candidate_id = f"survey_{idx}"
        if not article_text:
            return _reject("survey", candidate_id, "missing_article_text")
        article_tokens = _token_len(self._encoding, article_text)
        if article_tokens > cfg.max_article_tokens:
            return _reject(
                "survey",
                candidate_id,
                f"article_tokens>{cfg.max_article_tokens}",
                token_counts={"article": article_tokens},
            )
        if article_tokens < cfg.min_article_tokens:
            return _reject("survey", candidate_id, "article_too_short")
        if _has_nul(article_text):
            return _reject("survey", candidate_id, "contains_nul")
        if _control_ratio(article_text) > cfg.max_control_ratio:
            return _reject("survey", candidate_id, "control_ratio_exceeded")
        language_ratio = _non_latin_ratio(article_text)
        if language_ratio > cfg.max_non_latin_ratio:
            return _reject(
                "survey",
                candidate_id,
                "non_latin_ratio_exceeded",
                metadata={"language_ratio": language_ratio},
            )
        if cfg.disambiguation_filter and title.lower().startswith("list of"):
            return _reject("survey", candidate_id, "disambiguation_filter")
        prompt = WIKI_PROMPT.format(title=title, body=article_text)
        system_prompt = "Return valid JSON following the provided schema."
        tokens = _chat_tokens(
            self._encoding, system_prompt, prompt, overhead=self._settings.per_message_overhead
        )
        if tokens["total"] > cfg.max_total_tokens:
            return _reject(
                "survey",
                candidate_id,
                f"total_tokens>{cfg.max_total_tokens}",
                token_counts=tokens,
            )
        payload = {
            "title": title,
            "article_text": article_text,
        }
        if _json_size_bytes(payload) > self._settings.max_json_bytes:
            return _reject("survey", candidate_id, "json_bytes_exceeded")
        digest_value = _digest(_normalize_for_digest(article_text))
        if self._settings.dedupe and digest_value in dedupe_keys:
            return _reject("survey", candidate_id, "duplicate_article")
        dedupe_keys.add(digest_value)
        record_id = str(record.get("id") or idx)
        suffix = _stable_suffix("survey", f"{record_id}_{idx}", self._plan_cfg.seed)
        sample_id = f"survey_{record_id}_{suffix}"
        source_metadata = {
            "source": "wikipedia",
            "dataset_index": int(idx),
            "title": title,
            "article_text": article_text,
            "input_text": article_text,
            "language_ratio": language_ratio,
        }
        # Don't include plan generation prompts in preflight accept records
        # The plan generation stage will build its own prompts
        accept_record = _make_accept_record(
            sample_id=sample_id,
            domain="survey",
            source="wikipedia",
            dataset_index=int(idx),
            messages=[],  # Empty messages for preflight
            prompt="",  # Empty prompt for preflight
            token_counts=tokens,
            source_metadata=source_metadata,
            digest_value=digest_value,
        )
        return {"accepted": True, "record": accept_record}

    def _iter_wiki_records(self, shard_paths: Iterable[Path]) -> Iterator[Mapping[str, Any]]:
        for path in shard_paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _extract_wiki_headings(article_text: str) -> list[str]:
    headings: list[str] = []
    for raw_line in article_text.splitlines():
        line = raw_line.strip()
        if len(line) < 5 or not line.startswith("==") or not line.endswith("=="):
            continue
        stripped = line.strip("=").strip()
        if stripped:
            headings.append(stripped)
    return headings


def _reject(
    domain: str,
    candidate_id: str,
    reason: str,
    *,
    token_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "accepted": False,
        "record": _make_reject_record(
            domain=domain,
            candidate_id=candidate_id,
            reason=reason,
            token_counts=token_counts,
            metadata=metadata,
        ),
    }


__all__ = [
    "PreflightRecord",
    "PreflightRunner",
    "PreflightSettings",
    "ReasoningGymPreflightConfig",
    "SquadPreflightConfig",
    "WikipediaClassifierConfig",
    "WikipediaPreflightConfig",
]
