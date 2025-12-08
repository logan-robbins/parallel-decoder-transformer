"""Plan generation pipeline with batched OpenAI Responses calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency guard
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
try:  # pragma: no cover
    from reasoning_gym.factory import create_dataset as rg_create_dataset  # type: ignore
except ImportError:  # pragma: no cover
    rg_create_dataset = None  # type: ignore[assignment]
from tqdm import tqdm

from parallel_decoder_transformer.datasets.async_llm import (
    AsyncStructuredLLMClient,
    SequentialStructuredLLMClient,
    StructuredOutputRequest,
    StructuredRequestError,
)
from parallel_decoder_transformer.datasets.config import GenerationConfig
from parallel_decoder_transformer.utils.llm_client_factory import create_llm_client

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# 1. PLAN SCHEMA
# -------------------------------------------------------------------------
# This schema defines the interface between the Planner (LLM) and the
# Data Pipeline. It is strict to ensure the downstream "Teacher"
# (True Note Extractor) can deterministically slice the ground truth text.

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "array",
            "description": "Step-by-step logic justifying the decomposition strategy.",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5,
        },
        "streams": {
            "type": "array",
            "description": "The parallel execution slots.",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "header": {
                        "type": "string",
                        "description": "Section title for the final output.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "High-level content descriptor for note seeding.",
                    },
                    "entities": {
                        "type": "array",
                        "description": "Initial ENT templates to seed the DNB.",
                        "items": {"type": "string"},
                    },
                    "constraints": {
                        "type": "array",
                        "description": "Initial FACT/COVERAGE templates to seed the DNB.",
                        "items": {"type": "string"},
                    },
                    "section_contract": {
                        "type": "object",
                        "description": "The EXACT slice of the source text this stream owns. Critical for Teacher signal extraction.",
                        "properties": {
                            "type": {"type": "string", "enum": ["source_slice"]},
                            "start_idx": {"type": "integer"},
                            "end_idx": {"type": "integer"},
                        },
                        "required": ["type", "start_idx", "end_idx"],
                        "additionalProperties": False,
                    },
                    "notes_contract": {
                        "type": "array",
                        "description": "Mandatory semantic keys this stream must cover.",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": [
                    "header",
                    "summary",
                    "entities",
                    "constraints",
                    "section_contract",
                    "notes_contract",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "streams"],
    "additionalProperties": False,
}


# -------------------------------------------------------------------------
# 2. WIKI PLANNER PROMPT (Data Generation Phase)
# -------------------------------------------------------------------------
# This prompt drives the offline data construction. It takes a raw Wikipedia
# article and "decomposes" it into a training example for the PDT.
#
# OBJECTIVE: Create a Plan P such that:
#   1. Sections are roughly equal length (Load Balancing).
#   2. Sections are distinct (Independence).
#   3. 'section_contract' accurately maps to the source text (Teacher Alignment).

BIO_SKIP_TOKEN = "SKIP_BIOGRAPHY"

WIKI_PROMPT = """You are an expert Task Decomposition Planner for a Parallel Decoder Transformer (PDT). Your goal is to convert a linear text into a parallel execution plan.

You will be given a `Topic` and the full `Article Text`. You must generate a JSON Plan that decomposes this article into exactly **3 parallel streams**.

### The Architecture: Why this matters
The PDT uses a **Dynamic Notes Bus (DNB)**. Streams decode in parallel and coordinate by exchanging semantic "Notes".
* **Teacher Model:** Extracts "True Notes" ($N$) from the text slice.
* **Student Model:** Generates text using only "Speculative Notes" ($\\widehat{{N}}$).
* **Robustness:** The architecture handles mid-paragraph dependencies via the Note Bus. Do not fear splitting dense sections.

### Requirements for a Valid Plan

1.  **Strict Partitioning (The Contract):**
    * You must slice the provided `Article Text` into 3 non-overlapping, contiguous segments.
    * Use `section_contract` to specify the exact `start_idx` and `end_idx`.
    * **Coverage:** Together, the 3 streams must cover the essence of the article.

2.  **Aggressive Load Balancing (CRITICAL):**
    * **The Golden Rule:** The latency of the system is determined by the *longest* stream.
    * **Target Metric:** Each stream should be roughly **1/3rd** of the total text length.
    * **Penalty:** A plan where one stream is >50% longer than another is a FAILURE.
    * **Action:** You must split long sections (e.g., a massive "History" or "Biology" block) into two different streams. Do not preserve topic boundaries if it destroys load balancing.

3.  **Semantic Independence via Pivot Points:**
    * While balancing load, look for logical pivot points (e.g., shift from "Early Career" to "Late Career", or "Physical Anatomy" to "Functional Anatomy").
    * Stream 2 should not *linguistically* depend on the last sentence of Stream 1 (e.g., avoid starting with "However, ..."). It should stand alone given the Plan.

4.  **The "Blind Start" Rule:**
    * Stream 2 **DOES NOT KNOW** what Stream 1 has written. It only knows the Plan.
    * **Correct:** "This section covers the legal aftermath." (Semantic reference)
    * **Incorrect:** "This section discusses the consequences of the event mentioned in Stream 1." (Temporal dependency)

### The Breaking Heuristic (How to handle Long Sections)
If the source text has one massive section (e.g., indices 5000-11000 are all "Paleobiology"):
* **DO NOT** assign 5000-11000 to Stream 3.
* **DO** split it.
    * Stream 2: 2600-7000 (Anatomy + Early Paleobiology)
    * Stream 3: 7000-11350 (Late Paleobiology + Classification)
* **It is acceptable** to split a thematic block as long as the cut happens at a sentence boundary.

### Schema Fields Guide

* `reasoning`: Explain *why* you chose these split points. Explicitly mention the character counts of each stream to prove balance.
* `header`: A title for the section.
* `summary`: A 1-sentence content summary.
* `entities`: List 2-3 key entities.
* `constraints`: List 1-2 key facts.
* `section_contract`:
    * `type`: "source_slice"
    * `start_idx`: Integer (0-based index)
    * `end_idx`: Integer (exclusive)
* `notes_contract`: A list of 2-3 high-level requirements.

### Forbidden Phrases (Automatic Rejection)
* "Stream 1", "Stream 2", "Stream 3"
* "The previous stream", "The prior section"
* "Continues from...", "Builds on..."

**DO NOT** spend more than 30000 tokens on your reasoning alone or your response will be rejected. This is a hard limit. Be concise and deliberate.
**DO** summarize/condense multi-year timelines, you must be careufl not to exceed total token limit of 100000 (100k) tokens if possible. 
**REFERENCE WINDOW:** You must rely *exclusively* on the Article Text provided below. Its approximate character length (post-trim) is {char_count}; every `section_contract` must stay within `[0, {char_count})`, and `end_idx` must always be greater than `start_idx`. Do not infer or assume any content that is not explicitly present in the provided text.

### Biography Guard (STRICT)
If the article is a biography/profile of a specific person (signals include birth/death dates, phrases such as "was a", life timeline headings, or detailed personal achievements), you **must not** generate a plan. Instead:
* Output the single token `{bio_skip_token}` (uppercase, no punctuation, no JSON).
* Do **not** emit any additional text, reasoning, or metadata.
This guard applies to all biographies, regardless of domain (e.g., politicians, athletes, artists).


### Output Format
Return **ONLY** a valid JSON object matching `PLAN_SCHEMA`. No markdown.

---
Topic: {title}
Approximate Character Count (post-trim): {char_count}
Article Text:
{body}
"""


_TRAILING_SECTION_REGEX = re.compile(
    r"\n\s*(references|see also|external links|further reading|notes|bibliography|sources|citations|updates)\s*\n",
    re.IGNORECASE,
)

# -------------------------------------------------------------------------
# 3. SQUAD PROMPT (QA / Long-Form Explanation)
# -------------------------------------------------------------------------
# Rationale:
# To train a Parallel Decoder, we cannot just output a short answer. We must
# distill a "Gold" sequential explanation (teacher) into 3 parallel components.
#
# Stream Strategy: "Triangulation"
#   1. Grounding: Extracts raw evidence/quotes from the Context.
#   2. Reasoning: Connects evidence to the Question (inference layer).
#   3. Synthesis: Formulates the final Answer string and confidence check.

SQUAD_PROMPT = """You are a Reasoning Distillation Planner for a Parallel Decoder.
Your goal is to decompose a "Reference Explanation" for a SQuAD question into 3 parallel streams.

We are training a Student model to answer questions in parallel. To do this, we take a sequential
Gold Explanation (provided below) and slice it into 3 cooperating streams.

### Inputs
* **Question**: {{question}}
* **Context**: {{context}}
* **Reference Explanation**: {{reference_response}}  <-- THIS is the text you must slice.

### The Architecture: Why this matters
The PDT executes all 3 streams **simultaneously**.
* **CRITICAL:** Stream 2 **CANNOT** wait for Stream 1 to finish. It must start generating immediately.
* Do NOT decompose into "Step 1 -> Step 2 -> Step 3" (sequential dependency).
* DO decompose into "Aspect A || Aspect B || Synthesis".

### The Plan (JSON)
Generate a JSON object matching `PLAN_SCHEMA`.

1.  **Decomposition Strategy (Aspect-Based)**:
    * **Stream 1 (Aspect A)**: Owns evidence regarding the *first* logical component (e.g., the first entity mentioned, the early timeline, or the "Yes" arguments).
    * **Stream 2 (Aspect B)**: Owns evidence regarding the *second* logical component (e.g., the second entity, the later timeline, or the "No" arguments).
    * **Stream 3 (Synthesis)**: Owns the final judgment. It synthesizes the partial evidence from Streams 1 & 2 into the final Answer string.

2.  **Section Contract (Indices)**:
    * The `section_contract` for each stream must define the `start_idx` and `end_idx` within the `Reference Explanation` string.
    * The union of these 3 slices must cover the entire `Reference Explanation`.

3.  **Notes & Constraints**:
    * Define `entities` (Context Entities) and `constraints` (Evidence Facts) that each stream needs to generate its slice.

Return ONLY the JSON.

Question: {{question}}
Reference Explanation:
{{reference_response}}
"""

# -------------------------------------------------------------------------
# 4. REASONING GYM PROMPT (Chain-of-Thought / Math)
# -------------------------------------------------------------------------
# Rationale:
# Math/Logic problems are inherently sequential. To parallelize them, we rely on
# "Checkpointing". The Reference Solution is sliced into 3 phases, where the
# Notes Bus acts as the state transfer (passing variable values) between phases.
#
# Stream Strategy: "State Hand-Off"
#   1. Setup: Variable definition, equation setup, parsing givens.
#   2. Execution: The heavy algebraic/logical manipulation.
#   3. Finalization: Result normalization, unit check, final formatting.

RG_PROMPT = """You are a Chain-of-Thought Decomposition Engine.
Your task is to convert a linear "Reference Solution" into a 3-stage Parallel Plan.

### The Architecture: Why this matters
The PDT executes all 3 streams **simultaneously**.
* **CRITICAL:** Stream 2 cannot wait for Stream 1's text. It must start solving immediately.
* We rely on the **Dynamic Notes Bus (DNB)** to pass variable values (e.g., "x=5") from Stream 1 to Stream 2 in real-time.

### Inputs
* **Dataset**: {{dataset_name}}
* **Question**: {{question}}
* **Reference Solution**: {{solution_trace}}  <-- THIS is the text you must slice.

### The Plan (JSON)
Generate a JSON object matching `PLAN_SCHEMA`.

1.  **Strict Load Balancing**:
    * Math solutions often have a long "middle". You must balance the character count of the `Reference Solution` across the 3 streams as evenly as possible.

2.  **Decomposition Strategy (Parallel State Hand-Off)**:
    * **Stream 1 (Setup & Givens)**: Owns the definitions, variable extractions, and setting up the initial equations. It broadcasts the "Givens".
    * **Stream 2 (Core Execution)**: Owns the algebraic manipulation and solving steps. It *assumes* Stream 1 is broadcasting valid variable states via the bus.
    * **Stream 3 (Verification & Answer)**: Owns the final result formatting, unit checks, and sanity verification (e.g., "Is the answer positive?").

3.  **Notes as State Transfer**:
    * **Stream 1 Notes**: Must broadcast the "Given" variables and Setup (e.g., "FACT:radius=5").
    * **Stream 2 Notes**: Must broadcast intermediate steps (e.g., "FACT:area=25pi").
    * **Stream 3 Notes**: Must broadcast the final answer for consistency.

Return ONLY the JSON.

Question: {{question}}
Reference Solution:
{{solution_trace}}
"""


# --------------------------------------------------------------------------- #
# Data classes                                                               #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PlanTask:
    sample_id: str
    domain: str
    prompt: str
    input_messages: Sequence[Mapping[str, str]]
    source_metadata: Mapping[str, Any]
    output_path: Path
    schema: Mapping[str, Any] = field(default_factory=lambda: PLAN_SCHEMA)

    def to_structured_request(self, cfg: GenerationConfig) -> StructuredOutputRequest:
        metadata = self._build_metadata()
        return StructuredOutputRequest(
            request_id=self.sample_id,
            messages=list(self.input_messages),
            schema_name=f"{self.domain}_plan",
            schema=self.schema,
            # For reasoning models (gpt-5.1), max_output_tokens includes reasoning
            # tokens. Use 4k to control cost.
            max_output_tokens=max(cfg.max_new_tokens, 4_096),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop_sequences=tuple(cfg.stop_sequences),
            seed=cfg.seed,
            metadata=metadata,
        )

    def _build_metadata(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "output_path": str(self.output_path),
            "source": str(self.source_metadata.get("source", "")),
            "dataset_index": str(self.source_metadata.get("dataset_index", "")),
        }
        metadata["prompt_digest"] = _digest(self.prompt)
        input_text = self.source_metadata.get("input_text")
        if isinstance(input_text, str) and input_text:
            metadata["input_digest"] = _digest(input_text)
        return metadata


@dataclass(slots=True)
class GeneratedPlan:
    sample_id: str
    domain: str
    path: Path
    payload: Mapping[str, Any]
    raw_text: str
    latency_ms: float


@dataclass(slots=True)
class PlanGenerationConfig:
    total_per_domain: Mapping[str, int]
    # Number of requests to group per async submission. The CLI typically
    # overrides this via --plan-batch-size; keep the default modest so that
    # "no‑args" runs stay conservative.
    batch_size: int = 24
    # Maximum number of in‑flight planner requests. Lowered from 15 to 8 to be
    # more cautious with API rate limits and avoid long‑running jobs stalling
    # under bursty traffic.
    concurrency: int = 8
    seed: int = 41
    output_root: Path = Path("data/prep/plans")
    wiki_manifest: Path = Path("data/manifests/wikipedia_20231101_en_train.json")
    wiki_shard_limit: int | None = None
    wiki_offset: int = 0
    squad_split: str = "train"
    reasoning_gym_dataset: str = "simple_equations"
    reasoning_gym_offset: int = 0
    use_async_client: bool = True
    sequential_sleep_seconds: float = 1.0
    resume_existing: bool = True
    preflight_manifest: Path | None = None


# --------------------------------------------------------------------------- #
# Task builders                                                               #
# --------------------------------------------------------------------------- #


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_article_text(text: Any, *, min_position_ratio: float = 0.4) -> str:
    """
    Trim trailing Wikipedia sections (References, See also, etc.) so planners never
    allocate tokens to boilerplate.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if not cleaned:
        return ""
    match = _TRAILING_SECTION_REGEX.search(cleaned)
    if match and match.start() >= int(len(cleaned) * min_position_ratio):
        cleaned = cleaned[: match.start()].rstrip()
    return cleaned


def _sample_indices(total: int, count: int, *, rng: random.Random) -> list[int]:
    if count >= total:
        return list(range(total))
    return rng.sample(range(total), count)


def build_tasks_from_manifest(cfg: PlanGenerationConfig) -> list[PlanTask]:
    manifest_path = cfg.preflight_manifest
    if manifest_path is None:
        return []
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Preflight manifest {path} does not exist.")
    limits = {
        domain: int(cfg.total_per_domain.get(domain, 0)) for domain in ("qa", "math", "survey")
    }
    counts: dict[str, int] = defaultdict(int)
    tasks: list[PlanTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            domain = str(record.get("domain") or "").strip()
            if not domain:
                continue
            if limits.get(domain, 0) <= 0:
                continue
            if counts[domain] >= limits[domain]:
                continue
            sample_id = str(record.get("sample_id") or "").strip()
            if not sample_id:
                continue

            source_metadata = dict(record.get("source_metadata") or {})
            if domain == "survey":
                article_text = source_metadata.get("article_text") or source_metadata.get(
                    "input_text", ""
                )
                prepared_text = _prepare_article_text(article_text)
                source_metadata["article_text"] = prepared_text
                source_metadata["input_text"] = prepared_text
                source_metadata["char_count"] = len(prepared_text)

            # Reconstruct prompts from source_metadata when messages are empty
            messages = record.get("messages")
            prompt = str(record.get("prompt") or record.get("user_prompt") or "")

            if not messages or (messages and not any(m.get("content") for m in messages)):
                # Rebuild prompts based on domain and source_metadata
                system_prompt = "Return valid JSON following the provided schema."

                if domain == "survey":
                    # Rebuild Wikipedia prompt from source_metadata
                    title = source_metadata.get("title", "Untitled")
                    article_text = source_metadata.get(
                        "article_text", source_metadata.get("input_text", "")
                    )
                    char_count = source_metadata.get("char_count")
                    if char_count is None:
                        char_count = len(article_text)
                        source_metadata["char_count"] = char_count
                    prompt = WIKI_PROMPT.format(
                        title=title,
                        body=article_text,
                        char_count=char_count,
                        bio_skip_token=BIO_SKIP_TOKEN,
                    )
                elif domain == "qa":
                    # Rebuild SQuAD prompt from source_metadata
                    question = source_metadata.get("question", "")
                    context = source_metadata.get("context", source_metadata.get("input_text", ""))
                    prompt = SQUAD_PROMPT.format(question=question, context=context)
                elif domain == "math":
                    # Rebuild Reasoning Gym prompt from source_metadata
                    dataset_name = source_metadata.get("source", cfg.reasoning_gym_dataset)
                    question = source_metadata.get("question", "")
                    answer = source_metadata.get("answer", "{}")
                    prompt = RG_PROMPT.format(
                        dataset_name=dataset_name, question=question, answer=answer
                    )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

            normalized_messages = [
                {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
                for msg in messages
            ]
            if not normalized_messages:
                continue

            output_path = cfg.output_root / domain / f"{sample_id}.json"
            tasks.append(
                PlanTask(
                    sample_id=sample_id,
                    domain=domain,
                    prompt=prompt or normalized_messages[-1]["content"],
                    input_messages=normalized_messages,
                    output_path=output_path,
                    source_metadata=source_metadata,
                )
            )
            counts[domain] += 1
    missing = {
        domain: max(0, limits.get(domain, 0) - counts.get(domain, 0))
        for domain in limits
        if limits.get(domain, 0) > 0
    }
    unmet = {domain: shortfall for domain, shortfall in missing.items() if shortfall > 0}
    if unmet:
        logger.warning("Preflight manifest %s lacked %s samples", path, unmet)
    return tasks


def build_squad_tasks(cfg: PlanGenerationConfig, *, rng: random.Random) -> list[PlanTask]:
    target = int(cfg.total_per_domain.get("qa", 0))
    if target <= 0:
        return []
    if load_dataset is None:
        raise RuntimeError("datasets package is required to build SQuAD tasks.")
    dataset = load_dataset("squad", split=cfg.squad_split)
    indices = _sample_indices(len(dataset), target, rng=rng)
    tasks: list[PlanTask] = []
    for idx in indices:
        record = dataset[int(idx)]
        question = (record.get("question") or "").strip()
        context = (record.get("context") or "").strip()
        prompt = SQUAD_PROMPT.format(question=question, context=context)
        record_id = str(record.get("id") or idx)
        suffix = _stable_suffix("qa", f"{record_id}_{idx}", cfg.seed)
        sample_id = f"qa_{record_id}_{suffix}"
        output_path = cfg.output_root / "qa" / f"{sample_id}.json"
        messages = [
            {"role": "system", "content": "You always return valid JSON for the provided schema."},
            {"role": "user", "content": prompt},
        ]
        tasks.append(
            PlanTask(
                sample_id=sample_id,
                domain="qa",
                prompt=prompt,
                input_messages=messages,
                output_path=output_path,
                source_metadata={
                    "question": question,
                    "input_text": context,
                    "dataset_index": int(idx),
                    "source": "squad",
                },
            )
        )
    return tasks


def build_reasoning_gym_tasks(cfg: PlanGenerationConfig, *, rng: random.Random) -> list[PlanTask]:
    target = int(cfg.total_per_domain.get("math", 0))
    if target <= 0:
        return []
    if rg_create_dataset is None:
        raise RuntimeError("reasoning_gym package is required to build math tasks.")
    dataset = rg_create_dataset(cfg.reasoning_gym_dataset)
    total = len(dataset)
    start = cfg.reasoning_gym_offset % max(total, 1)
    indices = [(start + i) % total for i in range(target)]
    tasks: list[PlanTask] = []
    for idx in indices:
        record = dataset[int(idx)]
        question = str(record.get("question", "")).strip()
        answer = json.dumps(record.get("answer"), ensure_ascii=False)
        prompt = RG_PROMPT.format(
            dataset_name=cfg.reasoning_gym_dataset, question=question, answer=answer
        )
        identifier = f"{cfg.reasoning_gym_dataset}_{idx}"
        suffix = _stable_suffix("math", identifier, cfg.seed)
        sample_id = f"math_{identifier}_{suffix}"
        output_path = cfg.output_root / "math" / f"{sample_id}.json"
        messages = [
            {"role": "system", "content": "Return valid JSON following the provided schema."},
            {"role": "user", "content": prompt},
        ]
        tasks.append(
            PlanTask(
                sample_id=sample_id,
                domain="math",
                prompt=prompt,
                input_messages=messages,
                output_path=output_path,
                source_metadata={
                    "question": question,
                    "input_text": question,
                    "dataset_index": int(idx),
                    "source": cfg.reasoning_gym_dataset,
                },
            )
        )
    return tasks


def _load_wiki_manifest(manifest_path: Path) -> list[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(data.get("output_dir", "data/raw/wiki-v6-500"))
    shards = data.get("shards", [])
    paths: list[Path] = []
    for shard in shards:
        rel_path = shard.get("relative_path") or shard.get("filename")
        if not rel_path:
            continue
        paths.append((output_dir / rel_path).resolve())
    return paths


def _iter_wiki_records(shard_paths: Sequence[Path]) -> Iterable[Mapping[str, Any]]:
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


def build_wikipedia_tasks(cfg: PlanGenerationConfig, *, rng: random.Random) -> list[PlanTask]:
    target = int(cfg.total_per_domain.get("survey", 0))
    if target <= 0:
        return []
    shard_paths = _load_wiki_manifest(cfg.wiki_manifest)
    if cfg.wiki_shard_limit is not None:
        shard_paths = shard_paths[: cfg.wiki_shard_limit]
    records = list(_iter_wiki_records(shard_paths))
    if not records:
        raise RuntimeError(
            f"No Wikipedia records found under {cfg.wiki_manifest}. "
            "Run the manifest download step before generating plans."
        )
    indices = _sample_indices(len(records), target, rng=rng)
    tasks: list[PlanTask] = []
    for idx in indices:
        record = records[int(idx)]
        title = (record.get("title") or record.get("id") or "Untitled").strip()
        text = _prepare_article_text(record.get("text") or "")
        char_count = len(text)
        prompt = WIKI_PROMPT.format(
            title=title,
            body=text,
            char_count=char_count,
            bio_skip_token=BIO_SKIP_TOKEN,
        )
        record_id = str(record.get("id", idx))
        suffix = _stable_suffix("survey", f"{record_id}_{idx}", cfg.seed)
        sample_id = f"survey_{record_id}_{suffix}"
        output_path = cfg.output_root / "survey" / f"{sample_id}.json"
        messages = [
            {"role": "system", "content": "Return valid JSON following the provided schema."},
            {"role": "user", "content": prompt},
        ]
        tasks.append(
            PlanTask(
                sample_id=sample_id,
                domain="survey",
                prompt=prompt,
                input_messages=messages,
                output_path=output_path,
                source_metadata={
                    "title": title,
                    "input_text": text,
                    "article_text": text,
                    "char_count": char_count,
                    "dataset_index": int(idx),
                    "source": "wikipedia",
                },
            )
        )
    return tasks


# --------------------------------------------------------------------------- #
# Generator                                                                   #
# --------------------------------------------------------------------------- #


def _chunked(seq: Sequence[PlanTask], size: int) -> Iterable[Sequence[PlanTask]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


class PlanGenerator:
    """Coordinates batched planner calls."""

    def __init__(
        self,
        *,
        llm_config,
        generation_cfg: GenerationConfig,
        plan_cfg: PlanGenerationConfig,
    ) -> None:
        if load_dataset is None or rg_create_dataset is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "datasets>=2.20 and reasoning_gym are required for PlanGenerator. "
                "Install them inside dataset_test_venv."
            )
        self._llm = create_llm_client(llm_config)
        self._async_client = (
            AsyncStructuredLLMClient(self._llm) if plan_cfg.use_async_client else None
        )
        self._sequential_client: SequentialStructuredLLMClient | None = None
        self._gen_cfg = generation_cfg
        self._plan_cfg = plan_cfg
        self._model_id = getattr(self._llm, "model", getattr(self._llm, "model_name", "unknown"))
        self._run_started_at = datetime.now(timezone.utc)
        timestamp = self._run_started_at.strftime("%Y%m%dT%H%M%SZ")
        self._failure_log_path = (
            plan_cfg.output_root / f"plan_generation_failures_{timestamp}.jsonl"
        )
        self._failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._failure_log_path.touch(exist_ok=True)
        logger.info("Planner failure log will be written to %s", self._failure_log_path)
        self._max_retries = 5
        # When reasoning models exhaust token budgets we retry once with an expanded cap.
        # For initial 16k budget, jump directly to 50k to handle long outputs.
        # For larger budgets, use 2x multiplier up to 200k max.
        self._incomplete_retry_multiplier = 2.0
        self._incomplete_retry_token_limit = 200_000
        self._incomplete_retry_minimum = 50_000  # Minimum retry budget for truncated outputs

    def build_tasks(self) -> list[PlanTask]:
        rng = random.Random(self._plan_cfg.seed)
        if self._plan_cfg.preflight_manifest:
            tasks = build_tasks_from_manifest(self._plan_cfg)
        else:
            tasks = (
                build_squad_tasks(self._plan_cfg, rng=rng)
                + build_reasoning_gym_tasks(self._plan_cfg, rng=rng)
                + build_wikipedia_tasks(self._plan_cfg, rng=rng)
            )
        logger.info("Prepared %d plan tasks", len(tasks))
        return tasks

    def _pending_tasks(self, tasks: Sequence[PlanTask]) -> list[PlanTask]:
        pending: list[PlanTask] = []
        for task in tasks:
            if self._plan_cfg.resume_existing and task.output_path.exists():
                logger.info("Skipping existing plan for %s (resume enabled)", task.sample_id)
                continue
            _ensure_output_dir(task.output_path)
            pending.append(task)
        return pending

    async def generate_async(self, tasks: Sequence[PlanTask]) -> list[GeneratedPlan]:
        if not tasks:
            return []
        if self._async_client is None:
            raise RuntimeError("Async Structured LLM client is not initialized.")
        task_list = list(tasks)
        for task in task_list:
            _ensure_output_dir(task.output_path)
        generated: list[GeneratedPlan] = []
        progress = tqdm(total=len(task_list), desc="plans", unit="sample")
        chunks = list(_chunked(task_list, self._plan_cfg.batch_size))
        total_chunks = len(chunks)
        for chunk_idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Submitting plan chunk %d/%d (%d samples)", chunk_idx, total_chunks, len(chunk)
            )
            request_pairs = [(task, task.to_structured_request(self._gen_cfg)) for task in chunk]
            requests = [req for _, req in request_pairs]
            request_map = {req.request_id: task for task, req in request_pairs}
            results = await self._async_client.submit_batch(
                requests,
                concurrency=self._plan_cfg.concurrency,
                max_retries=self._max_retries,
            )
            for result in results:
                task = request_map.get(result.request.request_id)
                if task is None:
                    progress.update(1)
                    continue
                if result.error:
                    if self._is_biography_skip_error(result.error):
                        self._record_biography_skip(task)
                        progress.update(1)
                        continue
                    error_type = getattr(result.error, "error_type", "")
                    if error_type in {"incomplete_response", "missing_structured_payload"}:
                        fallback_plan = await self._retry_with_expanded_budget(
                            task,
                            result.request,
                            reason=error_type,
                        )
                        if fallback_plan:
                            generated.append(fallback_plan)
                            progress.update(1)
                            continue
                    self._record_structured_error(task, result.error)
                    if self._should_abort_for_error(result.error):
                        raise RuntimeError(
                            f"Planner request {task.sample_id} failed: {result.error}"
                        )
                    progress.update(1)
                    continue
                response = result.response
                if response is None or not self._response_has_payload(response):
                    error = StructuredRequestError(
                        f"Empty response for {task.sample_id}",
                        error_type="empty_response",
                        retryable=False,
                    )
                    self._record_structured_error(
                        task, error, raw_text=response.output_text if response else None
                    )
                    if self._should_abort_for_error(error):
                        raise RuntimeError(f"Planner request {task.sample_id} failed: {error}")
                    progress.update(1)
                    continue
                plan = self._finalize_plan(task, response)
                if plan:
                    generated.append(plan)
                progress.update(1)
            logger.info("Completed plan chunk %d/%d", chunk_idx, total_chunks)
        progress.close()
        return generated

    async def _retry_with_expanded_budget(
        self,
        task: PlanTask,
        failed_request: StructuredOutputRequest,
        *,
        reason: str,
    ) -> GeneratedPlan | None:
        if self._async_client is None:
            return None
        current_budget = failed_request.max_output_tokens
        # If budget is below minimum threshold (e.g., 16k), jump to minimum (50k)
        # Otherwise use 2x multiplier up to 200k max
        if current_budget < self._incomplete_retry_minimum:
            next_budget = self._incomplete_retry_minimum
        else:
            next_budget = int(
                min(
                    self._incomplete_retry_token_limit,
                    max(current_budget * self._incomplete_retry_multiplier, current_budget + 1),
                )
            )
        if next_budget <= current_budget:
            return None
        logger.warning(
            "Retrying %s with expanded token budget (%d -> %d tokens) after %s",
            task.sample_id,
            current_budget,
            next_budget,
            reason.replace("_", " "),
        )
        retry_request = task.to_structured_request(self._gen_cfg)
        retry_request.max_output_tokens = max(retry_request.max_output_tokens, next_budget)
        retry_request.metadata["retry_reason"] = f"{reason}_retry"
        retry_results = await self._async_client.submit_batch(
            [retry_request],
            concurrency=1,
            max_retries=self._max_retries,
        )
        retry_result = retry_results[0]
        if retry_result.error:
            logger.error(
                "Expanded budget retry for %s failed (%s)",
                task.sample_id,
                getattr(retry_result.error, "error_type", "unknown_error"),
            )
            return None
        response = retry_result.response
        if response is None or not self._response_has_payload(response):
            logger.error(
                "Expanded budget retry for %s returned empty payload",
                task.sample_id,
            )
            return None
        plan = self._finalize_plan(task, response)
        if plan:
            logger.info(
                "Plan %s succeeded after expanded budget retry (%d tokens)",
                task.sample_id,
                retry_request.max_output_tokens,
            )
        return plan

    def _run_sequential_with_retries(
        self,
        task: PlanTask,
        request: StructuredOutputRequest,
    ) -> GeneratedPlan | None:
        if self._sequential_client is None:
            return None
        attempt = 0
        delay = max(self._plan_cfg.sequential_sleep_seconds or 0.5, 0.5)
        while attempt <= self._max_retries:
            try:
                response = self._sequential_client.submit(request)
                if not self._response_has_payload(response):
                    error = StructuredRequestError(
                        f"Empty response for {task.sample_id}",
                        error_type="empty_response",
                        retryable=False,
                    )
                    self._record_structured_error(task, error, raw_text=response.output_text)
                    if self._should_abort_for_error(error):
                        raise RuntimeError(f"Planner request {task.sample_id} failed: {error}")
                    return None
                return self._finalize_plan(task, response)
            except StructuredRequestError as exc:
                exc.attempts = attempt
                if exc.status_code == 429:
                    self._bump_sequential_sleep(exc.retry_after)
                if not exc.retryable or attempt >= self._max_retries:
                    self._record_structured_error(task, exc)
                    if self._should_abort_for_error(exc):
                        raise RuntimeError(f"Planner request {task.sample_id} failed: {exc}")
                    return None
                sleep_for = exc.retry_after if exc.retry_after is not None else delay
                time.sleep(AsyncStructuredLLMClient._jittered_delay(sleep_for))
                delay = min(delay * 1.5, AsyncStructuredLLMClient._MAX_BACKOFF_SECONDS)
                attempt += 1
        return None

    def _finalize_plan(
        self,
        task: PlanTask,
        response,
    ) -> GeneratedPlan | None:
        metadata = dict(response.request.metadata)
        output_path = Path(metadata["output_path"])
        try:
            payload = self._normalize_payload(response, metadata, task)
        except PlanValidationError as exc:
            self._handle_validation_failure(task, exc, response.output_text)
            return None
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Plan %s (%s) finalized in %.0f ms -> %s",
            response.request.request_id,
            metadata.get("domain", task.domain if task else "unknown"),
            response.latency_ms,
            output_path,
        )
        return GeneratedPlan(
            sample_id=response.request.request_id,
            domain=str(metadata.get("domain", task.domain)),
            path=output_path,
            payload=payload,
            raw_text=response.output_text,
            latency_ms=response.latency_ms,
        )

    @staticmethod
    def _response_has_payload(response) -> bool:
        return bool(response.parsed_json) or bool(response.output_text.strip())

    def _record_structured_error(
        self,
        task: PlanTask,
        error: StructuredRequestError,
        *,
        raw_text: str | None = None,
    ) -> None:
        metadata: dict[str, Any] = {}
        if raw_text:
            metadata["raw_response_excerpt"] = raw_text[:200]
        self._log_failure(
            task,
            error_type=getattr(error, "error_type", error.__class__.__name__),
            message=str(error),
            status_code=getattr(error, "status_code", None),
            retry_count=getattr(error, "attempts", 0),
            metadata=metadata,
        )

    def _handle_validation_failure(
        self,
        task: PlanTask,
        exc: Exception,
        raw_text: str | None,
    ) -> None:
        metadata: dict[str, Any] = {}
        if raw_text:
            metadata["raw_response_excerpt"] = raw_text[:200]
        self._log_failure(
            task,
            error_type="plan_validation",
            message=str(exc),
            status_code=None,
            retry_count=0,
            metadata=metadata,
        )

    def _log_failure(
        self,
        task: PlanTask,
        *,
        error_type: str,
        message: str,
        status_code: int | None,
        retry_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "sample_id": task.sample_id,
            "domain": task.domain,
            "model_id": self._model_id,
            "error_type": error_type,
            "error_message": message,
            "http_status": status_code,
            "retry_count": retry_count,
            "source": str(task.source_metadata.get("source", "")),
            "output_path": str(task.output_path),
            "timestamp": time.time(),
        }
        if metadata:
            entry.update(metadata)
        self._failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._failure_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.error("Plan %s failed (%s): %s", task.sample_id, error_type, message)

    @staticmethod
    def _is_biography_skip_error(error: StructuredRequestError) -> bool:
        if getattr(error, "error_type", "") != "missing_structured_payload":
            return False
        message = str(error).upper()
        return BIO_SKIP_TOKEN in message

    def _record_biography_skip(self, task: PlanTask) -> None:
        logger.info(
            "Planner identified biography for %s (%s); emitting skip token.",
            task.sample_id,
            task.domain,
        )
        entry: dict[str, Any] = {
            "sample_id": task.sample_id,
            "domain": task.domain,
            "model_id": self._model_id,
            "error_type": "skip_biography",
            "error_message": "Planner emitted biography skip token.",
            "http_status": None,
            "retry_count": 0,
            "source": str(task.source_metadata.get("source", "")),
            "output_path": str(task.output_path),
            "timestamp": time.time(),
        }
        self._failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._failure_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _bump_sequential_sleep(self, retry_after: float | None) -> None:
        base = max(self._plan_cfg.sequential_sleep_seconds or 0.5, 0.5)
        new_value = min(base * 1.5, 10.0)
        if retry_after:
            new_value = max(new_value, retry_after)
        self._plan_cfg.sequential_sleep_seconds = new_value
        if self._sequential_client:
            self._sequential_client.set_pause(new_value)
        logger.warning(
            "HTTP 429 detected for planner; increasing sequential sleep to %.2fs", new_value
        )

    def _generate_sequential(self, tasks: Sequence[PlanTask]) -> list[GeneratedPlan]:
        if not tasks:
            return []
        if self._sequential_client is None:
            self._sequential_client = SequentialStructuredLLMClient(
                self._llm,
                pause_seconds=self._plan_cfg.sequential_sleep_seconds,
            )
        generated: list[GeneratedPlan] = []
        progress = tqdm(total=len(tasks), desc="plans", unit="sample")
        total = len(tasks)
        for idx, task in enumerate(tasks, start=1):
            logger.info("Submitting plan %d/%d (%s)", idx, total, task.sample_id)
            request = task.to_structured_request(self._gen_cfg)
            plan = self._run_sequential_with_retries(task, request)
            if plan:
                generated.append(plan)
            progress.update(1)
        progress.close()
        return generated

    def generate(self, tasks: Sequence[PlanTask]) -> list[GeneratedPlan]:
        pending = self._pending_tasks(tasks)
        if not pending:
            logger.info("No plan tasks to run (resume skipped all targets)")
            return []
        if self._plan_cfg.use_async_client:
            if self._async_client is None:
                self._async_client = AsyncStructuredLLMClient(self._llm)
            return asyncio.run(self.generate_async(pending))
        return self._generate_sequential(pending)

    @staticmethod
    def _should_abort_for_error(error: StructuredRequestError) -> bool:
        status = getattr(error, "status_code", None)
        error_type = getattr(error, "error_type", "")
        # Only rate limiting (429), content policy (403), and gateway/server errors (500)
        # are treated as non-fatal; everything else should abort the pipeline so we do not
        # quietly continue after paid-but-bad responses.
        # 403 errors typically indicate content policy violations which we skip.
        nonfatal_types = {
            "http_400",
            "http_403",
            "http_429",
            "http_500",
            "incomplete_response",
            "missing_structured_payload",
        }
        if status in (400, 403, 429, 500) or error_type in nonfatal_types:
            return False
        return True

    @staticmethod
    def _normalize_payload(
        response,
        metadata: Mapping[str, Any],
        task: PlanTask | None,
    ) -> dict[str, Any]:
        payload = dict(response.parsed_json or {})
        payload["sample_id"] = metadata.get("sample_id")
        payload["domain"] = metadata.get("domain")
        if task is not None:
            payload["input_text"] = task.source_metadata.get("input_text", "")
            payload["prompt"] = task.prompt
            payload["source_metadata"] = dict(task.source_metadata)
        else:
            payload["input_text"] = ""
            payload["prompt"] = ""
            payload["source_metadata"] = dict(metadata)
        payload.setdefault("stream_count", len(payload.get("streams", [])))
        payload["topology"] = "all_to_all"
        payload.setdefault("expected_dnb_lag_delta", 1)
        payload.setdefault("note_cadence_M", 6)
        # Sectional independence is a hard invariant in this codebase: all released
        # plans must be decodable per-stream from the initial notes snapshot without
        # relying on cross-stream textual dependencies. We therefore ignore any
        # model-emitted flag here and always enforce True.
        payload["sectional_independence"] = True
        payload["generated_at"] = time.time()
        payload["raw_response_text"] = response.output_text
        payload["latency_ms"] = response.latency_ms
        payload["streams"] = _attach_stream_ids(payload.get("streams", []))
        input_text = payload.get("input_text")
        _validate_plan_contracts(payload, input_text if isinstance(input_text, str) else None)
        return payload


def _attach_stream_ids(streams: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, stream in enumerate(streams[:3]):
        stream_id = f"stream_{idx + 1}"
        normalized.append(
            {
                "stream_id": stream_id,
                "header": str(stream.get("header", "")),
                "summary": str(stream.get("summary", "")),
                "entities": [str(ent) for ent in stream.get("entities", [])],
                "constraints": [str(constraint) for constraint in stream.get("constraints", [])],
                "section_contract": _normalize_section_contract(stream.get("section_contract")),
                "notes_contract": _normalize_notes_contract(stream.get("notes_contract")),
            }
        )
    return normalized


def _normalize_section_contract(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    return {}


def _normalize_notes_contract(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        normalized = [str(entry).strip() for entry in value if str(entry).strip()]
        return normalized
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return default
    return default


class PlanValidationError(ValueError):
    """Raised when a plan violates sectional-independence invariants."""


_RANGE_CONTRACT_TYPES = {"alphabet_range", "section_index_range"}


def _cross_stream_lint_action() -> str:
    value = os.getenv("PDT_PLAN_CROSS_STREAM_LINT", "fail").strip().lower()
    if value not in {"fail", "warn", "ignore"}:
        return "fail"
    return value


_CROSS_STREAM_DEPENDENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bstream\s*(?:one|two|three|1|2|3)\b", re.IGNORECASE),
    re.compile(r"\bprevious\s+stream\b", re.IGNORECASE),
    re.compile(r"\bprior\s+stream\b", re.IGNORECASE),
    re.compile(r"\banother\s+stream\b", re.IGNORECASE),
    re.compile(r"\bother\s+streams?\b", re.IGNORECASE),
    re.compile(r"\bdepends?\s+on\s+stream\b", re.IGNORECASE),
    re.compile(r"\buses?\s+stream\b", re.IGNORECASE),
    re.compile(r"\bbuilds?\s+on\s+stream\b", re.IGNORECASE),
    re.compile(r"\brefin(?:e|ing)\s+stream\b", re.IGNORECASE),
)


def _validate_plan_contracts(payload: Mapping[str, Any], input_text: str | None = None) -> None:
    # Sectional independence is always enforced at normalization time, so the
    # validator assumes it is enabled for all plans that reach this point.
    streams: Sequence[Mapping[str, Any]] = payload.get("streams", [])  # type: ignore[assignment]
    if len(streams) != 3:
        raise PlanValidationError("Sectional-independence plans must define exactly 3 streams.")
    range_contract_type: str | None = None
    range_specs: list[tuple[int, int, int]] = []
    for idx, stream in enumerate(streams, start=1):
        _lint_cross_stream_references(stream, idx)
        contract = stream.get("section_contract")
        if not isinstance(contract, Mapping) or not contract:
            raise PlanValidationError(f"Stream {idx} must include a non-empty section_contract.")
        contract_type = contract.get("type")
        if not isinstance(contract_type, str) or not contract_type.strip():
            raise PlanValidationError(
                f"Stream {idx} section_contract must declare a string 'type'."
            )
        cleaned_contract = {str(key): value for key, value in contract.items()}
        contract_type_clean = contract_type.strip()
        if contract_type_clean == "source_slice":
            _sanitize_source_slice_contract(cleaned_contract, input_text, idx)
        stream["section_contract"] = cleaned_contract  # type: ignore[index]

        notes_contract = stream.get("notes_contract")
        if not isinstance(notes_contract, list) or not notes_contract:
            raise PlanValidationError(
                f"Stream {idx} must include at least one notes_contract bullet."
            )
        cleaned_notes = [str(entry).strip() for entry in notes_contract if str(entry).strip()]
        if not cleaned_notes:
            raise PlanValidationError(
                f"Stream {idx} notes_contract cannot be empty after normalization."
            )
        stream["notes_contract"] = cleaned_notes  # type: ignore[index]

        if contract_type_clean in _RANGE_CONTRACT_TYPES:
            normalized_type = contract_type_clean
            start, end = _extract_contract_range(normalized_type, cleaned_contract, stream_idx=idx)
            range_specs.append((start, end, idx))
            if range_contract_type is None:
                range_contract_type = normalized_type
            elif range_contract_type != normalized_type:
                raise PlanValidationError(
                    "Sectional range contracts must use the same type across all streams."
                )

    if range_contract_type in _RANGE_CONTRACT_TYPES:
        _enforce_range_ordering(range_specs, range_contract_type, len(input_text or ""))


def _sanitize_source_slice_contract(
    contract: dict[str, Any],
    input_text: str | None,
    stream_idx: int,
) -> None:
    if not isinstance(contract.get("start_idx"), int) or not isinstance(
        contract.get("end_idx"), int
    ):
        raise PlanValidationError(
            f"Stream {stream_idx} source_slice contract must include integer start_idx/end_idx."
        )
    start = int(contract["start_idx"])
    end = int(contract["end_idx"])
    if start < 0:
        logger.warning("Stream %d start_idx %d < 0; clamping to 0.", stream_idx, start)
        start = 0
    clamped = False
    if input_text is not None:
        total_len = len(input_text)
        if start >= total_len:
            raise PlanValidationError(
                f"Stream {stream_idx} start_idx {start} exceeds article length {total_len}."
            )
        if end > total_len:
            allowable_overrun = max(int(total_len * 0.05), 512)
            if end - total_len <= allowable_overrun:
                logger.warning(
                    "Stream %d end_idx %d exceeds article length %d by %d chars; clamping.",
                    stream_idx,
                    end,
                    total_len,
                    end - total_len,
                )
                end = total_len
                clamped = True
            else:
                raise PlanValidationError(
                    f"Stream {stream_idx} end_idx {end} exceeds article length {total_len} "
                    f"by {end - total_len} chars (limit {allowable_overrun})."
                )
    if end <= start:
        raise PlanValidationError(
            f"Stream {stream_idx} source_slice invalid range [{start}, {end}); "
            "end_idx must be greater than start_idx."
        )
    contract["start_idx"] = start
    contract["end_idx"] = end


def _extract_contract_range(
    contract_type: str,
    contract: Mapping[str, Any],
    *,
    stream_idx: int,
) -> tuple[int, int]:
    if contract_type == "section_index_range":
        start = contract.get("start_idx")
        end = contract.get("end_idx")
        if not isinstance(start, int) or not isinstance(end, int):
            raise PlanValidationError(
                f"Stream {stream_idx} section_index_range requires integer start_idx/end_idx."
            )
        if end < start:
            raise PlanValidationError(
                f"Stream {stream_idx} section_index_range end_idx must be >= start_idx."
            )
        return start, end
    if contract_type == "alphabet_range":
        start = contract.get("start")
        end = contract.get("end")
        if not isinstance(start, str) or not start.strip():
            raise PlanValidationError(f"Stream {stream_idx} alphabet_range.start must be a letter.")
        if not isinstance(end, str) or not end.strip():
            raise PlanValidationError(f"Stream {stream_idx} alphabet_range.end must be a letter.")
        start_ord = ord(start.strip().upper()[0])
        end_ord = ord(end.strip().upper()[0])
        if end_ord < start_ord:
            raise PlanValidationError(
                f"Stream {stream_idx} alphabet_range end must not precede start."
            )
        return start_ord, end_ord
    raise PlanValidationError(f"Unsupported section contract type '{contract_type}'.")


def _enforce_range_ordering(
    range_specs: Sequence[tuple[int, int, int]], contract_type: str, total_len: int
) -> None:
    ordered = sorted(range_specs, key=lambda spec: spec[0])
    prev_end: int | None = None
    lengths: list[int] = []
    for start, end, stream_idx in ordered:
        if prev_end is not None:
            if start <= prev_end:
                raise PlanValidationError(
                    f"Stream {stream_idx} {contract_type} overlaps with a previous stream."
                )
            if start != prev_end + 1:
                raise PlanValidationError(
                    f"Gap detected between {contract_type} assignments near stream {stream_idx}."
                )
        prev_end = end
        lengths.append(end - start + 1)
    if lengths and total_len > 0:
        avg_len = sum(lengths) / len(lengths)
        min_len = max(int(avg_len * 0.5), int(total_len * 0.05))
        for idx, length in enumerate(lengths, start=1):
            if length < min_len:
                raise PlanValidationError(
                    f"Stream {idx} {contract_type} length {length} is below minimum "
                    f"{min_len} (50% of average or 5% of article)."
                )


def _lint_cross_stream_references(stream: Mapping[str, Any], stream_idx: int) -> None:
    action = _cross_stream_lint_action()
    if action == "ignore":
        return
    fields: list[tuple[str, str]] = []
    header = stream.get("header")
    if isinstance(header, str) and header.strip():
        fields.append(("header", header))
    summary = stream.get("summary")
    if isinstance(summary, str) and summary.strip():
        fields.append(("summary", summary))
    constraints = stream.get("constraints", [])
    if isinstance(constraints, Sequence):
        for idx, constraint in enumerate(constraints):
            if isinstance(constraint, str) and constraint.strip():
                fields.append((f"constraint[{idx}]", constraint))

    for field_name, text in fields:
        match = _detect_cross_stream_dependency(text)
        if not match:
            continue
        snippet = text[match.start() : match.end()]
        message = (
            f"Stream {stream_idx} field '{field_name}' references other streams via '{snippet}'."
        )
        logger.warning("%s", message)
        if action == "warn":
            continue
        raise PlanValidationError(message)


def _detect_cross_stream_dependency(text: str) -> re.Match[str] | None:
    if not text or not text.strip():
        return None
    for pattern in _CROSS_STREAM_DEPENDENCY_PATTERNS:
        match = pattern.search(text)
        if match:
            return match
    return None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_suffix(domain: str, identifier: str, seed: int | None) -> str:
    base = f"{domain}:{identifier}:{seed or 0}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]


__all__ = [
    "GeneratedPlan",
    "PlanGenerationConfig",
    "PlanGenerator",
    "PlanTask",
]
