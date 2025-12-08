"""Generation of true/speculative notes conditioned on teacher plans."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from tqdm import tqdm

from parallel_decoder_transformer.data.extraction import StreamNotes, load_stream_notes
from parallel_decoder_transformer.datasets.async_llm import (
    AsyncStructuredLLMClient,
    StructuredOutputRequest,
    StructuredOutputResult,
    StructuredRequestError,
)
from parallel_decoder_transformer.datasets.config import (
    GenerationConfig,
    SpeculativeNotesNoiseConfig,
)
from parallel_decoder_transformer.datasets.plan_contract_notes import (
    derive_initial_notes_from_plan,
    merge_seed_notes,
)
from parallel_decoder_transformer.datasets.procedural_snapshots import (
    generate_procedural_snapshots,
)

logger = logging.getLogger(__name__)


# Schemas for ENT/FACT/COVERAGE notes. These mirror the Pydantic payload
# models used in scripts/generate_notes.py and the StreamNotes dataclass, and
# follow OpenAI Responses requirements that nested objects explicitly forbid
# unknown keys via additionalProperties=False.

# ==========================================
# 1. SCHEMA DEFINITIONS (Compact / Protobuf-style)
# ==========================================


_EVIDENCE_SPAN_SCHEMA = {
    "type": "array",
    "prefixItems": [
        {"type": "integer"},  # start index
        {"type": "integer"},  # end index
        # HARDENING: minLength 1 prevents the model from generating empty strings ""
        {"type": "string", "minLength": 1},
    ],
    # Additional entries are forbidden by forcing all remaining slots to null
    "items": {"type": "null"},
    "minItems": 3,
    "maxItems": 3,
}

_COMPACT_ENTITY_SCHEMA = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},  # id
        {"type": "string"},  # name
        {"type": "string"},  # type
        {"type": "boolean"},  # canonical_bool
        # HARDENING: Cap aliases to 5 to maintain compact token usage
        {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    ],
    "items": {"type": "null"},
    "minItems": 5,
    "maxItems": 5,
}

_COMPACT_FACT_SCHEMA = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},  # subj_id
        {"type": "string"},  # predicate
        {"type": "string"},  # object
        {"type": "number"},  # certainty
        _EVIDENCE_SPAN_SCHEMA,  # Nested Evidence Tuple
    ],
    "items": {"type": "null"},
    "minItems": 5,
    "maxItems": 5,
}

_COMPACT_COVERAGE_SCHEMA = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},  # plan_item_id
        {"type": "string"},  # status
    ],
    "items": {"type": "null"},
    "minItems": 2,
    "maxItems": 2,
}

# --- 2. Top-Level Schemas (Re-verified) ---

# Schema for per-stream true notes response (single stream per request)
TRUE_NOTES_SCHEMA_SINGLE_STREAM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stream_id": {"type": "string"},
        "ENT": {
            "type": "array",
            "items": _COMPACT_ENTITY_SCHEMA,
            "maxItems": 16,  # Enforces "Bounded Lists" prompt instruction
        },
        "FACT": {
            "type": "array",
            "items": _COMPACT_FACT_SCHEMA,
            "minItems": 3,  # Enforces "Minimum 3 Facts" prompt instruction
            "maxItems": 16,
        },
        "COVERAGE": {"type": "array", "items": _COMPACT_COVERAGE_SCHEMA, "maxItems": 16},
    },
    "required": ["stream_id", "ENT", "FACT", "COVERAGE"],
    "additionalProperties": False,
}

# Schema for per-stream speculative notes response (single stream per request)
SPEC_NOTES_SCHEMA_SINGLE_STREAM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stream_id": {"type": "string"},
        "variant_id": {"type": "string"},
        "ENT": {"type": "array", "items": _COMPACT_ENTITY_SCHEMA, "maxItems": 16},
        "FACT": {"type": "array", "items": _COMPACT_FACT_SCHEMA, "minItems": 3, "maxItems": 16},
        "COVERAGE": {"type": "array", "items": _COMPACT_COVERAGE_SCHEMA, "maxItems": 16},
    },
    "required": ["stream_id", "variant_id", "ENT", "FACT", "COVERAGE"],
    "additionalProperties": False,
}


# ==========================================
# 2. PROMPT DEFINITIONS (Behavioral Control)
# ==========================================

TRUE_NOTES_PROMPT = """You are the Oracle Teacher for the Parallel Decoder Transformer (PDT) runtime.
Your task is to generate the "Golden Record": the Ground Truth Notes for **A SINGLE STREAM**.

**CRITICAL CONTEXT**:
You are receiving a raw string slice of text. This text is your **ENTIRE UNIVERSE**.
- Treat the first character of the provided text as **index 0**.
- Do NOT attempt to calculate global document offsets. Use local 0-based indexing relative to the provided text only.

## INPUT DATA
1. **Stream Text Slice**: The actual text content to extract from.
2. **Metadata**: The plan header/summary.

## OBJECTIVES
1.  **Search First**: Find 3-16 atomic statements in the text.
2.  **Quote**: Copy the exact substring for each statement.
3.  **Index**: Calculate the start/end indices for that substring.
4.  **Format**: Package into the Compact Array schema.

## SCHEMA: COMPACT ARRAYS (TUPLES)
You must output a JSON object containing these specific arrays:

* **`stream_id`**: (String) e.g., "stream_1"
* **`ENT`**: `[id, name, type, canonical_bool, [aliases]]`
* **`FACT`**: `[subj_id, predicate, object, certainty, [start, end, quote_text]]`
* **`COVERAGE`**: `[plan_item_id, status]`

## EVIDENCE SPAN RULES (STRICT ENFORCEMENT)
The last element of every FACT tuple is the **Evidence Span**: `[start, end, quote_text]`.
1.  **NON-NEGOTIABLE**: `quote_text` must be an exact, case-sensitive substring found in the input.
2.  **NO LAZY SPANS**: `[0, 0, ""]` is **FORBIDDEN**. If you cannot find a valid non-zero length quote, **DELETE THE FACT**.
3.  **INDEXING**: `start` must be the integer index where `quote_text` begins in the provided string. `end` is `start + len(quote_text)`.

## STYLE & QUANTITY
* **Density**: High. Encyclopedia style.
* **Quantity**: Minimum 3 Facts. Maximum 16 Facts.
* **Entities**: Extract distinct nouns (People, Locations, Species) acting as subjects.

## ONE-SHOT EXAMPLE (Mental Model)
Input: "The project started in 1995."
Output FACT: ["E1", "started in", "1995", 1.0, [23, 27, "1995"]]
*(Note: "1995" starts at index 23 of the input string)*

Response must be **valid JSON only**. No markdown fencing or preamble.
"""

SPEC_NOTES_PROMPT = """You are the Speculation Head for the Parallel Decoder Transformer (PDT).
Your task is to generate a "Dirty Draft": the Speculative Notes for **A SINGLE STREAM**.

**CONTEXT**:
You are simulating a "noisy cache read" or a "corrupted memory" of the plan.
You will receive the **TRUE NOTES** (The Golden Record). Your job is to degrade them while maintaining the schema structure.

## THEORETICAL GOAL (SPECULATIVE INVARIANCE)
The runtime uses these notes to test the "Agreement Head". We need to trick the model into generating content based on slightly wrong information, so the Agreement Head learns to recognize contradictions.

## INSTRUCTIONS
1.  **Ingest True Notes**: Read the `ENT` and `FACT` arrays provided in the prompt.
2.  **Inject Noise (15-20% Rate)**:
    * **Subtle Corruption**: Change a specific detail (e.g., Year "1999" -> "1995", Name "John" -> "Jon").
    * **Hallucination**: Add an alias or attribute that doesn't exist.
    * **Omission**: Drop a non-critical fact.
3.  **Handling Evidence Spans during Corruption (CRITICAL)**:
    * If you corrupt a Fact (e.g., change "Red" to "Blue"), **KEEP THE ORIGINAL EVIDENCE SPAN**.
    * *Why?* This creates a distinct **CONTRADICTION** between the Fact Object (Blue) and the Evidence Quote (Red). The Agreement Head is trained to spot exactly this discrepancy.
    * **Do not** invent new fake evidence spans/indices.

## OUTPUT FORMAT: COMPACT ARRAYS
Maintain the exact schema of the input, but with your corrupted values.
* **`stream_id`**: Same as input.
* **`variant_id`**: "variant_1"
* **`ENT`**: `[id, name, type, canonical_bool, [aliases]]` (With Noise)
* **`FACT`**: `[subj_id, predicate, object, certainty, [start, end, quote_text]]` (With Noise; Keep original evidence)
* **`COVERAGE`**: `[plan_item_id, status]`

## BOUNDS
* Keep lists to a maximum of 16 items.
* Do not delete *all* facts; keep at least 3.

Response must be **valid JSON only**. No markdown fencing or preamble.
"""


@dataclass(slots=True)
class PlanDocument:
    sample_id: str
    domain: str
    path: Path
    payload: Mapping[str, Any]

    @property
    def lag_delta(self) -> int:
        return int(self.payload.get("expected_dnb_lag_delta", 1) or 1)

    @property
    def note_cadence(self) -> int:
        return int(self.payload.get("note_cadence_M", 6) or 6)

    @property
    def sectional_independence(self) -> bool:
        value = self.payload.get("sectional_independence")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return False


@dataclass(slots=True)
class NotesArtifact:
    sample_id: str
    domain: str
    path: Path
    true_notes: list[StreamNotes]
    speculative_variants: list[dict[str, Any]]
    z_true: str
    z_speculative: list[str]
    rollback: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(slots=True)
class NotesGenerationConfig:
    # Number of plan documents to process per async batch. We keep this fairly
    # small so that each batch represents a modest burst of tokens.
    batch_size: int = 24
    # Maximum number of concurrent notes requests. Reduced from 20 to 10 to
    # stay comfortably under typical Tier‑3 rate limits during long runs.
    concurrency: int = 10
    # Thread pool size for local post‑processing; 16 is usually ample on
    # workstation hardware without over‑scheduling.
    max_workers: int = 16
    variants_per_sample: int = 3
    rollback_probability_low: float = 0.15
    rollback_probability_high: float = 0.25
    output_root: Path = Path("data/prep/notes")
    resume_existing: bool = True


class NotesGenerator:
    """Generates true and speculative notes conditioned on saved plans."""

    def __init__(
        self,
        *,
        llm_config,
        true_cfg: GenerationConfig,
        speculative_cfg: GenerationConfig,
        noise_cfg: SpeculativeNotesNoiseConfig,
        notes_cfg: NotesGenerationConfig,
    ) -> None:
        # Import here to avoid circular dependency
        from parallel_decoder_transformer.utils.llm_client_factory import create_llm_client

        self._llm_client = create_llm_client(llm_config)

        # Use appropriate async client based on backend
        backend = getattr(llm_config, "backend", "openai").lower()
        if backend == "ollama":
            from parallel_decoder_transformer.utils.async_ollama_client import AsyncOllamaLLMClient

            self._async_client = AsyncOllamaLLMClient(self._llm_client)
        elif backend == "vllm":
            from parallel_decoder_transformer.utils.async_vllm_client import AsyncVLLMLLMClient

            self._async_client = AsyncVLLMLLMClient(self._llm_client)
        else:
            self._async_client = AsyncStructuredLLMClient(self._llm_client)

        self._true_cfg = true_cfg
        self._spec_cfg = speculative_cfg
        self._noise_cfg = noise_cfg
        self._notes_cfg = notes_cfg
        self._rng = random.Random(true_cfg.seed or 11)
        self._model_id = getattr(
            self._llm_client,
            "model",
            getattr(self._llm_client, "model_name", "unknown"),
        )
        # Extract reasoning_effort from llm_config if available
        self._reasoning_effort = None
        if hasattr(llm_config, "openai") and hasattr(llm_config.openai, "reasoning_effort"):
            self._reasoning_effort = llm_config.openai.reasoning_effort
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._failure_log_path = (
            self._notes_cfg.output_root / f"notes_generation_failures_{timestamp}.jsonl"
        )
        self._failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._failure_log_path.touch(exist_ok=True)
        self._failure_log_lock = Lock()
        self._max_retries = 5
        # When reasoning models exhaust token budgets we retry once with an expanded cap.
        # For initial 16k budget, jump directly to 50k to handle long outputs.
        # For larger budgets, use 2x multiplier up to 200k max.
        self._incomplete_retry_multiplier = 2.0
        self._incomplete_retry_token_limit = 200_000
        self._incomplete_retry_minimum = 50_000  # Minimum retry budget for truncated outputs

    def load_plan_documents(self, plan_dir: Path, limit: int | None = None) -> list[PlanDocument]:
        paths = sorted(plan_dir.rglob("*.json"))
        docs: list[PlanDocument] = []
        for path in paths:
            if limit is not None and len(docs) >= limit:
                break
            payload = json.loads(path.read_text(encoding="utf-8"))
            sample_id = str(payload.get("sample_id") or path.stem)
            domain = str(payload.get("domain") or path.parent.name)
            docs.append(
                PlanDocument(sample_id=sample_id, domain=domain, path=path, payload=payload)
            )
        return docs

    def _pending_plans(self, plans: Sequence[PlanDocument]) -> list[PlanDocument]:
        pending: list[PlanDocument] = []
        for plan in plans:
            artifact_path = self._artifact_path(plan)
            if self._notes_cfg.resume_existing and artifact_path.exists():
                logger.info("Skipping existing notes for %s (resume enabled)", plan.sample_id)
                continue
            self._ensure_output_dir(plan)
            pending.append(plan)
        return pending

    async def generate_async(self, plans: Sequence[PlanDocument]) -> list[NotesArtifact]:
        if not plans:
            return []
        generated: list[NotesArtifact] = []
        progress = tqdm(total=len(plans), desc="notes", unit="plan")
        for batch in _chunk(plans, self._notes_cfg.batch_size):
            # PASS 1: Teacher generates true notes (per-stream, 3x parallelization)
            true_requests: list[StructuredOutputRequest] = []
            for plan in batch:
                true_requests.extend(self._true_requests(plan))
            logger.info(
                "Pass 1 (Teacher): Generating true notes for batch of %d plans (%d per-stream requests)",
                len(batch),
                len(true_requests),
            )
            true_results = await self._async_client.submit_batch(
                true_requests, concurrency=self._notes_cfg.concurrency
            )

            # Retry failed requests with expanded token budget
            true_results_final: list[StructuredOutputResult] = []
            for result in true_results:
                if result.error:
                    error_type = getattr(result.error, "error_type", "")
                    if error_type in {"incomplete_response", "missing_structured_payload"}:
                        retry_result = await self._retry_request_with_expanded_budget(
                            result.request,
                            reason=error_type,
                        )
                        if retry_result and not retry_result.error:
                            true_results_final.append(retry_result)
                            continue
                true_results_final.append(result)

            self._record_async_failures(true_results_final, stage="true_notes")

            # Build true_map and extract compact notes for Pass 2
            # Group responses by sample_id (3 streams per sample)
            true_map_per_stream: dict[str, list[dict[str, Any]]] = {}
            for res in true_results_final:
                if res.response is None:
                    continue
                sample_id = str(res.request.metadata.get("sample_id", ""))
                if not sample_id:
                    continue
                # Store response with metadata for remapping
                true_map_per_stream.setdefault(sample_id, []).append(
                    {
                        "response": res.response,
                        "stream_id": res.request.metadata.get("stream_id"),
                        "stream_idx": int(res.request.metadata.get("stream_idx", 0)),
                        "slice_offset": int(res.request.metadata.get("slice_offset", 0)),
                    }
                )

            # Merge per-stream responses and remap evidence indices
            true_notes_map: dict[str, list[Any]] = {}
            true_map: dict[str, Any] = {}
            for plan in batch:
                stream_responses = true_map_per_stream.get(plan.sample_id, [])
                if len(stream_responses) == 3:
                    # Process and merge all 3 stream responses
                    merged_notes = self._merge_stream_responses(stream_responses, plan)
                    true_notes_map[plan.sample_id] = merged_notes
                    # Create mock response for postprocess compatibility
                    true_map[plan.sample_id] = type(
                        "Response", (), {"parsed_json": {"notes": merged_notes, "z_n": ""}}
                    )()
                else:
                    logger.warning(
                        "Plan %s has incomplete stream responses (%d/3). Skipping.",
                        plan.sample_id,
                        len(stream_responses),
                    )

            # PASS 2: Student generates speculative notes (with teacher's notes as input)
            # Per-stream, per-variant requests (3 streams * N variants per plan)
            spec_requests = self._spec_requests(batch, true_notes_map)
            logger.info(
                "Pass 2 (Student): Generating speculative notes for batch of %d plans (%d per-stream requests, %d variants each)",
                len(batch),
                len(spec_requests),
                self._notes_cfg.variants_per_sample,
            )
            spec_results = (
                await self._async_client.submit_batch(
                    spec_requests, concurrency=self._notes_cfg.concurrency
                )
                if spec_requests
                else []
            )

            # Retry failed requests with expanded token budget
            spec_results_final: list[StructuredOutputResult] = []
            for result in spec_results:
                if result.error:
                    error_type = getattr(result.error, "error_type", "")
                    if error_type in {"incomplete_response", "missing_structured_payload"}:
                        retry_result = await self._retry_request_with_expanded_budget(
                            result.request,
                            reason=error_type,
                        )
                        if retry_result and not retry_result.error:
                            spec_results_final.append(retry_result)
                            continue
                spec_results_final.append(result)

            self._record_async_failures(spec_results_final, stage="spec_notes")

            # Group speculative responses by sample_id and variant
            spec_map_per_stream: dict[str, dict[int, list[dict[str, Any]]]] = {}
            for res in spec_results_final:
                if res.response is None:
                    continue
                sample_id = str(res.request.metadata.get("sample_id", ""))
                variant_idx = int(res.request.metadata.get("variant_index", 0))
                if not sample_id:
                    continue
                # Organize by sample_id -> variant_idx -> list of stream responses
                if sample_id not in spec_map_per_stream:
                    spec_map_per_stream[sample_id] = {}
                if variant_idx not in spec_map_per_stream[sample_id]:
                    spec_map_per_stream[sample_id][variant_idx] = []
                spec_map_per_stream[sample_id][variant_idx].append(
                    {
                        "response": res.response,
                        "stream_id": res.request.metadata.get("stream_id"),
                        "stream_idx": int(res.request.metadata.get("stream_idx", 0)),
                        "slice_offset": int(res.request.metadata.get("slice_offset", 0)),
                    }
                )

            # Merge per-stream spec responses
            spec_map: dict[str, list[Any]] = {}
            for plan in batch:
                variants_data = spec_map_per_stream.get(plan.sample_id, {})
                merged_variants = []
                for variant_idx in sorted(variants_data.keys()):
                    stream_responses = variants_data[variant_idx]
                    if len(stream_responses) == 3:
                        merged_notes = self._merge_stream_responses(stream_responses, plan)
                        # Create mock response with variant metadata
                        merged_variants.append(
                            type(
                                "Response",
                                (),
                                {
                                    "parsed_json": {
                                        "variant_id": f"{plan.sample_id}_variant_{variant_idx}",
                                        "notes": merged_notes,
                                    }
                                },
                            )()
                        )
                    else:
                        logger.warning(
                            "Plan %s variant %d has incomplete stream responses (%d/3). Skipping.",
                            plan.sample_id,
                            variant_idx,
                            len(stream_responses),
                        )
                if merged_variants:
                    spec_map[plan.sample_id] = merged_variants

            artifacts = self._postprocess_batch(batch, true_map, spec_map)
            generated.extend(artifacts)
            progress.update(len(batch))
        progress.close()
        return generated

    def generate(self, plans: Sequence[PlanDocument]) -> list[NotesArtifact]:
        pending = self._pending_plans(plans)
        if not pending:
            logger.info("No notes tasks to run (resume skipped all targets)")
            return []
        return asyncio.run(self.generate_async(pending))

    def _true_requests(self, plan: PlanDocument) -> list[StructuredOutputRequest]:
        """
        Generate per-stream true notes requests.

        Returns 3 requests (one per stream) that run in parallel.
        Each request receives only the text slice for its stream.
        """
        requests: list[StructuredOutputRequest] = []
        input_text = plan.payload.get("input_text", "")
        streams = plan.payload.get("streams", [])

        if len(streams) != 3:
            logger.warning(
                "Plan %s has %d streams (expected 3). Skipping per-stream generation.",
                plan.sample_id,
                len(streams),
            )
            return []

        for stream_idx, stream in enumerate(streams):
            stream_id = stream.get("stream_id", f"stream_{stream_idx + 1}")
            section_contract = stream.get("section_contract")

            if not section_contract:
                logger.warning(
                    "Stream %s in plan %s missing section_contract. Skipping.",
                    stream_id,
                    plan.sample_id,
                )
                continue

            try:
                sliced_text, offset = self._extract_stream_slice(input_text, section_contract)
            except ValueError as exc:
                logger.warning(
                    "Failed to extract slice for stream %s in plan %s: %s",
                    stream_id,
                    plan.sample_id,
                    exc,
                )
                continue

            messages = [
                {"role": "system", "content": TRUE_NOTES_PROMPT},
                {
                    "role": "user",
                    "content": self._single_stream_notes_prompt(stream, sliced_text, stream_id),
                },
            ]
            metadata = {
                "sample_id": plan.sample_id,
                "domain": plan.domain,
                "plan_path": str(plan.path),
                "stream_id": stream_id,
                "stream_idx": str(stream_idx),
                "slice_offset": str(offset),
            }
            request_id = f"{plan.sample_id}_{stream_id}"
            requests.append(
                StructuredOutputRequest(
                    request_id=request_id,
                    messages=messages,
                    schema_name="true_notes_single_stream",
                    schema=TRUE_NOTES_SCHEMA_SINGLE_STREAM,
                    # Per-stream responses are ~1/3 size of full response
                    # Reduced from 100k to 35k for single-stream focus
                    max_output_tokens=max(self._true_cfg.max_new_tokens, 16_384),
                    temperature=self._true_cfg.temperature,
                    top_p=self._true_cfg.top_p,
                    stop_sequences=tuple(self._true_cfg.stop_sequences),
                    seed=self._true_cfg.seed,
                    reasoning_effort=self._reasoning_effort,
                    metadata=metadata,
                )
            )
        return requests

    def _spec_requests(
        self,
        batch: Sequence[PlanDocument],
        true_notes_map: Mapping[str, list[Any]],
    ) -> list[StructuredOutputRequest]:
        """
        Generate per-stream speculative notes requests.

        Returns 3 * variants_per_sample requests per plan (one per stream per variant).
        Each request receives only the text slice and true notes for its stream.
        """
        requests: list[StructuredOutputRequest] = []
        for plan in batch:
            true_notes = true_notes_map.get(plan.sample_id, [])
            if not true_notes or len(true_notes) != 3:
                logger.warning(
                    "Skipping speculative notes for %s (expected 3 true note streams, got %d)",
                    plan.sample_id,
                    len(true_notes),
                )
                continue

            input_text = plan.payload.get("input_text", "")
            streams = plan.payload.get("streams", [])

            for stream_idx, (stream, stream_true_notes) in enumerate(zip(streams, true_notes)):
                stream_id = stream.get("stream_id", f"stream_{stream_idx + 1}")
                section_contract = stream.get("section_contract")

                if not section_contract:
                    logger.warning(
                        "Stream %s in plan %s missing section_contract. Skipping.",
                        stream_id,
                        plan.sample_id,
                    )
                    continue

                try:
                    sliced_text, offset = self._extract_stream_slice(input_text, section_contract)
                except ValueError as exc:
                    logger.warning(
                        "Failed to extract slice for stream %s in plan %s: %s",
                        stream_id,
                        plan.sample_id,
                        exc,
                    )
                    continue

                # Generate variants for this stream
                for variant in range(self._notes_cfg.variants_per_sample):
                    variant_id = f"{plan.sample_id}_{stream_id}_variant_{variant}"
                    messages = [
                        {"role": "system", "content": SPEC_NOTES_PROMPT},
                        {
                            "role": "user",
                            "content": self._single_stream_spec_prompt(
                                stream, sliced_text, stream_id, variant, stream_true_notes
                            ),
                        },
                    ]
                    metadata = {
                        "sample_id": plan.sample_id,
                        "domain": plan.domain,
                        "plan_path": str(plan.path),
                        "stream_id": stream_id,
                        "stream_idx": str(stream_idx),
                        "slice_offset": str(offset),
                        "variant_index": str(variant),
                    }
                    requests.append(
                        StructuredOutputRequest(
                            request_id=variant_id,
                            messages=messages,
                            schema_name="speculative_notes_single_stream",
                            schema=SPEC_NOTES_SCHEMA_SINGLE_STREAM,
                            # Per-stream responses are ~1/3 size of full response
                            max_output_tokens=max(self._spec_cfg.max_new_tokens, 16_384),
                            temperature=self._spec_cfg.temperature,
                            top_p=self._spec_cfg.top_p,
                            stop_sequences=tuple(self._spec_cfg.stop_sequences),
                            seed=(self._spec_cfg.seed or 0) + variant,
                            reasoning_effort=self._reasoning_effort,
                            metadata=metadata,
                        )
                    )
        return requests

    def _single_stream_notes_prompt(
        self, stream: Mapping[str, Any], sliced_text: str, stream_id: str
    ) -> str:
        """Build prompt for single-stream true notes extraction."""
        prompt_parts = [
            f"Stream ID: {stream_id}",
            f"Header: {stream.get('header', '')}",
            f"Summary: {stream.get('summary', '')}",
            "",
            "Notes Contract (requirements):",
            json.dumps(stream.get("notes_contract", []), ensure_ascii=False),
            "",
            "Stream Text Slice:",
            sliced_text,
            "",
            "REMINDER: All evidence spans must use indices relative to the stream slice (0-indexed).",
        ]
        return "\n".join(prompt_parts)

    def _single_stream_spec_prompt(
        self,
        stream: Mapping[str, Any],
        sliced_text: str,
        stream_id: str,
        variant_index: int,
        stream_true_notes: Any,
    ) -> str:
        """Build prompt for single-stream speculative notes extraction."""
        prompt_parts = [
            f"Stream ID: {stream_id}",
            f"Variant: {variant_index}",
            "",
            "TRUE NOTES (from Oracle Teacher for this stream - compact format):",
            json.dumps(stream_true_notes, ensure_ascii=False),
            "",
            f"Noise config: {json.dumps(_noise_config_dict(self._noise_cfg), ensure_ascii=False)}",
            "",
            "Stream Text Slice:",
            sliced_text,
            "",
            "REMINDER: Generate noisy notes based on the TRUE NOTES above.",
            "REMINDER: All evidence spans must use indices relative to the stream slice (0-indexed).",
        ]
        return "\n".join(prompt_parts)

    def _extract_stream_slice(
        self,
        input_text: str,
        section_contract: Mapping[str, Any],
    ) -> tuple[str, int]:
        """
        Extract text slice for a single stream based on section_contract.

        Args:
            input_text: Full document text
            section_contract: Stream's section_contract (contains start_idx, end_idx)

        Returns:
            (sliced_text, offset) where offset is start_idx for remapping evidence spans

        Raises:
            ValueError: If section_contract is invalid or indices out of bounds
        """
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("input_text is empty; cannot extract stream slice")

        contract_type = section_contract.get("type")
        if contract_type != "source_slice":
            raise ValueError(
                f"Unsupported section_contract type: {contract_type}. "
                f"Only 'source_slice' is supported for per-stream note generation."
            )

        start_idx = section_contract.get("start_idx")
        end_idx = section_contract.get("end_idx")

        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            raise ValueError(
                f"section_contract.start_idx and end_idx must be integers, "
                f"got start_idx={type(start_idx).__name__}, end_idx={type(end_idx).__name__}"
            )

        if start_idx < 0:
            raise ValueError(f"start_idx must be non-negative, got {start_idx}")

        if end_idx > len(input_text):
            raise ValueError(f"end_idx ({end_idx}) exceeds input_text length ({len(input_text)})")

        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid range: start_idx ({start_idx}) must be < end_idx ({end_idx})"
            )

        sliced_text = input_text[start_idx:end_idx]
        if not sliced_text.strip():
            raise ValueError(
                f"Extracted slice for range [{start_idx}, {end_idx}) is empty. "
                "Check section_contract indices."
            )

        return sliced_text, start_idx

    def _plan_stream_slices(self, plan: PlanDocument) -> OrderedDict[str, str]:
        """
        Materialize per-stream text slices directly from the plan contract.
        """
        slices: OrderedDict[str, str] = OrderedDict()
        input_text = str(plan.payload.get("input_text", ""))
        streams = plan.payload.get("streams", [])
        for idx, stream in enumerate(streams or []):
            if not isinstance(stream, Mapping):
                continue
            stream_id = str(stream.get("stream_id") or f"stream_{idx + 1}")
            contract = stream.get("section_contract")
            if not contract:
                continue
            try:
                slice_text, _ = self._extract_stream_slice(input_text, contract)
            except ValueError:
                continue
            slices[stream_id] = slice_text
        if not slices and input_text.strip():
            slices["stream_1"] = input_text
        return slices

    def _remap_evidence_indices(
        self,
        notes: StreamNotes,
        offset: int,
    ) -> StreamNotes:
        """
        Remap evidence spans from stream-relative to document-absolute indices.

        During per-stream note generation, the LLM receives a text slice starting at
        character 0. This method adds the stream's start offset to all evidence spans
        to convert them back to document-absolute indices.

        Args:
            notes: StreamNotes with stream-relative evidence spans
            offset: Start index of stream slice in full document (from section_contract.start_idx)

        Returns:
            StreamNotes with absolute evidence spans (modifies in place and returns)
        """
        for fact in notes.facts:
            if fact.evidence_span:
                fact.evidence_span.start += offset
                fact.evidence_span.end += offset
        return notes

    def _merge_stream_responses(
        self,
        stream_responses: list[dict[str, Any]],
        plan: PlanDocument,
    ) -> list[Any]:
        """
        Merge per-stream true notes responses into verbose format with remapped evidence indices.

        Args:
            stream_responses: List of stream response dicts with metadata
            plan: Original plan document for error reporting

        Returns:
            List of verbose note dictionaries (compatible with load_stream_notes)
        """
        # Sort by stream_idx to maintain order
        stream_responses = sorted(stream_responses, key=lambda x: x.get("stream_idx", 0))

        merged_notes: list[Any] = []
        for stream_data in stream_responses:
            response = stream_data["response"]
            slice_offset = stream_data.get("slice_offset", 0)
            stream_id = stream_data.get("stream_id", "unknown")

            # DEBUG: Log slice offset for verification
            logger.info(
                "Merging stream %s (idx %s) with slice_offset %s",
                stream_id,
                stream_data.get("stream_idx"),
                slice_offset,
            )

            if not response or not response.parsed_json:
                logger.warning(
                    "Empty response for stream %s in plan %s",
                    stream_id,
                    plan.sample_id,
                )
                continue

            parsed = response.parsed_json

            # Handle case where LLM returns old multi-stream format with 'notes' array
            # This happens when OpenAI doesn't respect the single-stream schema
            if "notes" in parsed and isinstance(parsed["notes"], list):
                logger.info(
                    "  LLM returned multi-stream format for %s. Extracting from 'notes' array.",
                    stream_id,
                )
                # The 'notes' array may have multiple streams - find the one matching stream_id
                ent_compact = []
                fact_compact = []
                coverage_compact = []

                for stream_data in parsed["notes"]:
                    if isinstance(stream_data, Mapping):
                        if stream_data.get("stream_id") == stream_id:
                            ent_compact = stream_data.get("ENT", [])
                            fact_compact = stream_data.get("FACT", [])
                            coverage_compact = stream_data.get("COVERAGE", [])
                            logger.info("  Found matching stream data for %s", stream_id)
                            break

                if not ent_compact and not fact_compact and not coverage_compact:
                    logger.warning(
                        "  No matching stream data found for %s in notes array", stream_id
                    )
            else:
                # Extract from top-level (single-stream schema)
                ent_compact = parsed.get("ENT", [])
                fact_compact = parsed.get("FACT", [])
                coverage_compact = parsed.get("COVERAGE", [])

            # DEBUG: Log what LLM returned
            logger.info(
                "  LLM response for %s: ENT=%d items, FACT=%d items, COVERAGE=%d items",
                stream_id,
                len(ent_compact),
                len(fact_compact),
                len(coverage_compact),
            )

            # Normalize verbose format to compact if needed
            # Some models (like local gpt-oss) return verbose dicts instead of compact arrays
            def _normalize_to_compact(items: list[Any], item_type: str) -> list[Any]:
                """Convert verbose dict format to compact array format if needed."""
                if not items or not isinstance(items, list):
                    return []

                normalized = []
                for item in items:
                    # Skip non-dict/non-list items
                    if not isinstance(item, (list, Mapping)):
                        continue

                    # If already an array/tuple, keep it
                    if isinstance(item, (list, tuple)):
                        normalized.append(item)
                        continue

                    # Convert verbose dict to compact array
                    if item_type == "ENT":
                        normalized.append(
                            [
                                item.get("id", ""),
                                item.get("name", ""),
                                item.get("type", ""),
                                item.get("canonical", True),
                                item.get("aliases", []),
                            ]
                        )
                    elif item_type == "FACT":
                        ev_span = item.get("evidence_span", {})
                        normalized.append(
                            [
                                item.get("subj_id", ""),
                                item.get("predicate", ""),
                                item.get("object", ""),
                                item.get("certainty", 1.0),
                                [
                                    ev_span.get("start", 0),
                                    ev_span.get("end", 0),
                                    ev_span.get("text", ""),
                                ],
                            ]
                        )
                    elif item_type == "COVERAGE":
                        normalized.append([item.get("plan_item_id", ""), item.get("status", "")])

                return normalized

            ent_compact = _normalize_to_compact(ent_compact, "ENT")
            fact_compact = _normalize_to_compact(fact_compact, "FACT")
            coverage_compact = _normalize_to_compact(coverage_compact, "COVERAGE")

            # Convert compact format to verbose format with remapped evidence indices
            verbose_entities = []
            for ent_tuple in ent_compact:
                if len(ent_tuple) >= 5:
                    verbose_entities.append(
                        {
                            "id": ent_tuple[0],
                            "name": ent_tuple[1],
                            "type": ent_tuple[2],
                            "canonical": ent_tuple[3],
                            "aliases": ent_tuple[4] if isinstance(ent_tuple[4], list) else [],
                        }
                    )

            verbose_facts = []
            for i, fact_tuple in enumerate(fact_compact):
                if len(fact_tuple) >= 5 and isinstance(fact_tuple[4], list):
                    evidence = fact_tuple[4]
                    if len(evidence) >= 3:
                        # CRITICAL: Remap evidence indices from stream-relative to document-absolute
                        start_remapped = int(evidence[0]) + slice_offset
                        end_remapped = int(evidence[1]) + slice_offset

                        if i < 3:  # Log first few facts
                            logger.info(
                                "  Fact %d: raw_start=%s, offset=%s, remapped=%s",
                                i,
                                evidence[0],
                                slice_offset,
                                start_remapped,
                            )

                        verbose_facts.append(
                            {
                                "subj_id": fact_tuple[0],
                                "predicate": fact_tuple[1],
                                "object": fact_tuple[2],
                                "certainty": fact_tuple[3],
                                "evidence_span": {
                                    "start": start_remapped,
                                    "end": end_remapped,
                                    "text": evidence[2],
                                },
                            }
                        )

            verbose_coverage = []
            for cov_tuple in coverage_compact:
                if len(cov_tuple) >= 2:
                    verbose_coverage.append(
                        {
                            "plan_item_id": cov_tuple[0],
                            "status": cov_tuple[1],
                        }
                    )

            # Append stream notes in verbose format (ready for load_stream_notes)
            stream_notes = {
                "stream_id": stream_id,
                "ENT": verbose_entities,
                "FACT": verbose_facts,
                "COVERAGE": verbose_coverage,
            }

            merged_notes.append(stream_notes)

        return merged_notes

    def _postprocess_batch(
        self,
        batch: Sequence[PlanDocument],
        true_map,
        spec_map,
    ) -> list[NotesArtifact]:
        artifacts: list[NotesArtifact] = []
        with ThreadPoolExecutor(max_workers=self._notes_cfg.max_workers) as pool:
            futures: list[tuple[PlanDocument, Any]] = []
            for plan in batch:
                true_response = true_map.get(plan.sample_id)
                if true_response is None:
                    self._log_failure(
                        sample_id=plan.sample_id,
                        domain=plan.domain,
                        stage="true_notes",
                        error_type="missing_true_notes",
                        message="True notes response missing after async generation.",
                        plan_path=plan.path,
                    )
                    continue
                spec_responses = spec_map.get(plan.sample_id, [])
                futures.append(
                    (
                        plan,
                        pool.submit(
                            self._compile_artifact,
                            plan,
                            true_response,
                            spec_responses,
                        ),
                    )
                )
            for plan, future in futures:
                try:
                    artifact = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    self._log_failure(
                        sample_id=plan.sample_id,
                        domain=plan.domain,
                        stage="compile_artifact",
                        error_type=exc.__class__.__name__,
                        message=str(exc),
                        plan_path=plan.path,
                    )
                    continue
                artifacts.append(artifact)
        return artifacts

    def _compile_artifact(
        self,
        plan: PlanDocument,
        true_response,
        spec_responses: Sequence[Any],
    ) -> NotesArtifact:
        if true_response is None:
            raise RuntimeError(f"Missing true notes response for {plan.sample_id}")

        # LOG RAW RESPONSE FOR DEBUGGING
        logger.info("=" * 80)
        logger.info("RAW TRUE NOTES RESPONSE for %s:", plan.sample_id)
        logger.info("  parsed_json type: %s", type(getattr(true_response, "parsed_json", None)))
        logger.info(
            "  parsed_json keys: %s",
            (
                list(getattr(true_response, "parsed_json", {}).keys())
                if isinstance(getattr(true_response, "parsed_json", None), Mapping)
                else "N/A"
            ),
        )
        logger.info("  output_text length: %d", len(getattr(true_response, "output_text", "")))
        if hasattr(true_response, "parsed_json"):
            logger.info(
                "  parsed_json sample: %s", json.dumps(true_response.parsed_json, indent=2)[:1000]
            )
        logger.info("=" * 80)

        true_payload = true_response.parsed_json or {}

        # If parsed_json is empty, try parsing output_text directly
        if not true_payload:
            output_text = getattr(true_response, "output_text", "").strip()
            if output_text:
                try:
                    true_payload = json.loads(output_text)
                    logger.info(
                        "Successfully parsed output_text as fallback for %s", plan.sample_id
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse output_text as JSON for %s", plan.sample_id)
                    true_payload = {}

        # Rehydrate compact array format to verbose object format
        try:
            raw_notes = true_payload.get("notes", [])
            if not raw_notes:
                logger.warning(
                    "Empty notes array for %s, payload keys: %s",
                    plan.sample_id,
                    list(true_payload.keys()),
                )

            # Normalize structure: LLM may return various formats without strict schema
            logger.info(
                "Before normalization - raw_notes type: %s, len: %s",
                type(raw_notes),
                len(raw_notes) if hasattr(raw_notes, "__len__") else "N/A",
            )
            if isinstance(raw_notes, (list, dict)) and raw_notes:
                sample_str = (
                    str(raw_notes)[:300]
                    if isinstance(raw_notes, list)
                    else str(list(raw_notes.keys()))
                )
                logger.info("Before normalization - sample: %s", sample_str)

            if isinstance(raw_notes, Mapping):
                # If notes is a single dict with ENT/FACT/COVERAGE keys, wrap it in stream structure
                if any(key in raw_notes for key in ("ENT", "FACT", "COVERAGE")):
                    raw_notes = [dict(raw_notes, stream_id="stream_1")]
                    logger.info("Normalized: wrapped single ENT/FACT/COVERAGE dict")
                else:
                    # Otherwise convert dict values to list
                    raw_notes = list(raw_notes.values())
                    logger.info("Normalized: converted dict values to list")
            elif isinstance(raw_notes, list) and raw_notes:
                # Check if it's a flat list of compact arrays (not wrapped in streams)
                first_item = raw_notes[0]
                if isinstance(first_item, list):
                    # It's a flat list of compact entities/facts - need to wrap in stream structure
                    # Group them by type based on array length:
                    # - 5 elements = entity: [id, name, type, canonical, [aliases]]
                    # - 5 elements with nested array = fact: [subj_id, predicate, object, certainty, [start, end, text]]
                    # - 2 elements = coverage: [plan_item_id, status]
                    entities = []
                    facts = []
                    coverage = []
                    for item in raw_notes:
                        if not isinstance(item, list):
                            continue

                        # CRITICAL FIX: Strip type prefix if present
                        # LLM returns: ["ENT", id, name, type, canonical, [aliases]]
                        # We need:     [id, name, type, canonical, [aliases]]
                        if (
                            item
                            and isinstance(item[0], str)
                            and item[0] in ("ENT", "FACT", "COVERAGE")
                        ):
                            item = item[1:]

                        if len(item) == 5 and isinstance(item[4], list):
                            # Could be entity or fact - check if last element has 3 items (evidence span)
                            if len(item[4]) == 3 and all(
                                isinstance(x, (int, float, str)) for x in item[4][:2]
                            ):
                                facts.append(item)
                            else:
                                entities.append(item)
                        elif len(item) == 2:
                            coverage.append(item)
                    raw_notes = [
                        {
                            "stream_id": "stream_1",
                            "ENT": entities,
                            "FACT": facts,
                            "COVERAGE": coverage,
                        }
                    ]
                    logger.info(
                        "Normalized: wrapped flat list into stream structure (%d ENT, %d FACT, %d COV)",
                        len(entities),
                        len(facts),
                        len(coverage),
                    )

            logger.info(
                "After normalization - raw_notes type: %s, len: %s",
                type(raw_notes),
                len(raw_notes) if hasattr(raw_notes, "__len__") else "N/A",
            )

            # Check if notes are already in verbose format (from per-stream merging)
            # or need rehydration from compact format
            needs_rehydration = False
            if isinstance(raw_notes, list) and raw_notes:
                first_item = raw_notes[0]
                if isinstance(first_item, Mapping):
                    # Check if ENT/FACT are already verbose (dicts) or compact (arrays)
                    # Try ENT first, fall back to FACT if ENT is empty
                    ent_items = first_item.get("ENT", [])
                    fact_items = first_item.get("FACT", [])

                    # Check ENT or FACT to determine format
                    check_items = ent_items if ent_items else fact_items
                    if check_items:
                        is_verbose = isinstance(check_items[0], Mapping)
                        if is_verbose:
                            # Already verbose format, skip rehydration
                            needs_rehydration = False
                        else:
                            # Compact format, needs rehydration
                            needs_rehydration = True
                    else:
                        # Empty notes, doesn't matter
                        needs_rehydration = False
                else:
                    needs_rehydration = True

            if needs_rehydration:
                rehydrated_notes = self._rehydrate_compact_notes(raw_notes)
                parsed_true_notes = self._parse_streams(rehydrated_notes)
            else:
                # Notes are already verbose, directly parse them
                parsed_true_notes = self._parse_streams(raw_notes)
        except ValueError as exc:
            logger.error(
                "Rehydration failed for %s. Payload type: %s, keys: %s, notes type: %s",
                plan.sample_id,
                type(true_payload),
                list(true_payload.keys()) if isinstance(true_payload, Mapping) else "N/A",
                type(raw_notes),
            )
            raise RuntimeError(
                f"Failed to rehydrate compact notes for {plan.sample_id}: {exc}"
            ) from exc
        if not parsed_true_notes:
            parsed_true_notes = self._fallback_notes(plan)
        input_text = str(plan.payload.get("input_text", ""))
        plan_seed_notes = derive_initial_notes_from_plan(plan.payload, input_text=input_text)
        true_notes = merge_seed_notes(
            plan.payload.get("streams", []),
            parsed_true_notes,
            plan_seed_notes,
            input_text=input_text,
        )
        stream_slices = self._plan_stream_slices(plan)
        z_true = "\n\n".join(text.strip() for text in stream_slices.values() if text.strip())
        if not z_true:
            z_true = input_text
        # Generate versioned_notes procedurally from evidence spans
        try:
            versioned_notes = generate_procedural_snapshots(
                final_notes=true_notes,
                z_n=z_true,
                note_cadence_M=plan.note_cadence,
                lag_delta=plan.lag_delta,
            )
        except ValueError as exc:
            logger.error(
                "Procedural snapshot generation failed for %s: %s",
                plan.sample_id,
                exc,
            )
            raise RuntimeError(
                f"Evidence span validation failed for {plan.sample_id}: {exc}"
            ) from exc
        true_dict = [note.as_dict() for note in true_notes]
        spec_variants = self._build_spec_variants(plan, spec_responses, true_notes, z_true)
        rollback = self._simulate_rollback(plan)
        z_spec = [variant["z_hat"] for variant in spec_variants]
        kl_divergence = _approximate_kl(z_true, z_spec)
        artifact = {
            "sample_id": plan.sample_id,
            "domain": plan.domain,
            "plan_path": str(plan.path),
            "true_notes": true_dict,
            "speculative_notes": spec_variants,
            "z_n": z_true,
            "z_hat": z_spec,
            "versioned_notes": versioned_notes,
            "lag_delta": plan.lag_delta,
            "note_cadence_M": plan.note_cadence,
            "rollback": rollback,
            "kl_divergence": kl_divergence,
            "sectional_independence": plan.sectional_independence,
        }
        output_path = self._artifact_path(plan)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        return NotesArtifact(
            sample_id=plan.sample_id,
            domain=plan.domain,
            path=output_path,
            true_notes=true_notes,
            speculative_variants=spec_variants,
            z_true=z_true,
            z_speculative=z_spec,
            rollback=rollback,
            metadata={
                "versioned_notes": versioned_notes,
                "kl_divergence": kl_divergence,
            },
        )

    def _build_spec_variants(
        self,
        plan: PlanDocument,
        responses: Sequence[Any],
        true_notes: Sequence[StreamNotes],
        default_z_text: str,
    ) -> list[dict[str, Any]]:
        variants: list[dict[str, Any]] = []
        rng = random.Random(plan.sample_id)
        for idx, resp in enumerate(responses):
            try:
                # LOG RAW SPECULATIVE RESPONSE
                logger.debug("RAW SPEC NOTES RESPONSE for %s variant %d:", plan.sample_id, idx)
                logger.debug("  parsed_json type: %s", type(getattr(resp, "parsed_json", None)))
                if (
                    hasattr(resp, "parsed_json") and idx == 0
                ):  # Only log first variant to avoid spam
                    logger.debug(
                        "  parsed_json sample: %s", json.dumps(resp.parsed_json, indent=2)[:500]
                    )

                payload = getattr(resp, "parsed_json", None) or {}
                # Handle payload being a string
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse payload as JSON for %s variant %d",
                            plan.sample_id,
                            idx,
                        )
                        payload = {}

                if not isinstance(payload, Mapping):
                    logger.warning(
                        "Payload is not a dict for %s variant %d, got %s",
                        plan.sample_id,
                        idx,
                        type(payload),
                    )
                    payload = {}

                # Check if notes are already in verbose format (from per-stream merging)
                # or need rehydration from compact format
                raw_notes = payload.get("notes", [])
                needs_rehydration = False
                if isinstance(raw_notes, list) and raw_notes:
                    first_item = raw_notes[0]
                    if isinstance(first_item, Mapping):
                        # Check if ENT/FACT are already verbose (dicts) or compact (arrays)
                        # Try ENT first, fall back to FACT if ENT is empty
                        ent_items = first_item.get("ENT", [])
                        fact_items = first_item.get("FACT", [])

                        # Check ENT or FACT to determine format
                        check_items = ent_items if ent_items else fact_items
                        if check_items:
                            is_verbose = isinstance(check_items[0], Mapping)
                            logger.info(
                                "  Structure check for %s: Stream 1 first item type: %s, is_verbose: %s",
                                plan.sample_id,
                                type(check_items[0]),
                                is_verbose,
                            )
                            if is_verbose:
                                # Already verbose format, skip rehydration
                                needs_rehydration = False
                            else:
                                # Compact format, needs rehydration
                                needs_rehydration = True
                        else:
                            # Empty notes, doesn't matter
                            logger.info(
                                "  Structure check for %s: Stream 1 empty, needs_rehydration=False",
                                plan.sample_id,
                            )
                            needs_rehydration = False
                    else:
                        logger.info(
                            "  Structure check for %s: Stream 1 not Mapping, needs_rehydration=True",
                            plan.sample_id,
                        )
                        needs_rehydration = True

                logger.info(
                    "  Decision for %s: needs_rehydration=%s", plan.sample_id, needs_rehydration
                )

                if needs_rehydration:
                    rehydrated_notes = self._rehydrate_compact_notes(raw_notes)
                    notes = self._parse_streams(rehydrated_notes)
                else:
                    # Notes are already verbose, directly parse them
                    notes = self._parse_streams(raw_notes)
            except (ValueError, AttributeError) as exc:
                logger.warning(
                    "Failed to process speculative notes for %s variant %d: %s",
                    plan.sample_id,
                    idx,
                    exc,
                )
                notes = []

            if not notes:
                notes = self._noisy_copy(true_notes, rng)
            else:
                notes = self._apply_noise(notes, rng)
            variants.append(
                {
                    "variant_id": payload.get("variant_id") or resp.request.request_id,
                    "z_hat": str(payload.get("z_hat", "")).strip() or default_z_text,
                    "notes": [note.as_dict() for note in notes],
                    "noise_config": _noise_config_dict(self._noise_cfg),
                    "lag_delta": plan.lag_delta,
                }
            )
        if not variants:
            variants.append(
                {
                    "variant_id": f"{plan.sample_id}_synthetic",
                    "z_hat": default_z_text,
                    "notes": [note.as_dict() for note in self._noisy_copy(true_notes, rng)],
                    "noise_config": _noise_config_dict(self._noise_cfg),
                    "lag_delta": plan.lag_delta,
                }
            )
        return variants

    def _rehydrate_compact_notes(self, compact_notes: Sequence[Any]) -> list[dict[str, Any]]:
        """
        Convert compact array format to verbose object format.

        Compact format uses arrays to save tokens:
        - ENT: [id, name, type, canonical, [aliases]]
        - FACT: [subj_id, predicate, object, certainty, [start, end, text]]
        - COVERAGE: [plan_item_id, status]

        Raises:
            ValueError: If compact arrays are malformed
        """
        rehydrated: list[dict[str, Any]] = []

        for idx, stream in enumerate(compact_notes):
            # Handle case where LLM returns JSON-encoded strings
            if isinstance(stream, str):
                try:
                    stream = json.loads(stream)
                except json.JSONDecodeError:
                    logger.warning(
                        "Stream %d is a string but not valid JSON, skipping: %s",
                        idx,
                        stream[:100],
                    )
                    continue

            if not isinstance(stream, Mapping):
                logger.warning("Stream %d is not a mapping (got %s), skipping", idx, type(stream))
                continue

            stream_id = stream.get("stream_id")
            if not stream_id:
                stream_id = f"stream_{idx + 1}"
                logger.debug("Stream %d missing stream_id, using %s", idx, stream_id)

            # Rehydrate entities
            entities: list[dict[str, Any]] = []
            for idx, ent in enumerate(stream.get("ENT", [])):
                # Robustness check: if entity is already a dict, use it as is
                if isinstance(ent, Mapping):
                    entities.append(dict(ent))
                    continue

                if not isinstance(ent, Sequence) or isinstance(ent, (str, bytes)):
                    raise ValueError(f"Entity {idx} must be array, got {type(ent)}")
                if len(ent) != 5:
                    raise ValueError(f"Entity {idx} must have 5 elements, got {len(ent)}")

                entities.append(
                    {
                        "id": str(ent[0]),
                        "name": str(ent[1]),
                        "type": str(ent[2]),
                        "canonical": bool(ent[3]),
                        "aliases": [str(a) for a in ent[4]] if ent[4] else [],
                    }
                )

            # Rehydrate facts
            facts: list[dict[str, Any]] = []
            for idx, fact in enumerate(stream.get("FACT", [])):
                if not isinstance(fact, Sequence) or isinstance(fact, (str, bytes)):
                    logger.warning("Fact %d is not array, skipping: %s", idx, type(fact))
                    continue
                if len(fact) < 5:
                    logger.warning("Fact %d has only %d elements, skipping", idx, len(fact))
                    continue

                # Parse certainty as float, handling string values like "high", "medium", "low"
                certainty_raw = fact[3]
                if isinstance(certainty_raw, str):
                    certainty_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                    certainty = certainty_map.get(certainty_raw.lower(), 0.8)
                else:
                    try:
                        certainty = float(certainty_raw)
                    except (ValueError, TypeError):
                        certainty = 0.8

                evidence = fact[4]
                if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
                    logger.warning(
                        "Fact %d evidence_span must be array, got %s, using defaults",
                        idx,
                        type(evidence),
                    )
                    evidence = [0, 0, ""]
                elif len(evidence) < 3:
                    logger.warning(
                        "Fact %d evidence_span has only %d elements, padding", idx, len(evidence)
                    )
                    evidence = list(evidence) + [0, 0, ""][: 3 - len(evidence)]

                facts.append(
                    {
                        "subj_id": str(fact[0]),
                        "predicate": str(fact[1]),
                        "object": str(fact[2]),
                        "certainty": certainty,
                        "evidence_span": {
                            "start": (
                                int(evidence[0]) if isinstance(evidence[0], (int, float)) else 0
                            ),
                            "end": int(evidence[1]) if isinstance(evidence[1], (int, float)) else 0,
                            "text": str(evidence[2]),
                        },
                    }
                )

            # Rehydrate coverage
            coverage: list[dict[str, Any]] = []
            for idx, cov in enumerate(stream.get("COVERAGE", [])):
                if not isinstance(cov, Sequence) or isinstance(cov, (str, bytes)):
                    raise ValueError(f"Coverage {idx} must be array, got {type(cov)}")
                if len(cov) != 2:
                    raise ValueError(f"Coverage {idx} must have 2 elements, got {len(cov)}")

                coverage.append(
                    {
                        "plan_item_id": str(cov[0]),
                        "status": str(cov[1]),
                    }
                )

            rehydrated.append(
                {
                    "stream_id": stream_id,
                    "ENT": entities,
                    "FACT": facts,
                    "COVERAGE": coverage,
                }
            )

        return rehydrated

    def _parse_streams(self, raw_notes: Sequence[Mapping[str, Any]]) -> list[StreamNotes]:
        parsed: list[StreamNotes] = []
        for raw in raw_notes:
            try:
                parsed.append(load_stream_notes(raw))
            except ValueError as exc:
                logger.debug("Skipping malformed stream payload: %s", exc)
        return parsed

    def _fallback_notes(self, plan: PlanDocument) -> list[StreamNotes]:
        fallback = []
        streams = plan.payload.get("streams", [])
        for stream in streams:
            fallback.append(
                StreamNotes(
                    stream_id=str(stream.get("stream_id") or f"stream_{len(fallback)+1}"),
                    entities=[],
                    facts=[],
                    coverage=[],
                )
            )
        return fallback

    def _noisy_copy(
        self, true_notes: Sequence[StreamNotes], rng: random.Random
    ) -> list[StreamNotes]:
        copies: list[StreamNotes] = []
        for note in true_notes:
            copies.append(
                StreamNotes(
                    stream_id=note.stream_id,
                    entities=list(note.entities),
                    facts=list(note.facts),
                    coverage=list(note.coverage),
                )
            )
        return self._apply_noise(copies, rng)

    def _apply_noise(self, notes: Sequence[StreamNotes], rng: random.Random) -> list[StreamNotes]:
        mutated: list[StreamNotes] = []
        for note in notes:
            entities = list(note.entities)
            facts = list(note.facts)
            coverage = list(note.coverage)
            if entities and rng.random() < self._noise_cfg.paraphrase_ratio:
                entities = entities[:-1]
            if facts and rng.random() < self._noise_cfg.hallucination_ratio:
                facts = facts[:-1]
            if coverage and rng.random() < self._noise_cfg.drop_ratio:
                coverage = coverage[:-1]
            mutated.append(
                StreamNotes(
                    stream_id=note.stream_id, entities=entities, facts=facts, coverage=coverage
                )
            )
        if self._noise_cfg.shuffle_notes and len(mutated) > 1:
            rng.shuffle(mutated)
        return mutated

    def _simulate_rollback(self, plan: PlanDocument) -> Mapping[str, Any]:
        probability = self._rng.uniform(
            self._notes_cfg.rollback_probability_low, self._notes_cfg.rollback_probability_high
        )
        triggered = self._rng.random() < probability
        if not triggered:
            return {"triggered": False, "l_tokens": 0, "events": []}
        events = []
        steps = self._rng.randint(1, 3)
        for _ in range(steps):
            events.append(
                {
                    "position": self._rng.randint(1, 2048),
                    "reason": self._rng.choice(["agreement_head", "coverage_violation"]),
                    "l_tokens": self._rng.choice([16, 32, 48, 64]),
                }
            )
        return {
            "triggered": True,
            "events": events,
            "l_tokens": max(event["l_tokens"] for event in events),
        }

    def _ensure_output_dir(self, plan: PlanDocument) -> None:
        out_dir = self._notes_cfg.output_root / plan.domain
        out_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, plan: PlanDocument) -> Path:
        return self._notes_cfg.output_root / plan.domain / f"{plan.sample_id}.json"

    async def _retry_request_with_expanded_budget(
        self,
        failed_request: StructuredOutputRequest,
        *,
        reason: str,
    ) -> StructuredOutputResult | None:
        """Retry a failed request with expanded token budget (50k minimum, then 2x up to 200k max)."""
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

        sample_id = failed_request.metadata.get("sample_id", failed_request.request_id)
        logger.warning(
            "Retrying %s with expanded token budget (%d -> %d tokens) after %s",
            sample_id,
            current_budget,
            next_budget,
            reason.replace("_", " "),
        )

        # Create retry request with expanded budget, copying all fields from original
        retry_request = StructuredOutputRequest(
            request_id=f"{failed_request.request_id}_retry",
            messages=failed_request.messages,
            schema_name=failed_request.schema_name,
            schema=failed_request.schema,
            max_output_tokens=next_budget,
            temperature=failed_request.temperature,
            top_p=failed_request.top_p,
            stop_sequences=failed_request.stop_sequences,
            seed=failed_request.seed,
            reasoning_effort=failed_request.reasoning_effort,  # None for gpt-4o, "low" for gpt-5.1
            metadata={**failed_request.metadata, "retry_reason": f"{reason}_retry"},
        )

        retry_results = await self._async_client.submit_batch(
            [retry_request],
            concurrency=1,
            max_retries=self._max_retries,
        )

        retry_result = retry_results[0]
        if retry_result.error:
            logger.error(
                "Expanded budget retry for %s failed (%s)",
                sample_id,
                getattr(retry_result.error, "error_type", "unknown_error"),
            )
            return None

        if retry_result.response is None:
            logger.error(
                "Expanded budget retry for %s returned empty response",
                sample_id,
            )
            return None

        logger.info(
            "Request %s succeeded after expanded budget retry (%d tokens)",
            sample_id,
            next_budget,
        )
        return retry_result

    @staticmethod
    def _should_abort_for_error(error: StructuredRequestError) -> bool:
        status = getattr(error, "status_code", None)
        error_type = getattr(error, "error_type", "")
        # Non-fatal errors: rate limiting (429), content policy (403), gateway errors (500),
        # incomplete responses (token budget exhausted), and missing structured payloads (truncated JSON).
        # These are logged but don't abort the pipeline - we continue processing other samples.
        # All other errors (parse failures, bad schemas) should abort.
        if status in (403, 429, 500) or error_type in {
            "http_403",
            "http_429",
            "http_500",
            "incomplete_response",
            "missing_structured_payload",
        }:
            return False
        return True

    def _record_async_failures(
        self,
        results: Sequence[StructuredOutputResult],
        *,
        stage: str,
    ) -> None:
        for result in results:
            if result.error is None:
                continue
            request_metadata: MutableMapping[str, Any] = dict(result.request.metadata)
            sample_id = str(request_metadata.get("sample_id") or result.request.request_id)
            domain = str(request_metadata.get("domain") or "unknown")
            plan_path_value = request_metadata.get("plan_path")
            plan_path = Path(plan_path_value) if isinstance(plan_path_value, str) else None
            error = result.error
            extra: dict[str, Any] = {
                "request_id": result.request.request_id,
                "http_status": getattr(error, "status_code", None),
                "retry_count": getattr(error, "attempts", 0),
            }
            if "variant_index" in request_metadata:
                extra["variant_index"] = request_metadata["variant_index"]
            self._log_failure(
                sample_id=sample_id,
                domain=domain,
                stage=stage,
                error_type=getattr(error, "error_type", error.__class__.__name__),
                message=str(error),
                plan_path=plan_path,
                metadata=extra,
            )
            if self._should_abort_for_error(error):
                raise RuntimeError(f"Notes generation failed for {sample_id} ({stage}): {error}")

    def _log_failure(
        self,
        *,
        sample_id: str,
        domain: str,
        stage: str,
        error_type: str,
        message: str,
        plan_path: Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "domain": domain,
            "stage": stage,
            "model_id": self._model_id,
            "error_type": error_type,
            "error_message": message,
            "timestamp": time.time(),
            "output_path": str(self._notes_cfg.output_root / domain / f"{sample_id}.json"),
        }
        if plan_path:
            entry["plan_path"] = str(plan_path)
        if metadata:
            entry.update(metadata)
        with self._failure_log_lock:
            with self._failure_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.error("Notes generation failed for %s (%s): %s", sample_id, stage, message)


def _approximate_kl(z_true: str, z_specs: Sequence[str]) -> float:
    if not z_true or not z_specs:
        return 1.0
    base = len(z_true)
    divergences = []
    for candidate in z_specs:
        diff = abs(len(candidate) - base)
        divergences.append(diff / max(base, 1))
    return float(min(1.0, sum(divergences) / len(divergences)))


def _noise_config_dict(cfg: SpeculativeNotesNoiseConfig) -> dict[str, Any]:
    return {
        "paraphrase_ratio": cfg.paraphrase_ratio,
        "drop_ratio": cfg.drop_ratio,
        "hallucination_ratio": cfg.hallucination_ratio,
        "shuffle_notes": cfg.shuffle_notes,
    }


def _chunk(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


__all__ = ["NotesGenerator", "NotesGenerationConfig", "PlanDocument", "NotesArtifact"]
