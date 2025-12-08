"""Async wrapper for vLLM OpenAI-compatible API with structured output support."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Mapping, Sequence

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore[assignment]

from parallel_decoder_transformer.datasets.async_llm import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    StructuredOutputResult,
    StructuredRequestError,
)

logger = logging.getLogger(__name__)


class AsyncVLLMLLMClient:
    """Async wrapper for vLLM OpenAI-compatible API with structured JSON output."""

    _MAX_BACKOFF_SECONDS = 30.0

    def __init__(
        self,
        llm_client,
        *,
        timeout_seconds: float = 1800.0,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required. Install with: uv add httpx")
        if jsonschema is None:
            raise RuntimeError("jsonschema is required. Install with: uv add jsonschema")

        self._client = llm_client
        self._timeout = timeout_seconds
        # vLLM uses OpenAI-compatible /v1/chat/completions endpoint
        api_base = getattr(llm_client, "api_base", "http://localhost:8000")
        self._endpoint = f"{api_base.rstrip('/')}/v1/chat/completions"
        self._api_key = getattr(llm_client, "api_key", "EMPTY")

    async def submit_batch(
        self,
        requests: Sequence[StructuredOutputRequest],
        *,
        concurrency: int = 8,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        stop_on_error: bool = False,
    ) -> list[StructuredOutputResult]:
        """Submit batch with controlled concurrency."""
        if not requests:
            return []

        # vLLM with tensor parallelism can handle more concurrency than Ollama
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(timeout=self._timeout) as session:
            tasks = [
                asyncio.create_task(
                    self._run_with_capture(session, sem, request, max_retries, retry_backoff)
                )
                for request in requests
            ]
            pending: set[asyncio.Task[StructuredOutputResult]] = set(tasks)
            results: list[StructuredOutputResult] = []

            async def _cancel_pending() -> None:
                if not pending:
                    return
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending.clear()

            try:
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task.cancelled():
                            continue
                        result = await task
                        results.append(result)
                        if stop_on_error and result.error is not None:
                            logger.warning(
                                "Aborting remaining %d requests after %s failed",
                                len(pending),
                                result.request.request_id,
                            )
                            await _cancel_pending()
                            break
                return results
            finally:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _run_with_capture(
        self,
        session: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        request: StructuredOutputRequest,
        max_retries: int,
        retry_backoff: float,
    ) -> StructuredOutputResult:
        """Execute single request with retries and capture errors."""
        try:
            response = await self._execute_with_retries(
                session, sem, request, max_retries, retry_backoff
            )
            return StructuredOutputResult(request=request, response=response)
        except StructuredRequestError as exc:
            return StructuredOutputResult(request=request, error=exc)

    async def _execute_with_retries(
        self,
        session: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        request: StructuredOutputRequest,
        max_retries: int,
        retry_backoff: float,
    ) -> StructuredOutputResponse:
        """Execute with exponential backoff retry logic."""
        attempt = 0
        delay = 1.0

        while attempt <= max_retries:
            logger.debug(
                "vLLM request %s (%s) attempt %d/%d",
                request.request_id,
                request.schema_name,
                attempt + 1,
                max_retries + 1,
            )
            try:
                async with sem:
                    return await self._single_request(session, request)
            except StructuredRequestError as exc:
                exc.attempts = attempt
                logger.warning(
                    "vLLM request %s attempt %d failed (%s): %s",
                    request.request_id,
                    attempt + 1,
                    exc.error_type,
                    exc,
                )
                if not exc.retryable or attempt >= max_retries:
                    raise
                await asyncio.sleep(self._jittered_delay(delay))
                delay = min(delay * retry_backoff, self._MAX_BACKOFF_SECONDS)
                attempt += 1
            except Exception as exc:
                wrapped = StructuredRequestError(
                    f"{exc.__class__.__name__}: {exc}",
                    error_type=exc.__class__.__name__,
                    retryable=True,
                )
                wrapped.attempts = attempt
                logger.exception(
                    "vLLM request %s attempt %d raised %s",
                    request.request_id,
                    attempt + 1,
                    exc.__class__.__name__,
                )
                if attempt >= max_retries:
                    wrapped.retryable = False
                    raise wrapped from exc
                await asyncio.sleep(self._jittered_delay(delay))
                delay = min(delay * retry_backoff, self._MAX_BACKOFF_SECONDS)
                attempt += 1

        raise StructuredRequestError(
            f"Request {request.request_id} failed after {max_retries} retries",
            error_type="max_retries",
        )

    async def _single_request(
        self,
        session: httpx.AsyncClient,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        """Execute single vLLM request with JSON parsing and validation."""
        # Convert to OpenAI chat format
        messages = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in request.messages
        ]

        # Add JSON instruction to system message
        json_instruction = "\n\nYou must return ONLY valid JSON matching the required schema. Do not include any other text, explanations, or markdown formatting."
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += json_instruction
        else:
            messages.insert(0, {"role": "system", "content": json_instruction.strip()})

        payload = {
            "model": self._client.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens,
        }

        # Use vLLM's guided_json for structured output
        # This forces the model to generate valid JSON matching the schema
        if request.schema:
            payload["extra_body"] = {"guided_json": request.schema}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "⏳ Starting vLLM request %s (%s)...",
            request.request_id,
            request.schema_name,
        )
        start = time.perf_counter()
        try:
            response = await session.post(self._endpoint, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "✓ vLLM request %s completed in %.1f seconds",
                request.request_id,
                latency_ms / 1000.0,
            )

            if response.status_code >= 400:
                body = response.text
                logger.error(
                    "vLLM request %s failed with HTTP %d: %s",
                    request.request_id,
                    response.status_code,
                    body[:200],
                )
                raise StructuredRequestError(
                    f"HTTP {response.status_code} for {request.request_id}: {body[:200]}",
                    status_code=response.status_code,
                    error_type=f"http_{response.status_code}",
                    retryable=response.status_code >= 500,
                )

            data = response.json()

            # Extract response from OpenAI format
            choices = data.get("choices", [])
            if not choices:
                raise StructuredRequestError(
                    f"No choices in response for {request.request_id}",
                    error_type="no_choices",
                    retryable=True,
                )

            message = choices[0].get("message", {})

            # gpt-oss models may put output in either 'content' or 'reasoning' field
            # Check both fields (content first, then reasoning as fallback)
            response_text = message.get("content") or ""
            if not response_text.strip():
                # Fallback to reasoning field for gpt-oss models
                response_text = message.get("reasoning") or message.get("reasoning_content") or ""
                if response_text.strip():
                    logger.info(
                        "Using reasoning field for %s (content was empty)", request.request_id
                    )

            if not response_text or not response_text.strip():
                logger.error(
                    "vLLM returned empty response for %s (checked content, reasoning, and reasoning_content)",
                    request.request_id,
                )

            # Parse and validate JSON
            parsed = self._parse_and_validate_json(response_text, request.schema)

            if parsed is None:
                excerpt = response_text.strip().replace("\n", " ")[:500]
                logger.error(
                    "Failed to parse JSON for %s. Raw response: %s",
                    request.request_id,
                    excerpt if excerpt else "(empty)",
                )
                raise StructuredRequestError(
                    f"Failed to parse valid JSON for {request.request_id}: {excerpt if excerpt else '(empty response)'}",
                    error_type="invalid_json",
                    retryable=True,
                )

            return StructuredOutputResponse(
                request=request,
                raw_response=data,
                output_text=response_text,
                parsed_json=parsed,
                latency_ms=latency_ms,
            )

        except httpx.HTTPError as exc:
            raise StructuredRequestError(
                f"HTTP error for {request.request_id}: {exc}",
                error_type="http_error",
                retryable=True,
            ) from exc

    @staticmethod
    def _parse_and_validate_json(
        text: str,
        schema: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Parse JSON and validate against schema."""
        text = text.strip()
        if not text:
            return None

        parsed = None

        # Try 1: Parse as-is
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try 2: Extract from markdown code fence
        if parsed is None and "```" in text:
            try:
                if "```json" in text:
                    json_block = text.split("```json", 1)[1].split("```", 1)[0].strip()
                else:
                    json_block = text.split("```", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(json_block)
            except (IndexError, json.JSONDecodeError, ValueError):
                pass

        # Try 3: Extract first JSON object
        if parsed is None and "{" in text and "}" in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                parsed = json.loads(text[start:end])
            except (ValueError, json.JSONDecodeError):
                pass

        if parsed is None:
            return None

        if not isinstance(parsed, dict):
            return None

        # Validate against schema
        try:
            jsonschema.validate(instance=parsed, schema=schema)
            return parsed
        except jsonschema.ValidationError as e:
            logger.warning("Schema validation failed: %s", e.message)
            # Return parsed JSON even if validation fails
            return parsed

    @classmethod
    def _jittered_delay(cls, delay: float) -> float:
        """Add jitter to delay for retry backoff."""
        if delay <= 0:
            return 0.0
        factor = 0.8 + random.random() * 0.4
        return min(delay * factor, cls._MAX_BACKOFF_SECONDS)
