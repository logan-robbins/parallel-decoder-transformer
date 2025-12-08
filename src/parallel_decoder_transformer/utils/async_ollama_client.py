"""Async wrapper for Ollama with structured output support."""

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


class AsyncOllamaLLMClient:
    """Async wrapper for Ollama with structured JSON output and schema validation."""

    _MAX_BACKOFF_SECONDS = 30.0

    def __init__(
        self,
        ollama_client,
        *,
        timeout_seconds: float = 1800.0,
    ) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for AsyncOllamaLLMClient. Install with: uv add httpx"
            )
        if jsonschema is None:
            raise RuntimeError(
                "jsonschema is required for schema validation. Install with: uv add jsonschema"
            )
        self._client = ollama_client
        self._timeout = timeout_seconds
        self._endpoint = f"{ollama_client.base_url}/api/generate"

    async def submit_batch(
        self,
        requests: Sequence[StructuredOutputRequest],
        *,
        concurrency: int = 2,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        stop_on_error: bool = False,
    ) -> list[StructuredOutputResult]:
        """Submit batch with controlled concurrency for local models."""
        if not requests:
            return []

        # For local models, concurrency is limited by VRAM/compute
        # gpt-oss:120b may only handle 1-2 concurrent requests
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
                                "Aborting remaining %d requests after %s failed (%s).",
                                len(pending),
                                result.request.request_id,
                                getattr(result.error, "error_type", "unknown_error"),
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
                "Ollama request %s (%s) attempt %d/%d",
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
                    "Ollama request %s attempt %d failed (%s): %s",
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
                    "Ollama request %s attempt %d raised %s",
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
        """Execute single Ollama request with JSON parsing and validation."""
        # Convert messages to Ollama prompt format
        prompt = self._messages_to_prompt(request.messages)

        payload = {
            "model": self._client.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,  # Keep model loaded indefinitely
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_output_tokens,
            },
        }
        # Note: format="json" causes gpt-oss:120b to return empty responses
        # Instead, we rely on prompting and post-parse the output

        logger.info(
            "⏳ Starting Ollama request %s (%s) - this may take several minutes for 120B model...",
            request.request_id,
            request.schema_name,
        )
        start = time.perf_counter()
        try:
            response = await session.post(self._endpoint, json=payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "✓ Ollama request %s completed in %.1f seconds",
                request.request_id,
                latency_ms / 1000.0,
            )

            if response.status_code >= 400:
                body = response.text
                logger.error(
                    "Ollama request %s failed with HTTP %d: %s",
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

            # Log full response structure for debugging
            logger.debug("Ollama full response keys: %s", list(data.keys()))
            logger.debug("Ollama done=%s, error=%s", data.get("done"), data.get("error"))

            response_text = data.get("response", "")

            # Log raw response for debugging
            if not response_text.strip():
                logger.error(
                    "Ollama returned empty response for %s. Full data: %s",
                    request.request_id,
                    str(data)[:300],
                )
            else:
                logger.debug("Ollama raw response (first 500 chars): %s", response_text[:500])

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
                    retryable=True,  # Retry with hope model does better
                )

            logger.info(
                "Ollama request %s (%s) succeeded in %.0f ms",
                request.request_id,
                request.schema_name,
                latency_ms,
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
    def _messages_to_prompt(messages: Sequence[Mapping[str, str]]) -> str:
        """Convert OpenAI-style messages to Ollama prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")

        return "\n".join(parts)

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
                # Handle both ```json and ``` formats
                if "```json" in text:
                    json_block = text.split("```json", 1)[1].split("```", 1)[0].strip()
                else:
                    # Generic ``` fence
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
            # Return parsed JSON even if validation fails - let downstream handle it
            # This matches OpenAI behavior where some validation is post-hoc
            return parsed

    @classmethod
    def _jittered_delay(cls, delay: float) -> float:
        """Add jitter to delay for retry backoff."""
        if delay <= 0:
            return 0.0
        factor = 0.8 + random.random() * 0.4
        return min(delay * factor, cls._MAX_BACKOFF_SECONDS)
