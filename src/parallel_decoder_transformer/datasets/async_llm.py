"""Helpers for batching OpenAI Responses API calls with structured outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency guard
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

from parallel_decoder_transformer.utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StructuredOutputRequest:
    """Single structured generation task."""

    request_id: str
    messages: Sequence[Mapping[str, str]]
    schema_name: str
    schema: Mapping[str, Any]
    max_output_tokens: int
    temperature: float = 0.2
    top_p: float = 0.95
    stop_sequences: Sequence[str] = field(default_factory=tuple)
    seed: int | None = None
    reasoning_effort: str | None = None  # "low", "medium", "high" for reasoning models
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredOutputResponse:
    """Parsed result emitted by the async client."""

    request: StructuredOutputRequest
    raw_response: Mapping[str, Any]
    output_text: str
    parsed_json: Mapping[str, Any] | None
    latency_ms: float


@dataclass(slots=True)
class StructuredOutputResult:
    """Wrapper capturing either a successful response or a structured error."""

    request: StructuredOutputRequest
    response: StructuredOutputResponse | None = None
    error: "StructuredRequestError | None" = None


class StructuredRequestError(RuntimeError):
    """Raised when a structured request exhausts retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str = "structured_request_error",
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts: int = 0


class AsyncStructuredLLMClient:
    """Thin async wrapper around the OpenAI Responses API."""

    _RETRYABLE_STATUS = {408, 429}
    _MAX_BACKOFF_SECONDS = 30.0

    def __init__(
        self,
        llm_client: OpenAIClient,
        *,
        timeout_seconds: float = 600.0,
        api_base: str | None = None,
    ) -> None:
        if aiohttp is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "aiohttp is required for AsyncStructuredLLMClient. Install aiohttp>=3.13."
            )
        self._client = llm_client
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        base = api_base or getattr(llm_client, "api_base", "https://api.openai.com/v1")
        self._endpoint = f"{base.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {llm_client.api_key}",
            "Content-Type": "application/json",
        }
        if getattr(llm_client, "org_id", None):
            headers["OpenAI-Organization"] = llm_client.org_id  # type: ignore[assignment]
        self._headers = headers

    async def submit_batch(
        self,
        requests: Sequence[StructuredOutputRequest],
        *,
        concurrency: int = 10,
        max_retries: int = 5,
        retry_backoff: float = 1.5,
        stop_on_error: bool = False,
    ) -> list[StructuredOutputResult]:
        if not requests:
            return []
        sem = asyncio.Semaphore(concurrency)
        # Limit connections to prevent overwhelming the API
        connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency)
        async with aiohttp.ClientSession(timeout=self._timeout, connector=connector) as session:
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
                                "Aborting remaining %d structured requests after %s failed (%s).",
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
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        request: StructuredOutputRequest,
        max_retries: int,
        retry_backoff: float,
    ) -> StructuredOutputResult:
        try:
            response = await self._execute_with_retries(
                session, sem, request, max_retries, retry_backoff
            )
            return StructuredOutputResult(request=request, response=response)
        except StructuredRequestError as exc:
            return StructuredOutputResult(request=request, error=exc)

    async def _execute_with_retries(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        request: StructuredOutputRequest,
        max_retries: int,
        retry_backoff: float,
    ) -> StructuredOutputResponse:
        attempt = 0
        delay = 1.0
        while attempt <= max_retries:
            logger.debug(
                "Structured request %s (%s) attempt %d/%d",
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
                    "Structured request %s attempt %d failed (%s): %s",
                    request.request_id,
                    attempt + 1,
                    exc.error_type,
                    exc,
                )
                if not exc.retryable or attempt >= max_retries:
                    raise
                sleep_for = exc.retry_after if exc.retry_after is not None else delay
                await asyncio.sleep(self._jittered_delay(sleep_for))
                delay = min(delay * retry_backoff, self._MAX_BACKOFF_SECONDS)
                attempt += 1
            except Exception as exc:  # pragma: no cover - network errors hard to simulate
                wrapped = StructuredRequestError(
                    f"{exc.__class__.__name__}: {exc}",
                    error_type=exc.__class__.__name__,
                    retryable=True,
                )
                wrapped.attempts = attempt
                logger.exception(
                    "Structured request %s attempt %d raised %s",
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
        session: aiohttp.ClientSession,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        payload = _payload_from_request(self._client, request)
        start = time.perf_counter()
        async with session.post(self._endpoint, headers=self._headers, json=payload) as resp:
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = resp.status
            if status >= 400:
                body = await resp.text()
                retryable = status in self._RETRYABLE_STATUS or status >= 500
                retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                logger.error(
                    "Structured request %s failed with HTTP %d: %s",
                    request.request_id,
                    status,
                    body[:200],
                )
                raise StructuredRequestError(
                    f"Status {status} for {request.request_id}: {body[:200]}",
                    status_code=status,
                    error_type=f"http_{status}",
                    retryable=retryable,
                    retry_after=retry_after,
                )
            data = await resp.json()
        # Check for incomplete response (reasoning models can exhaust token budget
        # before completing output - OpenAI returns status: "incomplete")
        response_status = data.get("status")
        if response_status == "incomplete":
            incomplete_details = data.get("incomplete_details", {})
            reason = incomplete_details.get("reason", "unknown")
            logger.error(
                "Structured request %s returned incomplete response (reason: %s). "
                "Consider increasing max_output_tokens for reasoning models.",
                request.request_id,
                reason,
            )
            raise StructuredRequestError(
                f"Incomplete response for {request.request_id}: {reason}. "
                f"Reasoning models require larger token budgets (recommended: 50000+).",
                error_type="incomplete_response",
                retryable=False,
            )
        logger.info(
            "Structured request %s (%s) succeeded in %.0f ms (status %d)",
            request.request_id,
            request.schema_name,
            latency_ms,
            status,
        )
        output_text, parsed = self._extract_output_parts(data)
        if parsed is None:
            stripped = output_text.strip()
            if stripped.startswith(("{", "[")):
                parsed = self._maybe_parse_json(output_text)
        if parsed is None:
            excerpt = output_text.strip().replace("\n", " ")[:200]
            raise StructuredRequestError(
                f"Structured payload missing for {request.request_id}: {excerpt}",
                error_type="missing_structured_payload",
                retryable=False,
            )
        return StructuredOutputResponse(
            request=request,
            raw_response=data,
            output_text=output_text,
            parsed_json=parsed,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_output_parts(data: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
        """Return concatenated text plus the first structured payload (if any)."""

        text_chunks: list[str] = []
        parsed_candidates: list[Any] = []

        def consider_parsed(value: Any) -> None:
            if value is None:
                return
            parsed_candidates.append(value)

        def collect_text(value: Any) -> None:
            if isinstance(value, str):
                text_chunks.append(value)
            elif isinstance(value, list):
                text_chunks.extend(str(item) for item in value if item is not None)

        def collect_content(content: Any) -> None:
            if not isinstance(content, list):
                return
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                collect_text(block.get("text"))
                consider_parsed(block.get("parsed"))
                schema_payload = block.get("json_schema") or block.get("schema")
                if isinstance(schema_payload, Mapping):
                    consider_parsed(schema_payload.get("parsed"))

        def collect_outputs(outputs: Any) -> None:
            if not isinstance(outputs, Iterable):
                return
            for item in outputs:
                if not isinstance(item, Mapping):
                    continue
                collect_text(item.get("text"))
                consider_parsed(item.get("parsed"))
                collect_content(item.get("content"))

        # Legacy convenience fields
        collect_text(data.get("output_text"))
        # Some SDKs expose the structured payload via `output_parsed`; when this is
        # a mapping we should treat it as an already-parsed JSON candidate rather
        # than concatenating it into the text buffer.
        output_parsed = data.get("output_parsed")
        if isinstance(output_parsed, Mapping):
            consider_parsed(output_parsed)
        else:
            collect_text(output_parsed)

        collect_outputs(data.get("output"))
        collect_outputs(data.get("outputs"))

        response_payload = data.get("response")
        if isinstance(response_payload, Mapping):
            collect_text(response_payload.get("output_text"))
            collect_outputs(response_payload.get("output"))
            collect_outputs(response_payload.get("outputs"))

        structured_payload: Mapping[str, Any] | None = None
        for candidate in parsed_candidates:
            if isinstance(candidate, Mapping):
                structured_payload = candidate
                break

        collected_text = "".join(text_chunks)
        return collected_text, structured_payload

    @staticmethod
    def _maybe_parse_json(text: str) -> Mapping[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured output as JSON (truncated): %s", text[:200])
            return None
        if isinstance(parsed, Mapping):
            return parsed
        return None

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            seconds = float(value)
        except ValueError:
            return None
        return max(0.0, seconds)

    @classmethod
    def _jittered_delay(cls, delay: float) -> float:
        if delay <= 0:
            return 0.0
        factor = 0.8 + random.random() * 0.4
        return min(delay * factor, cls._MAX_BACKOFF_SECONDS)


__all__ = [
    "AsyncStructuredLLMClient",
    "SequentialStructuredLLMClient",
    "StructuredOutputRequest",
    "StructuredOutputResponse",
    "StructuredOutputResult",
    "StructuredRequestError",
]


class SequentialStructuredLLMClient:
    """Synchronous Responses runner with optional pauses between calls."""

    def __init__(self, llm_client: OpenAIClient, *, pause_seconds: float = 0.0) -> None:
        self._client = llm_client
        self._pause = max(0.0, pause_seconds)
        self._sdk = llm_client.sdk_client

    def set_pause(self, seconds: float) -> None:
        self._pause = max(0.0, seconds)

    def submit(
        self,
        request: StructuredOutputRequest,
    ) -> StructuredOutputResponse:
        payload = _payload_from_request(self._client, request)
        start = time.perf_counter()
        try:
            response = self._sdk.responses.create(**payload)
        except Exception as exc:  # pragma: no cover - network errors hard to simulate
            raise self._wrap_exception(request, exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        data = response.model_dump(mode="json")
        output_text, parsed = AsyncStructuredLLMClient._extract_output_parts(data)
        result = StructuredOutputResponse(
            request=request,
            raw_response=data,
            output_text=output_text,
            parsed_json=parsed,
            latency_ms=latency_ms,
        )
        if self._pause:
            time.sleep(self._pause)
        return result

    def submit_batch(
        self, requests: Sequence[StructuredOutputRequest]
    ) -> list[StructuredOutputResponse]:
        results: list[StructuredOutputResponse] = []
        for request in requests:
            results.append(self.submit(request))
        return results

    def _wrap_exception(
        self,
        request: StructuredOutputRequest,
        exc: Exception,
    ) -> StructuredRequestError:
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        retry_after = None
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = AsyncStructuredLLMClient._parse_retry_after(headers.get("Retry-After"))
        retryable = bool(
            status is not None
            and (status in AsyncStructuredLLMClient._RETRYABLE_STATUS or status >= 500)
        )
        return StructuredRequestError(
            f"{exc.__class__.__name__} for {request.request_id}: {exc}",
            status_code=status,
            error_type=exc.__class__.__name__,
            retryable=retryable,
            retry_after=retry_after,
        )


def _payload_from_request(
    llm_client: OpenAIClient, request: StructuredOutputRequest
) -> dict[str, Any]:
    # Plan generation uses json_schema mode for strict validation.
    # Notes generation typically uses json_object for token savings, but some models
    # (e.g., gpt-4o snapshots) require json_schema.
    model_name = getattr(llm_client, "model", "").lower()
    schema_name = request.schema_name.lower()
    use_strict_schema = "_plan" in schema_name  # e.g., "survey_plan", "qa_plan"
    requires_json_schema = "gpt-4o" in model_name or "gpt-5" in model_name

    if use_strict_schema or requires_json_schema:
        text_format = {
            "type": "json_schema",
            "name": request.schema_name,
            "schema": request.schema,
            "strict": True,
        }
    else:
        text_format = {
            "type": "json_object",
        }

    payload: dict[str, Any] = {
        "model": llm_client.model,
        "input": list(request.messages),
        "max_output_tokens": request.max_output_tokens,
        "text": {"format": text_format},
    }
    service_tier = getattr(llm_client, "service_tier", None)
    if service_tier:
        payload["service_tier"] = service_tier
    # Models using the Responses API with structured outputs (gpt-5-mini, gpt-5.1,
    # o-series) do not accept sampling parameters like `temperature` and `top_p`.
    # Only include these parameters for older models that support them.
    supports_sampling = (
        "mini" not in model_name
        and "gpt-5" not in model_name
        and not model_name.startswith("o1")
        and not model_name.startswith("o3")
    )
    if supports_sampling:
        payload["temperature"] = request.temperature
        payload["top_p"] = request.top_p
    # Reasoning effort for reasoning models (gpt-5.1, o-series)
    # OpenAI Responses API uses nested structure: reasoning.effort
    if request.reasoning_effort:
        payload["reasoning"] = {"effort": request.reasoning_effort}
    if request.stop_sequences:
        payload["stop_sequences"] = list(request.stop_sequences)
    if request.metadata:
        payload["metadata"] = dict(request.metadata)
    return payload
