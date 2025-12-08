"""Ollama API client for local model inference."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Sequence

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..datasets.config import GenerationConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API client compatible with the OpenAI client interface."""

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        base_url: str = "http://localhost:11434",
        client_timeout: float = 1800.0,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required for OllamaClient. Install with: uv add httpx")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client_timeout = client_timeout
        self._client = httpx.Client(timeout=client_timeout)
        logger.info("Initialized Ollama client for model %s at %s", model, base_url)

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[str]:
        """Generate completions using Ollama API."""
        outputs = []
        for prompt in prompts:
            response_text = self._generate_single(
                prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
            )
            outputs.append(response_text)
        return outputs

    def _generate_single(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        format_json: bool = False,
    ) -> str:
        """Generate a single response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,  # Keep model loaded indefinitely
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        # Note: format="json" causes gpt-oss:120b to return empty responses
        # Commented out for now - rely on prompting instead
        # if format_json:
        #     payload["format"] = "json"

        start = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = response.json()
            response_text = data.get("response", "")
            logger.debug(
                "Ollama generation completed in %.0f ms (%d tokens)",
                latency_ms,
                data.get("eval_count", 0),
            )
            return response_text
        except httpx.HTTPError as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
