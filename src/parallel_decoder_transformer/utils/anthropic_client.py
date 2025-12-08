from __future__ import annotations

import os
from typing import Sequence

try:  # pragma: no cover - optional dependency guard
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]

from ..datasets.config import GenerationConfig
from .env import load_repo_dotenv


class AnthropicClient:
    """Anthropic API client for dataset generation."""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-opus-20240229") -> None:
        load_repo_dotenv()
        if Anthropic is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "anthropic package is required for AnthropicClient. Install anthropic>=0.32."
            )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or "
                "provide api_key in config."
            )
        self.model = model
        self._client = Anthropic(api_key=self.api_key)

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[str]:
        """Generate completions using Anthropic API."""
        outputs = []

        for prompt in prompts:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=[s for s in config.stop_sequences],
                messages=[
                    {
                        "stream": "user",
                        "content": prompt,
                    }
                ],
            )
            output = message.content[0].text
            outputs.append(output)

        return outputs
