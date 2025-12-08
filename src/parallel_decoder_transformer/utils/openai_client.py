from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

from openai import OpenAI

from .env import load_repo_dotenv

if TYPE_CHECKING:
    from ..datasets.config import GenerationConfig


class OpenAIClient:
    """OpenAI API client for dataset generation."""

    def __init__(
        self,
        api_key: str | None = None,
        org_id: str | None = None,
        model: str = "gpt-5.1",
        service_tier: str | None = None,
        client_timeout: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        load_repo_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # For local endpoints (vLLM, ollama), API key may not be required
        if not self.api_key:
            api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            if "localhost" in api_base or "127.0.0.1" in api_base:
                self.api_key = "EMPTY"  # Local endpoint placeholder
            else:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable or "
                    "provide api_key in config."
                )
        self.model = model
        self.org_id = org_id or os.getenv("OPENAI_ORG_ID")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.service_tier = service_tier
        self.client_timeout = client_timeout
        self.reasoning_effort = reasoning_effort
        client_kwargs = {
            "api_key": self.api_key,
            "organization": self.org_id,
            "base_url": self.api_base,
        }
        if self.client_timeout is not None:
            client_kwargs["timeout"] = float(self.client_timeout)
        self._client = OpenAI(**client_kwargs)

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[str]:
        """Generate completions using OpenAI API."""
        outputs = []

        for prompt in prompts:
            messages = [{"stream": "user", "content": prompt}]

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
                stop=list(config.stop_sequences) if config.stop_sequences else None,
            )
            output = response.choices[0].message.content or ""
            outputs.append(output)

        return outputs

    @property
    def sdk_client(self) -> OpenAI:
        """Return the underlying OpenAI SDK client."""

        return self._client
