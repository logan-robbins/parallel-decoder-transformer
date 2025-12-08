from __future__ import annotations

from dataclasses import asdict

from ..datasets.config import LLMConfig
from .anthropic_client import AnthropicClient
from .local_llm_client import LocalLLMClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from .env import load_repo_dotenv


def create_llm_client(config: LLMConfig):
    """Factory resolving the appropriate LLM client based on config."""

    load_repo_dotenv()
    backend = config.backend.lower()
    if backend == "openai":
        return OpenAIClient(**asdict(config.openai))
    if backend == "anthropic":
        return AnthropicClient(**asdict(config.anthropic))
    if backend == "local":
        if config.local is None:
            raise ValueError("Local backend selected, but no local config provided.")
        return LocalLLMClient(config.local)
    if backend == "ollama":
        return OllamaClient(**asdict(config.ollama))
    if backend == "vllm":
        # vLLM uses OpenAI-compatible API, so reuse OpenAIClient
        return OpenAIClient(**asdict(config.openai))
    raise ValueError(
        f"Unsupported LLM backend: {config.backend}. Supported: openai, anthropic, local, ollama, vllm"
    )
