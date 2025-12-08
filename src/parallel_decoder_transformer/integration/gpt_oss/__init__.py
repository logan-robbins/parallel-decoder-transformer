"""GPT-OSS integration primitives for the Parallel Decoder Transformer package."""

from .trunk_adapter import GptOssTrunkAdapter, TrunkAdapterConfig
from .embedder import GptOssEmbedder, GptOssEmbedderConfig

__all__ = [
    "GptOssTrunkAdapter",
    "TrunkAdapterConfig",
    "GptOssEmbedder",
    "GptOssEmbedderConfig",
]
