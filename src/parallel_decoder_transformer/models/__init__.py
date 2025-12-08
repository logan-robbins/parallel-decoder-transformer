"""Model primitives for the GPT-OSS backed Parallel Decoder Transformer."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Iterable, Tuple

__all__ = [
    "ParallelDecoderTransformer",
    "ParallelDecoderModelConfig",
    "StreamAdapterConfig",
    "StreamAdapters",
    "PlannerHead",
    "PlannerHeadConfig",
    "NotesHead",
    "NotesHeadConfig",
    "SpeculationHead",
    "SpeculationHeadConfig",
    "AgreementHead",
    "AgreementHeadConfig",
    "SNCBackend",
    "PostTrunkSNC",
    "MidStackSNC",
]

_EXPORTS: Dict[str, Tuple[str, ...]] = {
    "parallel_decoder_transformer": ("ParallelDecoderTransformer", "ParallelDecoderModelConfig"),
    "stream_adapters": ("StreamAdapterConfig", "StreamAdapters"),
    "heads": (
        "PlannerHead",
        "PlannerHeadConfig",
        "NotesHead",
        "NotesHeadConfig",
        "SpeculationHead",
        "SpeculationHeadConfig",
        "AgreementHead",
        "AgreementHeadConfig",
    ),
    "snc_backend": ("SNCBackend", "PostTrunkSNC", "MidStackSNC"),
}


def _load_module(name: str) -> ModuleType:
    return importlib.import_module(f"parallel_decoder_transformer.models.{name}")


def __getattr__(name: str):
    for module_name, symbols in _EXPORTS.items():
        if name in symbols:
            module = _load_module(module_name)
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Iterable[str]:
    return sorted(set(__all__))
