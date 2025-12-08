"""Inference-time components for the GPT-OSS backed Parallel Decoder Transformer stack."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Iterable, Tuple

__all__ = [
    "AdvanceOutcome",
    "CadencePolicyConfig",
    "CounterfactualConfig",
    "FlickerMonitorConfig",
    "AgreementGate",
    "AgreementResult",
    "DecodeConfig",
    "GateAnnealingConfig",
    "DynamicNotesBus",
    "DynamicNotesBusConfig",
    "InferenceConfig",
    "LipschitzMonitorConfig",
    "KVCheckpoint",
    "NotesWindow",
    "NotesWindowBuilder",
    "TopologyMask",
    "PastKeyValues",
    "ScheduleTick",
    "StreamState",
    "Snapshot",
    "StepOutcome",
    "TriangularScheduler",
    "build_inference_config",
    "MultiStreamOrchestrator",
    "SharedNotesCrossAttention",
    "SharedNotesCrossAttentionConfig",
    "SafeguardConfig",
]

_EXPORTS: Dict[str, Tuple[str, ...]] = {
    "config": (
        "CadencePolicyConfig",
        "CounterfactualConfig",
        "DecodeConfig",
        "GateAnnealingConfig",
        "FlickerMonitorConfig",
        "LipschitzMonitorConfig",
        "InferenceConfig",
        "SafeguardConfig",
        "build_inference_config",
    ),
    "dnb_bus": ("DynamicNotesBus", "DynamicNotesBusConfig", "Snapshot"),
    "orchestrator": ("AgreementGate", "AgreementResult", "MultiStreamOrchestrator", "StepOutcome"),
    "snc_cross_attn": ("SharedNotesCrossAttention", "SharedNotesCrossAttentionConfig"),
    "scheduler": ("AdvanceOutcome", "ScheduleTick", "TriangularScheduler"),
    "state": ("KVCheckpoint", "PastKeyValues", "StreamState"),
    "window": ("NotesWindow", "NotesWindowBuilder", "TopologyMask"),
}


def _load_module(name: str) -> ModuleType:
    return importlib.import_module(f"parallel_decoder_transformer.inference.{name}")


def __getattr__(name: str):
    for module_name, symbols in _EXPORTS.items():
        if name in symbols:
            module = _load_module(module_name)
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Iterable[str]:
    return sorted(set(__all__))
