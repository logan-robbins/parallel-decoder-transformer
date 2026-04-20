"""Trainable sidecar tree (all of \u03c6): SNC, per-stream adapters, heads,
plan embedding, plan-notes projection."""

from pdt.sidecar.snc import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig
from pdt.sidecar.adapters import StreamAdapterConfig, StreamAdapterLayer, StreamAdapters
from pdt.sidecar.plan_embedding import PlanEmbedding

__all__ = [
    "PlanEmbedding",
    "SharedNotesCrossAttention",
    "SharedNotesCrossAttentionConfig",
    "StreamAdapterConfig",
    "StreamAdapterLayer",
    "StreamAdapters",
]
