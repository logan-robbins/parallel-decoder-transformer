"""Trainable sidecar tree (all of phi): SNC, per-stream adapters, heads,
and planner-seeded note projection."""

from pdt.sidecar.snc import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig
from pdt.sidecar.adapters import StreamAdapterConfig, StreamAdapterLayer, StreamAdapters

__all__ = [
    "SharedNotesCrossAttention",
    "SharedNotesCrossAttentionConfig",
    "StreamAdapterConfig",
    "StreamAdapterLayer",
    "StreamAdapters",
]
