"""Trainable PDT sidecar modules."""

from pdt.sidecar.product_vq import ProductVQOutput, ProductVectorQuantizer
from pdt.sidecar.snc import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig


__all__ = [
    "ProductVQOutput",
    "ProductVectorQuantizer",
    "SharedNotesCrossAttention",
    "SharedNotesCrossAttentionConfig",
]
