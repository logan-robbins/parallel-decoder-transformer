"""Trainable PDT sidecar modules."""

from pdt.config.schemas import PlanAdapterConfig
from pdt.sidecar.adapters import PlanConditionedAdapter
from pdt.sidecar.product_vq import ProductVQOutput, ProductVectorQuantizer
from pdt.sidecar.snc import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig


__all__ = [
    "PlanConditionedAdapter",
    "ProductVQOutput",
    "ProductVectorQuantizer",
    "SharedNotesCrossAttention",
    "SharedNotesCrossAttentionConfig",
    "PlanAdapterConfig",
]
