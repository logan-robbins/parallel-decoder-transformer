"""Canonical PDT baselines used to falsify causal coordination claims."""

from pdt.baselines.self_only import (
    ParameterMatchedSelfOnlyAttention,
    SelfOnlyMemory,
    build_self_only_memory,
)

__all__ = [
    "ParameterMatchedSelfOnlyAttention",
    "SelfOnlyMemory",
    "build_self_only_memory",
]
