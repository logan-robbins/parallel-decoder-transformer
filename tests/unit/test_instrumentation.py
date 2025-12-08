"""Unit tests covering instrumentation fail-fast behaviour."""

from __future__ import annotations

import pytest
import torch.nn as nn

from parallel_decoder_transformer.integration.gpt_oss.trunk_adapter import TrunkAdapterConfig
from parallel_decoder_transformer.integration.instrumentation import (
    InstrumentedTrunkAdapter,
    InstrumentedTrunkAdapterConfig,
    InstrumentationSpec,
)


class _DummyModel(nn.Module):
    """Minimal torch module without transformer layers."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


def test_instrumented_trunk_adapter_requires_transformer_layers() -> None:
    """Ensure mid-stack instrumentation raises when no layers can be wrapped."""

    config = InstrumentedTrunkAdapterConfig(
        trunk=TrunkAdapterConfig(base_model="dummy"),
        instrumentation=InstrumentationSpec(enabled=True, top_k_layers=1),
    )
    dummy_trunk = _DummyModel()
    with pytest.raises(RuntimeError, match="InstrumentedTrunkAdapter"):
        InstrumentedTrunkAdapter(config, model=dummy_trunk)
