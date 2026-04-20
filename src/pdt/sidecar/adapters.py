"""Per-stream bottleneck adapters for the instrumented Qwen3 decoder layers.

Each stream has its own independent ``down -> act -> up`` bottleneck. The
adapter is called per-sample (one stream per PDT forward) and the caller
applies the outer gate + residual add.

The outer gate lives on the *instrumented decoder layer*, not here, so that
the same symmetry-breaking logic used for SNC also applies to the adapter
contribution. Gate initialization uses the ``adapter_gate_init`` setting in
``InstrumentationConfig`` so pre-sigmoid values are under YAML control.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from pdt.config.schemas import StreamAdapterConfig


__all__ = ["StreamAdapters", "StreamAdapterLayer"]


class _BottleneckBlock(nn.Module):
    """Pure bottleneck delta. No LayerNorm, no residual -- caller adds both."""

    def __init__(self, config: StreamAdapterConfig) -> None:
        super().__init__()
        self.down = nn.Linear(config.hidden_size, config.bottleneck_size)
        self.up = nn.Linear(config.bottleneck_size, config.hidden_size)
        if config.activation == "relu":
            self.activation: nn.Module = nn.ReLU()
        elif config.activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.down(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.up(hidden)
        hidden = self.dropout(hidden)
        return hidden


class StreamAdapters(nn.Module):
    """Per-stream adapter container; routes by stream id."""

    def __init__(self, config: StreamAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict(
            {stream: _BottleneckBlock(config) for stream in config.streams}
        )

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(self.adapters.keys())

    def forward(self, stream: str, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            adapter = self.adapters[stream]
        except KeyError as exc:
            raise ValueError(
                f"Unknown stream adapter requested: {stream!r}. "
                f"Known streams: {self.streams}."
            ) from exc
        return adapter(hidden_states)


class StreamAdapterLayer(nn.Module):
    """Per-layer wrapper used inside an instrumented decoder layer.

    Holds a ``StreamAdapters`` instance. The instrumented layer's forward
    pass supplies the stream id and hidden states; this module returns the
    raw delta so the caller can apply the outer gate.
    """

    def __init__(self, config: StreamAdapterConfig) -> None:
        super().__init__()
        self.adapters = StreamAdapters(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        stream: str,
    ) -> torch.Tensor:
        return self.adapters(stream, hidden_states)
