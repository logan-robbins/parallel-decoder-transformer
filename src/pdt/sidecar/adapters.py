"""Per-stream bottleneck adapters for the instrumented Qwen3 decoder layers.

Each stream has its own independent ``down -> act -> up`` bottleneck. A
packed PDT forward supplies one stream ID per batch row. Rows are grouped by
adapter, transformed, then restored to their original frontier order.

The outer gate lives on the *instrumented decoder layer*, not here, so that
the same symmetry-breaking logic used for SNC also applies to the adapter
contribution. Gate initialization uses the ``adapter_gate_init`` setting in
``InstrumentationConfig`` so pre-sigmoid values are under YAML control.
"""

from __future__ import annotations

import torch
from torch import nn
from collections.abc import Sequence

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
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden = self.down(hidden_states.to(dtype=self.down.weight.dtype))
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.up(hidden)
        hidden = self.dropout(hidden)
        return hidden.to(dtype=input_dtype)


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
                f"Unknown stream adapter requested: {stream!r}. Known streams: {self.streams}."
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
        stream_ids: Sequence[str],
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError("Stream adapters require hidden_states with shape (B, T, H).")
        normalized = tuple(stream.lower() for stream in stream_ids)
        if len(normalized) != hidden_states.size(0):
            raise ValueError(
                "stream_ids must address every hidden-state batch row; "
                f"got {len(normalized)} IDs for batch {hidden_states.size(0)}."
            )

        grouped: dict[str, list[int]] = {}
        for row, stream in enumerate(normalized):
            if stream not in self.adapters.streams:
                raise ValueError(
                    f"Unknown stream adapter requested: {stream!r}. "
                    f"Known streams: {self.adapters.streams}."
                )
            grouped.setdefault(stream, []).append(row)

        output_rows: list[torch.Tensor | None] = [None] * hidden_states.size(0)
        for stream, row_indices in grouped.items():
            index = torch.tensor(row_indices, dtype=torch.long, device=hidden_states.device)
            transformed = self.adapters(stream, hidden_states.index_select(0, index))
            for grouped_row, original_row in enumerate(row_indices):
                output_rows[original_row] = transformed[grouped_row : grouped_row + 1]
        if any(row is None for row in output_rows):
            raise RuntimeError("Packed stream-adapter routing left an unassigned batch row.")
        return torch.cat([row for row in output_rows if row is not None], dim=0)
