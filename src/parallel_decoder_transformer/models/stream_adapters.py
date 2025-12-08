"""Stream adapters applied to the upper transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


@dataclass(slots=True)
class StreamAdapterConfig:
    hidden_size: int
    bottleneck_size: int = 1024
    streams: tuple[str, ...] = (
        "stream_0",
        "stream_1",
        "stream_2",
    )
    activation: str = "gelu"
    dropout: float = 0.0
    spectral_norm: bool = False
    spectral_norm_n_power_iterations: int = 1
    spectral_norm_eps: float = 1e-12


class _AdapterBlock(nn.Module):
    def __init__(self, config: StreamAdapterConfig) -> None:
        super().__init__()
        self.down = nn.Linear(config.hidden_size, config.bottleneck_size)
        self.up = nn.Linear(config.bottleneck_size, config.hidden_size)
        if config.spectral_norm:
            self.down = spectral_norm(
                self.down,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
            self.up = spectral_norm(
                self.up,
                n_power_iterations=config.spectral_norm_n_power_iterations,
                eps=config.spectral_norm_eps,
            )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if config.activation == "relu":
            act: nn.Module = nn.ReLU()
        elif config.activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.GELU()
        self.activation = act
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = hidden_states
        hidden = self.down(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.up(hidden)
        hidden = self.dropout(hidden)
        hidden = residual + hidden
        return self.layer_norm(hidden)


class StreamAdapters(nn.Module):
    """Container mapping stream identifiers to independent adapter blocks."""

    def __init__(self, config: StreamAdapterConfig) -> None:
        super().__init__()
        self.config = config
        modules = {
            stream: _AdapterBlock(
                config=config,
            )
            for stream in config.streams
        }
        self.adapters = nn.ModuleDict(modules)

    def forward(self, stream: torch.Tensor | str, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if isinstance(stream, torch.Tensor):
            streams = stream.tolist()
            outputs = []
            for index, stream_id in enumerate(streams):
                stream_name = self._stream_name(stream_id)
                adapter = self.adapters[stream_name]
                outputs.append(adapter(hidden_states[index : index + 1]))
            return torch.cat(outputs, dim=0)
        try:
            adapter = self.adapters[stream]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown stream adapter requested: {stream!r}") from exc
        return adapter(hidden_states)

    @property
    def available_streams(self) -> tuple[str, ...]:
        return tuple(self.adapters.keys())

    def state_dict_shallow(self) -> Dict[str, torch.Tensor]:
        """Collect a flattened state dict for PEFT snapshots."""

        payload: Dict[str, torch.Tensor] = {}
        for name, module in self.adapters.items():
            for param_name, tensor in module.state_dict().items():
                key = f"{name}.{param_name}"
                payload[key] = tensor.detach().cpu()
        return payload

    def _stream_name(self, index: int) -> str:
        try:
            return self.config.streams[index]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Stream index {index} is outside configured streams") from exc


__all__ = ["StreamAdapterConfig", "StreamAdapters"]
