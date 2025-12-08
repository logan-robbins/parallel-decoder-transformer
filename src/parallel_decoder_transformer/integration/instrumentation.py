"""Utilities for instrumenting GPT-NeoX trunks with mid-stack adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch
from torch import nn

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

from ..models.stream_adapters import StreamAdapterConfig, StreamAdapters
from ..inference.snc_cross_attn import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig
from ..models.heads.speculation import SpeculationHead, SpeculationHeadConfig
from .gpt_oss.trunk_adapter import (
    GptOssTrunkAdapter,
    TrunkAdapterConfig,
    _resolve_transformer_layers,
)

LOGGER = logging.getLogger("parallel decoder transformer.instrumentation")


class NotesProvider(Protocol):
    """Provides Dynamic Notes Bus windows for a stream during trunk execution."""

    def notes_for(self, stream: torch.Tensor | str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (notes, mask) tensors for the active stream."""


@dataclass(slots=True)
class LayerRuntimeContext:
    """Runtime payload shared with instrumented transformer blocks."""

    stream: Optional[torch.Tensor | str] = None
    notes: Optional[torch.Tensor] = None
    notes_mask: Optional[torch.Tensor] = None


class StreamAdapterLayer(nn.Module):
    """Leverage existing StreamAdapters to produce residual deltas."""

    def __init__(self, config: StreamAdapterConfig) -> None:
        super().__init__()
        self.adapters = StreamAdapters(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        stream: Optional[torch.Tensor | str],
    ) -> torch.Tensor:  # type: ignore[override]
        if stream is None:
            return torch.zeros_like(hidden_states)
        adapted = self.adapters(stream, hidden_states)
        return adapted - hidden_states


class SharedNotesResidual(nn.Module):
    """Convert SharedNotesCrossAttention output into a residual delta."""

    def __init__(self, config: SharedNotesCrossAttentionConfig) -> None:
        super().__init__()
        self.cross_attention = SharedNotesCrossAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        notes_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        attended = self.cross_attention(hidden_states, notes, notes_mask=notes_mask)
        return attended - hidden_states


class InstrumentedGPTNeoXLayer(GPTNeoXLayer):
    """GPT-NeoX layer augmented with stream adapters and SNC cross-attention."""

    def __init__(
        self,
        config,
        *,
        stream_adapter: Optional[StreamAdapterLayer],
        snc_residual: Optional[SharedNotesResidual],
        speculation_tap: Optional[SpeculationHead],
        gate_init: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.stream_adapter = stream_adapter
        self.stream_adapter_gate = nn.Parameter(torch.tensor(0.0))
        self.snc_residual = snc_residual
        self.notes_gate = nn.Parameter(torch.tensor(gate_init))
        self._runtime_context: Optional[LayerRuntimeContext] = None
        self._notes_provider: Optional[NotesProvider] = None
        self.speculation_tap = speculation_tap
        self._spec_sink: Optional[Callable[[torch.Tensor], None]] = None

    def set_runtime_context(self, context: Optional[LayerRuntimeContext]) -> None:
        self._runtime_context = context

    def set_notes_provider(self, provider: Optional[NotesProvider]) -> None:
        self._notes_provider = provider

    def set_speculation_sink(self, sink: Optional[Callable[[torch.Tensor], None]]) -> None:
        self._spec_sink = sink

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        context = self._runtime_context
        notes = None if context is None else context.notes
        notes_mask = None if context is None else context.notes_mask
        stream = None if context is None else context.stream
        if notes is None and stream is not None and self._notes_provider is not None:
            notes, notes_mask = self._notes_provider.notes_for(stream)

        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]
        attn_output = self.post_attention_dropout(attn_output)

        if self.snc_residual is not None and notes is not None and notes.size(1) > 0:
            gate = torch.sigmoid(self.notes_gate).to(
                dtype=attn_output.dtype, device=attn_output.device
            )
            delta = self.snc_residual(attn_output, notes, notes_mask=notes_mask)
            attn_output = attn_output + gate * delta

        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            mlp_output = self.post_mlp_dropout(mlp_output)
            if (
                self.speculation_tap is not None
                and stream is not None
                and self._spec_sink is not None
            ):
                speculation = self.speculation_tap(mlp_output)
                self._spec_sink(speculation)
            if self.stream_adapter is not None and stream is not None:
                gate = torch.sigmoid(self.stream_adapter_gate).to(
                    dtype=mlp_output.dtype, device=mlp_output.device
                )
                delta = self.stream_adapter(mlp_output, stream)
                mlp_output = mlp_output + gate * delta
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            attn_output = attn_output + hidden_states
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            mlp_output = self.post_mlp_dropout(mlp_output)
            if (
                self.speculation_tap is not None
                and stream is not None
                and self._spec_sink is not None
            ):
                speculation = self.speculation_tap(mlp_output)
                self._spec_sink(speculation)
            if self.stream_adapter is not None and stream is not None:
                gate = torch.sigmoid(self.stream_adapter_gate).to(
                    dtype=mlp_output.dtype, device=mlp_output.device
                )
                delta = self.stream_adapter(mlp_output, stream)
                mlp_output = mlp_output + gate * delta
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class InstrumentedGptOssDecoderLayer(GptOssDecoderLayer):
    """GptOss decoder layer augmented with stream adapters and SNC cross-attention."""

    def __init__(
        self,
        config,
        layer_idx: int,
        *,
        stream_adapter: Optional[StreamAdapterLayer],
        snc_residual: Optional[SharedNotesResidual],
        speculation_tap: Optional[SpeculationHead],
        gate_init: float = 0.0,
    ) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.stream_adapter = stream_adapter
        self.stream_adapter_gate = nn.Parameter(torch.tensor(0.0))
        self.snc_residual = snc_residual
        self.notes_gate = nn.Parameter(torch.tensor(gate_init))
        self._runtime_context: Optional[LayerRuntimeContext] = None
        self._notes_provider: Optional[NotesProvider] = None
        self.speculation_tap = speculation_tap
        self._spec_sink: Optional[Callable[[torch.Tensor], None]] = None

    def set_runtime_context(self, context: Optional[LayerRuntimeContext]) -> None:
        self._runtime_context = context

    def set_notes_provider(self, provider: Optional[NotesProvider]) -> None:
        self._notes_provider = provider

    def set_speculation_sink(self, sink: Optional[Callable[[torch.Tensor], None]]) -> None:
        self._spec_sink = sink

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        context = self._runtime_context
        notes = None if context is None else context.notes
        notes_mask = None if context is None else context.notes_mask
        stream = None if context is None else context.stream
        if notes is None and stream is not None and self._notes_provider is not None:
            notes, notes_mask = self._notes_provider.notes_for(stream)

        # Call parent's forward method
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Extract hidden states from output
        if isinstance(outputs, tuple):
            modified_hidden = outputs[0]
        else:
            modified_hidden = outputs

        # Apply SNC cross-attention residual
        if self.snc_residual is not None and notes is not None and notes.size(1) > 0:
            gate = torch.sigmoid(self.notes_gate).to(
                dtype=modified_hidden.dtype, device=modified_hidden.device
            )
            delta = self.snc_residual(modified_hidden, notes, notes_mask=notes_mask)
            modified_hidden = modified_hidden + gate * delta

        # Apply stream adapter residual
        if self.stream_adapter is not None and stream is not None:
            gate = torch.sigmoid(self.stream_adapter_gate).to(
                dtype=modified_hidden.dtype, device=modified_hidden.device
            )
            delta = self.stream_adapter(modified_hidden, stream)
            modified_hidden = modified_hidden + gate * delta

        # Apply speculation tap
        if self.speculation_tap is not None and stream is not None and self._spec_sink is not None:
            speculation = self.speculation_tap(modified_hidden)
            self._spec_sink(speculation)

        # Return output in the same format as received
        if isinstance(outputs, tuple):
            return (modified_hidden,) + outputs[1:]
        return modified_hidden


@dataclass(slots=True)
class InstrumentationSpec:
    """Declarative description of which blocks to instrument."""

    enabled: bool = False
    target_layers: Optional[Sequence[int]] = None
    top_k_layers: int = 0
    gate_init: float = 0.0
    stream_adapters: Optional[StreamAdapterConfig] = None
    cross_attention: Optional[SharedNotesCrossAttentionConfig] = None
    speculation: Optional[SpeculationHeadConfig] = None

    def resolve_indices(self, total_layers: int) -> Tuple[int, ...]:
        if not self.enabled:
            return tuple()
        if self.target_layers:
            indices = tuple(sorted(set(int(idx) for idx in self.target_layers)))
        else:
            k = self.top_k_layers
            if k <= 0:
                return tuple()
            start = max(total_layers - k, 0)
            indices = tuple(range(start, total_layers))
        return indices


def instrument_gpt_neox_layers(
    layers: Sequence[nn.Module],
    indices: Iterable[int],
    *,
    stream_adapter_config: Optional[StreamAdapterConfig],
    cross_attention_config: Optional[SharedNotesCrossAttentionConfig],
    speculation_config: Optional[SpeculationHeadConfig],
    gate_init: float = 0.0,
    model_config: Optional[object] = None,
) -> List[Tuple[int, InstrumentedGPTNeoXLayer]]:
    """Replace selected GPT-NeoX or GPT-OSS layers with instrumented versions."""

    instrumented: List[Tuple[int, InstrumentedGPTNeoXLayer]] = []
    for index in indices:
        try:
            original = layers[index]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Layer index {index} outside model range") from exc

        # Check if layer is compatible (GPTNeoXLayer or GptOssDecoderLayer)
        is_neox = isinstance(original, GPTNeoXLayer)
        is_gpt_oss = isinstance(original, GptOssDecoderLayer)

        if not (is_neox or is_gpt_oss):
            LOGGER.warning("skip_instrumentation | index=%d | type=%s", index, type(original))
            continue

        stream_adapter = (
            StreamAdapterLayer(stream_adapter_config) if stream_adapter_config else None
        )
        snc_residual = (
            SharedNotesResidual(cross_attention_config) if cross_attention_config else None
        )
        speculation_tap = (
            SpeculationHead(speculation_config) if speculation_config and not instrumented else None
        )

        # Create appropriate instrumented layer based on type
        if is_gpt_oss:
            # GptOssDecoderLayer doesn't have .config attribute, use model_config
            if model_config is None:
                raise ValueError("model_config required for instrumenting GptOssDecoderLayer")
            layer_idx = getattr(original, "layer_idx", index)
            replacement = InstrumentedGptOssDecoderLayer(
                model_config,
                layer_idx=layer_idx,
                stream_adapter=stream_adapter,
                snc_residual=snc_residual,
                speculation_tap=speculation_tap,
                gate_init=gate_init,
            )
        else:
            # GPTNeoXLayer has .config attribute
            replacement = InstrumentedGPTNeoXLayer(
                original.config,
                stream_adapter=stream_adapter,
                snc_residual=snc_residual,
                speculation_tap=speculation_tap,
                gate_init=gate_init,
            )

        replacement.load_state_dict(original.state_dict(), strict=False)
        device = next(original.parameters()).device
        dtype = next(original.parameters()).dtype
        replacement.to(device=device, dtype=dtype)
        layers[index] = replacement
        instrumented.append((index, replacement))
    return instrumented


__all__ = [
    "InstrumentationSpec",
    "InstrumentedGPTNeoXLayer",
    "InstrumentedGptOssDecoderLayer",
    "LayerRuntimeContext",
    "NotesProvider",
    "StreamAdapterLayer",
    "SharedNotesResidual",
    "instrument_gpt_neox_layers",
]


@dataclass(slots=True)
class InstrumentedTrunkAdapterConfig:
    """Configuration bundle for the instrumented GPT-OSS trunk."""

    trunk: TrunkAdapterConfig = field(default_factory=TrunkAdapterConfig)
    instrumentation: InstrumentationSpec = field(default_factory=InstrumentationSpec)


class _InstrumentationContextManager:
    """Context manager wiring runtime payload into instrumented layers."""

    def __init__(self, adapter: "InstrumentedTrunkAdapter", context: LayerRuntimeContext) -> None:
        self.adapter = adapter
        self.context = context

    def __enter__(self) -> LayerRuntimeContext:
        self.adapter._push_context(self.context)
        return self.context

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.adapter._pop_context()


class InstrumentedTrunkAdapter(GptOssTrunkAdapter):
    """GPT-OSS trunk adapter with optional mid-stack instrumentation."""

    def __init__(
        self,
        config: InstrumentedTrunkAdapterConfig,
        *,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.instrumentation = config.instrumentation
        self.instrumented_layers: list[InstrumentedGPTNeoXLayer] = []
        self._context_stack: list[LayerRuntimeContext] = []
        self._layer_indices: Tuple[int, ...] = tuple()
        self._notes_provider: Optional[NotesProvider] = None
        self._speculation: Optional[torch.Tensor] = None
        super().__init__(config.trunk, model=model)

    @property
    def selected_layers(self) -> Tuple[int, ...]:
        """Indices of layers currently instrumented."""

        return getattr(self, "_layer_indices", tuple())

    @property
    def instrumentation_enabled(self) -> bool:
        return self.instrumentation.enabled and bool(self.instrumented_layers)

    @property
    def last_speculation(self) -> Optional[torch.Tensor]:
        return self._speculation

    def configure(self, instrumentation: InstrumentationSpec) -> None:
        """Update instrumentation spec and re-apply if a trunk is attached."""

        self.instrumentation = instrumentation
        if self._model is not None:
            self._apply_instrumentation()

    def set_notes_provider(self, provider: Optional[NotesProvider]) -> None:
        """Attach a notes provider used to hydrate runtime contexts."""

        self._notes_provider = provider
        for layer in self.instrumented_layers:
            layer.set_notes_provider(provider)

    def consume_speculation(self) -> Optional[torch.Tensor]:
        payload = self._speculation
        self._speculation = None
        return payload

    def activate_context(
        self,
        *,
        stream: Optional[torch.Tensor | str],
        notes: Optional[torch.Tensor],
        notes_mask: Optional[torch.Tensor],
    ) -> _InstrumentationContextManager:
        """Install runtime tensors for subsequent forward passes."""

        context = LayerRuntimeContext(stream=stream, notes=notes, notes_mask=notes_mask)
        return _InstrumentationContextManager(self, context)

    def _push_context(self, context: LayerRuntimeContext) -> None:
        if (
            context.notes is None
            and context.stream is not None
            and self._notes_provider is not None
        ):
            notes, mask = self._notes_provider.notes_for(context.stream)
            context.notes = notes
            context.notes_mask = mask
        self._speculation = None
        self._context_stack.append(context)
        for layer in self.instrumented_layers:
            layer.set_runtime_context(context)

    def _pop_context(self) -> None:
        if not self._context_stack:
            return
        self._context_stack.pop()
        replacement = self._context_stack[-1] if self._context_stack else None
        for layer in self.instrumented_layers:
            layer.set_runtime_context(replacement)

    def _post_attach(self) -> None:
        super()._post_attach()
        self._apply_instrumentation()

    def _apply_instrumentation(self) -> None:
        self.instrumented_layers.clear()
        self._layer_indices = tuple()
        model = self._model
        if model is None:
            return
        spec = self.instrumentation
        if not spec.enabled:
            return
        layers = _resolve_transformer_layers(model)
        if not layers:
            raise RuntimeError(
                "InstrumentedTrunkAdapter: no transformer layers found for instrumentation."
            )
        indices = spec.resolve_indices(len(layers))
        if not indices:
            raise RuntimeError(
                "InstrumentedTrunkAdapter: instrumentation spec produced no target layer indices."
            )
        instrumented = instrument_gpt_neox_layers(
            layers,
            indices,
            stream_adapter_config=spec.stream_adapters,
            cross_attention_config=spec.cross_attention,
            speculation_config=spec.speculation,
            gate_init=spec.gate_init,
            model_config=getattr(model, "config", None),
        )
        if not instrumented:
            raise RuntimeError(
                "InstrumentedTrunkAdapter: no matching GPT-NeoX layers were instrumented."
            )
        self.instrumented_layers.extend(layer for _, layer in instrumented)
        for _, layer in instrumented:
            layer.set_runtime_context(None)
            layer.set_notes_provider(self._notes_provider)
            if layer.speculation_tap is not None:
                layer.set_speculation_sink(self._capture_speculation)
        self._layer_indices = tuple(index for index, _ in instrumented)

    def _capture_speculation(self, payload: torch.Tensor) -> None:
        self._speculation = payload.detach()


__all__.extend(
    [
        "InstrumentedTrunkAdapter",
        "InstrumentedTrunkAdapterConfig",
    ]
)
