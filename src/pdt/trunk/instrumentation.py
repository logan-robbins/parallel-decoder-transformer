"""Instrumented Qwen3 decoder layer.

This is the module that lands \u03c6 state (SNC + per-stream adapter) inside
the frozen trunk. The class subclasses ``Qwen3DecoderLayer`` and wraps
``super().forward(...)`` with two post-residual deltas:

    1. SNC cross-attention read from the visible notes window, applied via
       ``hidden + sigmoid(notes_gate) * snc(hidden, notes, notes_mask)``.
    2. Per-stream bottleneck adapter, applied via
       ``hidden + sigmoid(adapter_gate) * adapters(stream, hidden)``.

Both gates are initialized to sigmoid(-4.0) \u2248 0.018 so that at step 0 the
trunk's magnitude statistics are preserved; training opens the gates as the
auxiliary paths become reliable.

The layer reads its per-forward context (stream id, notes, notes mask) from
a ``LayerRuntimeContext`` threadable attribute set by
``pdt.runtime.orchestrator`` immediately before each trunk forward. This
decouples the layer from the orchestrator without relying on hooks.

**Install via** ``instrument_trunk(trunk, config, sidecar_modules)``. That
function asserts the post-install identity is correct -- catching the
ModuleList write-back bug that was present in the previous codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from pdt.baselines.self_only import ParameterMatchedSelfOnlyAttention, SelfOnlyMemory
from pdt.config.schemas import InstrumentationConfig, SidecarConfig
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.snc import SharedNotesCrossAttention
from pdt.trunk.qwen3_adapter import Qwen3TrunkAdapter


LOGGER = logging.getLogger("pdt.trunk.instrumentation")

__all__ = [
    "InstrumentedQwen3DecoderLayer",
    "LayerRuntimeContext",
    "instrument_trunk",
]


@dataclass(slots=True)
class LayerRuntimeContext:
    """Per-forward context threaded into every instrumented layer.

    Set on every instrumented layer via ``set_runtime_context`` just before
    a trunk forward. ``stream_ids`` addresses one adapter per batch row; a
    scalar stream selector is intentionally not supported because it would
    preserve the old physically sequential execution path.
    """

    stream_ids: Optional[Tuple[str, ...]] = None
    notes: Optional[torch.Tensor] = None  # (B, S, notes_dim) or None
    notes_mask: Optional[torch.Tensor] = None  # (B, S) bool or None
    note_producer_ids: Optional[torch.Tensor] = None  # (B, S) long or None
    note_kind_ids: Optional[torch.Tensor] = None  # (B, S) long; 0 anchor, 1 dynamic
    note_lags: Optional[torch.Tensor] = None  # (B, S) non-negative block ages
    self_only_memory: Optional[SelfOnlyMemory] = None
    self_only_query_positions: Optional[torch.Tensor] = None  # (B, T) long
    # Optional per-layer SNC gate override. None -> use trained gate; True -> force open;
    # False -> force closed; tensor -> per-batch override.
    snc_force_gate: Optional[object] = None


class InstrumentedQwen3DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder layer with \u03c6 injection points.

    Preserves the signature of ``Qwen3DecoderLayer.forward`` exactly and
    forwards all untouched keyword args to ``super().forward(...)``.
    """

    def __init__(
        self,
        config,  # Qwen3Config (HF)
        layer_idx: int,
        *,
        snc: Optional[SharedNotesCrossAttention],
        stream_adapter: Optional[StreamAdapterLayer],
        snc_gate_init: float,
        adapter_gate_init: float,
    ) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.pdt_layer_idx = layer_idx
        self.snc = snc
        self.stream_adapter = stream_adapter
        # Outer gates on the residual adds. Both start closed.
        if snc is not None:
            self.notes_gate = nn.Parameter(torch.tensor(float(snc_gate_init)))
        else:
            self.register_parameter("notes_gate", None)
        if stream_adapter is not None:
            self.adapter_gate = nn.Parameter(torch.tensor(float(adapter_gate_init)))
        else:
            self.register_parameter("adapter_gate", None)
        self._runtime_context: Optional[LayerRuntimeContext] = None

    def set_runtime_context(self, context: Optional[LayerRuntimeContext]) -> None:
        self._runtime_context = context

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Delegate the canonical Qwen3 forward unchanged.
        out = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # HF's Qwen3DecoderLayer returns a Tensor (not a tuple).
        modified = out if isinstance(out, torch.Tensor) else out[0]

        context = self._runtime_context
        batch = modified.size(0)

        if context is not None:
            if context.notes is not None and context.self_only_memory is not None:
                raise ValueError(
                    "A layer context cannot contain both bus notes and self-only memory."
                )
            if context.notes is not None and context.notes.size(0) != batch:
                raise ValueError(
                    "LayerRuntimeContext notes batch must match trunk hidden batch; "
                    f"got {context.notes.size(0)} and {batch}."
                )
            if context.notes_mask is not None and context.notes_mask.size(0) != batch:
                raise ValueError(
                    "LayerRuntimeContext notes_mask batch must match trunk hidden batch; "
                    f"got {context.notes_mask.size(0)} and {batch}."
                )
            metadata = (
                context.notes_mask,
                context.note_producer_ids,
                context.note_kind_ids,
                context.note_lags,
            )
            if context.notes is not None and any(tensor is None for tensor in metadata):
                raise ValueError(
                    "Addressed SNC requires mask, producer, kind, and lag metadata "
                    "whenever notes are present."
                )
            if context.notes is None and any(tensor is not None for tensor in metadata):
                raise ValueError("Note metadata cannot be supplied without notes.")
            for tensor in metadata[1:]:
                if tensor is not None and tensor.size(0) != batch:
                    raise ValueError("LayerRuntimeContext note metadata batch mismatch.")
            if context.stream_ids is not None and len(context.stream_ids) != batch:
                raise ValueError(
                    "LayerRuntimeContext stream_ids must address every trunk batch row; "
                    f"got {len(context.stream_ids)} IDs for batch {batch}."
                )
            if (
                context.self_only_memory is not None
                and context.self_only_memory.hidden_states.size(0) != batch
            ):
                raise ValueError(
                    "LayerRuntimeContext self-only memory batch must match trunk hidden batch."
                )
            if context.self_only_memory is not None:
                if context.self_only_query_positions is None:
                    raise ValueError("Self-only memory requires query-position metadata.")
                if context.stream_ids is None:
                    raise ValueError("Self-only memory requires row-addressed stream IDs.")
                if context.self_only_query_positions.shape != modified.shape[:2]:
                    raise ValueError(
                        "Self-only query positions must match trunk [batch, tokens]; "
                        f"got {tuple(context.self_only_query_positions.shape)} and "
                        f"{tuple(modified.shape[:2])}."
                    )
            elif context.self_only_query_positions is not None:
                raise ValueError("Self-only query positions cannot be supplied without memory.")

        # SNC residual add.
        if isinstance(self.snc, ParameterMatchedSelfOnlyAttention):
            if context is not None and context.notes is not None:
                raise ValueError("Self-only attention rejects bus-note layer contexts.")
            if context is not None and context.self_only_memory is not None:
                delta = self.snc(
                    modified,
                    context.self_only_memory,
                    query_positions=context.self_only_query_positions,
                    receiver_streams=context.stream_ids,
                    force_gate=context.snc_force_gate,
                )
                gate = torch.sigmoid(self.notes_gate).to(
                    dtype=modified.dtype, device=modified.device
                )
                modified = modified + gate * delta
        elif self.snc is not None and context is not None:
            if context.self_only_memory is not None:
                raise ValueError("Bus SNC rejects self-only layer contexts.")
            if context.notes is not None and context.notes.size(1) > 0:
                delta = self.snc(
                    modified,
                    context.notes,
                    notes_mask=context.notes_mask,
                    producer_ids=context.note_producer_ids,
                    kind_ids=context.note_kind_ids,
                    lags=context.note_lags,
                    force_gate=context.snc_force_gate,
                )
                gate = torch.sigmoid(self.notes_gate).to(
                    dtype=modified.dtype, device=modified.device
                )
                modified = modified + gate * delta

        # Per-stream adapter residual add.
        if (
            self.stream_adapter is not None
            and context is not None
            and context.stream_ids is not None
        ):
            delta = self.stream_adapter(modified, context.stream_ids)
            gate = torch.sigmoid(self.adapter_gate).to(dtype=modified.dtype, device=modified.device)
            modified = modified + gate * delta

        if isinstance(out, torch.Tensor):
            return modified
        # Defensive branch for older HF: propagate the tuple with swapped head.
        return (modified,) + out[1:]


def instrument_trunk(
    trunk: Qwen3TrunkAdapter,
    instrumentation: InstrumentationConfig,
    sidecar: SidecarConfig,
    *,
    make_snc,  # Callable[[], SharedNotesCrossAttention]
    make_adapter,  # Callable[[], StreamAdapterLayer]
) -> List[InstrumentedQwen3DecoderLayer]:
    """Replace selected trunk layers with instrumented subclasses.

    Called once at model construction. Returns the list of instrumented
    layers so callers can thread ``LayerRuntimeContext`` and reach \u03c6 state
    without re-indexing.

    Raises if any replacement fails the ``is``-identity check post-install.
    """
    if not instrumentation.enabled:
        return []

    total_layers = trunk.num_layers()
    targets = [idx for idx in instrumentation.target_layers if 0 <= idx < total_layers]
    if len(targets) != len(instrumentation.target_layers):
        invalid = set(instrumentation.target_layers) - set(targets)
        raise ValueError(
            f"Instrumentation target_layers out of range: {sorted(invalid)} "
            f"(total_layers={total_layers})."
        )

    hf_config = trunk.model.config
    instrumented: List[InstrumentedQwen3DecoderLayer] = []

    for idx in targets:
        src = trunk.layers[idx]
        if not isinstance(src, Qwen3DecoderLayer):
            raise TypeError(
                f"Layer {idx} is {type(src).__name__}, not Qwen3DecoderLayer. "
                f"PDT trunk instrumentation requires a vanilla Qwen3 trunk."
            )
        source_requires_grad = {
            name: parameter.requires_grad for name, parameter in src.named_parameters()
        }
        replacement = InstrumentedQwen3DecoderLayer(
            hf_config,
            layer_idx=idx,
            snc=make_snc(),
            stream_adapter=make_adapter(),
            snc_gate_init=instrumentation.snc_gate_init,
            adapter_gate_init=instrumentation.adapter_gate_init,
        )
        # Copy trunk weights into the replacement (so the self-attn + MLP
        # carry the frozen checkpoint's values).
        replacement.load_state_dict(src.state_dict(), strict=False)
        replacement_parameters = dict(replacement.named_parameters())
        for name, requires_grad in source_requires_grad.items():
            parameter = replacement_parameters.get(name)
            if parameter is None:
                raise RuntimeError(f"Replacement layer {idx} lost source parameter {name!r}.")
            parameter.requires_grad_(requires_grad)
        trunk.replace_layer(idx, replacement)
        # Post-install identity check -- catches the ModuleList bug that was
        # present in the previous codebase.
        if trunk.layers[idx] is not replacement:
            raise RuntimeError(
                f"Post-install identity check FAILED at layer {idx}. "
                f"The replacement is not wired into the trunk's forward graph."
            )
        instrumented.append(replacement)
        LOGGER.info("Instrumented layer %d -> %s", idx, type(replacement).__name__)

    trunk.record_instrumented_indices(tuple(targets))
    LOGGER.info("Instrumented %d/%d decoder layers.", len(targets), total_layers)
    return instrumented
