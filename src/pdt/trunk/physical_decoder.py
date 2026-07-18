"""Tensorized bank of three independently parameterized Qwen3 decoder stacks.

The lower Qwen3 layers remain one frozen representation trunk.  Every layer
above the configured fork is represented here by a parameter bank with an
explicit physical-decoder axis.  The three parameter slices are initialized
from the same pinned pretrained layer, but they are distinct ``Parameter``
storage and receive independent gradients.

Hidden states are shaped ``[documents, decoders, tokens, hidden]`` inside the
bank.  Linear algebra is grouped over the decoder axis, so one forward advances
all physical decoders without routing three rows through one weight set and
without three serial model calls.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import logging
from typing import Callable, Optional, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from pdt.baselines.self_only import ParameterMatchedSelfOnlyAttention
from pdt.config.schemas import InstrumentationConfig, SidecarConfig
from pdt.sidecar.snc import SharedNotesCrossAttention
from pdt.trunk.instrumentation import LayerRuntimeContext


LOGGER = logging.getLogger("pdt.trunk.physical_decoder")

__all__ = [
    "PhysicalDecoder",
    "PhysicalDecoderLayerBank",
    "PhysicalFrontierCache",
    "PlanMemoryCrossAttention",
]

PHYSICAL_ATTENTION_QUERY_TILE = 128


@dataclass(slots=True)
class PhysicalFrontierCache:
    """One shared-lower cache plus private upper caches for all decoder lanes."""

    shared: Cache
    branch_keys: list[Optional[torch.Tensor]]
    branch_values: list[Optional[torch.Tensor]]
    num_decoders: int

    @classmethod
    def empty(
        cls,
        *,
        shared: Cache,
        branch_layer_count: int,
        num_decoders: int,
    ) -> "PhysicalFrontierCache":
        if branch_layer_count <= 0:
            raise ValueError("Physical frontier cache requires at least one branch layer.")
        if num_decoders != 3:
            raise ValueError("The canonical physical frontier requires exactly three decoders.")
        return cls(
            shared=shared,
            branch_keys=[None] * branch_layer_count,
            branch_values=[None] * branch_layer_count,
            num_decoders=num_decoders,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the common temporal length after proving all caches agree."""

        shared_length = int(self.shared.get_seq_length(layer_idx))
        observed = {
            int(keys.size(-2))
            for keys in self.branch_keys
            if keys is not None
        }
        if len(observed) > 1:
            raise RuntimeError(f"Physical decoder caches disagree on length: {sorted(observed)}.")
        if observed and shared_length not in observed:
            raise RuntimeError(
                "Shared and physical decoder cache lengths disagree: "
                f"shared={shared_length}, branches={next(iter(observed))}."
            )
        return shared_length

    def update_branch(
        self,
        layer_offset: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one synchronous token block to every physical decoder cache."""

        if not 0 <= layer_offset < len(self.branch_keys):
            raise IndexError(f"Branch cache layer {layer_offset} is out of range.")
        if key_states.ndim != 5 or value_states.shape != key_states.shape:
            raise ValueError(
                "Physical branch key/value states must share "
                "[documents, decoders, kv_heads, tokens, head_dim]."
            )
        if key_states.size(1) != self.num_decoders:
            raise ValueError(
                f"Branch cache expected {self.num_decoders} decoders, "
                f"got {key_states.size(1)}."
            )
        prior_keys = self.branch_keys[layer_offset]
        prior_values = self.branch_values[layer_offset]
        if (prior_keys is None) != (prior_values is None):
            raise RuntimeError("Physical branch key/value cache presence is inconsistent.")
        if prior_keys is None:
            updated_keys = key_states
            updated_values = value_states
        else:
            assert prior_values is not None
            expected_prefix = key_states.shape[:3] + key_states.shape[-1:]
            observed_prefix = prior_keys.shape[:3] + prior_keys.shape[-1:]
            if observed_prefix != expected_prefix:
                raise ValueError(
                    "Physical branch cache shape changed across decoding steps: "
                    f"prior={tuple(prior_keys.shape)}, new={tuple(key_states.shape)}."
                )
            updated_keys = torch.cat((prior_keys, key_states), dim=-2)
            updated_values = torch.cat((prior_values, value_states), dim=-2)
        self.branch_keys[layer_offset] = updated_keys
        self.branch_values[layer_offset] = updated_values
        return updated_keys, updated_values


class _LinearBank(nn.Module):
    """Three FP32-master linear maps evaluated as one grouped contraction."""

    def __init__(self, source: nn.Linear, *, num_decoders: int) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = nn.Parameter(
            torch.stack(
                [
                    source.weight.detach().float().clone()
                    for _ in range(num_decoders)
                ],
                dim=0,
            )
        )
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                torch.stack(
                    [
                        source.bias.detach().float().clone()
                        for _ in range(num_decoders)
                    ],
                    dim=0,
                )
            )
        self._compute_weight: Optional[torch.Tensor] = None
        self._compute_bias: Optional[torch.Tensor] = None

    def begin_compute_session(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._compute_weight is not None or self._compute_bias is not None:
            raise RuntimeError("Physical linear compute session is already active.")
        self._compute_weight = self.weight.to(device=device, dtype=dtype)
        if self.bias is not None:
            self._compute_bias = self.bias.to(device=device, dtype=dtype)

    def end_compute_session(self) -> None:
        self._compute_weight = None
        self._compute_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4 or hidden_states.size(-1) != self.in_features:
            raise ValueError(
                "Grouped physical linear input must have "
                f"[documents, decoders, tokens, {self.in_features}]."
            )
        documents, decoders, tokens, _ = hidden_states.shape
        weight = self._compute_weight
        if weight is None:
            weight = self.weight.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif weight.device != hidden_states.device or weight.dtype != hidden_states.dtype:
            raise RuntimeError(
                "Physical linear compute session does not match its hidden-state "
                "device and dtype."
            )
        grouped_input = hidden_states.permute(1, 0, 2, 3).reshape(
            decoders,
            documents * tokens,
            self.in_features,
        )
        output = torch.bmm(grouped_input, weight.transpose(1, 2))
        output = output.reshape(
            decoders,
            documents,
            tokens,
            self.out_features,
        ).permute(1, 0, 2, 3)
        if self.bias is not None:
            bias = self._compute_bias
            if bias is None:
                bias = self.bias.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            output = output + bias.unsqueeze(0).unsqueeze(2)
        return output


class _RMSNormBank(nn.Module):
    """Independent RMSNorm weights on the physical-decoder axis."""

    def __init__(self, source: nn.Module, *, num_decoders: int) -> None:
        super().__init__()
        source_weight = getattr(source, "weight", None)
        epsilon = getattr(source, "variance_epsilon", None)
        if not isinstance(source_weight, nn.Parameter) or not isinstance(epsilon, float):
            raise TypeError("Qwen RMSNorm source has an unexpected implementation.")
        self.weight = nn.Parameter(
            torch.stack(
                [
                    source_weight.detach().float().clone()
                    for _ in range(num_decoders)
                ],
                dim=0,
            )
        )
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 3 or hidden_states.size(1) != self.weight.size(0):
            raise ValueError("Physical RMSNorm input does not match the decoder axis.")
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        weight_shape = [1, self.weight.size(0)]
        weight_shape.extend([1] * (hidden_states.ndim - 3))
        weight_shape.append(self.weight.size(1))
        weight = self.weight.to(dtype=input_dtype).view(*weight_shape)
        return normalized.to(dtype=input_dtype) * weight


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _QwenAttentionBank(nn.Module):
    """Grouped-query self-attention with independent parameters per decoder."""

    def __init__(
        self,
        source: Qwen3Attention,
        *,
        num_decoders: int,
        layer_offset: int,
    ) -> None:
        super().__init__()
        self.layer_offset = layer_offset
        self.num_decoders = num_decoders
        self.num_query_heads = int(source.config.num_attention_heads)
        self.num_key_value_heads = int(source.config.num_key_value_heads)
        self.head_dim = int(source.head_dim)
        self.scaling = float(source.scaling)
        self.attention_dropout = float(source.attention_dropout)
        self.q_proj = _LinearBank(source.q_proj, num_decoders=num_decoders)
        self.k_proj = _LinearBank(source.k_proj, num_decoders=num_decoders)
        self.v_proj = _LinearBank(source.v_proj, num_decoders=num_decoders)
        self.o_proj = _LinearBank(source.o_proj, num_decoders=num_decoders)
        self.q_norm = _RMSNormBank(source.q_norm, num_decoders=num_decoders)
        self.k_norm = _RMSNormBank(source.k_norm, num_decoders=num_decoders)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: PhysicalFrontierCache,
    ) -> torch.Tensor:
        documents, decoders, tokens, _ = hidden_states.shape
        if decoders != self.num_decoders:
            raise ValueError("Physical self-attention received the wrong decoder count.")
        query = self.q_proj(hidden_states).view(
            documents,
            decoders,
            tokens,
            self.num_query_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            documents,
            decoders,
            tokens,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            documents,
            decoders,
            tokens,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).permute(0, 1, 3, 2, 4)
        key = self.k_norm(key).permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)

        cos, sin = position_embeddings
        expected_rows = documents * decoders
        if (
            cos.ndim != 3
            or cos.size(0) not in (1, expected_rows)
            or cos.shape[1:] != (tokens, self.head_dim)
            or sin.shape != cos.shape
        ):
            raise ValueError(
                "Rotary position embeddings must have either one shared position row "
                "or one row per flattened physical decoder."
            )
        if cos.size(0) == 1:
            cos = cos.expand(expected_rows, -1, -1)
            sin = sin.expand(expected_rows, -1, -1)
        cos = cos.reshape(documents, decoders, tokens, self.head_dim).unsqueeze(2)
        sin = sin.reshape(documents, decoders, tokens, self.head_dim).unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        key, value = cache.update_branch(self.layer_offset, key, value)
        total_tokens = key.size(-2)
        if attention_mask is None:
            query_positions = torch.arange(
                total_tokens - tokens,
                total_tokens,
                device=hidden_states.device,
            )
            key_positions = torch.arange(total_tokens, device=hidden_states.device)
            attention_mask = key_positions.view(1, 1, 1, total_tokens) <= (
                query_positions.view(1, 1, tokens, 1)
            )
        else:
            if (
                attention_mask.ndim != 4
                or attention_mask.size(0) not in (1, expected_rows)
            ):
                raise ValueError(
                    "Physical attention requires a rank-four causal mask with either "
                    "one shared row or one row per flattened decoder."
                )
            attention_mask = attention_mask[..., :total_tokens]
            if attention_mask.size(0) == 1:
                attention_mask = attention_mask.expand(expected_rows, -1, -1, -1)
        query_flat = query.reshape(
            expected_rows,
            self.num_query_heads,
            tokens,
            self.head_dim,
        )
        key_flat = key.reshape(
            expected_rows,
            self.num_key_value_heads,
            total_tokens,
            self.head_dim,
        )
        value_flat = value.reshape_as(key_flat)
        attended = _tiled_grouped_query_attention(
            query=query_flat,
            key=key_flat,
            value=value_flat,
            attention_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attended = (
            attended.reshape(
                documents,
                decoders,
                self.num_query_heads,
                tokens,
                self.head_dim,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(documents, decoders, tokens, -1)
        )
        return self.o_proj(attended)


def _tiled_grouped_query_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout_p: float,
    scale: float,
) -> torch.Tensor:
    """Evaluate exact causal GQA in bounded query tiles."""

    query_tokens = query.size(-2)
    if attention_mask.size(-2) != query_tokens:
        raise ValueError("Physical attention mask query axis is misaligned.")
    outputs: list[torch.Tensor] = []
    for start in range(0, query_tokens, PHYSICAL_ATTENTION_QUERY_TILE):
        end = min(start + PHYSICAL_ATTENTION_QUERY_TILE, query_tokens)
        query_tile = query[..., start:end, :]
        mask_tile = attention_mask[..., start:end, :]

        def attend(
            query_value: torch.Tensor,
            key_value: torch.Tensor,
            value_value: torch.Tensor,
            mask_value: torch.Tensor,
        ) -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query_value,
                key_value,
                value_value,
                attn_mask=mask_value,
                dropout_p=dropout_p,
                scale=scale,
                is_causal=False,
                enable_gqa=True,
            )

        if torch.is_grad_enabled() and (
            query_tile.requires_grad or key.requires_grad or value.requires_grad
        ):
            outputs.append(
                checkpoint(
                    attend,
                    query_tile,
                    key,
                    value,
                    mask_tile,
                    use_reentrant=False,
                )
            )
        else:
            outputs.append(
                attend(
                    query_tile,
                    key,
                    value,
                    mask_tile,
                )
            )
    return torch.cat(outputs, dim=-2)


class _QwenMLPBank(nn.Module):
    """Independent SwiGLU feed-forward weights per physical decoder."""

    def __init__(self, source: Qwen3MLP, *, num_decoders: int) -> None:
        super().__init__()
        self.gate_proj = _LinearBank(source.gate_proj, num_decoders=num_decoders)
        self.up_proj = _LinearBank(source.up_proj, num_decoders=num_decoders)
        self.down_proj = _LinearBank(source.down_proj, num_decoders=num_decoders)
        self.activation = cast(
            Callable[[torch.Tensor], torch.Tensor],
            source.act_fn,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.activation(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class PlanMemoryCrossAttention(nn.Module):
    """Persistent hard-routed read over one decoder's individual plan nodes."""

    def __init__(
        self,
        *,
        hidden_size: int,
        plan_width: int,
        attention_width: int,
        num_heads: int,
        max_nodes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if attention_width % num_heads:
            raise ValueError("Plan attention width must be divisible by its head count.")
        self.hidden_size = hidden_size
        self.plan_width = plan_width
        self.attention_width = attention_width
        self.num_heads = num_heads
        self.head_dim = attention_width // num_heads
        self.max_nodes = max_nodes
        self.q_proj = nn.Linear(hidden_size, attention_width)
        self.k_proj = nn.Linear(plan_width, attention_width)
        self.v_proj = nn.Linear(plan_width, attention_width)
        self.o_proj = nn.Linear(attention_width, hidden_size)
        self.node_position = nn.Embedding(max_nodes, plan_width)
        self.dropout = dropout
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_memory: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("Plan queries must have [rows, tokens, hidden].")
        rows, tokens, hidden = hidden_states.shape
        if hidden != self.hidden_size:
            raise ValueError("Plan query hidden width does not match the branch width.")
        if plan_memory.ndim != 3 or plan_memory.size(0) != rows:
            raise ValueError("Plan memory must have [rows, nodes, plan_width].")
        if plan_memory.size(-1) != self.plan_width:
            raise ValueError("Plan memory width does not match plan attention.")
        nodes = plan_memory.size(1)
        if not 0 < nodes <= self.max_nodes:
            raise ValueError("Plan memory node count is outside the configured range.")
        if plan_mask.shape != (rows, nodes):
            raise ValueError("Plan mask does not match the hard-routed plan memory.")
        valid = plan_mask.to(dtype=torch.bool)
        if bool((~valid.any(dim=1)).any()):
            raise ValueError("Every physical decoder requires at least one plan node.")

        positions = torch.arange(nodes, device=plan_memory.device)
        addressed_plan = plan_memory + self.node_position(positions).unsqueeze(0)
        query = self.q_proj(hidden_states.to(dtype=self.q_proj.weight.dtype))
        key = self.k_proj(addressed_plan.to(dtype=self.k_proj.weight.dtype))
        value = self.v_proj(addressed_plan.to(dtype=self.v_proj.weight.dtype))
        query = query.view(rows, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(rows, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(rows, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(rows, tokens, self.attention_width)
        return self.o_proj(attended).to(dtype=hidden_states.dtype)


class PhysicalDecoderLayerBank(nn.Module):
    """One Qwen layer with three base parameter sets and shared plan/bus reads."""

    def __init__(
        self,
        source: Qwen3DecoderLayer,
        *,
        layer_offset: int,
        num_decoders: int,
        sidecar: SidecarConfig,
        instrumentation: InstrumentationConfig,
    ) -> None:
        super().__init__()
        source_attention = cast(Qwen3Attention, source.self_attn)
        self.pdt_layer_idx = int(source_attention.layer_idx)
        self.attention_type = str(source.attention_type)
        if self.attention_type != "full_attention":
            raise ValueError(
                "The physical decoder currently requires full-attention Qwen layers; "
                f"layer {self.pdt_layer_idx} is {self.attention_type!r}."
            )
        input_layernorm = cast(Qwen3RMSNorm, source.input_layernorm)
        post_attention_layernorm = cast(
            Qwen3RMSNorm,
            source.post_attention_layernorm,
        )
        source_mlp = cast(Qwen3MLP, source.mlp)
        self.input_layernorm = _RMSNormBank(
            input_layernorm,
            num_decoders=num_decoders,
        )
        self.self_attn = _QwenAttentionBank(
            source_attention,
            num_decoders=num_decoders,
            layer_offset=layer_offset,
        )
        self.post_attention_layernorm = _RMSNormBank(
            post_attention_layernorm,
            num_decoders=num_decoders,
        )
        self.mlp = _QwenMLPBank(source_mlp, num_decoders=num_decoders)
        self.plan_attention = PlanMemoryCrossAttention(
            hidden_size=sidecar.hidden_size,
            plan_width=sidecar.notes_dim,
            attention_width=sidecar.snc.attention_width,
            num_heads=sidecar.snc.num_heads,
            max_nodes=sidecar.planner_head.max_nodes_per_stream,
            dropout=sidecar.snc.dropout,
        )
        attention_type = (
            SharedNotesCrossAttention
            if instrumentation.coordination_source == "bus"
            else ParameterMatchedSelfOnlyAttention
        )
        self.snc = attention_type(
            sidecar.snc,
            num_producers=num_decoders,
            gating_init=instrumentation.snc_gate_init,
        )
        self.plan_gate = nn.Parameter(
            torch.tensor(float(instrumentation.plan_gate_init))
        )
        self.notes_gate = nn.Parameter(
            torch.tensor(float(instrumentation.snc_gate_init))
        )

    def base_parameters(self) -> Iterator[nn.Parameter]:
        for module in (
            self.input_layernorm,
            self.self_attn,
            self.post_attention_layernorm,
            self.mlp,
        ):
            yield from module.parameters()

    def extension_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.plan_attention.parameters()
        yield from self.snc.parameters()
        yield self.plan_gate
        yield self.notes_gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: PhysicalFrontierCache,
        context: LayerRuntimeContext,
        num_decoders: int,
    ) -> torch.Tensor:
        rows, tokens, hidden = hidden_states.shape
        if rows % num_decoders:
            raise ValueError("Physical decoder rows must be divisible by three.")
        documents = rows // num_decoders
        grouped = hidden_states.reshape(documents, num_decoders, tokens, hidden)

        residual = grouped
        grouped = self.input_layernorm(grouped)
        LOGGER.debug(
            "Physical layer %d entering self-attention with shape=%s.",
            self.pdt_layer_idx,
            tuple(grouped.shape),
        )
        grouped = residual + self.self_attn(
            grouped,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache=cache,
        )
        LOGGER.debug("Physical layer %d completed self-attention.", self.pdt_layer_idx)
        residual = grouped
        grouped = self.post_attention_layernorm(grouped)
        mlp_delta = (
            checkpoint(self.mlp, grouped, use_reentrant=False)
            if torch.is_grad_enabled() and grouped.requires_grad
            else self.mlp(grouped)
        )
        grouped = residual + mlp_delta
        LOGGER.debug("Physical layer %d completed MLP.", self.pdt_layer_idx)
        modified = grouped.reshape(rows, tokens, hidden)

        if context.plan_memory is None or context.plan_mask is None:
            raise ValueError(
                "Every physical branch layer requires persistent plan memory and mask."
            )
        plan_delta = self.plan_attention(
            modified,
            context.plan_memory,
            context.plan_mask,
        )
        plan_gate = torch.sigmoid(self.plan_gate).to(
            device=modified.device,
            dtype=modified.dtype,
        )
        modified = modified + plan_gate * plan_delta

        if isinstance(self.snc, ParameterMatchedSelfOnlyAttention):
            if context.self_only_memory is not None:
                if (
                    context.self_only_query_positions is None
                    or context.stream_ids is None
                ):
                    raise ValueError("Self-only physical decoding context is incomplete.")
                notes_delta = self.snc(
                    modified,
                    context.self_only_memory,
                    query_positions=context.self_only_query_positions,
                    receiver_streams=context.stream_ids,
                    force_gate=context.snc_force_gate,
                )
                notes_gate = torch.sigmoid(self.notes_gate).to(
                    device=modified.device,
                    dtype=modified.dtype,
                )
                modified = modified + notes_gate * notes_delta
        elif context.notes is not None and context.notes.size(1) > 0:
            metadata = (
                context.notes_mask,
                context.note_producer_ids,
                context.note_kind_ids,
                context.note_lags,
            )
            if any(value is None for value in metadata):
                raise ValueError("Dynamic notes require complete addressed metadata.")
            assert context.notes_mask is not None
            assert context.note_producer_ids is not None
            assert context.note_kind_ids is not None
            assert context.note_lags is not None
            notes_delta = self.snc(
                modified,
                context.notes,
                notes_mask=context.notes_mask,
                producer_ids=context.note_producer_ids,
                kind_ids=context.note_kind_ids,
                lags=context.note_lags,
                force_gate=context.snc_force_gate,
            )
            notes_gate = torch.sigmoid(self.notes_gate).to(
                device=modified.device,
                dtype=modified.dtype,
            )
            modified = modified + notes_gate * notes_delta
        return modified


class PhysicalDecoder(nn.Module):
    """All independently parameterized upper layers for three decoder lanes."""

    def __init__(
        self,
        source_layers: Sequence[Qwen3DecoderLayer],
        *,
        fork_layer: int,
        num_decoders: int,
        sidecar: SidecarConfig,
        instrumentation: InstrumentationConfig,
    ) -> None:
        super().__init__()
        if not source_layers:
            raise ValueError("Physical decoder requires at least one upper source layer.")
        if num_decoders != 3:
            raise ValueError("Physical decoder requires exactly three parameter banks.")
        expected_indices = tuple(range(fork_layer, fork_layer + len(source_layers)))
        observed_indices = tuple(
            int(cast(Qwen3Attention, layer.self_attn).layer_idx)
            for layer in source_layers
        )
        if observed_indices != expected_indices:
            raise ValueError(
                "Physical decoder source layers must be consecutive above the fork: "
                f"expected={expected_indices}, observed={observed_indices}."
            )
        self.fork_layer = fork_layer
        self.num_decoders = num_decoders
        self.layers = nn.ModuleList(
            [
                PhysicalDecoderLayerBank(
                    layer,
                    layer_offset=offset,
                    num_decoders=num_decoders,
                    sidecar=sidecar,
                    instrumentation=instrumentation,
                )
                for offset, layer in enumerate(source_layers)
            ]
        )
        self._runtime_context: Optional[LayerRuntimeContext] = None

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(
            cast(PhysicalDecoderLayerBank, layer).pdt_layer_idx
            for layer in self.layers
        )

    def set_runtime_context(self, context: Optional[LayerRuntimeContext]) -> None:
        self._runtime_context = context

    def base_parameters(self) -> Iterator[nn.Parameter]:
        for layer in self.layers:
            yield from cast(PhysicalDecoderLayerBank, layer).base_parameters()

    def extension_parameters(self) -> Iterator[nn.Parameter]:
        for layer in self.layers:
            yield from cast(PhysicalDecoderLayerBank, layer).extension_parameters()

    def begin_compute_session(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        banks = [
            module for module in self.modules() if isinstance(module, _LinearBank)
        ]
        if not banks:
            raise RuntimeError("Physical decoder contains no grouped linear banks.")
        for bank in banks:
            bank.begin_compute_session(device=device, dtype=dtype)

    def end_compute_session(self) -> None:
        for module in self.modules():
            if isinstance(module, _LinearBank):
                module.end_compute_session()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        causal_masks: dict[str, Optional[torch.Tensor]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: PhysicalFrontierCache,
    ) -> torch.Tensor:
        context = self._runtime_context
        if context is None:
            raise RuntimeError("Physical decoder runtime context was not set.")
        if context.plan_memory is None or context.plan_memory.size(0) != hidden_states.size(0):
            raise ValueError("Physical decoder plan memory must address every frontier row.")
        for raw_layer in self.layers:
            layer = cast(PhysicalDecoderLayerBank, raw_layer)
            try:
                attention_mask = causal_masks[layer.attention_type]
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing causal mask for branch attention type {layer.attention_type!r}."
                ) from exc
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache=cache,
                context=context,
                num_decoders=self.num_decoders,
            )
        return hidden_states
