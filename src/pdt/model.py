"""Shared frozen Qwen3 knowledge trunk plus three physical decoder stacks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Optional, cast

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from pdt.config.schemas import PDTConfig, SidecarConfig
from pdt.sidecar.heads.plan_memory import PlanMemoryProjection
from pdt.sidecar.heads.planner import PlannerHead
from pdt.sidecar.heads.semantic import SemanticSupervisionHeads
from pdt.sidecar.heads.speculation import SpeculationHead
from pdt.trunk.instrumentation import LayerRuntimeContext
from pdt.trunk.physical_decoder import (
    PhysicalDecoder,
    PhysicalDecoderLayerBank,
    PhysicalFrontierCache,
)
from pdt.trunk.qwen3_adapter import Qwen3TrunkAdapter, SharedTrunkOutput


LOGGER = logging.getLogger("pdt.model")

__all__ = ["PDTModel", "Sidecar"]


class Sidecar(nn.Module):
    """Prompt-time planner and source-grounded supervision heads."""

    def __init__(self, config: SidecarConfig) -> None:
        super().__init__()
        self.config = config
        self.planner_head = PlannerHead(config.planner_head)
        self.plan_memory_proj = PlanMemoryProjection(config.plan_memory_proj)
        self.semantic_heads = SemanticSupervisionHeads(config.semantic_supervision)
        self.speculation_head = SpeculationHead(config.speculation_head)


class PDTModel(nn.Module):
    """One frozen lower trunk and three tensorized, independent upper decoders."""

    def __init__(self, config: PDTConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk_adapter = Qwen3TrunkAdapter(config.trunk)
        self.sidecar = Sidecar(config.sidecar)
        upper_layers = self.trunk_adapter.take_upper_layers(
            config.instrumentation.fork_layer
        )
        self.physical_decoder = PhysicalDecoder(
            upper_layers,
            fork_layer=config.instrumentation.fork_layer,
            num_decoders=config.sidecar.num_streams,
            sidecar=config.sidecar,
            instrumentation=config.instrumentation,
        )
        # Read-only compatibility surface used by diagnostics and runtime
        # context threading.  Module ownership remains physical_decoder.layers.
        self.instrumented_layers = list(self.physical_decoder.layers)
        self._validate_parameter_partition()
        LOGGER.info(
            "PDTModel ready: trunk=%s shared_layers=%d fork=%d "
            "physical_decoders=%d branch_layers=%d shared_params=%s "
            "branch_base_params=%s branch_extension_params=%s sidecar_params=%s",
            config.trunk.base_model,
            self.trunk_adapter.num_layers(),
            self.physical_decoder.fork_layer,
            self.physical_decoder.num_decoders,
            len(self.physical_decoder.layers),
            _fmt_params(self.trunk_parameters()),
            _fmt_params(self.decoder_branch_parameters()),
            _fmt_params(self.per_layer_phi_parameters()),
            _fmt_params(self.sidecar_parameters()),
        )

    # ------------------------------------------------------------------ #
    # Parameter ownership
    # ------------------------------------------------------------------ #

    def trunk_parameters(self) -> Iterator[nn.Parameter]:
        """Frozen shared embeddings, lower layers, final norm, and LM head."""

        return iter(self.trunk_adapter.frozen_parameters())

    def decoder_branch_parameters(self) -> Iterator[nn.Parameter]:
        """All independently parameterized Qwen weights above the fork."""

        return self.physical_decoder.base_parameters()

    def sidecar_parameters(self) -> Iterator[nn.Parameter]:
        return self.sidecar.parameters()

    def per_layer_phi_parameters(self) -> Iterator[nn.Parameter]:
        """Persistent plan attention, notes attention, and their residual gates."""

        return self.physical_decoder.extension_parameters()

    def all_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Complete optimizer manifest, including the physical decoder weights."""

        yield from self.decoder_branch_parameters()
        yield from self.sidecar_parameters()
        yield from self.per_layer_phi_parameters()

    def _validate_parameter_partition(self) -> None:
        groups_as_parameters = {
            "trunk": tuple(self.trunk_parameters()),
            "decoder_branches": tuple(self.decoder_branch_parameters()),
            "sidecar": tuple(self.sidecar_parameters()),
            "per_layer_phi": tuple(self.per_layer_phi_parameters()),
        }
        groups = {
            name: {id(parameter) for parameter in parameters}
            for name, parameters in groups_as_parameters.items()
        }
        for name, parameters in groups_as_parameters.items():
            if len(groups[name]) != len(parameters):
                raise RuntimeError(f"Parameter aliases exist inside canonical group {name!r}.")
        names = tuple(groups)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if groups[left] & groups[right]:
                    raise RuntimeError(
                        f"Parameter partition overlap between {left!r} and {right!r}."
                    )
        if any(parameter.requires_grad for parameter in groups_as_parameters["trunk"]):
            raise RuntimeError("Frozen shared trunk contains a trainable parameter.")
        if not groups_as_parameters["decoder_branches"]:
            raise RuntimeError("Physical decoder parameter bank is empty.")
        expected_branch_axis = self.config.sidecar.num_streams
        for raw_layer in self.physical_decoder.layers:
            layer = cast(PhysicalDecoderLayerBank, raw_layer)
            for parameter in layer.base_parameters():
                if parameter.ndim < 2 or parameter.size(0) != expected_branch_axis:
                    raise RuntimeError(
                        "Every physical decoder parameter must retain an explicit "
                        f"leading branch axis of {expected_branch_axis}; "
                        f"observed {tuple(parameter.shape)}."
                    )
                if parameter.dtype != torch.float32:
                    raise RuntimeError(
                        "Physical decoder parameters must use FP32 master storage; "
                        f"observed {parameter.dtype}."
                    )

    # ------------------------------------------------------------------ #
    # Runtime context
    # ------------------------------------------------------------------ #

    def set_runtime_context(self, context: Optional[LayerRuntimeContext]) -> None:
        self.physical_decoder.set_runtime_context(context)

    # ------------------------------------------------------------------ #
    # Shared prompt encoder and physical frontier
    # ------------------------------------------------------------------ #

    def encode_planner_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a prompt once through the frozen lower knowledge trunk."""

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Planner prompt ids and mask must share [batch, tokens].")
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)
        cache_position = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
            dtype=torch.long,
        )
        shared = self.trunk_adapter.forward_shared(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
        )
        return self.trunk_adapter.model.model.norm(shared.hidden_states)

    def forward_frontier(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[PhysicalFrontierCache] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        exact_causal_mask: bool = False,
    ) -> CausalLMOutputWithPast:
        """Advance all three physical decoders in one grouped upper forward."""

        if not use_cache:
            raise ValueError("Physical frontier decoding requires its real persistent caches.")
        rows = input_ids.size(0)
        decoders = self.physical_decoder.num_decoders
        if rows % decoders:
            raise ValueError(
                f"Frontier rows must contain complete groups of {decoders}; got {rows}."
            )
        if attention_mask.ndim != 2 or attention_mask.size(0) != rows:
            raise ValueError("Physical frontier attention mask must address every row.")

        if past_key_values is None:
            shared_cache: Cache = DynamicCache(config=self.trunk_adapter.model.config)
        elif isinstance(past_key_values, PhysicalFrontierCache):
            shared_cache = past_key_values.shared
        else:
            raise TypeError(
                "Physical frontier past_key_values must be PhysicalFrontierCache or None."
            )
        shared_prefill = past_key_values is None and _groups_are_identical(
            input_ids,
            attention_mask,
            position_ids,
            group_size=decoders,
        )
        if shared_prefill:
            documents = rows // decoders
            grouped_shape = (documents, decoders, input_ids.size(1))
            shared_input_ids = input_ids.reshape(grouped_shape)[:, 0]
            shared_attention_mask = attention_mask.reshape(grouped_shape)[:, 0]
            shared_position_ids = (
                None
                if position_ids is None
                else position_ids.reshape(grouped_shape)[:, 0]
            )
        else:
            documents = rows
            shared_input_ids = input_ids
            shared_attention_mask = attention_mask
            shared_position_ids = position_ids
        # The lower trunk is immutable. Explicit no-grad is required so a
        # training caller cannot make SDPA retain a quadratic backward buffer
        # below the physical fork.
        with torch.no_grad():
            shared = self.trunk_adapter.forward_shared(
                input_ids=shared_input_ids,
                attention_mask=shared_attention_mask,
                past_key_values=shared_cache,
                position_ids=shared_position_ids,
                cache_position=cache_position,
                use_cache=True,
                exact_causal_mask=exact_causal_mask,
            )
        if shared.past_key_values is None:
            raise RuntimeError("Shared lower trunk failed to return its cache.")
        if shared_prefill:
            shared.past_key_values.batch_repeat_interleave(decoders)
            shared = _expand_shared_prefill(
                shared,
                documents=documents,
                decoders=decoders,
            )
        cache = past_key_values
        if cache is None:
            cache = PhysicalFrontierCache.empty(
                shared=cast(Cache, shared.past_key_values),
                branch_layer_count=len(self.physical_decoder.layers),
                num_decoders=decoders,
            )
        hidden_states = self.physical_decoder(
            shared.hidden_states,
            causal_masks=shared.causal_masks,
            position_embeddings=shared.position_embeddings,
            cache=cache,
        )
        hidden_states = self.trunk_adapter.model.model.norm(hidden_states)
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be non-negative.")
            indices: slice | torch.Tensor = (
                slice(-logits_to_keep, None)
                if logits_to_keep
                else slice(None)
            )
        else:
            indices = logits_to_keep
        logits = self.trunk_adapter.model.lm_head(hidden_states[:, indices, :])
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,  # type: ignore[arg-type]
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )

    def forward(self, *args, **kwargs):
        return self.forward_frontier(*args, **kwargs)


def _fmt_params(parameters: Iterator[nn.Parameter]) -> str:
    total = sum(parameter.numel() for parameter in parameters)
    if total >= 1_000_000:
        return f"{total / 1e6:.1f}M"
    if total >= 1_000:
        return f"{total / 1e3:.1f}K"
    return str(total)


def _groups_are_identical(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    *,
    group_size: int,
) -> bool:
    rows, tokens = input_ids.shape
    if rows % group_size:
        return False
    documents = rows // group_size
    shape = (documents, group_size, tokens)
    grouped_ids = input_ids.reshape(shape)
    grouped_mask = attention_mask.reshape(shape)
    if not torch.equal(grouped_ids, grouped_ids[:, :1].expand_as(grouped_ids)):
        return False
    if not torch.equal(grouped_mask, grouped_mask[:, :1].expand_as(grouped_mask)):
        return False
    if position_ids is None:
        return True
    grouped_positions = position_ids.reshape(shape)
    return torch.equal(
        grouped_positions,
        grouped_positions[:, :1].expand_as(grouped_positions),
    )


def _expand_shared_prefill(
    shared: SharedTrunkOutput,
    *,
    documents: int,
    decoders: int,
) -> SharedTrunkOutput:
    hidden = shared.hidden_states
    if hidden.size(0) != documents:
        raise ValueError("Shared prefill hidden states do not match document count.")
    expanded_hidden = (
        hidden[:, None]
        .expand(documents, decoders, hidden.size(1), hidden.size(2))
        .reshape(documents * decoders, hidden.size(1), hidden.size(2))
    )
    expanded_masks: dict[str, Optional[torch.Tensor]] = {}
    for name, mask in shared.causal_masks.items():
        if mask is None or mask.size(0) == 1:
            expanded_masks[name] = mask
            continue
        if mask.size(0) != documents:
            raise ValueError(
                f"Shared prefill mask {name!r} does not match document count."
            )
        expanded_masks[name] = (
            mask[:, None]
            .expand(documents, decoders, *mask.shape[1:])
            .reshape(documents * decoders, *mask.shape[1:])
        )
    expanded_positions: list[torch.Tensor] = []
    for values in shared.position_embeddings:
        if values.size(0) == 1:
            expanded_positions.append(values)
            continue
        if values.size(0) != documents:
            raise ValueError(
                "Shared prefill rotary embeddings do not match document count."
            )
        expanded_positions.append(
            values[:, None]
            .expand(documents, decoders, *values.shape[1:])
            .reshape(documents * decoders, *values.shape[1:])
        )
    return SharedTrunkOutput(
        hidden_states=expanded_hidden,
        past_key_values=shared.past_key_values,
        causal_masks=expanded_masks,
        position_embeddings=(expanded_positions[0], expanded_positions[1]),
    )
