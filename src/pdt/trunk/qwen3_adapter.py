"""Frozen Qwen3 trunk adapter.

Loads one revision-pinned dense Qwen3 trunk profile via
``AutoModelForCausalLM``, freezes every parameter, and exposes the trunk's
decoder layers for instrumentation. Canonical PDT prompts use the selected
trunk's existing chat vocabulary and never mutate frozen token rows.

**Critical fix relative to the previous codebase:** layer access returns the
actual ``nn.ModuleList`` (not a shallow Python list). Subclass replacement
via ``trunk.model.layers[idx] = replacement`` writes into the forward
graph. A post-instrumentation ``is``-identity assertion verifies the
installation did in fact land on the real module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, cast

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from pdt.config.schemas import TRUNK_PROFILES, TrunkConfig
from pdt.trunk.gqa_sdpa import register_pdt_gqa_sdpa


LOGGER = logging.getLogger("pdt.trunk")

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

__all__ = ["Qwen3TrunkAdapter", "SharedTrunkOutput"]


@dataclass(slots=True)
class SharedTrunkOutput:
    """Unnormalized fork states and attention state shared by all branches."""

    hidden_states: torch.Tensor
    past_key_values: Optional[Cache]
    causal_masks: dict[str, Optional[torch.Tensor]]
    position_embeddings: tuple[torch.Tensor, torch.Tensor]


class Qwen3TrunkAdapter:
    """Wrapper around a frozen Qwen3ForCausalLM.

    This is deliberately *not* an ``nn.Module`` -- it holds the underlying
    HuggingFace model by composition and exposes precisely the surface PDT
    needs (layer access, tokenizer access, frozen param iterator). Treating
    the trunk as a non-module prevents accidental inclusion of its
    parameters in the sidecar module tree and makes the \u03b8_pre / \u03c6 split
    inspectable by module path alone.
    """

    def __init__(self, config: TrunkConfig) -> None:
        self.config = config
        self.dtype = _resolve_dtype(config.torch_dtype)
        self.model: PreTrainedModel = self._load_model()
        self.tokenizer: PreTrainedTokenizerBase = self._load_tokenizer()
        self._freeze()

    def _load_model(self) -> PreTrainedModel:
        register_pdt_gqa_sdpa()
        source = self.config.local_path or self.config.base_model
        LOGGER.info(
            "Loading Qwen3 trunk from %s (dtype=%s, attn=%s)",
            source,
            self.dtype,
            self.config.attn_implementation,
        )
        kwargs = {
            "dtype": self.dtype,
            "attn_implementation": self.config.attn_implementation,
            "revision": self.config.revision,
        }
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        self._validate_loaded_architecture(model)
        # Incremental differentiable rollout requires the real KV cache.
        # Hugging Face disables use_cache when gradient checkpointing is active
        # in train mode, so the frozen trunk is intentionally kept in eval mode
        # and gradient checkpointing is not part of the canonical configuration.
        model.config.use_cache = True
        model.eval()
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        source = self.config.local_path or self.config.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            use_fast=True,
            revision=self.config.revision,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not tokenizer.chat_template:
            raise RuntimeError(
                f"Pinned trunk profile {self.config.profile!r} requires a tokenizer chat template."
            )
        return tokenizer

    def _validate_loaded_architecture(self, model: PreTrainedModel) -> None:
        try:
            profile = TRUNK_PROFILES[self.config.profile]
        except KeyError as exc:
            raise RuntimeError(f"Unknown loaded trunk profile {self.config.profile!r}.") from exc
        expected = {
            "model_type": "qwen3",
            "hidden_size": profile.hidden_size,
            "num_hidden_layers": profile.num_hidden_layers,
            "num_attention_heads": profile.num_attention_heads,
            "num_key_value_heads": profile.num_key_value_heads,
        }
        mismatches = {
            name: (getattr(model.config, name, None), value)
            for name, value in expected.items()
            if getattr(model.config, name, None) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Loaded checkpoint does not match trunk profile {profile.name!r}: {mismatches}."
            )

    def _freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Layer access
    # ------------------------------------------------------------------ #

    @property
    def layers(self) -> nn.ModuleList:
        """Return the actual ``nn.ModuleList`` of decoder layers.

        This is the canonical place to read and write trunk layers. **Do not
        wrap the result in ``list(...)``** -- that detaches the reference and
        mutations on the returned object will not propagate to the forward
        graph. The audit found this exact bug in the previous codebase.
        """
        inner = self.model.model  # Qwen3Model
        layers = inner.layers
        if not isinstance(layers, nn.ModuleList):
            raise RuntimeError(
                f"Expected model.model.layers to be nn.ModuleList, got {type(layers).__name__}. "
                f"This adapter assumes the standard Qwen3 layout."
            )
        return layers

    def num_layers(self) -> int:
        return len(self.layers)

    def replace_layer(self, index: int, replacement: nn.Module) -> None:
        """Swap the decoder layer at ``index`` with ``replacement`` in place.

        Verifies the identity post-swap: ``self.layers[index] is replacement``.
        """
        if not 0 <= index < self.num_layers():
            raise IndexError(f"Layer index {index} out of range [0, {self.num_layers()})")
        src_layer = self.layers[index]
        device = next(src_layer.parameters()).device
        dtype = next(src_layer.parameters()).dtype
        replacement.to(device=device, dtype=dtype)
        self.layers[index] = replacement
        if self.layers[index] is not replacement:
            raise RuntimeError(
                f"Layer replacement at index {index} failed identity check; "
                f"the swap did not land on model.model.layers."
            )

    def take_upper_layers(self, fork_layer: int) -> tuple[Qwen3DecoderLayer, ...]:
        """Detach the pretrained upper layers for physical parameter banking.

        The returned modules are authoritative initialization sources.  The
        underlying Hugging Face model is truncated in place so the removed
        shared upper path cannot accidentally remain executable or consume
        device memory alongside the physical decoder.
        """

        total_layers = self.num_layers()
        if type(fork_layer) is not int or not 0 < fork_layer < total_layers:
            raise ValueError(
                f"fork_layer must lie strictly inside [0, {total_layers}); "
                f"got {fork_layer!r}."
            )
        raw_upper = tuple(self.layers[fork_layer:])
        if not raw_upper or any(
            not isinstance(layer, Qwen3DecoderLayer)
            for layer in raw_upper
        ):
            raise TypeError("Physical decoder initialization requires vanilla Qwen3 upper layers.")
        upper = tuple(
            cast(Qwen3DecoderLayer, layer)
            for layer in raw_upper
        )
        self.model.model.layers = nn.ModuleList(self.layers[:fork_layer])
        if self.num_layers() != fork_layer:
            raise RuntimeError("Shared Qwen trunk truncation did not land in the forward graph.")
        return upper

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def frozen_parameters(self) -> List[torch.nn.Parameter]:
        """Frozen shared-trunk parameters after the upper layers are detached."""

        parameters = list(self.model.parameters())
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("A frozen base-trunk parameter unexpectedly requires gradients.")
        return parameters

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward_shared(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor],
        use_cache: bool,
        exact_causal_mask: bool = False,
    ) -> SharedTrunkOutput:
        """Execute only the frozen layers below the physical decoder fork."""

        if input_ids.ndim != 2:
            raise ValueError("Shared trunk input_ids must have shape [rows, tokens].")
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.model.config)
        if cache_position is None:
            past_seen = (
                int(past_key_values.get_seq_length())
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.size(1),
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        full_attention_mask = create_causal_mask(**mask_kwargs)
        if full_attention_mask is not None and not isinstance(
            full_attention_mask,
            torch.Tensor,
        ):
            raise TypeError(
                "The canonical Qwen SDPA path requires a tensor causal mask or None."
            )
        causal_masks: dict[str, Optional[torch.Tensor]] = {
            "full_attention": full_attention_mask,
        }
        if self.model.model.has_sliding_layers:
            sliding_attention_mask = create_sliding_window_causal_mask(
                **mask_kwargs
            )
            if sliding_attention_mask is not None and not isinstance(
                sliding_attention_mask,
                torch.Tensor,
            ):
                raise TypeError(
                    "The canonical Qwen SDPA path requires a tensor sliding mask or None."
                )
            causal_masks["sliding_attention"] = sliding_attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
        for raw_layer in self.layers:
            decoder_layer = cast(Qwen3DecoderLayer, raw_layer)
            attention_type = cast(str, decoder_layer.attention_type)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_masks[attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                pdt_exact_causal_mask=exact_causal_mask,
            )
            LOGGER.debug(
                "Shared lower layer %d completed for shape=%s.",
                int(decoder_layer.self_attn.layer_idx),
                tuple(hidden_states.shape),
            )
        return SharedTrunkOutput(
            hidden_states=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            causal_masks=causal_masks,
            position_embeddings=position_embeddings,
        )


def _resolve_dtype(alias: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[alias]
    except KeyError as exc:
        raise ValueError(f"Unsupported trunk dtype {alias!r}. Known: {sorted(_DTYPE_MAP)}") from exc
