"""Accelerated GPT-OSS model using PyTorch Flex Attention.

This module monkey-patches the GPT-OSS architecture to use `torch.nn.attention.flex_attention`
instead of the standard eager implementation. This enables FlashAttention-like speeds
on H100s despite the architecture's custom "Sliding Window + Global Sinks" attention mask,
which is not supported by standard FA2 kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import torch
from torch import nn

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssAttention,
        GptOssForCausalLM,
        GptOssModel,
        apply_rotary_pos_emb,
    )
except ImportError:
    # Fallback for environments where transformers is not installed or GPT-OSS is missing
    GptOssForCausalLM = object  # type: ignore
    GptOssModel = object  # type: ignore
    GptOssAttention = object  # type: ignore

    def apply_rotary_pos_emb(q, k, cos, sin):
        return q, k  # type: ignore


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False


def gpt_oss_mask_mod(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    window_size: int = 2048,
    sinks: int = 4,
) -> torch.Tensor:
    """Flex Attention mask for "Sliding Window + Global Sinks".

    Logic:
    1. Causal: q_idx >= kv_idx
    2. Window: q_idx - kv_idx <= window_size
    3. Sinks: kv_idx < sinks (always visible)

    Combined: Causal AND (Window OR Sinks)
    """
    # Causal masking
    causal = q_idx >= kv_idx
    # Sliding window constraint
    window = (q_idx - kv_idx) <= window_size
    # Global sinks are always visible
    is_sink = kv_idx < sinks
    return causal & (window | is_sink)


class AcceleratedGptOssAttention(GptOssAttention):
    """Monkey-patched Attention layer using Flex Attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # If we are not on CUDA or Flex is unavailable, fall back to super()
        if not _FLEX_AVAILABLE or hidden_states.device.type != "cuda":
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

        # Note: Flex Attention currently doesn't support past_key_value cache easily
        # For training (where we have full context), we can use it.
        # For inference, we might need to fall back if caching is required.
        if past_key_values is not None:
            # Fallback for inference/generation which relies on KV cache
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Flex Attention
        window_size = self.sliding_window if self.sliding_window is not None else 2048
        sinks = getattr(self, "sinks", 4)

        def mask_mod_factory(b, h, q, kv):
            return gpt_oss_mask_mod(b, h, q, kv, window_size=window_size, sinks=sinks)

        block_mask = create_block_mask(
            mask_mod_factory,
            B=query_states.shape[0],
            H=query_states.shape[1],
            Q_LEN=query_states.shape[2],
            KV_LEN=key_states.shape[2],
            device=query_states.device,
        )

        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=True,
        )

        # (bsz, num_heads, q_len, head_dim) -> (bsz, q_len, num_heads*head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class AcceleratedGptOssModel(GptOssModel):
    """GPT-OSS model with Accelerated Attention."""

    def _init_acceleration(self):
        """Swap out standard Attention layers for Accelerated ones."""
        for layer in self.layers:
            # We preserve the weights by copying the state dict
            old_attn = layer.self_attn
            new_attn = AcceleratedGptOssAttention(self.config, layer_idx=layer.layer_idx)
            new_attn.load_state_dict(old_attn.state_dict())
            new_attn.to(old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
            layer.self_attn = new_attn


class AcceleratedGptOssForCausalLM(GptOssForCausalLM):
    """GPT-OSS Causal LM with Accelerated Attention."""

    def __init__(self, config):
        super().__init__(config)
        # We need to replace the inner model *after* init
        # But GptOssForCausalLM init creates self.model = GptOssModel(config)
        # We can't easily intercept that call without rewriting __init__.
        # Instead, we can swap the attention layers in-place after construction.
        self._accelerate_attention()

    def _accelerate_attention(self):
        # Iterate over all layers and replace GptOssAttention with AcceleratedGptOssAttention
        for layer in self.model.layers:
            old_attn = layer.self_attn
            new_attn = AcceleratedGptOssAttention(self.config, layer_idx=layer.layer_idx)
            # Transfer weights
            new_attn.load_state_dict(old_attn.state_dict())
            # Ensure device/dtype match
            target_device = old_attn.q_proj.weight.device
            target_dtype = old_attn.q_proj.weight.dtype
            new_attn.to(target_device, dtype=target_dtype)

            layer.self_attn = new_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # We rely on standard loading, then hot-swap the attention layers.
        # Note: attn_implementation="eager" must be passed to the super() call
        # to avoid the transformers library complaining about unsupported SDPA/Flash.
        # We strip our custom "flex" argument if it leaked here.
        if kwargs.get("attn_implementation") == "flex":
            kwargs["attn_implementation"] = "eager"

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Apply the acceleration
        _accelerate_model_in_place(model)

        return model


def _accelerate_model_in_place(model: nn.Module):
    """Replaces GptOssAttention with AcceleratedGptOssAttention in-place."""
    # This function assumes 'model' is a GptOssForCausalLM (or similar)
    # that contains model.model.layers OR model.layers if it's the base model

    # Try to find layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        # Fallback or unknown structure
        return

    for layer in layers:
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, GptOssAttention):
            old_attn = layer.self_attn
            new_attn = AcceleratedGptOssAttention(old_attn.config, layer_idx=old_attn.layer_idx)
            new_attn.load_state_dict(old_attn.state_dict())
            new_attn.to(old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
            layer.self_attn = new_attn
