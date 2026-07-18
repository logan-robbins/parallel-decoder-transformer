"""Masked SDPA that preserves Qwen3 grouped-query KV heads physically."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPAParams, can_use_flash_attention
from torch.nn.attention.bias import causal_lower_right
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


PDT_GQA_SDPA = "pdt_gqa_sdpa"


def pdt_gqa_sdpa_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Apply exact masked SDPA without expanding grouped KV heads.

    Transformers' generic SDPA integration repeats KV heads whenever an
    incremental causal mask is present. The repetition is mathematically
    redundant and retains a full expanded prefix for backward. PyTorch's
    native GQA operation accepts the same mask and preserves the physical
    Qwen3 KV-head count.
    """

    if bool(kwargs.get("output_attentions", False)):
        raise ValueError("pdt_gqa_sdpa does not materialize attention weights.")
    if kwargs.get("head_mask") is not None:
        raise ValueError("pdt_gqa_sdpa does not support head_mask.")
    groups = getattr(module, "num_key_value_groups", None)
    if type(groups) is not int or groups <= 1:
        raise ValueError(
            "pdt_gqa_sdpa requires a grouped-query attention module with "
            "num_key_value_groups > 1."
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("pdt_gqa_sdpa expects rank-four query, key, and value tensors.")
    if key.shape != value.shape:
        raise ValueError("pdt_gqa_sdpa requires identical key/value shapes.")
    if query.size(0) != key.size(0) or query.size(-1) != key.size(-1):
        raise ValueError("pdt_gqa_sdpa query and key batch/head widths must match.")
    if query.size(1) != key.size(1) * groups:
        raise ValueError(
            "pdt_gqa_sdpa query heads must equal KV heads times "
            f"num_key_value_groups; got query={query.size(1)}, kv={key.size(1)}, "
            f"groups={groups}."
        )
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 4:
            raise ValueError("pdt_gqa_sdpa requires a rank-four tensor causal mask.")
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    exact_causal_mask = kwargs.get("pdt_exact_causal_mask", False)
    if type(exact_causal_mask) is not bool:
        raise TypeError("pdt_exact_causal_mask must be boolean.")
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None

    if exact_causal_mask:
        if query.size(0) != 1:
            raise ValueError("The exact cached causal path requires batch size one.")
        if query.size(-2) > key.size(-2):
            raise ValueError("Cached causal queries cannot be longer than their KV prefix.")
        if query.is_cuda:
            params = SDPAParams(query, key, value, None, dropout, False, True)
            if not can_use_flash_attention(params):
                raise RuntimeError(
                    "The canonical CUDA pdt_gqa_sdpa path requires fused flash SDPA "
                    "with native grouped-query attention."
                )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_lower_right(query.size(-2), key.size(-2)),
            dropout_p=dropout,
            scale=scaling,
            is_causal=False,
            enable_gqa=True,
        )
    else:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            enable_gqa=True,
        )
    return output.transpose(1, 2).contiguous(), None


def register_pdt_gqa_sdpa() -> None:
    """Register the one canonical attention and matching causal-mask path."""

    ALL_ATTENTION_FUNCTIONS.register(PDT_GQA_SDPA, pdt_gqa_sdpa_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(PDT_GQA_SDPA, sdpa_mask)


__all__ = ["PDT_GQA_SDPA", "pdt_gqa_sdpa_forward", "register_pdt_gqa_sdpa"]
