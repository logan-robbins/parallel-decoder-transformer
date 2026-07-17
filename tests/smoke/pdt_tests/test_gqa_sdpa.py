from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from pdt.trunk.gqa_sdpa import (
    PDT_GQA_SDPA,
    pdt_gqa_sdpa_forward,
    register_pdt_gqa_sdpa,
)


def test_native_gqa_matches_repeated_kv_sdpa_with_incremental_causal_mask() -> None:
    torch.manual_seed(1729)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_query = torch.randn(1, 8, 5, 16, device=device, dtype=dtype)
    base_key = torch.randn(1, 2, 13, 16, device=device, dtype=dtype)
    base_value = torch.randn(1, 2, 13, 16, device=device, dtype=dtype)
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (base_query, base_key, base_value)
    )
    actual_inputs = tuple(
        tensor.detach().clone().requires_grad_()
        for tensor in (base_query, base_key, base_value)
    )
    query_positions = torch.arange(8, 13, device=device)
    key_positions = torch.arange(13, device=device)
    mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))[None, None]

    reference, _ = sdpa_attention_forward(
        module,
        *reference_inputs,
        mask,
        dropout=0.0,
        scaling=16**-0.5,
    )
    actual, _ = pdt_gqa_sdpa_forward(
        module,
        *actual_inputs,
        mask,
        dropout=0.0,
        scaling=16**-0.5,
        pdt_exact_causal_mask=True,
    )

    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    absolute_tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(
        actual,
        reference,
        rtol=tolerance,
        atol=absolute_tolerance,
    )
    reference.square().mean().backward()
    actual.square().mean().backward()
    for reference_tensor, actual_tensor in zip(reference_inputs, actual_inputs):
        assert reference_tensor.grad is not None
        assert actual_tensor.grad is not None
        torch.testing.assert_close(
            actual_tensor.grad,
            reference_tensor.grad,
            rtol=tolerance,
            atol=absolute_tolerance,
        )


def test_native_gqa_registration_and_fail_fast_contracts() -> None:
    register_pdt_gqa_sdpa()
    assert ALL_ATTENTION_FUNCTIONS[PDT_GQA_SDPA] is pdt_gqa_sdpa_forward
    assert ALL_MASK_ATTENTION_FUNCTIONS[PDT_GQA_SDPA] is not None

    query = torch.zeros(1, 8, 1, 4)
    key = torch.zeros(1, 2, 1, 4)
    with pytest.raises(ValueError, match="num_key_value_groups > 1"):
        pdt_gqa_sdpa_forward(
            SimpleNamespace(num_key_value_groups=1),
            query,
            key,
            key,
            None,
        )
