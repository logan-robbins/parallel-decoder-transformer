"""Real Qwen-layer plumbing for the tensorized physical decoder frontier."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

import pdt.model as model_module
from pdt.config.schemas import (
    InstrumentationConfig,
    PDTConfig,
    PlanMemoryProjectionConfig,
    PlannerHeadConfig,
    SemanticSupervisionConfig,
    SidecarConfig,
    SNCConfig,
    SpeculationHeadConfig,
)
from pdt.model import PDTModel
from pdt.trunk.instrumentation import LayerRuntimeContext
from pdt.trunk.physical_decoder import (
    PhysicalFrontierCache,
    _tiled_grouped_query_attention,
)
from pdt.trunk.qwen3_adapter import Qwen3TrunkAdapter


class _TinyQwenAdapter(Qwen3TrunkAdapter):
    """Use the production adapter methods without loading an external checkpoint."""

    def __init__(self, _config: object) -> None:
        qwen = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            layer_types=["full_attention"] * 4,
            attention_dropout=0.0,
            use_cache=True,
        )
        self.config = SimpleNamespace()
        self.dtype = torch.float32
        self.model = Qwen3ForCausalLM(qwen).to(dtype=torch.bfloat16)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self._freeze()


def _config() -> PDTConfig:
    sidecar = SidecarConfig(
        hidden_size=16,
        notes_dim=4,
        num_streams=3,
        snc=SNCConfig(
            hidden_size=16,
            notes_dim=4,
            attention_width=8,
            num_heads=2,
            dropout=0.0,
        ),
        planner_head=PlannerHeadConfig(
            hidden_size=16,
            planner_width=8,
            num_streams=3,
            max_nodes_per_stream=8,
            num_layers=1,
            num_heads=2,
            feedforward_width=32,
            dropout=0.0,
        ),
        plan_memory_proj=PlanMemoryProjectionConfig(
            planner_width=8,
            notes_dim=4,
        ),
        semantic_supervision=SemanticSupervisionConfig(
            hidden_size=16,
            planner_width=8,
            fact_embedding_dim=1024,
            attention_width=8,
            num_fact_roles=3,
            max_facts=128,
            max_nodes_per_stream=8,
            dropout=0.0,
        ),
        speculation_head=SpeculationHeadConfig(
            hidden_size=16,
            notes_dim=4,
            num_codebooks=2,
            codes_per_codebook=4,
            dropout=0.0,
        ),
    )
    return PDTConfig(
        instrumentation=InstrumentationConfig(
            instrumented_layer_count=2,
            fork_layer=2,
            target_layers=(2, 3),
            snc_gate_init=-4.0,
            plan_gate_init=-4.0,
        ),
        sidecar=sidecar,
    )


def _context(plan_memory: torch.Tensor) -> LayerRuntimeContext:
    return LayerRuntimeContext(
        stream_ids=("stream_0", "stream_1", "stream_2"),
        plan_nodes=torch.randn(3, 8, 8),
        plan_mask=torch.ones(3, 8, dtype=torch.bool),
        plan_memory=plan_memory,
        plan_producer_ids=torch.arange(3).unsqueeze(1).expand(3, 8),
    )


def test_tiled_grouped_attention_matches_full_forward_and_backward() -> None:
    query = torch.randn(3, 4, 257, 4, requires_grad=True)
    key = torch.randn(3, 2, 257, 4, requires_grad=True)
    value = torch.randn(3, 2, 257, 4, requires_grad=True)
    mask = torch.ones(3, 1, 257, 257, dtype=torch.bool).tril()
    full = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=0.5,
        is_causal=False,
        enable_gqa=True,
    )
    full.square().mean().backward()
    full_gradients = (
        query.grad.detach().clone(),
        key.grad.detach().clone(),
        value.grad.detach().clone(),
    )

    tiled_query = query.detach().clone().requires_grad_()
    tiled_key = key.detach().clone().requires_grad_()
    tiled_value = value.detach().clone().requires_grad_()
    tiled = _tiled_grouped_query_attention(
        query=tiled_query,
        key=tiled_key,
        value=tiled_value,
        attention_mask=mask,
        dropout_p=0.0,
        scale=0.5,
    )
    torch.testing.assert_close(tiled, full.detach(), rtol=1e-5, atol=1e-6)
    tiled.square().mean().backward()
    for actual, expected in zip(
        (tiled_query.grad, tiled_key.grad, tiled_value.grad),
        full_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_real_qwen_layers_advance_grouped_private_branch_caches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "Qwen3TrunkAdapter", _TinyQwenAdapter)
    model = PDTModel(_config())
    shared_row_counts: list[int] = []
    original_shared = model.trunk_adapter.forward_shared

    def audited_shared(**kwargs):
        shared_row_counts.append(kwargs["input_ids"].size(0))
        return original_shared(**kwargs)

    monkeypatch.setattr(model.trunk_adapter, "forward_shared", audited_shared)
    assert all(
        parameter.dtype == torch.float32
        for parameter in model.decoder_branch_parameters()
    )
    model.set_runtime_context(_context(torch.randn(3, 8, 4)))
    prompt = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    mask = torch.ones_like(prompt, dtype=torch.bool)

    prefill = model.forward_frontier(
        input_ids=prompt,
        attention_mask=mask,
        use_cache=True,
        output_hidden_states=True,
    )
    assert isinstance(prefill.past_key_values, PhysicalFrontierCache)
    assert shared_row_counts == [1]
    assert prefill.past_key_values.get_seq_length() == 3
    for keys in prefill.past_key_values.branch_keys:
        assert keys is not None
        assert keys.shape[:3] == (1, 3, 2)
        assert keys.size(-2) == 3

    step = model.forward_frontier(
        input_ids=torch.tensor([[4], [5], [6]]),
        attention_mask=torch.ones(3, 4, dtype=torch.bool),
        past_key_values=prefill.past_key_values,
        position_ids=torch.full((3, 1), 3, dtype=torch.long),
        cache_position=torch.tensor([3]),
        use_cache=True,
        output_hidden_states=True,
    )
    assert step.logits.shape == (3, 1, 64)
    assert shared_row_counts == [1, 3]
    assert step.past_key_values.get_seq_length() == 4
    assert all(
        keys is not None and keys.size(-2) == 4
        for keys in step.past_key_values.branch_keys
    )

    loss = step.logits.float().square().mean()
    loss.backward()
    gradient = model.physical_decoder.layers[0].self_attn.q_proj.weight.grad
    assert gradient is not None
    assert gradient.shape[0] == 3
    assert all(float(gradient[lane].abs().sum()) > 0 for lane in range(3))
    assert all(
        keys is not None and keys.dtype == torch.bfloat16
        for keys in step.past_key_values.branch_keys
    )


def test_plan_memory_swap_moves_conditioning_without_swapping_decoder_weights(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "Qwen3TrunkAdapter", _TinyQwenAdapter)
    model = PDTModel(_config()).eval()
    for layer in model.physical_decoder.layers:
        with torch.no_grad():
            layer.plan_attention.o_proj.weight.normal_(std=0.2)
            layer.plan_gate.fill_(4.0)
    prompt = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    mask = torch.ones_like(prompt, dtype=torch.bool)
    plans = torch.randn(3, 8, 4)
    model.set_runtime_context(_context(plans))
    original = model.forward_frontier(
        input_ids=prompt,
        attention_mask=mask,
        use_cache=True,
    ).logits

    permutation = torch.tensor([2, 0, 1])
    model.set_runtime_context(_context(plans.index_select(0, permutation)))
    swapped = model.forward_frontier(
        input_ids=prompt,
        attention_mask=mask,
        use_cache=True,
    ).logits
    torch.testing.assert_close(
        swapped,
        original.index_select(0, permutation),
        rtol=1e-5,
        atol=1e-6,
    )


def test_physical_weight_edit_is_isolated_to_one_decoder(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "Qwen3TrunkAdapter", _TinyQwenAdapter)
    model = PDTModel(_config()).eval()
    model.set_runtime_context(_context(torch.randn(3, 8, 4)))
    prompt = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    mask = torch.ones_like(prompt, dtype=torch.bool)
    original = model.forward_frontier(
        input_ids=prompt,
        attention_mask=mask,
        use_cache=True,
    ).logits

    with torch.no_grad():
        model.physical_decoder.layers[0].self_attn.q_proj.weight[1].add_(0.1)
    changed = model.forward_frontier(
        input_ids=prompt,
        attention_mask=mask,
        use_cache=True,
    ).logits

    torch.testing.assert_close(changed[0], original[0], rtol=0, atol=0)
    torch.testing.assert_close(changed[2], original[2], rtol=0, atol=0)
    assert not torch.equal(changed[1], original[1])


def test_fp32_branch_master_weight_moves_after_bfloat16_forward(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "Qwen3TrunkAdapter", _TinyQwenAdapter)
    model = PDTModel(_config()).train()
    model.trunk_adapter.model.eval()
    model.set_runtime_context(_context(torch.randn(3, 8, 4)))
    parameter = model.physical_decoder.layers[0].self_attn.q_proj.weight
    optimizer = torch.optim.AdamW([parameter], lr=2e-4, weight_decay=0.0)
    before = parameter.detach().clone()
    output = model.forward_frontier(
        input_ids=torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        attention_mask=torch.ones(3, 3, dtype=torch.bool),
        use_cache=True,
    )
    targets = torch.tensor([7, 8, 9])
    loss = F.cross_entropy(output.logits[:, -1].float(), targets)
    loss.backward()
    assert parameter.grad is not None
    assert parameter.grad.dtype == torch.float32
    optimizer.step()
    assert not torch.equal(parameter, before)
