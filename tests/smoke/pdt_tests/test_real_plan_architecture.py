"""Continuous planning and physical three-decoder architecture contracts."""

from __future__ import annotations

import itertools

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from pdt.config.schemas import (
    InstrumentationConfig,
    PlanMemoryProjectionConfig,
    PlannerHeadConfig,
    SemanticSupervisionConfig,
    SidecarConfig,
    SNCConfig,
    SpeculationHeadConfig,
)
from pdt.sidecar.heads.plan_memory import PlanMemoryProjection
from pdt.sidecar.heads.planner import PlannerHead
from pdt.sidecar.heads.semantic import SemanticSupervisionHeads
from pdt.training.losses import match_unordered_plans
from pdt.trunk.physical_decoder import PhysicalDecoder, PlanMemoryCrossAttention


def test_planner_emits_three_continuous_structured_outlines_with_gradients() -> None:
    planner = PlannerHead(
        PlannerHeadConfig(
            hidden_size=16,
            planner_width=8,
            num_streams=3,
            max_nodes_per_stream=8,
            num_layers=1,
            num_heads=2,
            feedforward_width=32,
            dropout=0.0,
        )
    )
    prompt = torch.randn(2, 11, 16)
    mask = torch.tensor(
        [
            [True] * 11,
            [True] * 8 + [False] * 3,
        ]
    )
    output = planner(prompt, mask)

    assert output.nodes.shape == (2, 3, 8, 8)
    assert output.node_validity_logits.shape == (2, 3, 8)
    assert output.presentation_order_logits.shape == (2, 3)
    assert not hasattr(output, "indices")
    assert not any("codebook" in name for name, _ in planner.named_parameters())

    output.nodes.square().mean().backward()
    assert planner.queries.grad is not None
    assert float(planner.queries.grad.abs().sum()) > 0


def test_exact_six_way_matching_recovers_teacher_physical_axes() -> None:
    teacher = torch.randn(2, 3, 8, 8)
    mask = torch.ones(2, 3, 8, dtype=torch.bool)
    permutations = ((2, 0, 1), (1, 2, 0))
    predicted = torch.stack(
        [teacher[row, permutation] for row, permutation in enumerate(permutations)]
    ).requires_grad_()
    result = match_unordered_plans(
        predicted_nodes=predicted,
        validity_logits=torch.full((2, 3, 8), 10.0, requires_grad=True),
        presentation_order_logits=torch.randn(2, 3, requires_grad=True),
        teacher_nodes=teacher,
        teacher_node_mask=mask,
    )

    torch.testing.assert_close(result.nodes, teacher)
    assert {
        tuple(row.tolist()) for row in result.permutation
    } <= set(itertools.permutations(range(3)))
    result.semantic_loss.backward()
    assert predicted.grad is not None


def test_plan_memory_preserves_nodes_and_hard_routed_reads_follow_plan_rows() -> None:
    projection = PlanMemoryProjection(
        PlanMemoryProjectionConfig(planner_width=8, notes_dim=4)
    )
    nodes = torch.randn(2, 3, 8, 8)
    node_mask = torch.ones(2, 3, 8, dtype=torch.bool)
    node_mask[:, :, -1] = False
    memory = projection(nodes, node_mask)
    assert memory.shape == (2, 3, 8, 4)
    assert torch.equal(memory[:, :, -1], torch.zeros_like(memory[:, :, -1]))

    attention = PlanMemoryCrossAttention(
        hidden_size=16,
        plan_width=4,
        attention_width=8,
        num_heads=2,
        max_nodes=8,
        dropout=0.0,
    )
    with torch.no_grad():
        attention.o_proj.weight.normal_(std=0.1)
    hidden = torch.randn(1, 5, 16).expand(3, -1, -1).clone()
    plans = torch.randn(3, 8, 4)
    mask = torch.ones(3, 8, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])
    original = attention(hidden, plans, mask)
    permuted = attention(
        hidden,
        plans.index_select(0, permutation),
        mask.index_select(0, permutation),
    )
    torch.testing.assert_close(permuted, original.index_select(0, permutation))


def test_upper_qwen_weights_are_three_independent_parameter_banks() -> None:
    qwen = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention", "full_attention"],
    )
    source = Qwen3DecoderLayer(qwen, layer_idx=1)
    sidecar = SidecarConfig(
        hidden_size=16,
        notes_dim=4,
        num_streams=3,
        snc=SNCConfig(
            hidden_size=16,
            notes_dim=4,
            attention_width=8,
            num_heads=2,
        ),
        planner_head=PlannerHeadConfig(
            hidden_size=16,
            planner_width=8,
            num_streams=3,
            max_nodes_per_stream=8,
            num_layers=1,
            num_heads=2,
            feedforward_width=32,
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
        ),
        speculation_head=SpeculationHeadConfig(
            hidden_size=16,
            notes_dim=4,
            num_codebooks=2,
            codes_per_codebook=4,
        ),
    )
    physical = PhysicalDecoder(
        (source,),
        fork_layer=1,
        num_decoders=3,
        sidecar=sidecar,
        instrumentation=InstrumentationConfig(
            instrumented_layer_count=1,
            fork_layer=1,
            target_layers=(1,),
        ),
    )
    q_weight = physical.layers[0].self_attn.q_proj.weight
    assert q_weight.shape == (3, 16, 16)
    torch.testing.assert_close(q_weight[0], q_weight[1])
    before_lane_one = q_weight[1].detach().clone()
    with torch.no_grad():
        q_weight[0, 0, 0].add_(1.0)
    torch.testing.assert_close(q_weight[1], before_lane_one)
    assert not torch.equal(q_weight[0], q_weight[1])


def test_semantic_heads_keep_lane_block_node_and_fact_query_axes() -> None:
    heads = SemanticSupervisionHeads(
        SemanticSupervisionConfig(
            hidden_size=16,
            planner_width=8,
            fact_embedding_dim=12,
            attention_width=8,
            num_fact_roles=3,
            max_facts=24,
            max_nodes_per_stream=8,
            dropout=0.0,
        )
    )
    plans = torch.randn(2, 3, 8, 8)
    facts = torch.randn(2, 24, 12)
    blocks = torch.randn(2, 3, 7, 16)

    assert heads.fact_route_logits(plans, facts).shape == (2, 3, 8, 24)
    assert heads.outline_progress_logits(blocks, plans).shape == (2, 3, 7, 8)
    assert heads.fact_write_logits(blocks, facts).shape == (2, 3, 7, 24, 3)
