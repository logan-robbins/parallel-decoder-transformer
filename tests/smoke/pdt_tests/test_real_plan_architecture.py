"""Canonical continuous-plan, shared-adapter, and semantic-head contracts."""

from __future__ import annotations

import itertools

import torch

from pdt.config.schemas import (
    PlanMemoryProjectionConfig,
    PlannerHeadConfig,
    SemanticSupervisionConfig,
    PlanAdapterConfig,
)
from pdt.sidecar.adapters import PlanConditionedAdapter
from pdt.sidecar.heads.plan_memory import PlanMemoryProjection
from pdt.sidecar.heads.planner import PlannerHead
from pdt.sidecar.heads.semantic import SemanticSupervisionHeads
from pdt.training.losses import match_unordered_plans


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


def test_plan_memory_preserves_nodes_and_shared_adapter_has_no_lane_parameters() -> None:
    projection = PlanMemoryProjection(
        PlanMemoryProjectionConfig(planner_width=8, notes_dim=4)
    )
    nodes = torch.randn(2, 3, 8, 8)
    node_mask = torch.ones(2, 3, 8, dtype=torch.bool)
    node_mask[:, :, -1] = False
    memory = projection(nodes, node_mask)
    assert memory.shape == (2, 3, 8, 4)
    assert torch.equal(memory[:, :, -1], torch.zeros_like(memory[:, :, -1]))

    adapter = PlanConditionedAdapter(
        PlanAdapterConfig(
            hidden_size=16,
            bottleneck_size=6,
            plan_width=8,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        adapter.up.weight.normal_(std=0.1)
    hidden = torch.randn(3, 5, 16)
    plans = torch.randn(3, 8, 8)
    mask = torch.ones(3, 8, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])
    original = adapter(hidden, plans, mask)
    permuted = adapter(
        hidden.index_select(0, permutation),
        plans.index_select(0, permutation),
        mask.index_select(0, permutation),
    )
    torch.testing.assert_close(permuted, original.index_select(0, permutation))
    assert not any(
        "stream_" in name or "lane" in name
        for name, _ in adapter.named_parameters()
    )


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
