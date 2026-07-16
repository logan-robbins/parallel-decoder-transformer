"""Finite-alphabet contracts for dynamic PDT messages."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pdt.config.schemas import (
    LossWeights,
    NotesBusConfig,
    PDTConfig,
    SidecarConfig,
    SpeculationHeadConfig,
)
from pdt.runtime.dnb_bus import DynamicNotesBus
from pdt.sidecar.heads.speculation import SpeculationHead
from pdt.sidecar.product_vq import ProductVectorQuantizer
from pdt.training.losses import compute_pdt_losses


def test_product_vq_capacity_is_exact_and_straight_through_trainable() -> None:
    torch.manual_seed(5)
    quantizer = ProductVectorQuantizer(width=8, num_codebooks=2, codes_per_codebook=4)
    vectors = torch.randn(3, 8, requires_grad=True)
    output = quantizer(vectors)

    assert output.capacity_bits == 4
    assert output.indices.shape == (3, 2)
    assert output.assignment_logits.shape == (3, 2, 4)
    assert bool(((0 <= output.indices) & (output.indices < 4)).all())
    selected = torch.stack(
        [
            quantizer.codebook[group, output.indices[:, group]]
            for group in range(quantizer.num_codebooks)
        ],
        dim=1,
    ).reshape_as(vectors)
    torch.testing.assert_close(output.quantized.detach(), selected.detach())

    loss = output.quantized.square().mean() + output.commitment_loss + output.codebook_loss
    loss.backward()
    assert vectors.grad is not None and torch.count_nonzero(vectors.grad) > 0
    assert quantizer.codebook.grad is not None
    assert torch.count_nonzero(quantizer.codebook.grad) > 0


def test_canonical_speculation_head_emits_four_uint8_equivalent_codes() -> None:
    head = SpeculationHead(
        SpeculationHeadConfig(
            hidden_size=16,
            notes_dim=8,
            num_codebooks=4,
            codes_per_codebook=256,
        )
    )
    output = head(torch.randn(2, 3, 16))

    assert output.quantized.shape == (2, 3, 8)
    assert output.indices.shape == (2, 3, 4)
    assert output.capacity_bits == 32
    assert bool(((0 <= output.indices) & (output.indices < 256)).all())


def test_dynamic_bus_rejects_any_message_without_the_exact_code_tuple() -> None:
    bus = DynamicNotesBus(
        NotesBusConfig(snapshot_dim=8, lag=1, dtype="float32"),
        producers=("stream_0",),
        device=torch.device("cpu"),
        codec=ProductVectorQuantizer(width=8, num_codebooks=4, codes_per_codebook=256),
    )
    bus.seed_anchor("stream_0", torch.zeros(8))

    with pytest.raises(ValueError, match="tuple length"):
        bus.publish(
            "stream_0",
            published_block=0,
            stride=32,
            code_indices=(1, 2, 3),
        )
    with pytest.raises(ValueError, match=r"\[0, 256\)"):
        bus.publish(
            "stream_0",
            published_block=0,
            stride=32,
            code_indices=(1, 2, 3, 256),
        )

    snapshot = bus.publish(
        "stream_0",
        published_block=0,
        stride=32,
        code_indices=(1, 2, 3, 4),
    )
    assert snapshot.code_indices == (1, 2, 3, 4)


def test_config_rejects_nonintegral_or_mismatched_dynamic_rate() -> None:
    sidecar = SidecarConfig(
        speculation_head=replace(
            SpeculationHeadConfig(),
            num_codebooks=3,
        )
    )
    with pytest.raises(ValueError, match="divisible by num_codebooks"):
        PDTConfig(sidecar=sidecar).validate()

    sidecar = SidecarConfig(
        speculation_head=replace(
            SpeculationHeadConfig(),
            codes_per_codebook=255,
        )
    )
    with pytest.raises(ValueError, match="power of two"):
        PDTConfig(sidecar=sidecar).validate()

    bus = replace(NotesBusConfig(), num_codebooks=2)
    with pytest.raises(ValueError, match="num_codebooks must match"):
        PDTConfig(runtime=replace(PDTConfig().runtime, notes_bus=bus)).validate()


def test_dynamic_usage_loss_penalizes_collapsed_codebooks() -> None:
    balanced = torch.full((4, 2, 4), -10.0)
    collapsed = torch.full((4, 2, 4), -10.0)
    for sample in range(4):
        balanced[sample, :, sample] = 10.0
    collapsed[:, :, 0] = 10.0
    weights = LossWeights(
        lm_ce=0.0,
        kd_lm=0.0,
        planner_vq_commit=0.0,
        planner_vq_codebook=0.0,
        dynamic_vq_commit=0.0,
        dynamic_vq_codebook=0.0,
        planner_codebook_usage=0.0,
        dynamic_codebook_usage=1.0,
        stream_classifier=0.0,
    )

    balanced_loss = compute_pdt_losses(
        stage=0,
        weights=weights,
        dynamic_vq_logits=balanced,
    )
    collapsed_loss = compute_pdt_losses(
        stage=0,
        weights=weights,
        dynamic_vq_logits=collapsed,
    )

    assert balanced_loss.dynamic_codebook_usage < 1e-6
    assert collapsed_loss.dynamic_codebook_usage > 0.99
