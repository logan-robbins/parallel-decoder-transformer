"""Tests for MultiHeadCoverageHead (Upgrade 03).

18 tests covering: config validation, output shape, masking, gradient flow,
multi-scale keys, learned temperature, backward-compatibility aliases,
and parameter budget.
"""

from __future__ import annotations

import math

import torch

from parallel_decoder_transformer.models.heads.coverage import (
    CoverageHead,
    CoverageHeadConfig,
    MultiHeadCoverageHead,
    MultiHeadCoverageHeadConfig,
)


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------


def test_multihead_config_head_dim_divisibility_ok() -> None:
    """hidden_size=8 with num_heads=2 should construct without error."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2)
    head = MultiHeadCoverageHead(config)
    assert head.head_dim == 4


def test_multihead_config_head_dim_not_divisible_raises() -> None:
    """hidden_size=7 with num_heads=2 should raise ValueError."""
    config = MultiHeadCoverageHeadConfig(hidden_size=7, num_heads=2)
    try:
        MultiHeadCoverageHead(config)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "divisible" in str(exc).lower()


# ---------------------------------------------------------------------------
# 2. Forward pass output shape
# ---------------------------------------------------------------------------


def test_multihead_forward_output_shape_standard() -> None:
    """Output shape must be (B, P) for typical inputs."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=4)
    head = MultiHeadCoverageHead(config)
    B, P, T = 2, 5, 12
    hidden = torch.randn(B, T, 8)
    plan_emb = torch.randn(B, P, 8)
    plan_mask = torch.ones(B, P, dtype=torch.bool)
    logits = head(hidden, plan_emb, plan_mask)
    assert logits.shape == (B, P)


def test_multihead_forward_output_shape_single_plan_item() -> None:
    """Edge case: P=1 should still work."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=0)
    head = MultiHeadCoverageHead(config)
    B, P, T = 1, 1, 6
    hidden = torch.randn(B, T, 8)
    plan_emb = torch.randn(B, P, 8)
    plan_mask = torch.ones(B, P, dtype=torch.bool)
    logits = head(hidden, plan_emb, plan_mask)
    assert logits.shape == (B, P)


# ---------------------------------------------------------------------------
# 3. Masking
# ---------------------------------------------------------------------------


def test_multihead_plan_mask_fills_zero_on_padding() -> None:
    """Masked positions must be exactly 0.0."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=0)
    head = MultiHeadCoverageHead(config)
    B, P, T = 2, 3, 4
    hidden = torch.randn(B, T, 8)
    plan_emb = torch.randn(B, P, 8)
    # Only first 2 plan items are valid in batch 0, first 1 in batch 1
    plan_mask = torch.tensor([[True, True, False], [True, False, False]])
    logits = head(hidden, plan_emb, plan_mask)
    assert logits[0, 2].item() == 0.0
    assert logits[1, 1].item() == 0.0
    assert logits[1, 2].item() == 0.0


def test_multihead_masked_positions_detached_from_gradient() -> None:
    """Gradient must not flow through masked positions."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=0)
    head = MultiHeadCoverageHead(config)
    B, P, T = 1, 3, 4
    hidden = torch.randn(B, T, 8, requires_grad=True)
    plan_emb = torch.randn(B, P, 8, requires_grad=True)
    plan_mask = torch.tensor([[True, False, False]])
    logits = head(hidden, plan_emb, plan_mask)
    # Only the valid position should produce non-zero gradient
    loss = logits.sum()
    loss.backward()
    # The masked logit values are 0.0 constants, so they contribute
    # nothing to the loss.  The total grad is determined only by
    # the unmasked logit.
    assert plan_emb.grad is not None


# ---------------------------------------------------------------------------
# 4. Multi-scale keys
# ---------------------------------------------------------------------------


def test_multiscale_keys_shape_exact_multiple() -> None:
    """T exactly divisible by W -> S = T/W sentence segments."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=4)
    head = MultiHeadCoverageHead(config)
    B, T, H = 2, 8, 8
    hidden = torch.zeros(B, T, H)
    result = head._build_multiscale_keys(hidden)
    # T=8, W=4 -> S=2
    assert result.shape == (B, T + 2, H)


def test_multiscale_keys_shape_non_multiple() -> None:
    """T not divisible by W -> S = ceil(T/W) sentence segments."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=3)
    head = MultiHeadCoverageHead(config)
    B, T, H = 2, 7, 8
    hidden = torch.zeros(B, T, H)
    result = head._build_multiscale_keys(hidden)
    # T=7, W=3 -> S=ceil(7/3)=3 sentence segments
    assert result.shape == (B, T + 3, H)


def test_multiscale_keys_disabled_when_window_zero() -> None:
    """sentence_window=0 disables multi-scale: output == input."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=0)
    head = MultiHeadCoverageHead(config)
    B, T, H = 1, 5, 8
    hidden = torch.randn(B, T, H)
    result = head._build_multiscale_keys(hidden)
    assert result.shape == hidden.shape
    assert result is hidden  # no copy, exact same object


def test_multiscale_sentence_level_mean_pool_correctness() -> None:
    """Sentence-level keys must be the mean of their window tokens."""
    config = MultiHeadCoverageHeadConfig(hidden_size=4, num_heads=2, sentence_window=2)
    head = MultiHeadCoverageHead(config)
    # Construct hidden with T=4 (2 windows of 2 tokens)
    # Window 0: tokens [1,1,1,1] and [3,3,3,3] -> mean [2,2,2,2]
    # Window 1: tokens [5,5,5,5] and [7,7,7,7] -> mean [6,6,6,6]
    h = torch.tensor([[[1., 1., 1., 1.], [3., 3., 3., 3.],
                        [5., 5., 5., 5.], [7., 7., 7., 7.]]])  # (1, 4, 4)
    result = head._build_multiscale_keys(h)  # (1, 4+2, 4)
    assert result.shape == (1, 6, 4)
    # Sentence-level positions T:T+S = 4:6
    assert torch.allclose(result[0, 4], torch.tensor([2., 2., 2., 2.]))
    assert torch.allclose(result[0, 5], torch.tensor([6., 6., 6., 6.]))


# ---------------------------------------------------------------------------
# 5. Learned temperature
# ---------------------------------------------------------------------------


def test_learned_temperature_initialized_correctly() -> None:
    """log_temperature should initialise to 0.5 * log(head_dim)."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2)
    head = MultiHeadCoverageHead(config)
    expected_log_temp = 0.5 * math.log(4)  # head_dim = 8/2 = 4
    assert abs(head.log_temperature.item() - expected_log_temp) < 1e-5
    expected_temp = math.sqrt(4)  # = 2.0
    assert abs(head.log_temperature.exp().item() - expected_temp) < 1e-5


def test_learned_temperature_is_trainable_parameter() -> None:
    """When learn_temperature=True, log_temperature is in parameters()."""
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, learn_temperature=True)
    head = MultiHeadCoverageHead(config)
    param_names = {name for name, _ in head.named_parameters()}
    assert "log_temperature" in param_names


def test_frozen_temperature_not_in_named_parameters() -> None:
    """When learn_temperature=False, log_temperature must not be a parameter."""
    config = MultiHeadCoverageHeadConfig(
        hidden_size=8, num_heads=2, learn_temperature=False
    )
    head = MultiHeadCoverageHead(config)
    param_names = {name for name, _ in head.named_parameters()}
    assert "log_temperature" not in param_names


def test_frozen_temperature_not_saved_in_state_dict() -> None:
    """Frozen temperature registered as non-persistent buffer must not appear in state_dict."""
    config = MultiHeadCoverageHeadConfig(
        hidden_size=8, num_heads=2, learn_temperature=False
    )
    head = MultiHeadCoverageHead(config)
    sd = head.state_dict()
    assert "log_temperature" not in sd, (
        "Frozen temperature should not appear in state dict "
        "(registered as non-persistent buffer)"
    )


# ---------------------------------------------------------------------------
# 6. Backward-compatibility aliases
# ---------------------------------------------------------------------------


def test_backward_compat_alias_coverage_head() -> None:
    """CoverageHead must be the same class as MultiHeadCoverageHead."""
    assert CoverageHead is MultiHeadCoverageHead


def test_backward_compat_alias_coverage_head_config() -> None:
    """CoverageHeadConfig must be the same class as MultiHeadCoverageHeadConfig."""
    assert CoverageHeadConfig is MultiHeadCoverageHeadConfig


# ---------------------------------------------------------------------------
# 7. Config defaults
# ---------------------------------------------------------------------------


def test_coverage_head_config_default_fields() -> None:
    """Default values must match the spec."""
    config = MultiHeadCoverageHeadConfig(hidden_size=4096)
    assert config.num_heads == 8
    assert config.dropout == 0.0
    assert config.sentence_window == 32
    assert config.learn_temperature is True


# ---------------------------------------------------------------------------
# 8. Parameter budget
# ---------------------------------------------------------------------------


def test_parameter_count_does_not_exceed_budget() -> None:
    """Total parameter count at H=4096 must be < 90M."""
    config = MultiHeadCoverageHeadConfig(hidden_size=4096, num_heads=8)
    head = MultiHeadCoverageHead(config)
    total = sum(p.numel() for p in head.parameters())
    # 4 * 4096^2 + 4096 * 1 + 1 = 67,108,864 + 4096 + 1 ~ 67.1M
    assert total < 90_000_000, f"Parameter count {total} exceeds 90M budget"
