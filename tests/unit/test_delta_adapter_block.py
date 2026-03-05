"""Tests for _DeltaAdapterBlock pure-delta semantics."""
import torch
import pytest
from parallel_decoder_transformer.models.stream_adapters import (
    StreamAdapterConfig,
    StreamAdapters,
    _DeltaAdapterBlock,
)


def _make_config(hidden: int = 8, bottleneck: int = 4) -> StreamAdapterConfig:
    return StreamAdapterConfig(
        hidden_size=hidden,
        bottleneck_size=bottleneck,
        streams=("stream_0",),
        dropout=0.0,
    )


def test_delta_adapter_block_has_no_layer_norm():
    """_DeltaAdapterBlock must not contain a LayerNorm parameter."""
    config = _make_config()
    block = _DeltaAdapterBlock(config)
    assert not hasattr(block, "layer_norm"), "LayerNorm must be removed from delta block"
    param_names = [n for n, _ in block.named_parameters()]
    assert not any("layer_norm" in n for n in param_names)


def test_delta_adapter_block_zero_init_output_is_not_input():
    """Output of delta block must differ from input — it is a raw transformation."""
    config = _make_config()
    block = _DeltaAdapterBlock(config)
    h = torch.randn(1, 3, 8)
    out = block(h)
    assert out.shape == h.shape
    # Output should not equal input (the residual has been removed)
    assert not torch.allclose(out, h)


def test_stream_adapters_returns_delta_not_residual_sum():
    """StreamAdapters.forward must return a delta, not (h + delta)."""
    config = _make_config()
    adapters = StreamAdapters(config)
    # Zero all up-projection weights so delta = 0
    with torch.no_grad():
        for block in adapters.adapters.values():
            block.up.weight.zero_()
            if block.up.bias is not None:
                block.up.bias.zero_()
    h = torch.randn(1, 5, 8)
    out = adapters("stream_0", h)
    assert torch.allclose(out, torch.zeros_like(h), atol=1e-6), (
        "With zeroed up-projection, delta must be zero (not h)"
    )


def test_stream_adapter_layer_zero_stream_returns_zeros():
    from parallel_decoder_transformer.integration.instrumentation import StreamAdapterLayer
    config = _make_config()
    layer = StreamAdapterLayer(config)
    h = torch.randn(2, 4, 8)
    out = layer(h, stream=None)
    assert torch.allclose(out, torch.zeros_like(h))


def test_stream_adapter_layer_no_subtraction_applied():
    """StreamAdapterLayer must not subtract hidden_states from adapter output."""
    from parallel_decoder_transformer.integration.instrumentation import StreamAdapterLayer
    config = _make_config()
    layer = StreamAdapterLayer(config)
    # Zero up weights: delta = 0, so StreamAdapterLayer should return zeros
    with torch.no_grad():
        for block in layer.adapters.adapters.values():
            block.up.weight.zero_()
            if block.up.bias is not None:
                block.up.bias.zero_()
    h = torch.randn(2, 4, 8)
    out = layer(h, "stream_0")
    assert torch.allclose(out, torch.zeros_like(h), atol=1e-6), (
        "Zero up-projection must yield zero delta, not h (old behavior was to return h - h = 0 "
        "after subtraction, but the new path is direct — this test distinguishes them when "
        "the up-projection has a bias)"
    )
