from __future__ import annotations

import torch

from parallel_decoder_transformer.models import StreamAdapterConfig, StreamAdapters


def test_stream_adapter_single_stream() -> None:
    config = StreamAdapterConfig(hidden_size=8, bottleneck_size=2, streams=("intro",))
    adapters = StreamAdapters(config)
    hidden = torch.randn(1, 3, 8)
    output = adapters("intro", hidden)
    assert output.shape == hidden.shape


def test_stream_adapter_batch_streams() -> None:
    config = StreamAdapterConfig(hidden_size=8, bottleneck_size=2)
    adapters = StreamAdapters(config)
    hidden = torch.randn(2, 3, 8)
    stream_ids = torch.tensor([0, 1])
    output = adapters(stream_ids, hidden)
    assert output.shape == hidden.shape
