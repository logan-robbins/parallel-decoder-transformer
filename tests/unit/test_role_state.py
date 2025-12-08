"""Unit tests for StreamState rollback behaviour."""

from __future__ import annotations

from typing import Tuple

import torch

from parallel_decoder_transformer.inference.state import PastKeyValues, StreamState


def _make_state(
    *, stream: str = "intro", commit_stride: int = 2, commit_horizon: int = 4
) -> StreamState:
    input_ids = torch.tensor([[101, 102]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return StreamState(
        stream=stream,
        input_ids=input_ids,
        attention_mask=attention_mask,
        commit_stride=commit_stride,
        commit_horizon=commit_horizon,
    )


def _dummy_cache(value: float, *, seq_len: int, hidden_size: int = 4) -> PastKeyValues:
    tensor = torch.full((1, seq_len, hidden_size), float(value))
    layer: Tuple[torch.Tensor, torch.Tensor] = (tensor.clone(), tensor.clone())
    return (layer,)


def test_stream_state_rollback_restores_checkpoint() -> None:
    state = _make_state(commit_stride=2, commit_horizon=4)
    expected_cache: PastKeyValues | None = None

    # Seed prompt checkpoint explicitly.
    state.past_key_values = _dummy_cache(0.0, seq_len=state.total_tokens)
    state.register_commit()

    appended_tokens = [11, 12, 13]
    for step, token_id in enumerate(appended_tokens, start=1):
        cache = _dummy_cache(step, seq_len=state.total_tokens + 1)
        state.append_token(token_id, past_key_values=cache)
        state.past_key_values = cache
        state.register_commit()
        if state.tokens_since_commit == 0:
            expected_cache = cache

    removed_tokens, restored_cache = state.rollback()

    assert removed_tokens == [13]
    assert state.generated_tokens == [11, 12]
    assert state.total_tokens == state.commit_pointer == 4
    assert state.tokens_since_commit == 0
    assert state.tokens_since_snapshot == len(appended_tokens) - len(removed_tokens)

    assert restored_cache is not None
    assert expected_cache is not None
    restored_tensor = restored_cache[0][0]
    expected_tensor = expected_cache[0][0]
    assert torch.equal(restored_tensor, expected_tensor)
