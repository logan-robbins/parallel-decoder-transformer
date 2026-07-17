"""Packed-frontier cache, position, and batch-routing contracts."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pdt.runtime.state import PackedFrontierState, pack_token_rows


def _tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        head_dim=8,
        pad_token_id=0,
    )
    return Qwen3ForCausalLM(config).eval()


@torch.no_grad()
def test_unequal_packed_histories_match_independent_qwen3_caches() -> None:
    """Logical RoPE positions remove prompt/transition padding from semantics."""

    torch.manual_seed(11)
    model = _tiny_qwen3()
    streams = ("stream_0", "stream_1")
    prompts = {
        "stream_0": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "stream_1": torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
    }

    separate_prefills = [model(input_ids=prompts[stream], use_cache=True) for stream in streams]
    separate_caches = [output.past_key_values for output in separate_prefills]

    packed_prompt = pack_token_rows(streams, prompts, pad_token_id=0)
    packed_prefill = model(
        input_ids=packed_prompt.input_ids,
        attention_mask=packed_prompt.valid_mask,
        position_ids=packed_prompt.position_ids,
        cache_position=packed_prompt.cache_position,
        use_cache=True,
    )
    for row, separate in enumerate(separate_prefills):
        torch.testing.assert_close(
            packed_prefill.logits[row, -1],
            separate.logits[0, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    frontier = PackedFrontierState(
        streams=streams,
        attention_mask=packed_prompt.valid_mask,
        past_key_values=packed_prefill.past_key_values,
    )

    generated = {
        "stream_0": torch.tensor([[9]], dtype=torch.long),
        "stream_1": torch.tensor([[10]], dtype=torch.long),
    }
    packed_generated = frontier.prepare_append(generated, pad_token_id=0)
    packed_generated_out = model(
        input_ids=packed_generated.rows.input_ids,
        attention_mask=packed_generated.attention_mask,
        past_key_values=frontier.past_key_values,
        position_ids=packed_generated.rows.position_ids,
        cache_position=packed_generated.rows.cache_position,
        use_cache=True,
    )
    frontier.commit(packed_generated, past_key_values=packed_generated_out.past_key_values)

    separate_generated = []
    for row, stream in enumerate(streams):
        cache = separate_caches[row]
        output = model(
            input_ids=generated[stream],
            attention_mask=torch.ones((1, cache.get_seq_length() + 1), dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
        )
        separate_caches[row] = output.past_key_values
        separate_generated.append(output)
        torch.testing.assert_close(
            packed_generated_out.logits[row, -1],
            output.logits[0, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    transitions = {
        "stream_0": torch.tensor([[11, 12]], dtype=torch.long),
        "stream_1": torch.tensor([[13]], dtype=torch.long),
    }
    packed_transition = frontier.prepare_append(transitions, pad_token_id=0)
    assert packed_transition.rows.valid_mask.tolist() == [[True, True], [False, True]]
    assert packed_transition.rows.position_ids.tolist() == [[4, 5], [0, 6]]
    packed_transition_out = model(
        input_ids=packed_transition.rows.input_ids,
        attention_mask=packed_transition.attention_mask,
        past_key_values=frontier.past_key_values,
        position_ids=packed_transition.rows.position_ids,
        cache_position=packed_transition.rows.cache_position,
        use_cache=True,
    )
    frontier.commit(packed_transition, past_key_values=packed_transition_out.past_key_values)

    for row, stream in enumerate(streams):
        cache = separate_caches[row]
        transition = transitions[stream]
        output = model(
            input_ids=transition,
            attention_mask=torch.ones(
                (1, cache.get_seq_length() + transition.size(1)),
                dtype=torch.bool,
            ),
            past_key_values=cache,
            use_cache=True,
        )
        torch.testing.assert_close(
            packed_transition_out.logits[row, -1],
            output.logits[0, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    assert frontier.physical_length == 8
    assert frontier.logical_lengths.tolist() == [6, 7]


def test_packed_frontier_rejects_non_cache_and_partial_stream_rows() -> None:
    streams = ("stream_0", "stream_1")
    mask = torch.ones((2, 3), dtype=torch.bool)
    try:
        PackedFrontierState(streams=streams, attention_mask=mask, past_key_values=3)
    except TypeError as exc:
        assert "Hugging Face Cache" in str(exc)
    else:
        raise AssertionError("A scalar per-stream-style cache must be rejected.")

    prompts = {"stream_0": torch.tensor([[1]], dtype=torch.long)}
    try:
        pack_token_rows(streams, prompts, pad_token_id=0)
    except ValueError as exc:
        assert "exactly match" in str(exc)
    else:
        raise AssertionError("A partial frontier must be rejected.")


def test_packed_rows_keep_completed_lanes_physical_but_logically_masked() -> None:
    rows = pack_token_rows(
        ("stream_0", "stream_1", "stream_2"),
        {
            "stream_0": torch.tensor([[7]], dtype=torch.long),
            "stream_1": torch.tensor([[0]], dtype=torch.long),
            "stream_2": torch.tensor([[8]], dtype=torch.long),
        },
        pad_token_id=0,
        prior_logical_lengths=torch.tensor([2, 2, 2]),
        active_streams=("stream_0", "stream_2"),
    )
    assert rows.input_ids.shape == (3, 1)
    assert rows.valid_mask.tolist() == [[True], [False], [True]]
    assert rows.position_ids.tolist() == [[2], [0], [2]]
