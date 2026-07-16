"""Exact entropy, chat retokenization, and CE aggregation contracts."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping

import pytest
from transformers import AutoTokenizer

from pdt.datasets.retokenize import retokenize_record, validate_retokenized_record
from pdt.prompts import (
    block_observation_text,
    privileged_teacher_user_text,
    stream_observation_update_text,
    stream_user_text,
)
from pdt.training.dataset import PDTCollator, PDTDependencyDataset
from scripts.generate_dependency_dataset import (
    BITS_PER_CODEWORD,
    CODEWORDS,
    DEFAULT_BLOCKS,
    DEFAULT_DELTA,
    DEFAULT_SLOTS,
    DECODED_NOTE_STORAGE_BITS,
    DYNAMIC_CODES_PER_CODEBOOK,
    DYNAMIC_NOTE_CODEBOOKS,
    NOTE_DTYPE_BITS,
    NOTES_DIM,
    TRANSMITTED_NOTE_BITS,
    attainable_eta,
    generate_examples,
    validate_generation_args,
)
from scripts.validate_dependency_dataset import CESums, ExampleAudit, aggregate_report


PINNED_QWEN_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
PINNED_QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class _CharacterChatTokenizer:
    """Fast offset tokenizer with a deterministic assistant chat boundary."""

    is_fast = True
    chat_template = "test-template"
    name_or_path = "test-qwen-instruct"

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [
            100 + sum((idx + 1) * ord(char) for idx, char in enumerate(match.group()))
            for match in re.finditer(r"\n|[A-Za-z0-9_]+|[^\w\s]", text)
        ]

    @staticmethod
    def _offsets(text: str) -> list[tuple[int, int]]:
        return [match.span() for match in re.finditer(r"\n|[A-Za-z0-9_]+|[^\w\s]", text)]

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        return {
            "input_ids": self._encode(text),
            "offset_mapping": self._offsets(text),
        }

    def apply_chat_template(
        self,
        messages: list[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        continue_final_message: bool = False,
    ) -> list[int]:
        assert tokenize is True
        assert enable_thinking is False
        rendered: list[int] = []
        for message in messages:
            if message["role"] == "user":
                rendered.extend([1, *self._encode(message["content"]), 2])
            elif message["role"] == "assistant":
                rendered.extend([3, *self._encode(message["content"])])
                if not continue_final_message:
                    rendered.append(4)
            else:
                raise AssertionError(f"unexpected role: {message['role']}")
        if add_generation_prompt:
            rendered.append(3)
        return rendered


def _real_chat_ids(
    tokenizer: object,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    assert isinstance(ids, list)
    return [int(token_id) for token_id in ids]


def _retokenized_record(*, rho: float = 1.0) -> dict[str, object]:
    record = next(generate_examples(num_examples=1, rho=rho, seed=17))
    retokenize_record(record, _CharacterChatTokenizer(), line_no=1)
    validate_retokenized_record(record, line_ref="test", expected_streams=3)
    return record


def _parse_registers(stream: Mapping[str, object]) -> list[list[str]]:
    observations = stream["block_observations"]
    assert isinstance(observations, list)
    return [str(row["text"]).removeprefix("private register: ").split() for row in observations]


def test_exact_entropy_default_and_source_lag_contract() -> None:
    assert len(CODEWORDS) == 64
    assert len(set(CODEWORDS)) == 64
    assert BITS_PER_CODEWORD == 6
    assert DEFAULT_SLOTS == 3
    assert DEFAULT_DELTA == 1
    assert DECODED_NOTE_STORAGE_BITS == NOTES_DIM * NOTE_DTYPE_BITS == 4096
    assert DYNAMIC_NOTE_CODEBOOKS == 4
    assert DYNAMIC_CODES_PER_CODEBOOK == 256
    assert TRANSMITTED_NOTE_BITS == 32
    assert math.isclose(attainable_eta(DEFAULT_SLOTS), 18 / 32)

    record = next(generate_examples(num_examples=1, seed=99))
    assert record["visibility_lag_blocks"] == 1
    assert record["entropy_accounting"] == {
        "codebook_size": 64,
        "bits_per_codeword": 6,
        "slots": 3,
        "exact_bits_per_block": 18,
        "notes_dim": 256,
        "note_dtype_bits": 16,
        "decoded_note_storage_bits": 4096,
        "dynamic_note_codebooks": 4,
        "codes_per_codebook": 256,
        "transmitted_note_bits": 32,
        "note_representation": "product_vq_indices",
        "attainable_eta_ceiling": 18 / 32,
        "rho": 1.0,
    }
    streams = record["stream_inputs"]
    assert isinstance(streams, list)
    registers = [_parse_registers(stream) for stream in streams]
    for receiver, stream in enumerate(streams):
        assert "local_observation" not in stream
        assert [row["block_index"] for row in stream["block_observations"]] == list(
            range(DEFAULT_BLOCKS)
        )
        spans = stream["dependency_spans"]
        blocks = stream["target_blocks"]
        assert len(spans) == DEFAULT_BLOCKS - 1
        for span in spans:
            block_idx = span["block_index"]
            source = (receiver + 1) % 3
            expected = " ".join(registers[source][block_idx - 1])
            assert span["token_span_text"] == expected
            assert span["source_stream"] == f"stream_{source}"
            assert span["source_block_index"] == block_idx - 1
            assert span["exact_bits"] == 18
            assert blocks[block_idx].startswith("\n")


def test_seed_is_reproducible_per_example_and_null_is_self_relay() -> None:
    first = list(generate_examples(num_examples=3, seed=123))
    repeated = list(generate_examples(num_examples=3, seed=123))
    changed = list(generate_examples(num_examples=3, seed=124))
    assert first == repeated
    assert first != changed
    assert len({record["generator"]["seed"] for record in first}) == 3
    assert [record["example_id"] for record in first] == [
        "xdep_train_000000",
        "xdep_train_000001",
        "xdep_train_000002",
    ]
    validation = list(generate_examples(num_examples=3, seed=123, split="validation"))
    assert {record["example_id"] for record in first}.isdisjoint(
        record["example_id"] for record in validation
    )

    null = next(generate_examples(num_examples=1, rho=0.0, seed=123))
    streams = null["stream_inputs"]
    assert isinstance(streams, list)
    registers = [_parse_registers(stream) for stream in streams]
    assert null["entropy_accounting"]["exact_bits_per_block"] == 0
    assert null["entropy_accounting"]["attainable_eta_ceiling"] == 0.0
    for receiver, stream in enumerate(streams):
        for span in stream["dependency_spans"]:
            assert span["source_stream"] == f"stream_{receiver}"
            expected = " ".join(registers[receiver][span["source_block_index"]])
            assert span["token_span_text"] == expected
            assert span["kind"] == "self_register_null"
            assert span["exact_bits"] == 0


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"num_examples": 0}, "num_examples"),
        ({"streams": 1}, "at least 2"),
        ({"blocks": 1}, "must exceed"),
        ({"slots": 0}, "slots must be positive"),
        ({"delta": 0}, "locked to delta=1"),
        ({"delta": 2}, "locked to delta=1"),
        ({"rho": 0.5}, "rho must be exactly"),
        ({"split": ""}, "non-empty"),
    ],
)
def test_generation_boundaries_fail_fast(override: dict[str, object], match: str) -> None:
    kwargs: dict[str, object] = {
        "num_examples": 1,
        "streams": 3,
        "blocks": 8,
        "slots": 3,
        "delta": 1,
        "rho": 1.0,
        "split": "train",
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        validate_generation_args(**kwargs)  # type: ignore[arg-type]


def test_chat_retokenization_masks_exact_span_and_block_boundaries() -> None:
    record = _retokenized_record()
    assert "shared_ids" not in record
    assert record["prompt_schema_version"] == "qwen3-instruct-temporal-chat-v2"
    assert record["temporal_visibility"] == "one_private_observation_per_block"
    assert record["planner_prompt_ids"]
    assert len(record["teacher_block_prompt_ids"]) == DEFAULT_BLOCKS
    for stream in record["stream_inputs"]:
        assert "local_ids" not in stream
        assert "local_observation" not in stream
        assert stream["stream_prompt_ids"]
        assert stream["block_transition_ids"][0] == []
        assert all(stream["block_transition_ids"][idx] for idx in range(1, DEFAULT_BLOCKS))
        concatenated = [token for block in stream["target_block_ids"] for token in block]
        expected_text = "".join(stream["target_blocks"])
        assert concatenated == _CharacterChatTokenizer._encode(expected_text)
        for ids, dep, non in zip(
            stream["target_block_ids"],
            stream["dependency_token_mask"],
            stream["nondependency_token_mask"],
        ):
            assert len(ids) == len(dep) == len(non) == 32
            assert all(is_dep != is_non for is_dep, is_non in zip(dep, non))
        for span in stream["dependency_spans"]:
            block_idx = span["block_index"]
            dep = stream["dependency_token_mask"][block_idx]
            assert sum(dep) == 3


def test_pinned_qwen_multiturn_prefix_and_temporal_horizon_are_exact() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        PINNED_QWEN_TOKENIZER,
        revision=PINNED_QWEN_REVISION,
        local_files_only=True,
        use_fast=True,
    )
    record = next(generate_examples(num_examples=1, seed=73))
    retokenize_record(record, tokenizer, line_no=1)
    validate_retokenized_record(record, line_ref="pinned-qwen", expected_streams=3)

    shared = str(record["shared_context"])
    streams = record["stream_inputs"]
    assert isinstance(streams, list)
    for stream in streams:
        stream_id = str(stream["stream_id"])
        observations = [str(row["text"]) for row in stream["block_observations"]]
        initial_user = stream_user_text(
            shared,
            stream_id,
            block_observation_text(0, observations[0]),
        )
        messages = [{"role": "user", "content": initial_user}]
        current_prompt = _real_chat_ids(tokenizer, messages, add_generation_prompt=True)
        assert current_prompt == stream["stream_prompt_ids"]
        for block_idx, (target_text, target_ids) in enumerate(
            zip(stream["target_blocks"], stream["target_block_ids"], strict=True)
        ):
            completed_prefix = current_prompt + target_ids
            completed_chat = _real_chat_ids(
                tokenizer,
                [*messages, {"role": "assistant", "content": target_text}],
                add_generation_prompt=False,
            )
            assert completed_chat[: len(completed_prefix)] == completed_prefix
            if block_idx + 1 == len(stream["target_blocks"]):
                continue
            messages.extend(
                (
                    {"role": "assistant", "content": target_text},
                    {
                        "role": "user",
                        "content": stream_observation_update_text(
                            stream_id,
                            block_idx + 1,
                            observations[block_idx + 1],
                        ),
                    },
                )
            )
            next_prompt = _real_chat_ids(tokenizer, messages, add_generation_prompt=True)
            assert completed_prefix + stream["block_transition_ids"][block_idx + 1] == next_prompt
            current_prompt = next_prompt

    for block_idx, actual_prompt in enumerate(record["teacher_block_prompt_ids"]):
        visible_observations = [
            (
                str(stream["stream_id"]),
                "\n".join(
                    block_observation_text(
                        observation_idx,
                        stream["block_observations"][observation_idx]["text"],
                    )
                    for observation_idx in range(block_idx + 1)
                ),
            )
            for stream in streams
        ]
        completed = [
            [
                (str(stream["stream_id"]), str(stream["target_blocks"][prior_block]))
                for stream in streams
            ]
            for prior_block in range(block_idx)
        ]
        expected_text = privileged_teacher_user_text(shared, visible_observations, completed)
        assert actual_prompt == _real_chat_ids(
            tokenizer,
            [{"role": "user", "content": expected_text}],
            add_generation_prompt=True,
        )


def test_retokenized_contract_rejects_legacy_tokens_and_zero_dependency_mask() -> None:
    record = _retokenized_record()
    legacy = copy.deepcopy(record)
    legacy["shared_ids"] = [1]
    with pytest.raises(ValueError, match="removed split-prompt"):
        validate_retokenized_record(legacy, line_ref="legacy")

    leaky = copy.deepcopy(record)
    leaky["stream_inputs"][0]["local_observation"] = "future block log"
    with pytest.raises(ValueError, match="leak future observations"):
        validate_retokenized_record(leaky, line_ref="leaky")

    empty = copy.deepcopy(record)
    for stream in empty["stream_inputs"]:
        stream["dependency_token_mask"] = [[False] * len(row) for row in stream["target_block_ids"]]
        stream["nondependency_token_mask"] = [
            [True] * len(row) for row in stream["target_block_ids"]
        ]
    with pytest.raises(ValueError, match="zero dependency tokens"):
        validate_retokenized_record(empty, line_ref="empty")

    wrong_source = copy.deepcopy(record)
    wrong_source["stream_inputs"][0]["dependency_spans"][0]["source_stream"] = "stream_2"
    with pytest.raises(ValueError, match="exact source/lag/payload contract"):
        validate_retokenized_record(wrong_source, line_ref="wrong-source")


def test_temporal_contract_rejects_bad_observations_and_transition_rows() -> None:
    record = _retokenized_record()

    missing_observation = copy.deepcopy(record)
    missing_observation["stream_inputs"][0]["block_observations"].pop()
    with pytest.raises(ValueError, match="exactly one row per target block"):
        validate_retokenized_record(missing_observation, line_ref="missing-observation")

    wrong_observation_index = copy.deepcopy(record)
    wrong_observation_index["stream_inputs"][0]["block_observations"][1]["block_index"] = 7
    with pytest.raises(ValueError, match="contiguous from zero"):
        validate_retokenized_record(wrong_observation_index, line_ref="wrong-index")

    nonempty_row_zero = copy.deepcopy(record)
    nonempty_row_zero["stream_inputs"][0]["block_transition_ids"][0] = [1]
    with pytest.raises(ValueError, match="transition 0 must be empty"):
        validate_retokenized_record(nonempty_row_zero, line_ref="row-zero")

    empty_later_row = copy.deepcopy(record)
    empty_later_row["stream_inputs"][0]["block_transition_ids"][1] = []
    with pytest.raises(ValueError, match="transition 1 must be non-empty"):
        validate_retokenized_record(empty_later_row, line_ref="empty-later")


def test_collator_shapes_and_refuses_empty_or_truncating_batches() -> None:
    record = _retokenized_record()
    collator = PDTCollator(
        pad_token_id=0,
        num_streams=3,
        max_planner_prompt_length=1024,
        max_stream_prompt_length=2048,
        max_block_transition_length=64,
        max_teacher_prompt_length=4096,
        max_blocks=8,
        max_block_length=32,
    )
    batch = collator([record])
    assert batch.planner_prompt_ids.shape == (1, 1024)
    assert batch.stream_prompt_ids.shape == (1, 3, 2048)
    assert batch.block_transition_ids.shape == (1, 3, 8, 64)
    assert not batch.block_transition_attention_mask[:, :, 0].any()
    assert (batch.block_transition_attention_mask[:, :, 1:].sum(dim=-1) > 0).all()
    assert batch.teacher_block_prompt_ids.shape == (1, 8, 4096)
    assert batch.dependency_token_mask.any()
    with pytest.raises(ValueError, match="empty batch"):
        collator([])

    too_short = PDTCollator(
        pad_token_id=0,
        num_streams=3,
        max_planner_prompt_length=1,
        max_stream_prompt_length=2048,
        max_block_transition_length=64,
        max_teacher_prompt_length=4096,
        max_blocks=8,
        max_block_length=32,
    )
    with pytest.raises(ValueError, match="would truncate"):
        too_short([record])

    transition_too_short = PDTCollator(
        pad_token_id=0,
        num_streams=3,
        max_planner_prompt_length=1024,
        max_stream_prompt_length=2048,
        max_block_transition_length=1,
        max_teacher_prompt_length=4096,
        max_blocks=8,
        max_block_length=32,
    )
    with pytest.raises(ValueError, match="block_transition_ids.*would truncate"):
        transition_too_short([record])

    wrong_tau = PDTCollator(
        pad_token_id=0,
        num_streams=3,
        max_planner_prompt_length=1024,
        max_stream_prompt_length=2048,
        max_block_transition_length=64,
        max_teacher_prompt_length=4096,
        max_blocks=8,
        max_block_length=64,
    )
    with pytest.raises(ValueError, match="train/runtime tau must match exactly"):
        wrong_tau([record])


def test_dataset_rejects_empty_jsonl(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no PDT examples"):
        PDTDependencyDataset(path)


def test_ce_report_is_globally_token_weighted() -> None:
    audits = [
        ExampleAudit(
            dependency=CESums(local_nats=4.0, privileged_nats=1.0, tokens=1),
            nondependency=CESums(local_nats=1.0, privileged_nats=1.0, tokens=1),
        ),
        ExampleAudit(
            dependency=CESums(local_nats=3.0, privileged_nats=0.0, tokens=3),
            nondependency=CESums(local_nats=3.0, privileged_nats=3.0, tokens=9),
        ),
    ]
    report = aggregate_report(audits)
    # Per-example averaging would give (3 + 1) / 2 = 2.0.  The identified
    # corpus statistic is the token-weighted (7 - 1) / 4 = 1.5 nats/token.
    assert report["mean_dependency_gap_nats_per_token"] == 1.5
    assert report["dependency_tokens"] == 4
    assert report["nondependency_tokens"] == 10
    assert report["passes"] is True


def test_ce_report_rejects_empty_inputs_and_zero_dependency_tokens() -> None:
    with pytest.raises(ValueError, match="zero examples"):
        aggregate_report([])
    with pytest.raises(ValueError, match="zero dependency tokens"):
        aggregate_report(
            [
                ExampleAudit(
                    dependency=CESums(),
                    nondependency=CESums(local_nats=1.0, privileged_nats=1.0, tokens=1),
                )
            ]
        )
