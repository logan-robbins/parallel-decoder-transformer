"""Canonical Qwen-Instruct retokenization for dependency JSONL records.

The token schema stores complete chat-template prompts.  A Qwen assistant
generation marker cannot be split safely across independently encoded shared
and local fragments, so the removed ``shared_ids``/``local_ids`` representation
is rejected rather than emulated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES
from pdt.datasets.document_contract import (
    DOCUMENT_BLOCKS,
    DOCUMENT_BLOCK_TOKENS,
    DOCUMENT_CODEWORDS,
    DOCUMENT_CONTRACT_VERSION,
    DOCUMENT_DEPENDENCY_LAGS,
    DOCUMENT_DEPENDENCY_SCHEDULE,
    DOCUMENT_HISTORY_BLOCKS,
    DOCUMENT_SECTION_ROLES,
    DOCUMENT_TOKENS_PER_STREAM,
)
from pdt.prompts import (
    block_observation_text,
    planner_user_text,
    privileged_teacher_user_text,
    stream_observation_update_text,
    stream_user_text,
)


LEGACY_TOKEN_FIELDS = {"shared_ids", "local_ids"}
LEGACY_LEAKY_FIELDS = {"local_observation"}
EXACT_CODEBOOK_SIZE = 64
EXACT_BITS_PER_CODEWORD = 6
EXACT_NOTES_DIM = 256
EXACT_NOTE_DTYPE_BITS = 16
EXACT_DECODED_NOTE_STORAGE_BITS = EXACT_NOTES_DIM * EXACT_NOTE_DTYPE_BITS
EXACT_DYNAMIC_NOTE_CODEBOOKS = 4
EXACT_CODES_PER_CODEBOOK = 256
EXACT_TRANSMITTED_NOTE_BITS = 32
EXACT_DELTA = 1
EXACT_BLOCK_TOKENS = DOCUMENT_BLOCK_TOKENS


@dataclass(frozen=True, slots=True)
class RetokenizeConfig:
    input_path: Path
    output_path: Path
    trunk_profile: str = DEFAULT_TRUNK_PROFILE
    force: bool = False


def run_retokenize(config: RetokenizeConfig) -> int:
    """Retokenize one JSONL file and return the number of written records."""
    input_path = config.input_path.resolve()
    output_path = config.output_path.resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must differ; source JSONL is immutable.")
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {input_path}")
    if output_path.exists() and not config.force:
        raise FileExistsError(f"output already exists: {output_path}; pass --force to replace it.")
    profile = TRUNK_PROFILES.get(config.trunk_profile)
    if profile is None:
        raise ValueError(
            f"trunk_profile must name one of {tuple(TRUNK_PROFILES)}, "
            f"got {config.trunk_profile!r}."
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        profile.base_model,
        revision=profile.revision,
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("retokenization requires a fast tokenizer for exact offset masks.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"tokenizer {profile.base_model!r} has no chat template; use the locked "
            "Qwen3 Instruct tokenizer."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        input_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, MutableMapping):
                raise ValueError(f"line {line_no}: each JSONL row must be an object.")
            retokenize_record(record, tokenizer, line_no=line_no)
            record["tokenizer"] = profile.base_model
            record["tokenizer_profile"] = profile.name
            record["tokenizer_revision"] = profile.revision
            validate_retokenized_record(record, line_ref=f"line {line_no}")
            dst.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"input JSONL contains no records: {input_path}")
    return count


def retokenize_record(
    record: MutableMapping[str, Any],
    tokenizer: Any,
    *,
    line_no: int,
) -> None:
    """Attach temporally composable student/teacher prompts and target masks."""
    _reject_legacy_token_fields(record, line_ref=f"line {line_no}")
    shared = _required_text(record, "shared_context", line_ref=f"line {line_no}")
    streams = record.get("stream_inputs")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"line {line_no}: stream_inputs must be a non-empty list.")

    observations_by_stream: list[list[str]] = []
    for stream_idx, stream in enumerate(streams):
        if not isinstance(stream, MutableMapping):
            raise ValueError(f"line {line_no}: stream_inputs[{stream_idx}] must be an object.")
        _required_text(
            stream,
            "stream_id",
            line_ref=f"line {line_no} stream {stream_idx}",
        )
        blocks = stream.get("target_blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(
                f"line {line_no} stream {stream_idx}: target_blocks must be non-empty."
            )
        observations_by_stream.append(
            _observation_texts(
                stream,
                expected_blocks=len(blocks),
                context=f"line {line_no} stream {stream_idx}",
            )
        )

    record["planner_prompt_ids"] = _chat_prompt_ids(tokenizer, planner_user_text(shared))
    record["tokenizer"] = str(getattr(tokenizer, "name_or_path", ""))
    record["prompt_schema_version"] = "qwen3-instruct-temporal-chat-v2"
    record["temporal_visibility"] = "one_private_observation_per_block"
    record["block_size_tokens"] = EXACT_BLOCK_TOKENS
    record["chat_template_kwargs"] = {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }

    for stream_idx, stream in enumerate(streams):
        stream_id = str(stream["stream_id"])
        observations = observations_by_stream[stream_idx]
        initial_observation = block_observation_text(0, observations[0])
        stream_text = stream_user_text(shared, stream_id, initial_observation)
        stream["stream_prompt_ids"] = _chat_prompt_ids(tokenizer, stream_text)
        blocks = stream.get("target_blocks")
        assert isinstance(blocks, list)
        for block_idx, block in enumerate(blocks[1:], start=1):
            if not str(block).startswith("\n"):
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: target block {block_idx} must "
                    "start with the canonical newline separator."
                )

        block_texts: list[str] = []
        block_ids: list[list[int]] = []
        block_offsets: list[list[tuple[int, int]]] = []
        for block_idx, block in enumerate(blocks):
            padded_text, ids, offsets = _pad_block_to_tau(
                tokenizer,
                str(block),
                context=f"line {line_no} stream {stream_idx} block {block_idx}",
            )
            block_texts.append(padded_text)
            block_ids.append(ids)
            block_offsets.append(offsets)

        dependency_mask = [[False] * EXACT_BLOCK_TOKENS for _ in block_ids]
        spans = stream.get("dependency_spans", [])
        if not isinstance(spans, list):
            raise ValueError(
                f"line {line_no} stream {stream_idx}: dependency_spans must be a list."
            )
        for span_idx, span in enumerate(spans):
            if not isinstance(span, Mapping):
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: span {span_idx} must be an object."
                )
            block_idx = int(span.get("block_index", -1))
            if not 0 <= block_idx < len(blocks):
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: span block_index {block_idx} "
                    f"outside [0, {len(blocks)})."
                )
            span_text = str(span.get("token_span_text", ""))
            if not span_text:
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: dependency span text is empty."
                )
            block_text = block_texts[block_idx]
            start = block_text.find(span_text)
            if start < 0:
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: dependency span {span_text!r} "
                    f"not found in block {block_idx}."
                )
            if block_text.find(span_text, start + 1) >= 0:
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: dependency span {span_text!r} "
                    f"is ambiguous in block {block_idx}."
                )
            end = start + len(span_text)
            marked = False
            for token_idx, (token_start, token_end) in enumerate(block_offsets[block_idx]):
                if token_start < end and token_end > start:
                    dependency_mask[block_idx][token_idx] = True
                    marked = True
            if not marked:
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: span {span_text!r} aligned to no tokens."
                )

        stream["target_blocks"] = block_texts
        stream["target_block_ids"] = block_ids
        stream["dependency_token_mask"] = dependency_mask
        stream["nondependency_token_mask"] = [
            [not value for value in row] for row in dependency_mask
        ]
        messages: list[dict[str, str]] = [{"role": "user", "content": stream_text}]
        current_prompt = list(stream["stream_prompt_ids"])
        transitions: list[list[int]] = [[]]
        for block_idx, (target_text, target_ids) in enumerate(
            zip(block_texts, block_ids, strict=True)
        ):
            _validate_messages_continuation(
                tokenizer,
                messages=messages,
                prompt_ids=current_prompt,
                continuation_text=target_text,
                continuation_ids=target_ids,
                context=f"line {line_no} stream {stream_idx} block {block_idx}",
            )
            if block_idx + 1 == len(block_texts):
                continue
            messages.append({"role": "assistant", "content": target_text})
            update_text = stream_observation_update_text(
                stream_id,
                block_idx + 1,
                observations[block_idx + 1],
            )
            messages.append({"role": "user", "content": update_text})
            next_prompt = _chat_messages_ids(tokenizer, messages)
            completed_prefix = current_prompt + target_ids
            if next_prompt[: len(completed_prefix)] != completed_prefix:
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: target block {block_idx} "
                    "does not compose into the next multi-turn prompt."
                )
            transition = next_prompt[len(completed_prefix) :]
            if not transition:
                raise ValueError(
                    f"line {line_no} stream {stream_idx}: transition into block "
                    f"{block_idx + 1} tokenized empty."
                )
            transitions.append(transition)
            if completed_prefix + transition != next_prompt:
                raise AssertionError("block transition failed exact prefix reconstruction.")
            current_prompt = next_prompt
        stream["block_transition_ids"] = transitions

    block_count = len(streams[0]["target_blocks"])
    stream_index = {str(stream["stream_id"]): idx for idx, stream in enumerate(streams)}
    teacher_block_prompts: list[list[int]] = []
    for block_idx in range(block_count):
        required_observations = {(idx, block_idx) for idx in range(len(streams))}
        for receiver in streams:
            spans = receiver.get("dependency_spans", [])
            assert isinstance(spans, list)
            for span in spans:
                if not isinstance(span, Mapping) or int(span.get("block_index", -1)) != block_idx:
                    continue
                source_name = str(span.get("source_stream", ""))
                if source_name not in stream_index:
                    raise ValueError(
                        f"line {line_no}: dependency span names unknown source {source_name!r}."
                    )
                source_block = int(span.get("source_block_index", -1))
                if not 0 <= source_block < block_count:
                    raise ValueError(
                        f"line {line_no}: dependency source block {source_block} is out of range."
                    )
                required_observations.add((stream_index[source_name], source_block))
        required_by_stream: dict[int, list[int]] = {}
        for owner, source_block in sorted(required_observations):
            required_by_stream.setdefault(owner, []).append(source_block)
        visible_observations = [
            (
                str(streams[owner]["stream_id"]),
                "\n".join(
                    block_observation_text(source_block, observations_by_stream[owner][source_block])
                    for source_block in source_blocks
                ),
            )
            for owner, source_blocks in required_by_stream.items()
        ]
        teacher_text = privileged_teacher_user_text(
            shared,
            visible_observations,
        )
        prompt = _chat_prompt_ids(tokenizer, teacher_text)
        teacher_block_prompts.append(prompt)
        for stream_idx, stream in enumerate(streams):
            target_text = str(stream["target_blocks"][block_idx])
            target_ids = stream["target_block_ids"][block_idx]
            _validate_messages_continuation(
                tokenizer,
                messages=[{"role": "user", "content": teacher_text}],
                prompt_ids=prompt,
                continuation_text=target_text,
                continuation_ids=target_ids,
                context=f"line {line_no} teacher block {block_idx} stream {stream_idx}",
            )
    record["teacher_block_prompt_ids"] = teacher_block_prompts


def validate_retokenized_record(
    record: Mapping[str, Any],
    *,
    line_ref: str,
    expected_streams: int | None = None,
    expected_tokenizer: str | None = None,
    expected_tokenizer_revision: str | None = None,
) -> None:
    """Validate the exact-entropy tensor contract before collation or auditing."""
    _reject_legacy_token_fields(record, line_ref=line_ref)
    planner = _token_ids(record.get("planner_prompt_ids"), f"{line_ref} planner_prompt_ids")
    if not planner:
        raise ValueError(f"{line_ref}: planner prompt must tokenize non-empty.")
    if record.get("prompt_schema_version") != "qwen3-instruct-temporal-chat-v2":
        raise ValueError(
            f"{line_ref}: missing qwen3-instruct-temporal-chat-v2 prompt schema marker."
        )
    if record.get("temporal_visibility") != "one_private_observation_per_block":
        raise ValueError(f"{line_ref}: temporal visibility contract is missing or invalid.")
    if record.get("chat_template_kwargs") != {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }:
        raise ValueError(f"{line_ref}: chat-template kwargs do not match the locked schema.")
    if int(record.get("block_size_tokens", -1)) != EXACT_BLOCK_TOKENS:
        raise ValueError(f"{line_ref}: block_size_tokens must be exactly {EXACT_BLOCK_TOKENS}.")
    if expected_tokenizer is not None and record.get("tokenizer") != expected_tokenizer:
        raise ValueError(
            f"{line_ref}: tokenizer identity does not match the selected trunk; "
            f"expected {expected_tokenizer!r}, got {record.get('tokenizer')!r}."
        )
    if (
        expected_tokenizer_revision is not None
        and record.get("tokenizer_revision") != expected_tokenizer_revision
    ):
        raise ValueError(
            f"{line_ref}: tokenizer revision does not match the selected trunk; expected "
            f"{expected_tokenizer_revision!r}, got {record.get('tokenizer_revision')!r}."
        )

    lag = int(record.get("visibility_lag_blocks", EXACT_DELTA))
    if lag != EXACT_DELTA:
        raise ValueError(f"{line_ref}: visibility lag must be exactly Delta=1, got {lag}.")
    streams = record.get("stream_inputs")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"{line_ref}: stream_inputs must be a non-empty list.")
    if expected_streams is not None and len(streams) != expected_streams:
        raise ValueError(
            f"{line_ref}: expected exactly {expected_streams} stream_inputs, got {len(streams)}."
        )
    if "k" in record and int(record["k"]) != len(streams):
        raise ValueError(f"{line_ref}: k does not match stream_inputs length.")
    _validate_document_contract(record, streams=streams, line_ref=line_ref)
    first_blocks = streams[0].get("target_blocks") if isinstance(streams[0], Mapping) else None
    if not isinstance(first_blocks, list):
        raise ValueError(f"{line_ref}: first stream is missing target_blocks.")
    teacher_prompts = record.get("teacher_block_prompt_ids")
    if not isinstance(teacher_prompts, list) or len(teacher_prompts) != len(first_blocks):
        raise ValueError(
            f"{line_ref}: teacher_block_prompt_ids must have one prompt per target block."
        )
    for block_idx, prompt in enumerate(teacher_prompts):
        if not _token_ids(prompt, f"{line_ref} teacher block {block_idx} prompt"):
            raise ValueError(f"{line_ref}: teacher block {block_idx} prompt is empty.")

    entropy = record.get("entropy_accounting")
    if entropy is not None:
        _validate_entropy_accounting(entropy, line_ref=line_ref)
        assert isinstance(entropy, Mapping)
        _validate_exact_dependency_spans(
            streams,
            entropy=entropy,
            lag=lag,
            line_ref=line_ref,
        )

    labels: set[str] = set()
    dependency_tokens = 0
    for stream_idx, stream in enumerate(streams):
        if not isinstance(stream, Mapping):
            raise ValueError(f"{line_ref}: stream_inputs[{stream_idx}] must be an object.")
        stream_id = str(stream.get("stream_id", ""))
        if not stream_id or stream_id in labels:
            raise ValueError(f"{line_ref}: stream IDs must be non-empty and unique.")
        labels.add(stream_id)
        prompt = _token_ids(
            stream.get("stream_prompt_ids"),
            f"{line_ref} {stream_id} stream_prompt_ids",
        )
        if not prompt:
            raise ValueError(f"{line_ref} {stream_id}: stream prompt must tokenize non-empty.")
        blocks = stream.get("target_blocks")
        observations = _observation_texts(
            stream,
            expected_blocks=len(blocks) if isinstance(blocks, list) else -1,
            context=f"{line_ref} {stream_id}",
        )
        block_ids = stream.get("target_block_ids")
        transitions = stream.get("block_transition_ids")
        dep_mask = stream.get("dependency_token_mask")
        non_mask = stream.get("nondependency_token_mask")
        if not isinstance(blocks, list) or len(blocks) <= lag:
            raise ValueError(f"{line_ref} {stream_id}: Delta=1 requires at least two blocks.")
        for block_idx, block in enumerate(blocks[1:], start=1):
            if (
                not isinstance(block, str)
                or not block.endswith("\n")
                or block.endswith("\n\n")
                or block.startswith("\n")
            ):
                raise ValueError(
                    f"{line_ref} {stream_id}: target block {block_idx} must end in exactly "
                    "the canonical newline boundary."
                )
        if not isinstance(blocks[0], str) or not blocks[0].endswith("\n"):
            raise ValueError(f"{line_ref} {stream_id}: target block 0 must end in a newline.")
        for name, value in (
            ("target_block_ids", block_ids),
            ("block_transition_ids", transitions),
            ("dependency_token_mask", dep_mask),
            ("nondependency_token_mask", non_mask),
        ):
            if not isinstance(value, list) or len(value) != len(blocks):
                raise ValueError(
                    f"{line_ref} {stream_id}: {name} must have one row per target block."
                )
        assert isinstance(block_ids, list)
        assert isinstance(transitions, list)
        assert isinstance(dep_mask, list)
        assert isinstance(non_mask, list)
        if _token_ids(transitions[0], f"{line_ref} {stream_id} block transition 0"):
            raise ValueError(f"{line_ref} {stream_id}: block transition 0 must be empty.")
        for block_idx, transition in enumerate(transitions[1:], start=1):
            if not _token_ids(
                transition,
                f"{line_ref} {stream_id} block transition {block_idx}",
            ):
                raise ValueError(
                    f"{line_ref} {stream_id}: block transition {block_idx} must be non-empty."
                )
        if len(observations) != len(blocks):
            raise AssertionError("validated observations and target blocks diverged.")
        for block_idx, ids_value in enumerate(block_ids):
            ids = _token_ids(ids_value, f"{line_ref} {stream_id} block {block_idx}")
            if len(ids) != EXACT_BLOCK_TOKENS:
                raise ValueError(
                    f"{line_ref} {stream_id} block {block_idx}: expected exactly "
                    f"{EXACT_BLOCK_TOKENS} target tokens, got {len(ids)}."
                )
            dep = _bool_row(dep_mask[block_idx], len(ids), f"{line_ref} dependency mask")
            non = _bool_row(non_mask[block_idx], len(ids), f"{line_ref} nondependency mask")
            if any(d == n for d, n in zip(dep, non)):
                raise ValueError(
                    f"{line_ref} {stream_id} block {block_idx}: dependency and "
                    "nondependency masks must partition every target token."
                )
            dependency_tokens += sum(dep)
    if dependency_tokens == 0:
        raise ValueError(f"{line_ref}: record contains zero dependency tokens.")


def _chat_prompt_ids(tokenizer: Any, user_text: str) -> list[int]:
    return _chat_messages_ids(tokenizer, [{"role": "user", "content": user_text}])


def _chat_messages_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(ids, Mapping):
        ids = ids.get("input_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise ValueError("chat template did not return a token-id sequence.")
    result = [int(value) for value in ids]
    if not result:
        raise ValueError("chat template returned an empty prompt.")
    return result


def _validate_messages_continuation(
    tokenizer: Any,
    *,
    messages: Sequence[Mapping[str, str]],
    prompt_ids: Sequence[int],
    continuation_text: str,
    continuation_ids: Sequence[int],
    context: str,
) -> None:
    full_ids = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": continuation_text}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if isinstance(full_ids, Mapping):
        full_ids = full_ids.get("input_ids")
    if not isinstance(full_ids, Sequence) or isinstance(full_ids, (str, bytes)):
        raise ValueError(f"{context}: full chat template did not return token IDs.")
    expected_prefix = [int(value) for value in prompt_ids] + [
        int(value) for value in continuation_ids
    ]
    actual = [int(value) for value in full_ids]
    if actual[: len(expected_prefix)] != expected_prefix:
        raise ValueError(
            f"{context}: prompt + target block IDs are not a prefix of the canonical "
            "chat-template assistant transcript."
        )


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("fast tokenizer did not return input_ids.")
    ids = [int(value) for value in encoded["input_ids"]]
    offsets_value = encoded.get("offset_mapping")
    if not isinstance(offsets_value, Sequence) or len(offsets_value) != len(ids):
        raise ValueError("fast tokenizer did not return one offset per token.")
    offsets = [(int(pair[0]), int(pair[1])) for pair in offsets_value]
    return ids, offsets


def _pad_block_to_tau(
    tokenizer: Any,
    text: str,
    *,
    context: str,
) -> tuple[str, list[int], list[tuple[int, int]]]:
    core = text.lstrip("\n").rstrip("\n")
    if not core.endswith("."):
        raise ValueError(f"{context}: target block must end in a period before padding.")
    shortest = core + "\n"
    shortest_ids, _ = _tokenize_with_offsets(tokenizer, shortest)
    if len(shortest_ids) > EXACT_BLOCK_TOKENS:
        raise ValueError(
            f"{context}: target has {len(shortest_ids)} tokens, exceeding "
            f"tau={EXACT_BLOCK_TOKENS}."
        )
    for candidate in _natural_block_candidates(core):
        ids, offsets = _tokenize_with_offsets(tokenizer, candidate)
        if len(ids) == EXACT_BLOCK_TOKENS:
            return candidate, ids, offsets
    raise ValueError(
        f"{context}: no grammatical completion reached exactly tau={EXACT_BLOCK_TOKENS} "
        f"from {len(shortest_ids)} core tokens."
    )


def _natural_block_candidates(core: str) -> Sequence[str]:
    """Return deterministic prose completions without synthetic filler tokens."""

    candidates = [core + "\n"]
    stem = core[:-1]
    trailing_adverbs = (
        "overall",
        "thereafter",
        "deliberately",
        "continuously",
        "coherently",
    )
    candidates.extend(f"{stem} {adverb}.\n" for adverb in trailing_adverbs)
    decapitalized = core[0].lower() + core[1:]
    candidates.extend(
        f"{discourse}, {decapitalized}\n"
        for discourse in ("Notably", "Importantly", "Consequently", "Meanwhile")
    )
    inline_extensions = (
        "with care",
        "with deliberate continuity",
        "with deliberate editorial continuity",
        "throughout the report",
        "throughout the developing report",
        "for the sections that follow",
        "for all the sections that follow",
        "as the larger argument develops",
        "as the larger document steadily develops",
        "within the report's continuing analysis",
        "within the report's continuing coordinated analysis",
        "with enough context for later synthesis",
        "without breaking the document's continuous reasoning",
    )
    candidates.extend(f"{stem}, {extension}.\n" for extension in inline_extensions)

    subjects = (
        "This passage",
        "The drafting record",
        "The section's argument",
        "The working narrative",
    )
    qualities = (
        "clear",
        "coherent",
        "available",
        "stable and clear",
        "coherent and available",
        "explicit, stable, and available",
    )
    purposes = (
        "for later synthesis",
        "for the sections that follow",
        "as the report develops",
        "within the continuing document",
        "for subsequent cross-section analysis",
        "as the broader argument continues to develop",
    )
    for subject in subjects:
        for quality in qualities:
            for purpose in purposes:
                candidates.append(f"{core} {subject} remains {quality} {purpose}.\n")
    return candidates


def _reject_legacy_token_fields(record: Mapping[str, Any], *, line_ref: str) -> None:
    present = sorted(LEGACY_TOKEN_FIELDS.intersection(record))
    for stream in record.get("stream_inputs", []) or []:
        if isinstance(stream, Mapping):
            present.extend(sorted(LEGACY_TOKEN_FIELDS.intersection(stream)))
            leaky = sorted(LEGACY_LEAKY_FIELDS.intersection(stream))
            if leaky:
                raise ValueError(
                    f"{line_ref}: legacy full-log private fields leak future observations: {leaky}."
                )
    if present:
        raise ValueError(
            f"{line_ref}: removed split-prompt token fields are present: {sorted(set(present))}."
        )


def _observation_texts(
    stream: Mapping[str, Any],
    *,
    expected_blocks: int,
    context: str,
) -> list[str]:
    observations = stream.get("block_observations")
    if not isinstance(observations, list) or len(observations) != expected_blocks:
        raise ValueError(
            f"{context}: block_observations must contain exactly one row per target block."
        )
    texts: list[str] = []
    for block_idx, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != {"block_index", "text"}:
            raise ValueError(
                f"{context}: block observation {block_idx} must contain exactly "
                "block_index and text."
            )
        if type(observation["block_index"]) is not int or observation["block_index"] != block_idx:
            raise ValueError(f"{context}: block observation indices must be contiguous from zero.")
        text = observation["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{context}: block observation {block_idx} text is empty.")
        texts.append(text.strip())
    return texts


def _validate_document_contract(
    record: Mapping[str, Any],
    *,
    streams: Sequence[Any],
    line_ref: str,
) -> None:
    contract = record.get("document_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{line_ref}: document_contract must be an object.")
    expected = {
        "version": DOCUMENT_CONTRACT_VERSION,
        "form": "continuous_expository_prose",
        "question_answering": False,
        "blocks_per_stream": DOCUMENT_BLOCKS,
        "tokens_per_block": DOCUMENT_BLOCK_TOKENS,
        "tokens_per_stream": DOCUMENT_TOKENS_PER_STREAM,
        "history_blocks": DOCUMENT_HISTORY_BLOCKS,
        "dependency_lags": list(DOCUMENT_DEPENDENCY_LAGS),
        "dependency_uses_per_stream": len(DOCUMENT_DEPENDENCY_SCHEDULE),
        "local_control_blocks_per_stream": (
            DOCUMENT_BLOCKS - len(DOCUMENT_DEPENDENCY_SCHEDULE)
        ),
        "source_privacy": "one_private_document_packet_per_stream_and_block",
    }
    mismatches = {
        name: (contract.get(name), value)
        for name, value in expected.items()
        if contract.get(name) != value
    }
    if mismatches:
        raise ValueError(f"{line_ref}: invalid long-form document contract: {mismatches}.")
    if len(streams) != len(DOCUMENT_SECTION_ROLES):
        raise ValueError(
            f"{line_ref}: long-form documents require {len(DOCUMENT_SECTION_ROLES)} streams."
        )
    for stream_idx, stream in enumerate(streams):
        if not isinstance(stream, Mapping):
            raise ValueError(f"{line_ref}: stream_inputs[{stream_idx}] must be an object.")
        if stream.get("section_role") != DOCUMENT_SECTION_ROLES[stream_idx]:
            raise ValueError(
                f"{line_ref}: stream {stream_idx} has an invalid or misplaced section role."
            )
        blocks = stream.get("target_blocks")
        if not isinstance(blocks, list) or len(blocks) != DOCUMENT_BLOCKS:
            raise ValueError(
                f"{line_ref}: every long-form stream must contain {DOCUMENT_BLOCKS} blocks."
            )


def _validate_entropy_accounting(value: Any, *, line_ref: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{line_ref}: entropy_accounting must be an object.")
    codebook = int(value.get("codebook_size", -1))
    bits_per_word = int(value.get("bits_per_codeword", -1))
    slots = int(value.get("slots", 0))
    notes_dim = int(value.get("notes_dim", -1))
    dtype_bits = int(value.get("note_dtype_bits", -1))
    decoded_storage_bits = int(value.get("decoded_note_storage_bits", -1))
    dynamic_codebooks = int(value.get("dynamic_note_codebooks", -1))
    codes_per_codebook = int(value.get("codes_per_codebook", -1))
    transmitted_bits = int(value.get("transmitted_note_bits", -1))
    rho = float(value.get("rho", -1.0))
    if codebook != EXACT_CODEBOOK_SIZE or bits_per_word != EXACT_BITS_PER_CODEWORD:
        raise ValueError(f"{line_ref}: exact corpus requires 64 codewords at 6 bits each.")
    if (
        slots <= 0
        or notes_dim != EXACT_NOTES_DIM
        or dtype_bits != EXACT_NOTE_DTYPE_BITS
        or decoded_storage_bits != EXACT_DECODED_NOTE_STORAGE_BITS
        or dynamic_codebooks != EXACT_DYNAMIC_NOTE_CODEBOOKS
        or codes_per_codebook != EXACT_CODES_PER_CODEBOOK
        or transmitted_bits != EXACT_TRANSMITTED_NOTE_BITS
        or value.get("note_representation") != "product_vq_indices"
        or rho not in (0.0, 1.0)
    ):
        raise ValueError(f"{line_ref}: invalid finite-note transport or rho entropy contract.")
    exact_bits = int(value.get("exact_bits_per_dependency", -1))
    expected_bits = slots * bits_per_word if rho == 1.0 else 0
    if exact_bits != expected_bits:
        raise ValueError(
            f"{line_ref}: exact_bits_per_dependency={exact_bits}, expected {expected_bits}."
        )
    uses = int(value.get("dependency_uses_per_stream", -1))
    expected_uses = len(DOCUMENT_DEPENDENCY_SCHEDULE)
    if uses != expected_uses:
        raise ValueError(
            f"{line_ref}: dependency_uses_per_stream={uses}, expected {expected_uses}."
        )
    total_bits = int(value.get("total_exact_bits_per_stream", -1))
    if total_bits != expected_bits * expected_uses:
        raise ValueError(
            f"{line_ref}: total_exact_bits_per_stream={total_bits}, expected "
            f"{expected_bits * expected_uses}."
        )
    eta = float(value.get("attainable_eta_ceiling", math.nan))
    expected_eta = expected_bits / transmitted_bits
    if not math.isclose(eta, expected_eta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{line_ref}: attainable_eta_ceiling={eta}, expected {expected_eta}.")


def _validate_exact_dependency_spans(
    streams: Sequence[Any],
    *,
    entropy: Mapping[str, Any],
    lag: int,
    line_ref: str,
) -> None:
    slots = int(entropy["slots"])
    rho = float(entropy["rho"])
    exact_bits = slots * EXACT_BITS_PER_CODEWORD if rho == 1.0 else 0
    packets: list[list[list[str]]] = []
    for stream_idx, stream in enumerate(streams):
        assert isinstance(stream, Mapping)
        blocks = stream.get("target_blocks")
        assert isinstance(blocks, list)
        observation_texts = _observation_texts(
            stream,
            expected_blocks=len(blocks),
            context=f"{line_ref} stream {stream_idx}",
        )
        stream_packets: list[list[str]] = []
        for block_idx, observation in enumerate(observation_texts):
            prefix = "private document packet: marker="
            if not observation.startswith(prefix):
                raise ValueError(
                    f"{line_ref} stream {stream_idx}: malformed private document packet "
                    f"observation {block_idx}."
                )
            marker_text = observation[len(prefix) :].split(";", maxsplit=1)[0]
            marker_words = marker_text.split("|")
            if len(marker_words) != slots or any(
                word not in DOCUMENT_CODEWORDS for word in marker_words
            ):
                raise ValueError(
                    f"{line_ref} stream {stream_idx}: packet {block_idx} must contain "
                    f"exactly {slots} codewords."
                )
            stream_packets.append(marker_words)
        packets.append(stream_packets)

    for receiver, stream in enumerate(streams):
        assert isinstance(stream, Mapping)
        blocks = stream["target_blocks"]
        spans = stream.get("dependency_spans")
        assert isinstance(blocks, list)
        if not isinstance(spans, list) or len(spans) != len(DOCUMENT_DEPENDENCY_SCHEDULE):
            raise ValueError(
                f"{line_ref} stream {receiver}: expected exactly "
                f"{len(DOCUMENT_DEPENDENCY_SCHEDULE)} document dependency spans."
            )
        source = (receiver + 1) % len(streams) if rho == 1.0 else receiver
        by_target = {
            int(span.get("block_index", -1)): span
            for span in spans
            if isinstance(span, Mapping)
        }
        if set(by_target) != set(DOCUMENT_DEPENDENCY_SCHEDULE):
            raise ValueError(
                f"{line_ref} stream {receiver}: dependency targets do not match the "
                "registered long-form schedule."
            )
        for expected_block, (source_block, required_lag) in (
            DOCUMENT_DEPENDENCY_SCHEDULE.items()
        ):
            span = by_target[expected_block]
            if not isinstance(span, Mapping):
                raise ValueError(
                    f"{line_ref} stream {receiver}: dependency span must be an object."
                )
            expected_words = packets[source][source_block]
            expected_payload = _render_codewords(expected_words)
            expected_kind = (
                "cross_section_constraint"
                if rho == 1.0
                else "self_section_constraint_null"
            )
            if (
                int(span.get("block_index", -1)) != expected_block
                or str(span.get("source_stream", "")) != f"stream_{source}"
                or int(span.get("source_block_index", -1)) != source_block
                or int(span.get("lag_blocks", -1)) != required_lag
                or str(span.get("kind", "")) != expected_kind
                or int(span.get("exact_bits", -1)) != exact_bits
                or str(span.get("token_span_text", "")) != expected_payload
                or span.get("payload_codewords") != expected_words
                or expected_payload not in str(blocks[expected_block])
            ):
                raise ValueError(
                    f"{line_ref} stream {receiver}: dependency span {expected_block} "
                    "does not match the long-form source/lag/payload contract."
                )


def _render_codewords(words: Sequence[str]) -> str:
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"


def _required_text(record: Mapping[str, Any], field: str, *, line_ref: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{line_ref}: {field} must be a non-empty string.")
    return value


def _token_ids(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of token IDs.")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context} contains an invalid token ID: {item!r}.")
        result.append(item)
    return result


def _bool_row(value: Any, length: int, context: str) -> list[bool]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} length must equal target token count {length}.")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{context} must contain only booleans.")
    return value
