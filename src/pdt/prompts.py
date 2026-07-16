"""Pure canonical user-text builders shared by data, training, and runtime."""

from __future__ import annotations

from collections.abc import Sequence


AddressedText = tuple[str, str]
CompletedBlocks = Sequence[Sequence[AddressedText]]


def planner_user_text(shared_context: str) -> str:
    """Return the shared-only planner user message."""

    return _required_text(shared_context, "shared_context")


def stream_user_text(
    shared_context: str,
    stream_id: str,
    local_observation: str,
) -> str:
    """Return one addressed stream's private Instruct user message."""

    shared = planner_user_text(shared_context)
    stream = _required_text(stream_id, "stream_id").lower()
    local = _required_text(local_observation, "local_observation")
    return f"{shared}\n\nPrivate observation:\n[{stream}]\n{local}"


def block_observation_text(block_index: int, observation: str) -> str:
    """Return one temporally addressed private observation body."""

    if type(block_index) is not int or block_index < 0:
        raise ValueError(f"block_index must be a non-negative integer, got {block_index!r}.")
    text = _required_text(observation, "observation")
    return f"[block={block_index}]\n{text}"


def stream_observation_update_text(
    stream_id: str,
    block_index: int,
    observation: str,
) -> str:
    """Return the user turn that reveals one later private observation."""

    stream = _required_text(stream_id, "stream_id").lower()
    body = block_observation_text(block_index, observation)
    return f"Private observation update:\n[{stream}]\n{body}"


def privileged_teacher_user_text(
    shared_context: str,
    observations: Sequence[AddressedText],
    completed_blocks: CompletedBlocks = (),
) -> str:
    """Return the full-context teacher message at one synchronization block."""

    shared = planner_user_text(shared_context)
    if not observations:
        raise ValueError("observations must contain at least one addressed stream.")
    normalized = _normalize_addressed(observations, value_name="local observation")
    rendered_observations = "\n\n".join(f"[{stream}]\n{text}" for stream, text in normalized)
    result = f"{shared}\n\nPrivate stream observations:\n{rendered_observations}"
    if completed_blocks:
        result += "\n\nCompleted prior stream blocks:\n" + completed_blocks_transcript(
            completed_blocks
        )
    return result


def sequential_oracle_user_text(
    shared_context: str,
    receiver_stream: str,
    observations: Sequence[AddressedText],
    completed_blocks: CompletedBlocks = (),
) -> str:
    """Return a causal full-information prompt for one explicit receiver."""

    receiver = _required_text(receiver_stream, "receiver_stream").lower()
    normalized = _normalize_addressed(observations, value_name="local observation")
    if receiver not in {stream for stream, _ in normalized}:
        raise ValueError(f"receiver_stream {receiver!r} has no visible observation.")
    context = privileged_teacher_user_text(
        shared_context,
        normalized,
        completed_blocks=completed_blocks,
    )
    return (
        f"{context}\n\nReceiver to continue: [{receiver}]. "
        "Write only this receiver's next document block."
    )


def completed_blocks_transcript(completed_blocks: CompletedBlocks) -> str:
    """Serialize prior outputs in explicit block-major, addressed order."""

    rendered: list[str] = []
    expected_streams: tuple[str, ...] | None = None
    for block_idx, addressed in enumerate(completed_blocks):
        if not addressed:
            raise ValueError(f"completed block {block_idx} contains no streams.")
        normalized = _normalize_addressed(
            addressed,
            value_name=f"completed block {block_idx} text",
        )
        streams = tuple(stream for stream, _ in normalized)
        if expected_streams is None:
            expected_streams = streams
        elif streams != expected_streams:
            raise ValueError(
                "Every completed block must use the same addressed stream order: "
                f"expected {expected_streams}, got {streams}."
            )
        rendered.extend(
            f"[block={block_idx} stream={stream}]\n{text.lstrip()}" for stream, text in normalized
        )
    if not rendered:
        raise ValueError("completed_blocks must contain at least one block.")
    return "\n\n".join(rendered)


def _normalize_addressed(
    addressed: Sequence[AddressedText],
    *,
    value_name: str,
) -> tuple[AddressedText, ...]:
    normalized: list[AddressedText] = []
    seen: set[str] = set()
    for stream_id, value in addressed:
        stream = _required_text(stream_id, "stream_id").lower()
        if stream in seen:
            raise ValueError(f"Duplicate addressed stream {stream!r}.")
        seen.add(stream)
        normalized.append((stream, _required_text(value, value_name)))
    return tuple(normalized)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text.")
    return value.strip()


__all__ = [
    "block_observation_text",
    "completed_blocks_transcript",
    "planner_user_text",
    "privileged_teacher_user_text",
    "sequential_oracle_user_text",
    "stream_user_text",
    "stream_observation_update_text",
]
