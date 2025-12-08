"""Deterministic stride scheduler for multi-stream decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass(slots=True)
class ScheduleTick:
    """Descriptor for the next decode action."""

    stride_index: int
    stream: str
    token_index: int  # within current stride (0-based)


@dataclass(slots=True)
class AdvanceOutcome:
    """Result of advancing the scheduler by one token."""

    stream_completed: bool
    stride_completed: bool


class TriangularScheduler:
    """Schedules deterministic stride-aligned decoding for multiple streams.

    Each stream produces exactly ``stride`` tokens per cycle. Streams are decoded in
    the supplied order (stream_0 → stream_1 → stream_2, …). Optional ``levels`` allow
    grouping streams into hierarchical fan-in layers; streams within a level finish
    before the scheduler progresses to the next level.
    """

    def __init__(
        self,
        streams: Sequence[str],
        *,
        stride: int,
        levels: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if not streams:
            raise ValueError("TriangularScheduler requires at least one stream.")
        normalized_streams = tuple(stream.lower() for stream in streams)
        if levels is None:
            level_plan = tuple((stream,) for stream in normalized_streams)
        else:
            if not levels:
                raise ValueError("levels cannot be empty when provided.")
                # unreachable but prevents static analysis flags
            level_plan = tuple(
                tuple(stream.lower() for stream in level if stream is not None) for level in levels
            )
            if any(not level for level in level_plan):
                raise ValueError("levels must not contain empty groups.")
            flattened = [stream for level in level_plan for stream in level]
            if set(flattened) != set(normalized_streams) or len(flattened) != len(
                normalized_streams
            ):
                raise ValueError("levels must list each stream exactly once.")
        self.streams = normalized_streams
        self.stride = stride
        self.levels = level_plan
        self.stride_index = 0
        self._remaining_by_stream: Dict[str, int] = {stream: stride for stream in self.streams}
        self._current_level = 0
        self._current_stream_offset = 0

    def tick(self) -> ScheduleTick:
        stream = self._current_stream()
        consumed = self.stride - self._remaining_by_stream[stream]
        return ScheduleTick(stride_index=self.stride_index, stream=stream, token_index=consumed)

    def advance(self) -> AdvanceOutcome:
        """Mark one token as emitted for the current stream."""

        stream = self._current_stream()
        self._remaining_by_stream[stream] -= 1
        stream_completed = self._remaining_by_stream[stream] == 0
        stride_completed = False
        if stream_completed:
            self._advance_stream_pointer()
            if self._all_streams_completed():
                stride_completed = True
                self._start_next_stride()
        return AdvanceOutcome(stream_completed=stream_completed, stride_completed=stride_completed)

    def stream_progress(self) -> Dict[str, int]:
        """Return tokens produced in the current stride per stream."""

        return {
            stream: self.stride - remaining
            for stream, remaining in self._remaining_by_stream.items()
        }

    def _all_streams_completed(self) -> bool:
        return all(remaining == 0 for remaining in self._remaining_by_stream.values())

    def _start_next_stride(self) -> None:
        self.stride_index += 1
        self._remaining_by_stream = {stream: self.stride for stream in self.streams}
        self._current_level = 0
        self._current_stream_offset = 0

    def _advance_stream_pointer(self) -> None:
        if self._current_level >= len(self.levels):
            return
        self._current_stream_offset += 1
        while self._current_level < len(self.levels):
            level_streams = self.levels[self._current_level]
            if self._current_stream_offset < len(level_streams):
                candidate = level_streams[self._current_stream_offset]
                if self._remaining_by_stream[candidate] > 0:
                    return
                self._current_stream_offset += 1
                continue
            self._current_level += 1
            if self._current_level >= len(self.levels):
                break
            self._current_stream_offset = 0
            next_stream = self.levels[self._current_level][self._current_stream_offset]
            if self._remaining_by_stream[next_stream] > 0:
                return
            self._current_stream_offset += 1

    def _current_stream(self) -> str:
        if self._current_level >= len(self.levels):
            # All streams completed for this stride; restart pointer for next tick.
            self._start_next_stride()
        level_streams = self.levels[self._current_level]
        if self._current_stream_offset >= len(level_streams):
            self._current_stream_offset = 0
        return level_streams[self._current_stream_offset]


__all__ = ["AdvanceOutcome", "ScheduleTick", "TriangularScheduler"]
