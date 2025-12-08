"""Deterministic scheduling tests for multi-stream decoding."""

from __future__ import annotations

from parallel_decoder_transformer.inference.scheduler import TriangularScheduler


def test_triangular_scheduler_cycles_streams_per_stride() -> None:
    scheduler = TriangularScheduler(["stream_0", "stream_1", "stream_2"], stride=2)

    # We expect:
    # T=0: stream_0 (0,1)
    # T=1: stream_0 (2,3)
    # T=2: stream_1 (0,1)
    # ...
    stream_sequence: list[str] = []
    token_indices: dict[str, list[int]] = {"stream_0": [], "stream_1": [], "stream_2": []}
    stride_markers: list[int] = []

    for _ in range(6):
        tick = scheduler.tick()
        stream_sequence.append(tick.stream)
        token_indices[tick.stream].append(tick.token_index)
        stride_markers.append(tick.stride_index)
        scheduler.advance()

    assert stream_sequence == [
        "stream_0",
        "stream_0",
        "stream_1",
        "stream_1",
        "stream_2",
        "stream_2",
    ]
    for stream, indices in token_indices.items():
        expected = [0, 1]
        assert indices == expected
    assert stride_markers[0] == stride_markers[1] == 0
    assert stride_markers[2] == stride_markers[3] == 0
    assert stride_markers[4] == stride_markers[5] == 0

    # Next tick should advance to stride 1 and restart cycle.
    tick = scheduler.tick()
    assert tick.stride_index == 1
    assert tick.stream == "stream_0"
    assert tick.token_index == 0
