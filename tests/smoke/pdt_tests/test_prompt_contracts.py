"""Canonical prompt text is shared by retokenization and runtime."""

from __future__ import annotations

import pytest

from pdt.prompts import (
    block_observation_text,
    completed_blocks_transcript,
    planner_user_text,
    privileged_teacher_user_text,
    stream_user_text,
    stream_observation_update_text,
)


def test_addressed_prompt_builders_have_one_canonical_serialization():
    assert planner_user_text("shared task") == "shared task"
    assert stream_user_text("shared task", "STREAM_0", "private alpha") == (
        "shared task\n\nPrivate observation:\n[stream_0]\nprivate alpha"
    )
    assert block_observation_text(2, "private gamma") == "[block=2]\nprivate gamma"
    assert stream_observation_update_text("STREAM_0", 2, "private gamma") == (
        "Private observation update:\n[stream_0]\n[block=2]\nprivate gamma"
    )
    observations = (("stream_0", "private alpha"), ("stream_1", "private beta"))
    completed = (
        (("stream_0", "first zero"), ("stream_1", "first one")),
        (("stream_0", "\nsecond zero"), ("stream_1", "\nsecond one")),
    )
    transcript = completed_blocks_transcript(completed)
    assert transcript == (
        "[block=0 stream=stream_0]\nfirst zero\n\n"
        "[block=0 stream=stream_1]\nfirst one\n\n"
        "[block=1 stream=stream_0]\nsecond zero\n\n"
        "[block=1 stream=stream_1]\nsecond one"
    )
    assert privileged_teacher_user_text("shared task", observations, completed) == (
        "shared task\n\nPrivate stream observations:\n"
        "[stream_0]\nprivate alpha\n\n[stream_1]\nprivate beta\n\n"
        "Completed prior stream blocks:\n" + transcript
    )


def test_prompt_builders_reject_ambiguous_addressing():
    with pytest.raises(ValueError, match="block_index"):
        block_observation_text(-1, "alpha")
    with pytest.raises(ValueError, match="Duplicate addressed stream"):
        privileged_teacher_user_text(
            "shared",
            (("stream_0", "alpha"), ("STREAM_0", "beta")),
        )
    with pytest.raises(ValueError, match="same addressed stream order"):
        completed_blocks_transcript(
            (
                (("stream_0", "a"), ("stream_1", "b")),
                (("stream_1", "c"), ("stream_0", "d")),
            )
        )
