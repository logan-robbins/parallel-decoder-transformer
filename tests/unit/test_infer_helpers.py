from __future__ import annotations

from parallel_decoder_transformer.inference import CounterfactualConfig

from scripts import infer as infer_cli


def test_parse_cadence_overrides_handles_all_stream() -> None:
    overrides = infer_cli._parse_cadence_overrides(
        ["all=4", "stream_1=2"], ("stream_1", "stream_2")
    )
    assert overrides["stream_1"] == 2
    assert overrides["stream_2"] == 4


def test_summarize_counterfactuals_includes_all_toggles() -> None:
    cfg = CounterfactualConfig(
        swap_pairs=(("stream_1", "stream_2"),),
        shuffle_streams=("stream_1",),
        freeze_streams=("stream_2",),
        ablate_streams=("stream_3",),
        stale_overrides={"stream_1": 2},
        default_stale_extra=1,
        tag="demo",
    )
    tags = infer_cli._summarize_counterfactuals(cfg)
    assert any(tag.startswith("ablate:") for tag in tags)
    assert any(tag.startswith("swap:") for tag in tags)
    assert any(tag.startswith("stale_default") for tag in tags)
