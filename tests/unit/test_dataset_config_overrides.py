"""Tests for DatasetBuildConfig notes LLM overrides."""

from parallel_decoder_transformer.datasets import (
    DatasetBuildConfig,
    LLMConfig,
    resolve_notes_llm_config,
)


def test_notes_llm_default_is_none():
    cfg = DatasetBuildConfig()
    assert cfg.notes_llm is None
    assert resolve_notes_llm_config(cfg) is cfg.llm


def test_resolve_notes_llm_uses_override():
    cfg = DatasetBuildConfig()
    override = LLMConfig()
    override.openai.model = "gpt-4o"
    cfg.notes_llm = override

    selected = resolve_notes_llm_config(cfg)

    assert selected is override
    assert selected.openai.model == "gpt-4o"
