from parallel_decoder_transformer.datasets.notes_generation import (
    _approximate_kl,
    _noise_config_dict,
)
from parallel_decoder_transformer.datasets.config import SpeculativeNotesNoiseConfig


def test_approximate_kl_penalizes_length_gap() -> None:
    kl = _approximate_kl("answer", ["answer", "answer with drift"])
    assert 0 <= kl <= 1
    kl_zero = _approximate_kl("same", ["same"])
    assert kl_zero == 0.0


def test_noise_config_dict_exposes_ratios() -> None:
    cfg = SpeculativeNotesNoiseConfig(
        paraphrase_ratio=0.2, drop_ratio=0.1, hallucination_ratio=0.05
    )
    data = _noise_config_dict(cfg)
    assert data["paraphrase_ratio"] == 0.2
    assert data["drop_ratio"] == 0.1
