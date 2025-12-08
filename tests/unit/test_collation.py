import json
from pathlib import Path

import numpy as np
import pytest

from parallel_decoder_transformer.datasets.collation import CollateConfig, DatasetCollator


class _DummyTokenizer:
    pad_token = 0
    eos_token = 0

    def __call__(
        self, text: str, max_length: int, truncation: bool, padding: str, return_tensors: str
    ):
        del text, truncation, padding, return_tensors
        return {"input_ids": np.zeros((1, max_length), dtype=int)}


def test_dataset_collator_writes_splits(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    notes_dir = tmp_path / "notes" / "qa"
    notes_dir.mkdir(parents=True)
    plans_dir = tmp_path / "plans" / "qa"
    plans_dir.mkdir(parents=True)

    plan_payload = {
        "sample_id": "qa_sample",
        "domain": "qa",
        "input_text": "context paragraph",
        "streams": [
            {
                "stream_id": "stream_1",
                "header": "Part 1",
                "summary": "S",
                "entities": [],
                "constraints": [],
            }
        ],
    }
    plan_path = plans_dir / "qa_sample.json"
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    notes_payload = {
        "sample_id": "qa_sample",
        "domain": "qa",
        "plan_path": str(plan_path),
        "true_notes": [{"stream_id": "stream_1", "ENT": [], "FACT": [], "COVERAGE": []}],
        "speculative_notes": [
            {"variant_id": "v1", "notes": [], "z_hat": "", "noise_config": {}, "lag_delta": 1}
        ],
        "z_n": "answer",
        "z_hat": ["answer"],
        "lag_delta": 1,
        "note_cadence_M": 6,
        "rollback": {"triggered": False, "events": []},
        "kl_divergence": 0.0,
    }
    notes_path = notes_dir / "qa_sample.json"
    notes_path.write_text(json.dumps(notes_payload), encoding="utf-8")

    output_dir = tmp_path / "dataset"
    cfg = CollateConfig(
        notes_dir=tmp_path / "notes", output_dir=output_dir, tokenizer_path=Path("unused")
    )
    collator = DatasetCollator(cfg, tokenizer=_DummyTokenizer())
    exported = collator.collate()

    assert (output_dir / "manifest.json").exists()
    assert "train" in exported
