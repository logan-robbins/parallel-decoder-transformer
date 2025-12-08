from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from parallel_decoder_transformer.datasets.kd_export import KDExportConfig, KDExporter


def _write_parquet_row(path: Path) -> None:
    plan_entries = [
        {
            "stream_id": "stream_1",
            "summary": "Outline the context",
            "notes_contract": ["Provide historical context"],
            "section_contract": {"type": "section_index_range", "start_idx": 1, "end_idx": 1},
        },
        {
            "stream_id": "stream_2",
            "summary": "Explain the impact",
            "notes_contract": ["Discuss downstream effects"],
            "section_contract": {"type": "section_index_range", "start_idx": 2, "end_idx": 2},
        },
    ]
    true_notes = [
        {"stream_id": "stream_1", "ENT": [], "FACT": [], "COVERAGE": []},
        {"stream_id": "stream_2", "ENT": [], "FACT": [], "COVERAGE": []},
    ]
    spec_notes = [
        {
            "variant_id": "sample-1_var0",
            "z_hat": "draft",
            "lag_delta": 1,
            "notes": [
                {"stream_id": "stream_1", "ENT": [], "FACT": [], "COVERAGE": []},
                {"stream_id": "stream_2", "ENT": [], "FACT": [], "COVERAGE": []},
            ],
        }
    ]
    versioned_notes = [
        {
            "snapshot_id": 0,
            "source": "plan_contract",
            "lag_delta": 0,
            "notes": [
                {"stream_id": "stream_1", "ENT": [], "FACT": [], "COVERAGE": []},
                {"stream_id": "stream_2", "ENT": [], "FACT": [], "COVERAGE": []},
            ],
        }
    ]
    row = {
        "sample_id": "sample-1",
        "domain": "qa",
        "x_text": "Paragraph one.\n\nParagraph two.",
        "plan_text": json.dumps(plan_entries),
        "notes_true": json.dumps(true_notes),
        "notes_speculative": json.dumps(spec_notes),
        "notes_versioned": json.dumps(versioned_notes),
        "z_n": "final answer",
        "z_hat": json.dumps(["draft"]),
        "rollback_flags": json.dumps({"triggered": False}),
        "lag_delta": 1,
        "note_cadence_M": 4,
        "kl_divergence": 0.5,
        "sectional_independence": True,
        "x_tokens": [1, 2, 3],
        "plan_tokens": [4, 5, 6],
        "z_n_tokens": [7, 8, 9],
        "z_hat_tokens": [10, 11, 12],
    }
    table = pa.Table.from_pylist([row])
    pq.write_table(table, path)


def test_exporter_emits_records(tmp_path) -> None:
    dataset_dir = tmp_path / "data" / "datasets" / "demo_run"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = dataset_dir / "train.parquet"
    _write_parquet_row(parquet_path)

    processed_dir = tmp_path / "data" / "processed" / "demo_run"
    config = KDExportConfig(
        dataset_dir=dataset_dir,
        output_dir=processed_dir,
        splits=("train",),
        notes_dim=8,
    )
    exporter = KDExporter(config)
    counts = exporter.export()

    assert counts["train"] == 2  # one record per stream
    output_path = processed_dir / "kd.jsonl"
    lines = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    sample_record = lines[0]
    assert sample_record["student_ids"] == [10, 11, 12]
    assert sample_record["planner_ids"] == [4, 5, 6]
    assert sample_record["metadata"]["teacher_plan"]["plan"][0]["stream_id"] == "stream_1"
    assert len(sample_record["teacher_snapshots"]) >= 1
