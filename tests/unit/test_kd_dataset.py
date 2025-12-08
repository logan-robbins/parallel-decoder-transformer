from __future__ import annotations

import json

import torch

from parallel_decoder_transformer.training.dataset import KDJsonlDataset


def test_kd_dataset_decodes_snapshots(tmp_path) -> None:
    record = {
        "student_ids": [1, 2, 3],
        "student_labels": [1, 2, 3],
        "planner_ids": [4, 5, 6],
        "notes_student": [[0.1, 0.1], [0.2, 0.2]],
        "notes_teacher": [[0.9, 0.9], [1.0, 1.0]],
        "notes_schema_version": "2.0",
        "true_notes": [
            {
                "stream": "intro",
                "ENT": [
                    {
                        "id": "ent-intro",
                        "name": "Context",
                        "aliases": [],
                        "type": "entity",
                        "canonical": True,
                    }
                ],
                "FACT": [],
                "COVERAGE": [
                    {"plan_item_id": "set context", "status": "missing"},
                    {"plan_item_id": "Intro summary", "status": "covered"},
                ],
            },
            {
                "stream": "core",
                "ENT": [
                    {
                        "id": "ent-core",
                        "name": "Tactics",
                        "aliases": ["strategies"],
                        "type": "entity",
                        "canonical": True,
                    }
                ],
                "FACT": [
                    {
                        "subj_id": "ent-core",
                        "predicate": "involves",
                        "object": "criticisms",
                        "certainty": 0.9,
                        "evidence_span": {
                            "start": 0,
                            "end": 10,
                            "text": "Discuss tactics and criticisms",
                        },
                    }
                ],
                "COVERAGE": [
                    {"plan_item_id": "Discuss tactics and criticisms", "status": "covered"},
                    {"plan_item_id": "Core summary", "status": "partial"},
                ],
            },
            {
                "stream": "wrap",
                "ENT": [],
                "FACT": [],
                "COVERAGE": [
                    {"plan_item_id": "Summarise key takeaways", "status": "missing"},
                    {"plan_item_id": "Wrap summary", "status": "missing"},
                ],
            },
        ],
        "stream": "core",
        "teacher_snapshots": [
            {
                "notes": [[0.9, 0.9], [1.0, 1.0]],
                "stride": 0,
                "version": 1,
                "stream": "core",
                "coverage_flags": [1, 0],
            }
        ],
        "student_snapshots": [
            {
                "notes": [[0.3, 0.3], [0.4, 0.4]],
                "stride": 0,
                "version": 1,
                "stream": "core",
            }
        ],
        "agreement_labels": [1],
        "stride_ids": [0],
        "commit_points": [2],
        "plan_tokens": [
            "intro::set the context",
            "core::Discuss tactics and criticisms",
            "wrap::Summarise key takeaways",
        ],
        "notes_tokens": ["discuss", "core", "tactics"],
        "example_id": "abc-123",
        "metadata": {
            "document_text": "Intro paragraph.\n\nCore paragraph describing tactics.",
            "document_paragraphs": ["Intro paragraph.", "Core paragraph describing tactics."],
            "coverage_provenance": {
                "method": "teacher_llm",
                "schema_version": "2.0",
                "strength": "confirmed",
                "confirmed": True,
                "confirmed_plan_items": {
                    "intro": ["set context", "Intro summary"],
                    "core": ["Discuss tactics and criticisms", "Core summary"],
                    "wrap": ["Summarise key takeaways", "Wrap summary"],
                },
            },
            "teacher_plan": {
                "plan": [
                    {"stream": "intro", "summary": "Intro summary", "notes": ["set context"]},
                    {
                        "stream": "core",
                        "summary": "Core summary",
                        "notes": ["Discuss tactics and criticisms"],
                    },
                    {
                        "stream": "wrap",
                        "summary": "Wrap summary",
                        "notes": ["Summarise key takeaways"],
                    },
                ],
                "segments": [
                    {"stream": "intro", "paragraph_start": 0, "paragraph_end": 1},
                    {"stream": "core", "paragraph_start": 1, "paragraph_end": 2},
                    {"stream": "wrap", "paragraph_start": 2, "paragraph_end": 3},
                ],
            },
            "teacher_notes": {
                "intro": ["set context"],
                "core": ["Discuss tactics and criticisms"],
                "wrap": ["Summarise key takeaways"],
            },
        },
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = KDJsonlDataset(path)
    example = dataset[0]

    assert example["student_ids"].dtype == torch.long
    assert example["notes_teacher"].shape == torch.Size([2, 2])
    assert len(example["teacher_snapshots"]) == 1
    snapshot = example["teacher_snapshots"][0]
    assert snapshot.notes.shape == torch.Size([2, 2])
    assert example["agreement_labels"].shape == torch.Size([1])
    assert example["example_id"] == "abc-123"
    assert example["plan_items"] == ["Discuss tactics and criticisms", "Core summary"]
    assert example["plan_catalog"] == [
        "set context",
        "Intro summary",
        "Discuss tactics and criticisms",
        "Core summary",
        "Summarise key takeaways",
        "Wrap summary",
    ]
    assert example["plan_catalog_streams"] == [
        "stream_intro",
        "stream_intro",
        "stream_core",
        "stream_core",
        "stream_wrap",
        "stream_wrap",
    ]
    assert example["coverage_targets"] == [0.0, 1.0, 1.0, 0.5, 0.0, 0.0]
    assert example["coverage_supervision_mask"] == [True, True, True, True, True, True]
    assert "tactics" in example["notes_text"]
    assert example["raw_teacher_notes"]["stream_core"] == ["Discuss tactics and criticisms"]


def test_kd_dataset_marks_hint_coverage(tmp_path) -> None:
    record = {
        "student_ids": [1, 2],
        "planner_ids": [1, 2],
        "notes_student": [[0.1, 0.2], [0.3, 0.4]],
        "notes_teacher": [[0.5, 0.6], [0.7, 0.8]],
        "stream": "intro",
        "true_notes": [
            {
                "stream": "intro",
                "ENT": [],
                "FACT": [],
                "COVERAGE": [
                    {"plan_item_id": "item one", "status": "covered"},
                    {"plan_item_id": "item two", "status": "missing"},
                ],
            }
        ],
        "metadata": {
            "document_text": "Doc",
            "document_paragraphs": ["Doc"],
            "teacher_plan": {
                "plan": [
                    {"stream": "intro", "summary": "item two", "notes": ["item one"]},
                ]
            },
            "coverage_provenance": {
                "method": "text_hint",
                "schema_version": "2.0",
                "strength": "hint",
            },
        },
    }
    path = tmp_path / "hint.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = KDJsonlDataset(path)
    example = dataset[0]
    assert example["coverage_targets"] == [1.0, 0.0]
    assert example["coverage_supervision_mask"] == [False, False]
