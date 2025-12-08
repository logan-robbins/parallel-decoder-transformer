from __future__ import annotations

import torch

from parallel_decoder_transformer.data.collator_kd import (
    TwoBranchKDCollatorConfig,
    TwoBranchKnowledgeDistillationCollator,
)


def test_two_branch_collator_shapes() -> None:
    config = TwoBranchKDCollatorConfig(pad_token_id=0, notes_dim=2)
    collator = TwoBranchKnowledgeDistillationCollator(config)
    batch = [
        {
            "student_ids": torch.tensor([1, 2, 3]),
            "student_labels": torch.tensor([1, 2, 3]),
            "planner_ids": torch.tensor([4, 5]),
            "notes_student": torch.ones(2, 2),
            "notes_teacher": torch.ones(2, 2),
            "teacher_snapshots": [
                {
                    "notes": [[1.0, 1.0], [1.0, 1.0]],
                    "stride": 0,
                    "version": 0,
                    "stream_id": "stream_core",
                }
            ],
            "student_snapshots": [
                {
                    "notes": [[0.5, 0.5], [0.5, 0.5]],
                    "stride": 0,
                    "version": 0,
                    "stream_id": "stream_core",
                }
            ],
            "agreement_labels": [1],
            "plan_items": ["Discuss tactics"],
            "coverage_targets": [1],
            "coverage_supervision_mask": [True],
            "notes_text": "Discuss tactics",
            "stream_id": "stream_core",
        },
        {
            "student_ids": torch.tensor([1, 2]),
            "student_labels": torch.tensor([1, 2]),
            "planner_ids": torch.tensor([6]),
            "notes_student": torch.ones(2, 2),
            "notes_teacher": torch.ones(2, 2),
            "teacher_snapshots": [
                {
                    "notes": [[1.0, 1.0], [1.0, 1.0]],
                    "stride": 0,
                    "version": 0,
                    "stream_id": "stream_intro",
                }
            ],
            "student_snapshots": [
                {
                    "notes": [[0.5, 0.5], [0.5, 0.5]],
                    "stride": 0,
                    "version": 0,
                    "stream_id": "stream_intro",
                }
            ],
            "agreement_labels": [0],
            "plan_items": ["Provide overview"],
            "coverage_targets": [0],
            "coverage_supervision_mask": [False],
            "notes_text": "intro overview",
            "stream_id": "stream_0",
            "metadata": {"teacher_plan": {}},
        },
    ]
    output = collator(batch)
    assert output["input_ids"].shape[0] == 2
    assert output["notes_student"].shape[-1] == 2
    assert torch.all(output["stream_ids"] >= 0)
    assert output["teacher_notes_bus"].shape[0] == 2
    assert output["teacher_notes_bus"].shape[1] == config.max_snapshots
    assert output["teacher_notes_bus"].shape[2] == 2
    assert output["commit_mask"].dtype == torch.bool
    assert output["agreement_labels"].shape == torch.Size([2, config.max_snapshots])
    assert output["coverage_targets"].shape == torch.Size([2, 1])
    assert output["coverage_mask"].tolist() == [[True], [False]]
    assert output["plan_item_ids"].shape[0] == 2
    assert output["plan_item_stream_ids"].shape == output["plan_item_ids"].shape
    assert output["plan_text"][0] == ["Discuss tactics"]


def test_collator_builds_sectional_labels_mask() -> None:
    config = TwoBranchKDCollatorConfig(pad_token_id=0, notes_dim=2, max_length=6)
    collator = TwoBranchKnowledgeDistillationCollator(config)
    base_example = {
        "notes_student": torch.ones(2, 2),
        "notes_teacher": torch.ones(2, 2),
        "teacher_snapshots": [
            {
                "notes": [[1.0, 1.0], [1.0, 1.0]],
                "stride": 0,
                "version": 0,
                "stream_id": "stream_intro",
            }
        ],
        "student_snapshots": [
            {
                "notes": [[0.5, 0.5], [0.5, 0.5]],
                "stride": 0,
                "version": 0,
                "stream_id": "stream_intro",
            }
        ],
        "agreement_labels": [0],
        "plan_items": ["Provide overview"],
        "coverage_targets": [0],
        "coverage_supervision_mask": [False],
        "notes_text": "intro overview",
    }
    sectional_example = dict(
        base_example,
        student_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        student_labels=torch.tensor([10, 11, 12, 13, 14, 15]),
        planner_ids=torch.tensor([6]),
        stream_id="stream_core",
        sectional_independence=True,
        metadata={
            "sectional_independence": True,
            "teacher_plan": {
                "segments": [
                    {"stream": "stream_0", "paragraph_start": 0},
                    {"stream": "stream_2", "paragraph_start": 2},
                ]
            },
            "role_surface_lengths": {
                "stream_intro": 2,
                "stream_core": 3,
                "stream_wrap": 1,
            },
        },
    )
    non_sectional = dict(
        base_example,
        student_ids=torch.tensor([7, 8, 9]),
        student_labels=torch.tensor([7, 8, 9]),
        planner_ids=torch.tensor([3]),
        stream_id="stream_0",
        sectional_independence=False,
    )
    payload = collator([sectional_example, non_sectional])
    labels_mask = payload["labels_mask"]
    assert labels_mask.dtype == torch.bool
    assert labels_mask.shape == payload["labels"].shape
    # Sectional example: intro length 2, so core spans indices 2,3,4.
    assert labels_mask[0].tolist() == [False, False, True, True, True, False]
    # Non-sectional example defaults to non-pad mask.
    assert labels_mask[1].tolist()[:3] == [True, True, True]
