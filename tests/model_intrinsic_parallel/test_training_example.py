from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from model_intrinsic_parallel.training_example import (
    TrainingExample,
    load_training_example,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIRECTORY = ROOT / "data" / "model_intrinsic_parallel" / "examples"
EXAMPLES = {
    "baltimore_railroad_strike_1877": {
        "facts": 24,
        "owner_counts": {"s00": 8, "s01": 8, "s02": 8},
        "dependencies": 3,
        "target_tokens": {"s00": 765, "s01": 777, "s02": 771},
    },
    "blackwater_fire_of_1937": {
        "facts": 18,
        "owner_counts": {"s00": 6, "s01": 6, "s02": 6},
        "dependencies": 0,
        "target_tokens": {"s00": 761, "s01": 731, "s02": 705},
    },
    "battle_of_sluys": {
        "facts": 18,
        "owner_counts": {"s00": 6, "s01": 6, "s02": 6},
        "dependencies": 1,
        "target_tokens": {"s00": 812, "s01": 751, "s02": 733},
    },
    "great_stink": {
        "facts": 18,
        "owner_counts": {"s00": 6, "s01": 6, "s02": 6},
        "dependencies": 2,
        "target_tokens": {"s00": 771, "s01": 731, "s02": 775},
    },
}


@pytest.mark.parametrize(("name", "expected"), EXAMPLES.items())
def test_real_inspection_examples_satisfy_compiled_contract(
    name: str,
    expected: dict[str, object],
) -> None:
    example = load_training_example(EXAMPLE_DIRECTORY / f"{name}.json")

    assert example.creation_method == "manual-source-grounded-inspection"
    assert example.empirical_status == "data-contract-inspection-only"
    assert example.teacher_audit.article_quality == "featured-article"
    assert len(example.facts) == expected["facts"]
    assert len(example.lanes) == 3
    assert len(example.allowed_physical_bindings) == 6
    assert example.presentation_order == ("s00", "s01", "s02")
    assert len(example.dependencies) == expected["dependencies"]
    assert {
        lane.lane_id: sum(label.role == "OWNER" for label in lane.fact_labels)
        for lane in example.lanes
    } == expected["owner_counts"]
    assert {
        lane_id: target.token_count
        for lane_id, target in example.tokenization.decoder_targets.items()
    } == expected["target_tokens"]


def test_compiled_example_rejects_token_mutation() -> None:
    example = load_training_example(
        EXAMPLE_DIRECTORY / "baltimore_railroad_strike_1877.json"
    )
    mutated = example.model_dump(mode="json")
    mutated["tokenization"]["decoder_targets"]["s00"]["input_ids"][0] += 1

    with pytest.raises(ValidationError, match="input_ids_sha256"):
        TrainingExample.model_validate(mutated)


def test_compiled_example_rejects_future_lane_dependency() -> None:
    example = load_training_example(
        EXAMPLE_DIRECTORY / "baltimore_railroad_strike_1877.json"
    )
    mutated = example.model_dump(mode="json")
    mutated["presentation_order"] = ["s01", "s00", "s02"]

    with pytest.raises(ValidationError, match="violates presentation order"):
        TrainingExample.model_validate(mutated)
