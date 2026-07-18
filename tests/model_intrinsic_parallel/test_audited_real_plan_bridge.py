from __future__ import annotations

from pathlib import Path

from pdt.config.schemas import TRUNK_PROFILES
from pdt.datasets.audited_real_plan import (
    build_audited_real_plan_example,
    load_audited_catalog,
    load_catalog_raw_revision,
    load_training_example,
)
from pdt.datasets.real_plan_schema import FactRole, validate_real_plan_example


class _CountingTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text
        assert add_special_tokens is False
        return [1] * 4_000


def test_all_audited_examples_bridge_to_canonical_real_plan() -> None:
    root = Path.cwd()
    entries = load_audited_catalog(
        root
        / "data/model_intrinsic_parallel/candidates/trainer_record_catalog.jsonl"
    )
    assert len(entries) == 4
    dependency_counts: dict[str, int] = {}
    for entry in entries:
        source = load_training_example(
            root / "data/model_intrinsic_parallel/examples" / entry.example_file
        )
        converted = build_audited_real_plan_example(
            example=source,
            catalog=entry,
            raw_revision=load_catalog_raw_revision(root, entry),
            tokenizer=_CountingTokenizer(),
            trunk_profile=TRUNK_PROFILES["qwen3_4b_instruct_2507"],
        )
        validate_real_plan_example(converted)
        assert converted.source.source_id == source.source_document.source.source_id
        assert len(converted.facts.facts) == len(source.facts)
        assert [target.text for target in converted.teacher.target_sections] == [
            lane.target_text for lane in source.lanes
        ]
        assert [
            sum(
                label.role is FactRole.OWNER
                for label in target.fact_labels
            )
            for target in converted.teacher.target_sections
        ] == [
            sum(label.role == "OWNER" for label in lane.fact_labels)
            for lane in source.lanes
        ]
        dependency_counts[entry.example_file] = sum(
            len(node.dependencies)
            for plan in converted.teacher.plans
            for node in plan.nodes
        )

    assert dependency_counts == {
        "baltimore_railroad_strike_1877.json": 3,
        "battle_of_sluys.json": 1,
        "blackwater_fire_of_1937.json": 0,
        "great_stink.json": 2,
    }
