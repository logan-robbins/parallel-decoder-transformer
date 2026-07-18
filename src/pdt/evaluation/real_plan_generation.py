"""Held-out long-form generation evaluation for the physical three-lane PDT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import torch

from pdt.checkpoint import load_checkpoint
from pdt.config import load_config
from pdt.datasets.real_plan_retokenize import (
    MAX_TARGET_TOKENS,
    MIN_TARGET_TOKENS,
    render_planner_prompt,
)
from pdt.datasets.immutable_io import write_bytes_new
from pdt.datasets.real_plan_schema import (
    FactRole,
    RealPlanExample,
    validate_real_plan_example,
)
from pdt.evaluation.fact_ownership import (
    FactOwnershipCounts,
    NLI_MODEL,
    NLI_MODEL_REVISION,
    NliEntailmentScorer,
    calibrate_entailment_threshold,
    fact_ownership_counts,
)
from pdt.evaluation.paired_causal import bootstrap_mean
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import CounterfactualConfig
from pdt.runtime.orchestrator import MultiStreamOrchestrator, OrchestrationResult
from pdt.training.dataset import RealPlanDataset
from pdt.training.losses import match_unordered_plans


CONDITIONS = ("oracle_plan", "learned_plan", "plan_zero", "lane_swap")

__all__ = [
    "CONDITIONS",
    "GenerationEvaluationConfig",
    "long_form_statistics",
    "remap_role_targets",
    "run_generation_evaluation",
    "swap_teacher_to_physical",
]


@dataclass(frozen=True, slots=True)
class GenerationEvaluationConfig:
    config_path: Path
    checkpoint_path: Path
    calibration_raw_path: Path
    calibration_tokenized_path: Path
    evaluation_raw_path: Path
    evaluation_tokenized_path: Path
    output_path: Path
    max_new_tokens: int = MAX_TARGET_TOKENS
    device: str | None = None
    coordination_source: Literal["bus", "self_only"] | None = None
    entailment_device: str = "cuda"
    entailment_batch_size: int = 64


@dataclass(frozen=True, slots=True)
class _ExamplePair:
    raw: RealPlanExample
    tokenized: Mapping[str, object]


def run_generation_evaluation(config: GenerationEvaluationConfig) -> dict[str, object]:
    """Run the four required conditions and write one auditable JSON result."""

    _validate_paths(config)
    pdt_config = load_config(config.config_path)
    if config.coordination_source is not None:
        pdt_config.instrumentation.coordination_source = config.coordination_source
        pdt_config.validate()
    calibration = _load_pairs(
        config.calibration_raw_path,
        config.calibration_tokenized_path,
        expected_tokenizer=pdt_config.trunk.base_model,
        expected_tokenizer_revision=pdt_config.trunk.revision,
    )
    evaluation = _load_pairs(
        config.evaluation_raw_path,
        config.evaluation_tokenized_path,
        expected_tokenizer=pdt_config.trunk.base_model,
        expected_tokenizer_revision=pdt_config.trunk.revision,
    )
    overlap = {
        pair.raw.source.source_id for pair in calibration
    } & {
        pair.raw.source.source_id for pair in evaluation
    }
    if overlap:
        raise ValueError(
            "Calibration and held-out generation examples must be disjoint; "
            f"overlap={sorted(overlap)}."
        )

    entailment_scorer = NliEntailmentScorer(
        device=config.entailment_device,
        batch_size=config.entailment_batch_size,
    )
    threshold, calibration_metrics = _calibrate_threshold(
        calibration,
        scorer=entailment_scorer,
    )

    device = _resolve_device(config.device or pdt_config.training.device)
    model = PDTModel(pdt_config)
    metadata = load_checkpoint(config.checkpoint_path, model)
    active_conditions = CONDITIONS if metadata.stage >= 1 else ("oracle_plan",)
    trunk_model: torch.nn.Module = model.trunk_adapter.model
    trunk_model.to(device)
    model.to(device)
    model.eval()
    model.trunk_adapter.model.eval()

    document_payloads: list[dict[str, object]] = []
    document_counts: dict[str, list[FactOwnershipCounts]] = {
        condition: [] for condition in active_conditions
    }
    lane_swap_unmoved_counts: list[FactOwnershipCounts] = []
    for pair in evaluation:
        payload, counts, unmoved = _evaluate_example(
            pair,
            model=model,
            pdt_config=pdt_config,
            scorer=entailment_scorer,
            threshold=threshold,
            max_new_tokens=config.max_new_tokens,
            include_planner_conditions=metadata.stage >= 1,
        )
        document_payloads.append(payload)
        for condition in active_conditions:
            document_counts[condition].append(counts[condition])
        if unmoved is not None:
            lane_swap_unmoved_counts.append(unmoved)

    inference = {
        condition: _summarize_condition(
            document_counts[condition],
            bootstrap_samples=pdt_config.training.causal_eval_bootstrap_samples,
            confidence_level=pdt_config.training.causal_eval_confidence_level,
            seed=pdt_config.training.causal_eval_seed + index * 10,
        )
        for index, condition in enumerate(active_conditions)
    }
    checks = _evidence_checks(
        document_payloads=document_payloads,
        counts=document_counts,
        lane_swap_unmoved_counts=lane_swap_unmoved_counts,
        minimum_documents=pdt_config.training.causal_eval_min_documents,
        bootstrap_samples=pdt_config.training.causal_eval_bootstrap_samples,
        confidence_level=pdt_config.training.causal_eval_confidence_level,
        seed=pdt_config.training.causal_eval_seed + 100,
        active_conditions=active_conditions,
    )
    result: dict[str, object] = {
        "schema_version": "pdt-real-plan-generation-eval-v2",
        "checkpoint": {
            "path": str(config.checkpoint_path),
            **asdict(metadata),
        },
        "device": str(device),
        "coordination_source": pdt_config.instrumentation.coordination_source,
        "max_new_tokens_per_lane": config.max_new_tokens,
        "physical_lane_order": list(pdt_config.runtime.streams),
        "fact_entailment_model": NLI_MODEL,
        "fact_entailment_model_revision": NLI_MODEL_REVISION,
        "automatic_entailment_screen": {
            "not_final_evidence": True,
            "examples": len(calibration),
            "threshold": threshold,
            **calibration_metrics,
        },
        "human_fact_audit_required": True,
        "held_out_examples": len(evaluation),
        "active_conditions": active_conditions,
        "condition_inference": inference,
        "automatic_configured_checks": checks,
        "documents": document_payloads,
    }
    write_bytes_new(
        config.output_path,
        (
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    )
    return result


def remap_role_targets(
    owner_teacher_lane: torch.Tensor,
    reference_teacher_mask: torch.Tensor,
    absent_teacher_mask: torch.Tensor,
    teacher_to_physical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move exact teacher-plan roles onto the physical lane addresses."""

    if owner_teacher_lane.dim() != 1:
        raise ValueError("owner_teacher_lane must have shape [facts].")
    facts = owner_teacher_lane.numel()
    if reference_teacher_mask.shape != (3, facts):
        raise ValueError("reference_teacher_mask must have shape [3, facts].")
    if absent_teacher_mask.shape != (3, facts):
        raise ValueError("absent_teacher_mask must have shape [3, facts].")
    if teacher_to_physical.shape != (3,) or sorted(
        teacher_to_physical.tolist()
    ) != [0, 1, 2]:
        raise ValueError("teacher_to_physical must be a permutation of 0, 1, 2.")
    mapping = teacher_to_physical.long().cpu()
    owner_physical = mapping.index_select(0, owner_teacher_lane.long().cpu())
    reference_physical = torch.zeros_like(reference_teacher_mask, dtype=torch.bool)
    absent_physical = torch.zeros_like(absent_teacher_mask, dtype=torch.bool)
    for teacher_lane, physical_lane in enumerate(mapping.tolist()):
        reference_physical[physical_lane] = reference_teacher_mask[teacher_lane]
        absent_physical[physical_lane] = absent_teacher_mask[teacher_lane]
    return owner_physical, reference_physical, absent_physical


def swap_teacher_to_physical(
    teacher_to_physical: torch.Tensor,
    *,
    lane_pair: tuple[int, int] = (0, 1),
) -> torch.Tensor:
    """Return the expected plan addresses after a physical-lane swap."""

    if teacher_to_physical.shape != (3,) or sorted(
        teacher_to_physical.tolist()
    ) != [0, 1, 2]:
        raise ValueError("teacher_to_physical must be a permutation of 0, 1, 2.")
    left, right = lane_pair
    if (
        type(left) is not int
        or type(right) is not int
        or left == right
        or not 0 <= left < 3
        or not 0 <= right < 3
    ):
        raise ValueError("lane_pair must name two distinct physical lanes.")
    swapped = teacher_to_physical.clone()
    swapped[teacher_to_physical == left] = right
    swapped[teacher_to_physical == right] = left
    return swapped


def long_form_statistics(text: str, *, token_count: int) -> dict[str, object]:
    """Measure the explicit length and sentence-form constraints."""

    if not isinstance(text, str):
        raise TypeError("Generated lane text must be a string.")
    if type(token_count) is not int or token_count < 0:
        raise ValueError("Generated lane token_count must be non-negative.")
    if not text.strip():
        return {
            "token_count": token_count,
            "paragraph_count": 0,
            "sentence_count": 0,
            "median_sentence_words": 0.0,
            "short_sentence_rate": 1.0,
            "passes": False,
        }
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    sentence_word_counts = [
        len(re.findall(r"\b[\w’'-]+\b", sentence))
        for sentence in sentences
    ]
    short_sentences = sum(count < 8 for count in sentence_word_counts)
    short_sentence_rate = short_sentences / len(sentence_word_counts)
    return {
        "token_count": token_count,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "median_sentence_words": float(
            torch.tensor(sentence_word_counts, dtype=torch.float32).median().item()
        ),
        "short_sentence_rate": short_sentence_rate,
        "passes": (
            token_count >= MIN_TARGET_TOKENS
            and len(paragraphs) >= 4
            and short_sentence_rate <= 0.10
        ),
    }


def _evaluate_example(
    pair: _ExamplePair,
    *,
    model: PDTModel,
    pdt_config: Any,
    scorer: NliEntailmentScorer,
    threshold: float,
    max_new_tokens: int,
    include_planner_conditions: bool,
) -> tuple[
    dict[str, object],
    dict[str, FactOwnershipCounts],
    FactOwnershipCounts | None,
]:
    example = pair.raw
    record = pair.tokenized
    prompt = render_planner_prompt(example)
    teacher_nodes = torch.tensor(
        record["plan_semantic_targets"],
        dtype=torch.float32,
        device=next(model.sidecar.parameters()).device,
    ).unsqueeze(0)
    teacher_mask = torch.tensor(
        record["plan_node_mask"],
        dtype=torch.bool,
        device=teacher_nodes.device,
    ).unsqueeze(0)
    oracle_nodes = teacher_nodes * math.sqrt(teacher_nodes.size(-1))

    oracle = _generate(
        model,
        pdt_config,
        prompt,
        max_new_tokens=max_new_tokens,
        plan_nodes_override=oracle_nodes,
        plan_mask_override=teacher_mask,
    )
    owner_teacher, reference_teacher, absent_teacher = _teacher_roles(example)
    positive_fact_count_value = record.get("positive_fact_count")
    if type(positive_fact_count_value) is not int:
        raise ValueError("Tokenized positive_fact_count must be an integer.")
    positive_fact_count = positive_fact_count_value
    hypotheses = [
        fact.statement for fact in example.facts.facts
    ] + [
        fact.hard_negative for fact in example.facts.facts
    ]
    if len(hypotheses) != 2 * positive_fact_count:
        raise ValueError(
            "Raw fact inventory and tokenized positive_fact_count are inconsistent."
        )
    results = {"oracle_plan": oracle}
    mappings = {"oracle_plan": torch.arange(3)}
    learned_mapping: torch.Tensor | None = None
    learned_semantic_loss: float | None = None
    if include_planner_conditions:
        learned = _generate(
            model,
            pdt_config,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        match = match_unordered_plans(
            predicted_nodes=learned.plan_nodes,
            validity_logits=learned.planner_node_validity_logits,
            presentation_order_logits=learned.presentation_order_logits,
            teacher_nodes=teacher_nodes.to(dtype=learned.plan_nodes.dtype),
            teacher_node_mask=teacher_mask,
        )
        learned_mapping = match.permutation[0].detach().cpu()
        learned_semantic_loss = float(match.semantic_loss.detach().float().item())
        results.update(
            {
                "learned_plan": learned,
                "plan_zero": _generate(
                    model,
                    pdt_config,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    counterfactual_mode="plan_zero",
                ),
                "lane_swap": _generate(
                    model,
                    pdt_config,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    counterfactual_mode="lane_swap",
                ),
            }
        )
        mappings.update(
            {
                "learned_plan": learned_mapping,
                "plan_zero": learned_mapping,
                "lane_swap": swap_teacher_to_physical(learned_mapping),
            }
        )
    counts: dict[str, FactOwnershipCounts] = {}
    condition_payloads: dict[str, object] = {}
    lane_swap_scores: torch.Tensor | None = None
    for condition in results:
        result = results[condition]
        lane_texts = [
            result.text_by_stream[stream] for stream in pdt_config.runtime.streams
        ]
        scores = scorer.score(
            lane_texts,
            hypotheses,
        )
        if condition == "lane_swap":
            lane_swap_scores = scores
        owner, references, absences = remap_role_targets(
            owner_teacher,
            reference_teacher,
            absent_teacher,
            mappings[condition],
        )
        condition_counts = fact_ownership_counts(
            scores,
            positive_fact_count=positive_fact_count,
            owner_lane_by_fact=owner,
            reference_mask=references,
            absent_mask=absences,
            threshold=threshold,
        )
        counts[condition] = condition_counts
        condition_payloads[condition] = _condition_payload(
            result,
            streams=pdt_config.runtime.streams,
            mapping=mappings[condition],
            counts=condition_counts,
            eos_token_id=model.trunk_adapter.tokenizer.eos_token_id,
        )
    lane_swap_unmoved: FactOwnershipCounts | None = None
    if include_planner_conditions:
        if lane_swap_scores is None or learned_mapping is None:
            raise RuntimeError("Lane-swap condition did not produce fact scores.")
        unmoved_owner, unmoved_references, unmoved_absences = remap_role_targets(
            owner_teacher,
            reference_teacher,
            absent_teacher,
            learned_mapping,
        )
        lane_swap_unmoved = fact_ownership_counts(
            lane_swap_scores,
            positive_fact_count=positive_fact_count,
            owner_lane_by_fact=unmoved_owner,
            reference_mask=unmoved_references,
            absent_mask=unmoved_absences,
            threshold=threshold,
        )
        lane_swap_payload = condition_payloads["lane_swap"]
        if not isinstance(lane_swap_payload, dict):
            raise RuntimeError("Lane-swap payload construction failed.")
        lane_swap_payload["unmoved_address_control"] = lane_swap_unmoved.to_dict()
    return (
        {
            "example_id": example.source.source_id,
            "source_title": example.source.title,
            "prompt": prompt,
            "learned_plan_semantic_loss": learned_semantic_loss,
            "learned_teacher_to_physical_lane": (
                None if learned_mapping is None else learned_mapping.tolist()
            ),
            "conditions": condition_payloads,
        },
        counts,
        lane_swap_unmoved,
    )


def _generate(
    model: PDTModel,
    pdt_config: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    counterfactual_mode: Literal["none", "plan_zero", "lane_swap"] = "none",
    plan_nodes_override: torch.Tensor | None = None,
    plan_mask_override: torch.Tensor | None = None,
) -> OrchestrationResult:
    orchestrator = MultiStreamOrchestrator(
        model,
        model.trunk_adapter.tokenizer,
        pdt_config,
        counterfactual=CounterfactualConfig(mode=counterfactual_mode),
    )
    return orchestrator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        plan_nodes_override=plan_nodes_override,
        plan_mask_override=plan_mask_override,
    )


def _condition_payload(
    result: OrchestrationResult,
    *,
    streams: Sequence[str],
    mapping: torch.Tensor,
    counts: FactOwnershipCounts,
    eos_token_id: int | None,
) -> dict[str, object]:
    texts = {stream: result.text_by_stream[stream] for stream in streams}
    return {
        "expected_teacher_to_physical_lane": mapping.tolist(),
        "fact_ownership": counts.to_dict(),
        "long_form_by_physical_lane": {
            stream: long_form_statistics(
                texts[stream],
                token_count=_prose_token_count(
                    result.tokens_by_stream[stream],
                    eos_token_id=eos_token_id,
                ),
            )
            for stream in streams
        },
        "presentation_order": sorted(
            streams,
            key=lambda stream: float(
                result.presentation_order_logits[
                    0,
                    streams.index(stream),
                ].item()
            ),
            reverse=True,
        ),
        "text_by_physical_lane": texts,
    }


def _prose_token_count(tokens: Sequence[int], *, eos_token_id: int | None) -> int:
    if not tokens:
        raise ValueError("Generation produced no physical token for a lane.")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("Generation evaluation requires one non-negative EOS token ID.")
    return len(tokens) - int(tokens[-1] == eos_token_id)


def _teacher_roles(
    example: RealPlanExample,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plan_ids = [plan.plan_id for plan in example.teacher.plans]
    target_by_id = {
        target.plan_id: target for target in example.teacher.target_sections
    }
    fact_ids = [fact.fact_id for fact in example.facts.facts]
    owner = torch.empty(len(fact_ids), dtype=torch.long)
    reference = torch.zeros(3, len(fact_ids), dtype=torch.bool)
    absent = torch.zeros_like(reference)
    for fact_index, fact_id in enumerate(fact_ids):
        owners: list[int] = []
        for teacher_lane, plan_id in enumerate(plan_ids):
            labels = {
                label.fact_id: label for label in target_by_id[plan_id].fact_labels
            }
            role = labels[fact_id].role
            if role is FactRole.OWNER:
                owners.append(teacher_lane)
            elif role is FactRole.REFERENCE:
                reference[teacher_lane, fact_index] = True
            else:
                absent[teacher_lane, fact_index] = True
        if len(owners) != 1:
            raise ValueError(f"{fact_id} must have exactly one teacher-plan owner.")
        owner[fact_index] = owners[0]
    return owner, reference, absent


def _calibrate_threshold(
    pairs: Sequence[_ExamplePair],
    *,
    scorer: NliEntailmentScorer,
) -> tuple[float, dict[str, float | int]]:
    score_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for pair in pairs:
        owner, references, _ = _teacher_roles(pair.raw)
        facts = owner.numel()
        present = references.clone()
        present[owner, torch.arange(facts)] = True
        hypotheses = [
            fact.statement for fact in pair.raw.facts.facts
        ] + [
            fact.hard_negative for fact in pair.raw.facts.facts
        ]
        plan_ids = [plan.plan_id for plan in pair.raw.teacher.plans]
        target_by_id = {
            target.plan_id: target
            for target in pair.raw.teacher.target_sections
        }
        scores = scorer.score(
            [target_by_id[plan_id].text for plan_id in plan_ids],
            hypotheses,
        )
        labels = torch.cat(
            (present, torch.zeros(3, facts, dtype=torch.bool)),
            dim=1,
        )
        score_rows.append(scores.flatten())
        label_rows.append(labels.flatten())
    flat_scores = torch.cat(score_rows)
    flat_labels = torch.cat(label_rows)
    threshold = calibrate_entailment_threshold(flat_scores, flat_labels)
    predicted = flat_scores >= threshold
    positive_recall = float(
        (predicted & flat_labels).sum().item() / flat_labels.sum().item()
    )
    negative_recall = float(
        ((~predicted) & (~flat_labels)).sum().item()
        / (~flat_labels).sum().item()
    )
    return threshold, {
        "observations": flat_scores.numel(),
        "teacher_present_recall": positive_recall,
        "teacher_absent_recall": negative_recall,
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
    }


def _summarize_condition(
    counts: Sequence[FactOwnershipCounts],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    aggregate = _sum_counts(counts)
    return {
        "aggregate": aggregate.to_dict(),
        "owner_recall_document_bootstrap": bootstrap_mean(
            [value.owner_recall for value in counts],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        ).to_dict(),
        "unauthorized_leakage_document_bootstrap": bootstrap_mean(
            [value.unauthorized_leakage for value in counts],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 1,
        ).to_dict(),
        "hard_negative_rate_document_bootstrap": bootstrap_mean(
            [value.hard_negative_rate for value in counts],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed + 2,
        ).to_dict(),
    }


def _evidence_checks(
    *,
    document_payloads: Sequence[Mapping[str, object]],
    counts: Mapping[str, Sequence[FactOwnershipCounts]],
    lane_swap_unmoved_counts: Sequence[FactOwnershipCounts],
    minimum_documents: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    active_conditions: Sequence[str],
) -> dict[str, object]:
    oracle_owner = bootstrap_mean(
        [value.owner_recall for value in counts["oracle_plan"]],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    oracle_leakage = bootstrap_mean(
        [value.unauthorized_leakage for value in counts["oracle_plan"]],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 1,
    )
    long_form_passes = {
        condition: all(
            _document_condition_long_form_passes(document, condition)
            for document in document_payloads
        )
        for condition in active_conditions
    }
    enough_documents = len(document_payloads) >= minimum_documents
    if tuple(active_conditions) == ("oracle_plan",):
        checks: dict[str, object] = {
            "minimum_documents": minimum_documents,
            "enough_documents": enough_documents,
            "oracle_owner_recall_ci_lower_at_least_0_90": (
                oracle_owner.lower >= 0.90
            ),
            "oracle_unauthorized_leakage_ci_upper_at_most_0_10": (
                oracle_leakage.upper <= 0.10
            ),
            "long_form_passes": long_form_passes,
            "paired_document_intervals": {
                "oracle_owner_recall": oracle_owner.to_dict(),
                "oracle_unauthorized_leakage": oracle_leakage.to_dict(),
            },
        }
        checks["passes"] = (
            enough_documents
            and oracle_owner.lower >= 0.90
            and oracle_leakage.upper <= 0.10
            and all(long_form_passes.values())
        )
        return checks
    if tuple(active_conditions) != CONDITIONS:
        raise ValueError(f"Unsupported evidence condition set {tuple(active_conditions)!r}.")
    learned_retention = bootstrap_mean(
        [
            learned.owner_recall - 0.8 * oracle.owner_recall
            for learned, oracle in zip(
                counts["learned_plan"],
                counts["oracle_plan"],
                strict=True,
            )
        ],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 2,
    )
    zero_drop = bootstrap_mean(
        [
            learned.owner_recall - zero.owner_recall
            for learned, zero in zip(
                counts["learned_plan"],
                counts["plan_zero"],
                strict=True,
            )
        ],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 3,
    )
    swap_retention = bootstrap_mean(
        [
            swapped.owner_recall - 0.8 * learned.owner_recall
            for swapped, learned in zip(
                counts["lane_swap"],
                counts["learned_plan"],
                strict=True,
            )
        ],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 4,
    )
    swap_address_transfer = bootstrap_mean(
        [
            expected.owner_recall - unmoved.owner_recall
            for expected, unmoved in zip(
                counts["lane_swap"],
                lane_swap_unmoved_counts,
                strict=True,
            )
        ],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 5,
    )
    checks = {
        "minimum_documents": minimum_documents,
        "enough_documents": enough_documents,
        "oracle_owner_recall_ci_lower_at_least_0_90": oracle_owner.lower >= 0.90,
        "oracle_unauthorized_leakage_ci_upper_at_most_0_10": (
            oracle_leakage.upper <= 0.10
        ),
        "learned_owner_recall_at_least_80_percent_of_oracle_ci": (
            learned_retention.lower >= 0.0
        ),
        "plan_zero_owner_recall_drop_ci_lower_at_least_0_30": (
            zero_drop.lower >= 0.30
        ),
        "lane_swap_retains_80_percent_of_learned_owner_recall_ci": (
            swap_retention.lower >= 0.0
        ),
        "lane_swap_expected_address_advantage_ci_lower_at_least_0_30": (
            swap_address_transfer.lower >= 0.30
        ),
        "long_form_passes": long_form_passes,
        "paired_document_intervals": {
            "oracle_owner_recall": oracle_owner.to_dict(),
            "oracle_unauthorized_leakage": oracle_leakage.to_dict(),
            "learned_minus_80_percent_oracle_owner_recall": (
                learned_retention.to_dict()
            ),
            "learned_minus_plan_zero_owner_recall": zero_drop.to_dict(),
            "lane_swap_minus_80_percent_learned_owner_recall": (
                swap_retention.to_dict()
            ),
            "lane_swap_expected_minus_unmoved_owner_recall": (
                swap_address_transfer.to_dict()
            ),
        },
    }
    boolean_checks = [
        value
        for key, value in checks.items()
        if key not in {"minimum_documents", "paired_document_intervals", "long_form_passes"}
    ]
    checks["passes"] = (
        all(value is True for value in boolean_checks)
        and all(long_form_passes.values())
    )
    return checks


def _document_condition_long_form_passes(
    document: Mapping[str, object],
    condition: str,
) -> bool:
    conditions = document.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("Evaluation document is missing conditions.")
    payload = conditions.get(condition)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evaluation document is missing {condition}.")
    lanes = payload.get("long_form_by_physical_lane")
    if not isinstance(lanes, Mapping):
        raise ValueError(f"Evaluation condition {condition} is missing lane statistics.")
    return all(
        isinstance(value, Mapping) and value.get("passes") is True
        for value in lanes.values()
    )


def _sum_counts(counts: Sequence[FactOwnershipCounts]) -> FactOwnershipCounts:
    if not counts:
        raise ValueError("Fact ownership aggregation requires at least one document.")
    total = counts[0]
    for value in counts[1:]:
        total = total + value
    return total


def _load_pairs(
    raw_path: Path,
    tokenized_path: Path,
    *,
    expected_tokenizer: str,
    expected_tokenizer_revision: str,
) -> tuple[_ExamplePair, ...]:
    raw_by_id: dict[str, RealPlanExample] = {}
    with raw_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                example = RealPlanExample.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{raw_path}:{line_number} violates the schema.") from exc
            validate_real_plan_example(example)
            example_id = example.source.source_id
            if example_id in raw_by_id:
                raise ValueError(f"{raw_path} repeats source_id={example_id!r}.")
            raw_by_id[example_id] = example
    if not raw_by_id:
        raise ValueError(f"{raw_path} contains no real-plan examples.")
    tokenized = RealPlanDataset(
        tokenized_path,
        expected_tokenizer=expected_tokenizer,
        expected_tokenizer_revision=expected_tokenizer_revision,
    )
    tokenized_by_id = {
        str(tokenized[index]["example_id"]): tokenized[index]
        for index in range(len(tokenized))
    }
    if set(raw_by_id) != set(tokenized_by_id):
        raise ValueError(
            "Raw and tokenized example IDs must match exactly; "
            f"raw_only={sorted(set(raw_by_id) - set(tokenized_by_id))}, "
            f"tokenized_only={sorted(set(tokenized_by_id) - set(raw_by_id))}."
        )
    return tuple(
        _ExamplePair(raw=raw_by_id[example_id], tokenized=tokenized_by_id[example_id])
        for example_id in sorted(raw_by_id)
    )


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS evaluation was requested but MPS is unavailable.")
    return device


def _validate_paths(config: GenerationEvaluationConfig) -> None:
    inputs = (
        config.config_path,
        config.checkpoint_path,
        config.calibration_raw_path,
        config.calibration_tokenized_path,
        config.evaluation_raw_path,
        config.evaluation_tokenized_path,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required evaluation inputs do not exist: {missing}.")
    if config.output_path.exists():
        raise FileExistsError(f"Refusing to replace evaluation output: {config.output_path}")
    if config.output_path.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Evaluation output must not replace an input artifact.")
    if not MIN_TARGET_TOKENS <= config.max_new_tokens <= MAX_TARGET_TOKENS:
        raise ValueError(
            f"max_new_tokens must be {MIN_TARGET_TOKENS}-{MAX_TARGET_TOKENS} "
            "per physical lane."
        )
    if not config.entailment_device:
        raise ValueError("entailment_device must be non-empty.")
    if type(config.entailment_batch_size) is not int or config.entailment_batch_size <= 0:
        raise ValueError("entailment_batch_size must be a positive integer.")
