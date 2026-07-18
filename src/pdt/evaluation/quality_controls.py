"""Frozen-trunk blind and sequential-oracle quality controls.

Both controls score the exact target tokens from a processed long-form record.
The blind condition sees only receiver-local chat history.  The sequential
oracle sees every causally eligible private observation and every completed
prior stream block as text.  Per-document values are retained so a trained PDT
checkpoint can be compared on matched examples rather than pooled summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Literal

import torch
import torch.nn.functional as F

from pdt.datasets.retokenize import validate_retokenized_record
from pdt.evaluation.paired_causal import BootstrapMean, bootstrap_mean


__all__ = [
    "ConditionMetrics",
    "DocumentQualityControl",
    "QualityControlEvaluation",
    "score_quality_control_records",
]


Condition = Literal["blind", "sequential_oracle"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConditionMetrics:
    """Token-weighted cross entropy for one frozen-trunk condition."""

    dependency_tokens: int
    nondependency_tokens: int
    dependency_ce: float
    nondependency_ce: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentQualityControl:
    """Matched blind and oracle CE values for one complete document."""

    example_id: str
    dependency_tokens: int
    nondependency_tokens: int
    blind_dependency_ce: float
    blind_nondependency_ce: float
    sequential_oracle_dependency_ce: float
    sequential_oracle_nondependency_ce: float
    oracle_dependency_advantage: float
    oracle_nondependency_advantage: float
    oracle_selectivity_difference: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualityControlEvaluation:
    """Frozen-trunk lower/upper controls and document-paired evidence."""

    dataset_condition: Literal["dependency", "null"]
    documents: int
    scoring_tasks: int
    maximum_sequence_tokens: int
    blind: ConditionMetrics
    sequential_oracle: ConditionMetrics
    oracle_dependency_advantage: BootstrapMean
    oracle_nondependency_advantage: BootstrapMean
    oracle_selectivity_difference: BootstrapMean
    document_values: tuple[DocumentQualityControl, ...]
    minimum_documents: int
    enough_documents: bool
    dependency_ci_lower_positive: bool
    selectivity_ci_lower_positive: bool
    evidence_gate_passes: bool
    expected_outcome_passes: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_condition": self.dataset_condition,
            "documents": self.documents,
            "scoring_tasks": self.scoring_tasks,
            "maximum_sequence_tokens": self.maximum_sequence_tokens,
            "conditions": {
                "blind": self.blind.to_dict(),
                "sequential_oracle": self.sequential_oracle.to_dict(),
            },
            "oracle_advantage": {
                "dependency_ce_reduction": self.oracle_dependency_advantage.to_dict(),
                "nondependency_ce_reduction": self.oracle_nondependency_advantage.to_dict(),
                "selectivity_difference": self.oracle_selectivity_difference.to_dict(),
            },
            "document_values": [value.to_dict() for value in self.document_values],
            "evidence_gate": {
                "minimum_documents": self.minimum_documents,
                "enough_documents": self.enough_documents,
                "dependency_ci_lower_positive": self.dependency_ci_lower_positive,
                "selectivity_ci_lower_positive": self.selectivity_ci_lower_positive,
                "passes": self.evidence_gate_passes,
            },
            "expected_outcome_passes": self.expected_outcome_passes,
        }


@dataclass(frozen=True, slots=True)
class _ScoringTask:
    document_index: int
    condition: Condition
    context_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    dependency_mask: tuple[bool, ...]


@dataclass(slots=True)
class _LossSums:
    dependency_nats: float = 0.0
    nondependency_nats: float = 0.0
    dependency_tokens: int = 0
    nondependency_tokens: int = 0

    def add(self, nll: torch.Tensor, dependency_mask: Sequence[bool]) -> None:
        if nll.ndim != 1 or nll.numel() != len(dependency_mask):
            raise ValueError("Quality-control NLL and dependency mask lengths must match.")
        dep = torch.tensor(dependency_mask, dtype=torch.bool, device=nll.device)
        non = ~dep
        self.dependency_nats += _sum_float64(nll[dep])
        self.nondependency_nats += _sum_float64(nll[non])
        self.dependency_tokens += int(dep.sum().item())
        self.nondependency_tokens += int(non.sum().item())

    def merge(self, other: _LossSums) -> None:
        self.dependency_nats += other.dependency_nats
        self.nondependency_nats += other.nondependency_nats
        self.dependency_tokens += other.dependency_tokens
        self.nondependency_tokens += other.nondependency_tokens

    @property
    def dependency_ce(self) -> float:
        if self.dependency_tokens <= 0:
            raise ValueError("Cannot compute dependency CE without tokens.")
        return self.dependency_nats / self.dependency_tokens

    @property
    def nondependency_ce(self) -> float:
        if self.nondependency_tokens <= 0:
            raise ValueError("Cannot compute nondependency CE without tokens.")
        return self.nondependency_nats / self.nondependency_tokens

    def metrics(self) -> ConditionMetrics:
        return ConditionMetrics(
            dependency_tokens=self.dependency_tokens,
            nondependency_tokens=self.nondependency_tokens,
            dependency_ce=self.dependency_ce,
            nondependency_ce=self.nondependency_ce,
        )


@torch.inference_mode()
def score_quality_control_records(
    records: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    *,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    expected_tokenizer: str,
    expected_tokenizer_revision: str,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 1729,
    minimum_documents: int = 32,
) -> QualityControlEvaluation:
    """Score complete processed records under the two canonical controls."""

    if not records:
        raise ValueError("Quality-control scoring requires at least one document.")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer.")
    if type(minimum_documents) is not int or minimum_documents <= 1:
        raise ValueError("minimum_documents must be an integer greater than one.")
    if model.training:
        raise ValueError("Quality-control scoring requires the frozen trunk in eval mode.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Quality-control scoring requires every trunk parameter to be frozen.")
    parameter_devices = {parameter.device for parameter in model.parameters()}
    if parameter_devices and parameter_devices != {device}:
        raise ValueError(
            "Quality-control trunk parameters must all reside on the scoring device; "
            f"expected={device}, observed={sorted(map(str, parameter_devices))}."
        )
    model_config = getattr(model, "config", None)
    context_limit = getattr(model_config, "max_position_embeddings", None)
    if type(context_limit) is not int or context_limit <= 0:
        raise ValueError(
            "Quality-control scoring requires a positive model max_position_embeddings."
        )

    conditions = {_dataset_condition(record) for record in records}
    if len(conditions) != 1:
        raise ValueError("A quality-control report cannot mix dependency and null documents.")
    dataset_condition = conditions.pop()
    tasks: list[_ScoringTask] = []
    example_ids: list[str] = []
    for document_index, record in enumerate(records):
        validate_retokenized_record(
            record,
            line_ref=f"quality-control document {document_index}",
            expected_tokenizer=expected_tokenizer,
            expected_tokenizer_revision=expected_tokenizer_revision,
        )
        example_id = str(record.get("example_id", ""))
        if not example_id or example_id in example_ids:
            raise ValueError("Quality-control example IDs must be non-empty and unique.")
        example_ids.append(example_id)
        tasks.extend(_record_tasks(record, document_index=document_index))

    tasks.sort(key=lambda task: len(task.context_ids) + len(task.target_ids))
    maximum_sequence_tokens = len(tasks[-1].context_ids) + len(tasks[-1].target_ids)
    if maximum_sequence_tokens > context_limit:
        raise ValueError(
            "Quality-control sequence exceeds the pinned trunk context limit: "
            f"required={maximum_sequence_tokens}, limit={context_limit}."
        )
    document_sums = [
        {"blind": _LossSums(), "sequential_oracle": _LossSums()} for _ in records
    ]
    for start in range(0, len(tasks), batch_size):
        task_batch = tasks[start : start + batch_size]
        nll_rows = _score_task_batch(
            task_batch,
            model,
            device=device,
            pad_token_id=pad_token_id,
        )
        for task, nll in zip(task_batch, nll_rows, strict=True):
            document_sums[task.document_index][task.condition].add(
                nll,
                task.dependency_mask,
            )
        completed = min(start + batch_size, len(tasks))
        if start == 0 or completed == len(tasks) or completed % (100 * batch_size) == 0:
            LOGGER.info("quality-control scoring tasks=%d/%d", completed, len(tasks))

    aggregate = {"blind": _LossSums(), "sequential_oracle": _LossSums()}
    document_values: list[DocumentQualityControl] = []
    for example_id, sums in zip(example_ids, document_sums, strict=True):
        blind = sums["blind"]
        oracle = sums["sequential_oracle"]
        if (
            blind.dependency_tokens != oracle.dependency_tokens
            or blind.nondependency_tokens != oracle.nondependency_tokens
        ):
            raise RuntimeError(f"Quality-control token alignment drifted for {example_id!r}.")
        aggregate["blind"].merge(blind)
        aggregate["sequential_oracle"].merge(oracle)
        dependency_advantage = blind.dependency_ce - oracle.dependency_ce
        nondependency_advantage = blind.nondependency_ce - oracle.nondependency_ce
        document_values.append(
            DocumentQualityControl(
                example_id=example_id,
                dependency_tokens=blind.dependency_tokens,
                nondependency_tokens=blind.nondependency_tokens,
                blind_dependency_ce=blind.dependency_ce,
                blind_nondependency_ce=blind.nondependency_ce,
                sequential_oracle_dependency_ce=oracle.dependency_ce,
                sequential_oracle_nondependency_ce=oracle.nondependency_ce,
                oracle_dependency_advantage=dependency_advantage,
                oracle_nondependency_advantage=nondependency_advantage,
                oracle_selectivity_difference=(dependency_advantage - nondependency_advantage),
            )
        )

    dependency_bootstrap = bootstrap_mean(
        [value.oracle_dependency_advantage for value in document_values],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    nondependency_bootstrap = bootstrap_mean(
        [value.oracle_nondependency_advantage for value in document_values],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 1,
    )
    selectivity_bootstrap = bootstrap_mean(
        [value.oracle_selectivity_difference for value in document_values],
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + 2,
    )
    enough_documents = len(document_values) >= minimum_documents
    dependency_positive = dependency_bootstrap.lower > 0.0
    selectivity_positive = selectivity_bootstrap.lower > 0.0
    evidence_passes = enough_documents and dependency_positive and selectivity_positive
    expected_outcome = (
        evidence_passes
        if dataset_condition == "dependency"
        else enough_documents and not selectivity_positive
    )
    return QualityControlEvaluation(
        dataset_condition=dataset_condition,
        documents=len(document_values),
        scoring_tasks=len(tasks),
        maximum_sequence_tokens=maximum_sequence_tokens,
        blind=aggregate["blind"].metrics(),
        sequential_oracle=aggregate["sequential_oracle"].metrics(),
        oracle_dependency_advantage=dependency_bootstrap,
        oracle_nondependency_advantage=nondependency_bootstrap,
        oracle_selectivity_difference=selectivity_bootstrap,
        document_values=tuple(document_values),
        minimum_documents=minimum_documents,
        enough_documents=enough_documents,
        dependency_ci_lower_positive=dependency_positive,
        selectivity_ci_lower_positive=selectivity_positive,
        evidence_gate_passes=evidence_passes,
        expected_outcome_passes=expected_outcome,
    )


def _record_tasks(record: Mapping[str, Any], *, document_index: int) -> list[_ScoringTask]:
    streams = record.get("stream_inputs")
    if not isinstance(streams, list):
        raise ValueError("Quality-control record stream_inputs must be a list.")
    tasks: list[_ScoringTask] = []
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise ValueError("Quality-control stream rows must be mappings.")
        prompt = _ids(stream.get("stream_prompt_ids"), "stream_prompt_ids")
        targets = stream.get("target_block_ids")
        transitions = stream.get("block_transition_ids")
        oracle_prompts = stream.get("full_text_oracle_block_prompt_ids")
        dependency_masks = stream.get("dependency_token_mask")
        if not all(
            isinstance(value, list)
            for value in (targets, transitions, oracle_prompts, dependency_masks)
        ):
            raise ValueError("Quality-control stream rows are missing processed block fields.")
        assert isinstance(targets, list)
        assert isinstance(transitions, list)
        assert isinstance(oracle_prompts, list)
        assert isinstance(dependency_masks, list)
        lengths = {len(targets), len(transitions), len(oracle_prompts), len(dependency_masks)}
        if len(lengths) != 1:
            raise ValueError("Quality-control block fields must have identical lengths.")
        local_prior: list[int] = []
        for block_index, target_value in enumerate(targets):
            target = _ids(target_value, f"target block {block_index}")
            dependency_mask = _bool_mask(
                dependency_masks[block_index],
                expected=len(target),
            )
            tasks.append(
                _ScoringTask(
                    document_index=document_index,
                    condition="blind",
                    context_ids=tuple(prompt + local_prior),
                    target_ids=tuple(target),
                    dependency_mask=dependency_mask,
                )
            )
            tasks.append(
                _ScoringTask(
                    document_index=document_index,
                    condition="sequential_oracle",
                    context_ids=tuple(
                        _ids(
                            oracle_prompts[block_index],
                            f"full-text oracle block {block_index}",
                        )
                    ),
                    target_ids=tuple(target),
                    dependency_mask=dependency_mask,
                )
            )
            local_prior.extend(target)
            if block_index + 1 < len(targets):
                local_prior.extend(
                    _ids(
                        transitions[block_index + 1],
                        f"block transition {block_index + 1}",
                    )
                )
    return tasks


def _score_task_batch(
    tasks: Sequence[_ScoringTask],
    model: torch.nn.Module,
    *,
    device: torch.device,
    pad_token_id: int,
) -> tuple[torch.Tensor, ...]:
    if not tasks:
        raise ValueError("Cannot score an empty quality-control task batch.")
    sequences = [task.context_ids + task.target_ids for task in tasks]
    if any(not task.context_ids or not task.target_ids for task in tasks):
        raise ValueError("Quality-control contexts and targets must be non-empty.")
    target_lengths = {len(task.target_ids) for task in tasks}
    if len(target_lengths) != 1:
        raise ValueError("Every task in a quality-control batch must have one target length.")
    target_length = target_lengths.pop()
    logits_to_keep = target_length + 1
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(tasks), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        offset = max_length - length
        input_ids[row, offset:] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, offset:] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.clamp_min_(0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        logits_to_keep=logits_to_keep,
    )
    logits = getattr(output, "logits", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != (len(tasks), logits_to_keep)
    ):
        raise RuntimeError("Frozen-trunk quality control returned invalid logits.")
    results: list[torch.Tensor] = []
    vocabulary = logits.size(-1)
    for row, task in enumerate(tasks):
        target = torch.tensor(task.target_ids, dtype=torch.long, device=device)
        if bool((target < 0).any()) or bool((target >= vocabulary).any()):
            raise ValueError("Quality-control target token lies outside model vocabulary.")
        prediction_logits = logits[row, :target_length].float()
        nll = F.cross_entropy(prediction_logits, target, reduction="none")
        if not bool(torch.isfinite(nll).all()):
            raise ValueError("Frozen-trunk quality control produced non-finite CE.")
        results.append(nll.detach().cpu())
    return tuple(results)


def _dataset_condition(record: Mapping[str, Any]) -> Literal["dependency", "null"]:
    entropy = record.get("entropy_accounting")
    if not isinstance(entropy, Mapping):
        raise ValueError("Quality-control record is missing entropy_accounting.")
    rho = entropy.get("rho")
    if rho == 1.0:
        return "dependency"
    if rho == 0.0:
        return "null"
    raise ValueError(f"Quality-control rho must be exactly 0.0 or 1.0, got {rho!r}.")


def _ids(value: object, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a token-ID sequence.")
    result = [int(item) for item in value]
    if not result:
        raise ValueError(f"{label} must be non-empty.")
    return result


def _bool_mask(value: object, *, expected: int) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("dependency_token_mask must be a sequence.")
    result = tuple(bool(item) for item in value)
    if len(result) != expected:
        raise ValueError("dependency_token_mask length must match target length.")
    return result


def _sum_float64(tensor: torch.Tensor) -> float:
    value = float(tensor.detach().to(dtype=torch.float64, device="cpu").sum().item())
    if not math.isfinite(value):
        raise ValueError("Quality-control loss sum is non-finite.")
    return value
