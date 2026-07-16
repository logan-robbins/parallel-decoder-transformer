"""Validate retokenized dependency examples and run local-only CE audits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from pdt.datasets.retokenize import validate_retokenized_record


@dataclass(slots=True)
class CESums:
    """Sufficient statistics for one token class, measured in nats."""

    local_nats: float = 0.0
    privileged_nats: float = 0.0
    tokens: int = 0

    def add(self, *, local: Sequence[float], privileged: Sequence[float]) -> None:
        if len(local) != len(privileged):
            raise ValueError("local and privileged CE vectors must have equal length.")
        self.local_nats += sum(local)
        self.privileged_nats += sum(privileged)
        self.tokens += len(local)

    def merge(self, other: CESums) -> None:
        self.local_nats += other.local_nats
        self.privileged_nats += other.privileged_nats
        self.tokens += other.tokens

    @property
    def mean_gap(self) -> float:
        if self.tokens == 0:
            raise ValueError("cannot compute a CE gap with zero tokens.")
        return (self.local_nats - self.privileged_nats) / self.tokens


@dataclass(slots=True)
class ExampleAudit:
    dependency: CESums
    nondependency: CESums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    if args.max_examples <= 0:
        raise ValueError(f"max-examples must be positive, got {args.max_examples}.")

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    model.eval()

    audits: list[ExampleAudit] = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_retokenized_record(record, line_ref=f"line {line_no}")
            audits.append(_audit_example(record, model))
            if len(audits) >= args.max_examples:
                break

    report = aggregate_report(audits)
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not report["passes"]:
        raise SystemExit(f"CE audit failed thresholds: {report}")


def aggregate_report(audits: Iterable[ExampleAudit]) -> dict[str, int | float | bool]:
    """Aggregate raw CE sums/counts; never average per-example or per-batch means."""
    dependency = CESums()
    nondependency = CESums()
    examples = 0
    positive_examples = 0
    for audit in audits:
        if audit.dependency.tokens == 0:
            raise ValueError(f"example {examples} contains zero dependency tokens.")
        dependency.merge(audit.dependency)
        nondependency.merge(audit.nondependency)
        positive_examples += int(audit.dependency.mean_gap > 0.0)
        examples += 1
    if examples == 0:
        raise ValueError("dependency audit received zero examples.")
    if dependency.tokens == 0:
        raise ValueError("dependency audit contains zero dependency tokens.")
    if nondependency.tokens == 0:
        raise ValueError("dependency audit contains zero nondependency tokens.")

    mean_dependency_gap = dependency.mean_gap
    mean_nondependency_gap = nondependency.mean_gap
    positive_fraction = positive_examples / examples
    return {
        "examples": examples,
        "dependency_tokens": dependency.tokens,
        "nondependency_tokens": nondependency.tokens,
        "dependency_local_nats": dependency.local_nats,
        "dependency_privileged_nats": dependency.privileged_nats,
        "nondependency_local_nats": nondependency.local_nats,
        "nondependency_privileged_nats": nondependency.privileged_nats,
        "mean_dependency_gap_nats_per_token": mean_dependency_gap,
        "mean_nondependency_gap_nats_per_token": mean_nondependency_gap,
        "positive_dependency_gap_fraction": positive_fraction,
        "passes": (
            mean_dependency_gap >= 1.5
            and abs(mean_nondependency_gap) < 0.3
            and positive_fraction >= 0.8
        ),
    }


@torch.no_grad()
def _audit_example(record: Mapping[str, object], model: torch.nn.Module) -> ExampleAudit:
    dependency = CESums()
    nondependency = CESums()
    streams = record["stream_inputs"]
    assert isinstance(streams, list)
    teacher_prompts_value = record["teacher_block_prompt_ids"]
    assert isinstance(teacher_prompts_value, list)

    for stream in streams:
        assert isinstance(stream, Mapping)
        stream_prompt = _as_token_ids(stream["stream_prompt_ids"])
        transitions = stream["block_transition_ids"]
        target_blocks = stream["target_block_ids"]
        dep_masks = stream["dependency_token_mask"]
        non_masks = stream["nondependency_token_mask"]
        assert isinstance(target_blocks, list)
        assert isinstance(transitions, list)
        assert isinstance(dep_masks, list)
        assert isinstance(non_masks, list)
        local_prior: list[int] = []
        for block_idx, target_value in enumerate(target_blocks):
            target_ids = _as_token_ids(target_value)
            local_ce = _target_ce_ids(stream_prompt + local_prior, target_ids, model)
            privileged_ce = _target_ce_ids(
                _as_token_ids(teacher_prompts_value[block_idx]),
                target_ids,
                model,
            )
            dep_local, dep_privileged = _select_masked(
                local_ce,
                privileged_ce,
                dep_masks[block_idx],
            )
            non_local, non_privileged = _select_masked(
                local_ce,
                privileged_ce,
                non_masks[block_idx],
            )
            dependency.add(local=dep_local, privileged=dep_privileged)
            nondependency.add(local=non_local, privileged=non_privileged)
            local_prior.extend(target_ids)
            if block_idx + 1 < len(target_blocks):
                local_prior.extend(_as_token_ids(transitions[block_idx + 1]))
    return ExampleAudit(dependency=dependency, nondependency=nondependency)


def _target_ce_ids(
    context_ids: Sequence[int],
    target_ids: Sequence[int],
    model: torch.nn.Module,
) -> list[float]:
    if not context_ids or not target_ids:
        raise ValueError("CE audit requires non-empty context and target token IDs.")
    ids = torch.tensor([list(context_ids) + list(target_ids)], dtype=torch.long)
    logits = model(input_ids=ids).logits[:, :-1]
    labels = ids[:, 1:]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view(-1)
    start = len(context_ids) - 1
    values = ce[start : start + len(target_ids)]
    if values.numel() != len(target_ids):
        raise ValueError("model CE output does not cover every target token.")
    return [float(value) for value in values.tolist()]


def _select_masked(
    local: Sequence[float],
    privileged: Sequence[float],
    mask: object,
) -> tuple[list[float], list[float]]:
    if not isinstance(mask, list) or len(mask) != len(local):
        raise ValueError("CE mask length must match target token count.")
    selected_local = [value for value, keep in zip(local, mask) if keep]
    selected_privileged = [value for value, keep in zip(privileged, mask) if keep]
    return selected_local, selected_privileged


def _as_token_ids(value: object) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("token IDs must be stored as a list.")
    return [int(item) for item in value]


if __name__ == "__main__":
    main()
