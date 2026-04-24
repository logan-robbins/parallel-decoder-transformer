"""Validate retokenized dependency examples and run local-only CE audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    model.eval()

    dep_gaps: list[float] = []
    nondep_gaps: list[float] = []
    positives = 0
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no > args.max_examples:
                break
            rec = json.loads(line)
            _structural_validate(rec, line_no)
            dep_gap, nondep_gap = _audit_example(rec, tokenizer, model)
            dep_gaps.append(dep_gap)
            nondep_gaps.append(nondep_gap)
            positives += int(dep_gap > 0)

    mean_dep = sum(dep_gaps) / max(1, len(dep_gaps))
    mean_non = sum(nondep_gaps) / max(1, len(nondep_gaps))
    positive_fraction = positives / max(1, len(dep_gaps))
    report = {
        "examples": len(dep_gaps),
        "mean_dependency_gap": mean_dep,
        "mean_nondependency_gap": mean_non,
        "positive_dependency_gap_fraction": positive_fraction,
        "passes": mean_dep >= 1.5 and mean_non < 0.3 and positive_fraction >= 0.8,
    }
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passes"]:
        raise SystemExit(f"CE audit failed thresholds: {report}")


def _structural_validate(rec: dict[str, object], line_no: int) -> None:
    if int(rec.get("visibility_lag_blocks", 1)) == 1:
        for stream in rec["stream_inputs"]:
            if len(stream["target_block_ids"]) < 2:
                raise ValueError(f"line {line_no}: Delta=1 requires at least two blocks.")


@torch.no_grad()
def _audit_example(rec: dict[str, object], tokenizer, model) -> tuple[float, float]:
    dep_local: list[float] = []
    dep_full: list[float] = []
    non_local: list[float] = []
    non_full: list[float] = []
    shared = str(rec.get("shared_context", ""))
    sibling_state = " ".join(
        str(stream["target_blocks"][0]) for stream in rec["stream_inputs"]
    )
    for stream in rec["stream_inputs"]:
        local_context = f"{shared}\n{stream.get('local_observation', '')}\n"
        full_context = f"{local_context}{sibling_state}\n"
        for block_idx, block in enumerate(stream["target_blocks"]):
            target_ids = stream["target_block_ids"][block_idx]
            dep_mask = stream["dependency_token_mask"][block_idx]
            non_mask = stream["nondependency_token_mask"][block_idx]
            local_ce = _target_ce(local_context, str(block), tokenizer, model)
            full_ce = _target_ce(full_context, str(block), tokenizer, model)
            for i, is_dep in enumerate(dep_mask[: len(local_ce)]):
                if is_dep:
                    dep_local.append(local_ce[i])
                    dep_full.append(full_ce[i])
            for i, is_non in enumerate(non_mask[: len(local_ce)]):
                if is_non:
                    non_local.append(local_ce[i])
                    non_full.append(full_ce[i])
            if len(target_ids) != len(local_ce):
                raise ValueError("tokenized target length changed between retokenize and audit.")
    return _mean_gap(dep_local, dep_full), _mean_gap(non_local, non_full)


def _target_ce(context: str, target: str, tokenizer, model) -> list[float]:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    ids = torch.tensor([context_ids + target_ids], dtype=torch.long)
    logits = model(input_ids=ids).logits[:, :-1]
    labels = ids[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
    ce = ce.view(1, -1)[0]
    start = max(0, len(context_ids) - 1)
    return [float(x) for x in ce[start : start + len(target_ids)].tolist()]


def _mean_gap(local: list[float], full: list[float]) -> float:
    if not local:
        return 0.0
    return (sum(local) - sum(full)) / len(local)


if __name__ == "__main__":
    main()
