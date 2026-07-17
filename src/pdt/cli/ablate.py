"""Generation-level smoke runner for checkpoint counterfactuals.

Runs gate-zero, sibling-note norm scramble, and a targeted bus mutation plus
the baseline on a fixed prompt set.  It emits two generation-differentiation
metrics:

    - pairwise cosine distance between per-stream output embeddings
    - cross-stream ROUGE-L (on the generated text)

These are smoke diagnostics, not the dependency-span causal gate.  The latter
requires paired teacher-forced CE/KL metrics from the evaluation dataset.
Plan and source swaps require explicit cross-prompt donor state and are
therefore exposed by the runtime API rather than fabricated by this CLI.

Usage:
    uv run scripts/ablate.py --config configs/pdt_qwen3_4b.yaml \
        --checkpoint experiments/qwen3_4b/checkpoints/step_0050000.pt \
        --prompts-file evaluation/prompts.jsonl \
        --output experiments/qwen3_4b/ablations/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, List, Literal

import torch

from pdt.checkpoint import load_checkpoint
from pdt.config import load_config
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import CounterfactualConfig
from pdt.runtime.orchestrator import MultiStreamOrchestrator


_GenerationMode = Literal["baseline", "gate_zero", "norm_scramble", "bus_mutation"]


def _load_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, Mapping):
                    raise ValueError(
                        f"Prompt JSONL row must be an object with prompt/text, got "
                        f"{type(obj).__name__}."
                    )
                value = obj.get("prompt") or obj.get("text")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Prompt JSONL object must contain non-empty prompt or text.")
                prompts.append(value.strip())
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


def _pairwise_cosine_mean(texts: List[str], tokenizer) -> float:
    """Cosine distance mean over pairs of stream outputs (bag of token ids).

    A lightweight stand-in for an embedding-model cosine that runs on CPU.
    """
    if len(texts) < 2:
        return 0.0
    vecs: List[torch.Tensor] = []
    for t in texts:
        ids = tokenizer(t or "", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if ids.numel() == 0:
            vecs.append(torch.zeros(tokenizer.vocab_size, dtype=torch.float32))
            continue
        vec = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
        vec.index_add_(0, ids, torch.ones(ids.numel()))
        vec = vec / (vec.norm() + 1e-9)
        vecs.append(vec)
    stacked = torch.stack(vecs, dim=0)
    sim = torch.matmul(stacked, stacked.T)
    n = len(texts)
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pair_sim = sim[triu]
    return float((1.0 - pair_sim).mean().item())


def _rouge_l_mean(texts: List[str]) -> float:
    """Pairwise ROUGE-L mean across streams.

    Lower = more differentiated. Implements the standard LCS-based F1.
    """

    def _lcs(a: List[str], b: List[str]) -> int:
        if not a or not b:
            return 0
        na, nb = len(a), len(b)
        dp = [[0] * (nb + 1) for _ in range(na + 1)]
        for i in range(1, na + 1):
            for j in range(1, nb + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[na][nb]

    tokens = [t.split() for t in texts]
    n = len(tokens)
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tokens[i], tokens[j]
            lcs_len = _lcs(a, b)
            if lcs_len == 0:
                scores.append(0.0)
                continue
            p = lcs_len / max(1, len(b))
            r = lcs_len / max(1, len(a))
            if p + r == 0:
                scores.append(0.0)
                continue
            scores.append(2 * p * r / (p + r))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _run_condition(
    model: PDTModel,
    config,
    *,
    prompts: List[str],
    mode: _GenerationMode,
    max_new_tokens: int,
    seed: int,
    mutation_producer: str | None,
    mutation_block: int,
    mutation_code_offset: int,
) -> Dict[str, object]:
    cf_mode: Literal["gate_zero", "norm_scramble", "bus_mutation"] | None
    if mode == "baseline":
        cf_mode = None
    else:
        cf_mode = mode
    cf = CounterfactualConfig(
        mode=cf_mode,
        seed=seed,
        mutation_producer=mutation_producer,
        mutation_block=mutation_block,
        mutation_code_offset=mutation_code_offset,
    )
    orch = MultiStreamOrchestrator(model, model.trunk_adapter.tokenizer, config, counterfactual=cf)
    per_prompt = []
    cosine_sum = 0.0
    rouge_sum = 0.0
    count = 0
    for prompt in prompts:
        result = orch.generate(prompt, max_new_tokens=max_new_tokens)
        stream_texts = list(result.text_by_stream.values())
        cos = _pairwise_cosine_mean(stream_texts, model.trunk_adapter.tokenizer)
        rouge = _rouge_l_mean(stream_texts)
        cosine_sum += cos
        rouge_sum += rouge
        count += 1
        per_prompt.append(
            {
                "prompt": prompt,
                "pairwise_cosine_distance": cos,
                "rouge_l": rouge,
                "streams": result.text_by_stream,
            }
        )
    return {
        "mode": mode,
        "pairwise_cosine_distance_mean": cosine_sum / max(count, 1),
        "rouge_l_mean": rouge_sum / max(count, 1),
        "per_prompt": per_prompt,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--mutation-producer", default=None)
    parser.add_argument("--mutation-block", type=int, default=0)
    parser.add_argument("--mutation-code-offset", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.config.is_file():
        parser.error(f"--config must name an existing file: {args.config}")
    if not args.checkpoint.is_file():
        parser.error(f"--checkpoint must name an existing file: {args.checkpoint}")
    if not args.prompts_file.is_file():
        parser.error(f"--prompts-file must name an existing file: {args.prompts_file}")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.mutation_block < 0:
        parser.error("--mutation-block must be non-negative")
    if args.mutation_code_offset <= 0:
        parser.error("--mutation-code-offset must be positive")
    prompts = _load_prompts(args.prompts_file)
    if not prompts:
        parser.error(f"Prompt file contains no prompts: {args.prompts_file}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(args.config)
    if args.mutation_producer is not None and args.mutation_producer not in config.runtime.streams:
        parser.error(
            f"--mutation-producer must be one of {config.runtime.streams}, "
            f"got {args.mutation_producer!r}"
        )
    code_count = config.sidecar.speculation_head.codes_per_codebook
    if args.mutation_code_offset >= code_count:
        parser.error(f"--mutation-code-offset must be less than {code_count}")
    completed_blocks = args.max_new_tokens // config.runtime.block_size
    if completed_blocks == 0 or args.mutation_block >= completed_blocks:
        parser.error(
            "The bus-mutation condition requires --mutation-block to name a "
            f"published block; max-new-tokens={args.max_new_tokens}, "
            f"tau={config.runtime.block_size}."
        )
    model = PDTModel(config)
    load_checkpoint(args.checkpoint, model)
    device = (
        torch.device(config.training.device)
        if config.training.device is not None
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    trunk_model: torch.nn.Module = model.trunk_adapter.model
    trunk_model.to(device)
    model.to(device)
    trunk_model.eval()
    model.eval()

    print(f"Running ablations on {len(prompts)} prompts with device={device}.")

    manifest = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "num_prompts": len(prompts),
        "seed": args.seed,
        "mutation": {
            "producer": args.mutation_producer,
            "block": args.mutation_block,
            "code_offset": args.mutation_code_offset,
        },
        "conditions": {},
    }
    modes: tuple[_GenerationMode, ...] = (
        "baseline",
        "gate_zero",
        "norm_scramble",
        "bus_mutation",
    )
    for mode in modes:
        print(f"-- {mode} --")
        result = _run_condition(
            model,
            config,
            prompts=prompts,
            mode=mode,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            mutation_producer=args.mutation_producer,
            mutation_block=args.mutation_block,
            mutation_code_offset=args.mutation_code_offset,
        )
        manifest["conditions"][mode] = result

    # Descriptive generation-level differences. These are not acceptance gates.
    base_cos = manifest["conditions"]["baseline"]["pairwise_cosine_distance_mean"]
    base_rouge = manifest["conditions"]["baseline"]["rouge_l_mean"]
    results_summary = {}
    for condition in ("gate_zero", "norm_scramble", "bus_mutation"):
        cos_delta = base_cos - manifest["conditions"][condition]["pairwise_cosine_distance_mean"]
        rouge_delta = manifest["conditions"][condition]["rouge_l_mean"] - base_rouge
        results_summary[condition] = {
            "cosine_distance_delta_vs_baseline": cos_delta,
            "rouge_l_delta_vs_baseline": rouge_delta,
        }
    manifest["generation_difference_summary"] = results_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote ablation manifest to {args.output}")


if __name__ == "__main__":
    main()
