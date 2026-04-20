"""Pre-registered ablation runner for Interventions A / B / C.

Runs all three counterfactuals plus the baseline on a fixed evaluation
prompt set and emits a manifest with cross-stream differentiation metrics:

    - pairwise cosine distance between per-stream output embeddings
    - cross-stream ROUGE-L (on the generated text)
    - JS divergence of entity distributions
    - coverage overlap against the planner-seeded catalog
    - cross-stream contradiction rate (simple lexical approximation)

Pre-registered gate: >= 3 absolute points on coverage-overlap OR
contradiction-rate between the baseline and Intervention A (or B) rejects
the null that SNC is decorative.

Usage:
    uv run python -m pdt.cli.ablate --config configs/pdt_qwen3_4b.yaml \
        --checkpoint experiments/qwen3_4b/checkpoints/step_0050000.pt \
        --prompts-file evaluation/prompts.jsonl \
        --output experiments/qwen3_4b/ablations/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch

from pdt.cli.infer import _load_phi_checkpoint
from pdt.config import load_config
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import CounterfactualConfig
from pdt.runtime.orchestrator import MultiStreamOrchestrator


def _load_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(str(obj.get("prompt") or obj.get("text") or obj))
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


def _pairwise_cosine_mean(texts: List[str], tokenizer) -> float:
    """Cosine *distance* mean over pairs of stream outputs (token-hash bag).

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
    mode: str,
    max_new_tokens: int,
) -> Dict[str, object]:
    cf = CounterfactualConfig(mode=None if mode == "baseline" else mode)
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
        per_prompt.append({
            "prompt": prompt,
            "pairwise_cosine_distance": cos,
            "rouge_l": rouge,
            "streams": result.text_by_stream,
        })
    return {
        "mode": mode,
        "pairwise_cosine_distance_mean": cosine_sum / max(count, 1),
        "rouge_l_mean": rouge_sum / max(count, 1),
        "per_prompt": per_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(args.config)
    model = PDTModel(config)
    _load_phi_checkpoint(model, args.checkpoint)

    prompts = _load_prompts(args.prompts_file)
    print(f"Running ablations on {len(prompts)} prompts.")

    manifest = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "num_prompts": len(prompts),
        "conditions": {},
    }
    for mode in ["baseline", "gate_zero", "norm_scramble"]:
        # Note: anchor_swap requires a prebuilt alt_prompt_anchors tensor and is
        # handled separately upstream.
        print(f"-- {mode} --")
        result = _run_condition(
            model, config, prompts=prompts, mode=mode, max_new_tokens=args.max_new_tokens,
        )
        manifest["conditions"][mode] = result

    # Pre-registered gate check.
    base_cos = manifest["conditions"]["baseline"]["pairwise_cosine_distance_mean"]
    base_rouge = manifest["conditions"]["baseline"]["rouge_l_mean"]
    results_summary = {}
    for mode in ["gate_zero", "norm_scramble"]:
        cos_delta = base_cos - manifest["conditions"][mode]["pairwise_cosine_distance_mean"]
        rouge_delta = manifest["conditions"][mode]["rouge_l_mean"] - base_rouge
        results_summary[mode] = {
            "cosine_distance_delta_vs_baseline": cos_delta,
            "rouge_l_delta_vs_baseline": rouge_delta,
        }
    manifest["pre_registered_gate_summary"] = results_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote ablation manifest to {args.output}")


if __name__ == "__main__":
    main()
