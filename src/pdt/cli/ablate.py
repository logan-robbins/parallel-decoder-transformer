"""Capture raw generation outputs for checkpoint counterfactuals.

Runs gate-zero, sibling-note norm scramble, and a targeted bus mutation plus
the baseline on a fixed prompt set. It records exact text, token IDs, and finite
bus codes. It computes no bag-of-token, lexical-overlap, or embedding proxy.
The dependency-span causal gate is computed from the aligned evaluation dataset.
Plan and source swaps require explicit cross-prompt donor state and are
therefore exposed by the runtime API rather than synthesized by this CLI.

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
from pdt.datasets.immutable_io import write_bytes_new
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
                if set(obj) != {"prompt"}:
                    raise ValueError(
                        "Prompt JSONL objects must contain exactly one 'prompt' field."
                    )
                value = obj["prompt"]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Prompt JSONL object must contain a non-empty prompt.")
                prompts.append(value.strip())
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Prompt JSONL rows must be JSON objects with exactly one prompt field."
                ) from exc
    return prompts


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
    for prompt in prompts:
        result = orch.generate(prompt, max_new_tokens=max_new_tokens)
        per_prompt.append(
            {
                "prompt": prompt,
                "text_by_stream": result.text_by_stream,
                "tokens_by_stream": result.tokens_by_stream,
                "dynamic_codes_by_stream": result.dynamic_codes_by_stream,
            }
        )
    return {
        "mode": mode,
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
        "schema_version": "pdt-counterfactual-generation-capture-v1",
        "not_an_empirical_result": True,
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

    write_bytes_new(
        args.output,
        (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    )
    print(f"Wrote ablation manifest to {args.output}")


if __name__ == "__main__":
    main()
