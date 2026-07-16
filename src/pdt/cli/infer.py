"""Multi-stream inference entry point.

Usage:
    uv run scripts/infer.py --config configs/pdt_qwen3_4b.yaml \
        --checkpoint experiments/qwen3_4b/checkpoints/step_0025000.pt \
        --prompt "Tell me three facts about orcas." \
        --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

from pdt.checkpoint import load_checkpoint
from pdt.config import load_config
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import CounterfactualConfig
from pdt.runtime.orchestrator import MultiStreamOrchestrator


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cf",
        type=str,
        choices=["none", "gate_zero", "norm_scramble", "bus_mutation"],
        default="none",
        help="Counterfactual intervention to apply.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mutation-producer", default=None)
    parser.add_argument("--mutation-block", type=int, default=0)
    parser.add_argument("--mutation-magnitude", type=float, default=1.0)
    args = parser.parse_args(argv)

    if not args.config.is_file():
        parser.error(f"--config must name an existing file: {args.config}")
    if not args.checkpoint.is_file():
        parser.error(f"--checkpoint must name an existing file: {args.checkpoint}")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.mutation_block < 0:
        parser.error("--mutation-block must be non-negative")
    if not math.isfinite(args.mutation_magnitude) or args.mutation_magnitude == 0:
        parser.error("--mutation-magnitude must be finite and non-zero")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(args.config)
    if args.mutation_producer is not None and args.mutation_producer not in config.runtime.streams:
        parser.error(
            f"--mutation-producer must be one of {config.runtime.streams}, "
            f"got {args.mutation_producer!r}"
        )
    if args.cf == "bus_mutation":
        completed_blocks = args.max_new_tokens // config.runtime.block_size
        if completed_blocks == 0 or args.mutation_block >= completed_blocks:
            parser.error(
                "bus_mutation requires --mutation-block to name a block that "
                f"will be published; max-new-tokens={args.max_new_tokens}, "
                f"tau={config.runtime.block_size}."
            )
    model = PDTModel(config)
    metadata = load_checkpoint(args.checkpoint, model)
    logging.getLogger(__name__).info(
        "Loaded checkpoint step=%d stage=%d format=%d",
        metadata.global_step,
        metadata.stage,
        metadata.format_version,
    )
    cf = CounterfactualConfig(
        mode=args.cf,
        seed=args.seed,
        mutation_producer=args.mutation_producer,
        mutation_block=args.mutation_block,
        mutation_magnitude=args.mutation_magnitude,
    )
    orch = MultiStreamOrchestrator(model, model.trunk_adapter.tokenizer, config, counterfactual=cf)

    result = orch.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    payload = {
        "prompt": args.prompt,
        "cf_mode": args.cf,
        "text_by_stream": result.text_by_stream,
        "plan_slot_ids": result.plan_slot_ids.squeeze(0).tolist(),
        "snapshot0_anchors_shape": list(result.snapshot0_anchors.shape),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
