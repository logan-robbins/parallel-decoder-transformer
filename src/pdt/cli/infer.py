"""Multi-stream inference entry point.

Usage:
    uv run python -m pdt.cli.infer --config configs/pdt_qwen3_4b.yaml \
        --checkpoint experiments/qwen3_4b/checkpoints/step_0025000.pt \
        --prompt "Tell me three facts about orcas." \
        --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from pdt.config import load_config
from pdt.model import PDTModel
from pdt.runtime.counterfactuals import CounterfactualConfig
from pdt.runtime.orchestrator import MultiStreamOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cf",
        type=str,
        choices=["none", "gate_zero", "norm_scramble", "anchor_swap"],
        default="none",
        help="Counterfactual intervention to apply.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(args.config)
    model = PDTModel(config)
    if args.checkpoint:
        _load_phi_checkpoint(model, args.checkpoint)
    cf = CounterfactualConfig(mode=args.cf, seed=args.seed)
    orch = MultiStreamOrchestrator(model, model.trunk_adapter.tokenizer, config, counterfactual=cf)

    result = orch.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    payload = {
        "prompt": args.prompt,
        "cf_mode": args.cf,
        "text_by_stream": result.text_by_stream,
        "plan_slot_ids": result.plan_slot_ids.squeeze(0).tolist(),
        "snapshot0_anchors_shape": list(result.snapshot0_anchors.shape),
        "agreement_history": result.agreement_history[:20],
        "rollback_events": result.rollback_events,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def _load_phi_checkpoint(model: PDTModel, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "sidecar" in state:
        model.sidecar.load_state_dict(state["sidecar"], strict=False)
    per_layer = state.get("per_layer_phi", {})
    for layer in model.instrumented_layers:
        key = f"layer_{layer.pdt_layer_idx}"
        bundle = per_layer.get(key)
        if not bundle:
            continue
        if layer.snc is not None and bundle.get("snc") is not None:
            layer.snc.load_state_dict(bundle["snc"], strict=False)
        if layer.stream_adapter is not None and bundle.get("stream_adapter") is not None:
            layer.stream_adapter.load_state_dict(bundle["stream_adapter"], strict=False)
        if layer.notes_gate is not None and bundle.get("notes_gate") is not None:
            with torch.no_grad():
                layer.notes_gate.copy_(bundle["notes_gate"])
        if layer.adapter_gate is not None and bundle.get("adapter_gate") is not None:
            with torch.no_grad():
                layer.adapter_gate.copy_(bundle["adapter_gate"])


if __name__ == "__main__":
    main()
