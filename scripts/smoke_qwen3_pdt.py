"""Load the pinned real trunk locally and execute one instrumented PDT round."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import torch


os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pdt.config import load_config  # noqa: E402
from pdt.model import PDTModel  # noqa: E402
from pdt.runtime.orchestrator import MultiStreamOrchestrator  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pdt_qwen3_4b.yaml")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires torch.cuda.is_available()")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires torch.backends.mps.is_available()")

    config = load_config(args.config)
    device = torch.device(args.device)
    model = PDTModel(config)
    model.to(device)
    model.trunk_adapter.model.to(device).eval()
    model.eval()

    base_parameters = tuple(model.trunk_parameters())
    phi_parameters = tuple(model.all_trainable_parameters())
    if {id(parameter) for parameter in base_parameters} & {
        id(parameter) for parameter in phi_parameters
    }:
        raise RuntimeError("Real-model theta_pre/phi parameter partition overlaps.")
    phi_count = sum(parameter.numel() for parameter in phi_parameters)
    if phi_count != 401_325_095:
        raise RuntimeError(f"Canonical phi count drifted: expected 401325095, got {phi_count}.")

    orchestrator = MultiStreamOrchestrator(
        model,
        model.trunk_adapter.tokenizer,
        config,
    )
    result = orchestrator.generate("PDT real-model smoke test.", max_new_tokens=1)
    expected_streams = set(config.runtime.streams)
    if set(result.tokens_by_stream) != expected_streams:
        raise RuntimeError("Real-model runtime returned the wrong stream set.")
    if any(len(tokens) != 1 for tokens in result.tokens_by_stream.values()):
        raise RuntimeError("Real-model runtime did not emit exactly one token per stream.")
    if not torch.isfinite(result.planner_logits).all():
        raise RuntimeError("Real-model planner produced non-finite logits.")

    print("device:", device)
    print("base_parameters:", sum(parameter.numel() for parameter in base_parameters))
    print("phi_parameters:", phi_count)
    print("planner_dtype:", result.planner_logits.dtype)
    print("tokens_by_stream:", result.tokens_by_stream)
    print("real PDT forward contract passed")


if __name__ == "__main__":
    main()
