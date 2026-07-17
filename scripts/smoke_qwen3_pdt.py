"""Load the pinned real trunk locally and execute one instrumented PDT round."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from typing import cast
from unittest.mock import patch

import torch


os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pdt.config import TRUNK_PROFILES, apply_trunk_profile, load_config  # noqa: E402
from pdt.model import PDTModel  # noqa: E402
from pdt.runtime.orchestrator import MultiStreamOrchestrator  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pdt_qwen3_4b.yaml")
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=None,
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires torch.cuda.is_available()")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires torch.backends.mps.is_available()")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    config = load_config(args.config)
    if args.trunk_profile is not None:
        apply_trunk_profile(config, args.trunk_profile)
        config.validate()
    device = torch.device(args.device)
    model = PDTModel(config)
    model.to(device)
    trunk_module = cast(torch.nn.Module, model.trunk_adapter.model)
    trunk_module.to(device=device).eval()
    model.eval()

    base_parameters = tuple(model.trunk_parameters())
    phi_parameters = tuple(model.all_trainable_parameters())
    if {id(parameter) for parameter in base_parameters} & {
        id(parameter) for parameter in phi_parameters
    }:
        raise RuntimeError("Real-model theta_pre/phi parameter partition overlaps.")
    phi_count = sum(parameter.numel() for parameter in phi_parameters)
    expected_phi = {
        "qwen3_4b_instruct_2507": 156_484_647,
        "qwen3_14b": 305_374_247,
    }[config.trunk.profile]
    if phi_count != expected_phi:
        raise RuntimeError(
            f"Canonical phi count drifted: expected {expected_phi}, got {phi_count}."
        )

    orchestrator = MultiStreamOrchestrator(
        model,
        model.trunk_adapter.tokenizer,
        config,
    )
    call_shapes: list[tuple[int, int]] = []
    original_forward = model.trunk_adapter.forward

    def audited_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
            raise RuntimeError("Trunk audit requires rank-2 input_ids on every call.")
        call_shapes.append((input_ids.size(0), input_ids.size(1)))
        return original_forward(*args, **kwargs)

    _synchronize(device)
    started = time.perf_counter()
    with patch.object(model.trunk_adapter, "forward", side_effect=audited_forward):
        result = orchestrator.generate(
            "PDT real-model packed-frontier smoke test.",
            max_new_tokens=args.max_new_tokens,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started

    expected_streams = set(config.runtime.streams)
    if set(result.tokens_by_stream) != expected_streams:
        raise RuntimeError("Real-model runtime returned the wrong stream set.")
    if any(len(tokens) != args.max_new_tokens for tokens in result.tokens_by_stream.values()):
        raise RuntimeError("Real-model runtime emitted the wrong number of tokens per stream.")
    if not torch.isfinite(result.plan_nodes).all():
        raise RuntimeError("Real-model planner produced non-finite plan nodes.")
    if not torch.isfinite(result.planner_node_validity_logits).all():
        raise RuntimeError("Real-model planner produced non-finite validity logits.")
    expected_snapshots = args.max_new_tokens // config.runtime.block_size
    for stream, code_rows in result.dynamic_codes_by_stream.items():
        if len(code_rows) != expected_snapshots:
            raise RuntimeError(
                f"Expected {expected_snapshots} dynamic codes for {stream}, got {code_rows}."
            )
        if any(
            len(row) != config.runtime.notes_bus.num_codebooks
            or any(not 0 <= index < config.runtime.notes_bus.codes_per_codebook for index in row)
            for row in code_rows
        ):
            raise RuntimeError(f"Invalid finite dynamic-note code tuple for {stream}: {code_rows}.")
    expected_calls = 2 + args.max_new_tokens
    if len(call_shapes) != expected_calls:
        raise RuntimeError(
            f"Expected planner + packed prefill + decode = {expected_calls} calls, "
            f"got {len(call_shapes)}: {call_shapes}."
        )
    stream_count = len(config.runtime.streams)
    if call_shapes[0][0] != 1 or call_shapes[1][0] != stream_count:
        raise RuntimeError(f"Planner/prefill batch contract failed: {call_shapes[:2]}.")
    if any(shape != (stream_count, 1) for shape in call_shapes[2:]):
        raise RuntimeError(f"Continuation was not physically K-row packed: {call_shapes[2:]}.")

    print("device:", device)
    print("base_parameters:", sum(parameter.numel() for parameter in base_parameters))
    print("phi_parameters:", phi_count)
    print("planner_dtype:", result.plan_nodes.dtype)
    print("tokens_by_stream:", result.tokens_by_stream)
    print("dynamic_codes_by_stream:", result.dynamic_codes_by_stream)
    print("trunk_call_shapes:", call_shapes)
    print("elapsed_seconds:", f"{elapsed:.6f}")
    print(
        "aggregate_generated_tokens_per_second:",
        f"{stream_count * args.max_new_tokens / elapsed:.6f}",
    )
    print("real packed PDT forward contract passed")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    main()
