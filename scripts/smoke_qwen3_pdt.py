"""Load the pinned model and execute the real three-decoder physical frontier."""

from __future__ import annotations

import argparse
import gc
import math
import os
import resource
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import torch


os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pdt.config import TRUNK_PROFILES, apply_trunk_profile, load_config  # noqa: E402
from pdt.datasets.real_plan_retokenize import (  # noqa: E402
    PLAN_SEMANTIC_DIM,
    render_planner_prompt,
)
from pdt.datasets.real_plan_schema import RealPlanExample, validate_real_plan_example  # noqa: E402
from pdt.training.dataset import RealPlanDataset  # noqa: E402
from pdt.model import PDTModel  # noqa: E402
from pdt.runtime.orchestrator import MultiStreamOrchestrator  # noqa: E402
from pdt.trunk.physical_decoder import PhysicalFrontierCache  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pdt_qwen3_4b.yaml")
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=None,
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument(
        "--raw-records",
        type=Path,
        default=Path(
            "data/processed/model_intrinsic_parallel/"
            "qwen3_4b_instruct_2507/audited_real_plan_v2.jsonl"
        ),
    )
    parser.add_argument(
        "--tokenized-records",
        type=Path,
        default=Path(
            "data/processed/model_intrinsic_parallel/"
            "qwen3_4b_instruct_2507/audited_real_plan_tokenized_v3.jsonl"
        ),
    )
    parser.add_argument(
        "--example-id",
        default="enwiki-1666662-1361217759",
    )
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
    profile = TRUNK_PROFILES[config.trunk.profile]
    prompt, oracle_nodes, oracle_mask = _load_audited_oracle_plan(
        raw_path=args.raw_records,
        tokenized_path=args.tokenized_records,
        example_id=args.example_id,
        tokenizer_name=profile.base_model,
        tokenizer_revision=profile.revision,
    )
    device = torch.device(args.device)
    model = PDTModel(config)
    gc.collect()
    model.to(device)
    trunk_module: torch.nn.Module = model.trunk_adapter.model
    trunk_module.to(device=device).eval()
    model.eval()

    shared_parameters = tuple(model.trunk_parameters())
    branch_parameters = tuple(model.decoder_branch_parameters())
    extension_parameters = tuple(model.per_layer_phi_parameters())
    sidecar_parameters = tuple(model.sidecar_parameters())
    trainable_parameters = (
        branch_parameters + extension_parameters + sidecar_parameters
    )
    if {id(parameter) for parameter in shared_parameters} & {
        id(parameter) for parameter in trainable_parameters
    }:
        raise RuntimeError("Real-model shared/trainable parameter partition overlaps.")
    if any(parameter.requires_grad for parameter in shared_parameters):
        raise RuntimeError("The shared knowledge trunk must remain frozen.")
    if not trainable_parameters or any(
        not parameter.requires_grad for parameter in trainable_parameters
    ):
        raise RuntimeError("Every physical-decoder and sidecar parameter must be trainable.")
    if model.physical_decoder.num_decoders != 3:
        raise RuntimeError("The real-model smoke requires exactly three physical decoders.")
    if model.physical_decoder.layer_indices != config.instrumentation.target_layers:
        raise RuntimeError("The physical decoder did not materialize the configured upper layers.")

    orchestrator = MultiStreamOrchestrator(
        model,
        model.trunk_adapter.tokenizer,
        config,
    )
    planner_shapes: list[tuple[int, int]] = []
    shared_shapes: list[tuple[int, int]] = []
    frontier_shapes: list[tuple[int, int]] = []
    cache_shapes: list[tuple[int, ...]] = []
    original_planner = model.encode_planner_prompt
    original_shared = model.trunk_adapter.forward_shared
    original_frontier = model.forward_frontier

    def audited_planner(input_ids: torch.Tensor, attention_mask: torch.Tensor):
        planner_shapes.append((input_ids.size(0), input_ids.size(1)))
        return original_planner(input_ids, attention_mask)

    def audited_shared(**kwargs):
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RuntimeError("Shared trunk audit requires rank-2 input_ids.")
        shared_shapes.append((input_ids.size(0), input_ids.size(1)))
        return original_shared(**kwargs)

    def audited_frontier(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RuntimeError("Physical frontier audit requires rank-2 input_ids.")
        frontier_shapes.append((input_ids.size(0), input_ids.size(1)))
        output = original_frontier(*args, **kwargs)
        cache = output.past_key_values
        if not isinstance(cache, PhysicalFrontierCache):
            raise RuntimeError("Physical frontier did not return its grouped private cache.")
        for keys in cache.branch_keys:
            if keys is None or keys.ndim != 5 or keys.size(1) != 3:
                raise RuntimeError(
                    "Every upper cache must retain "
                    "[documents, three_decoders, kv_heads, tokens, head_dim]."
                )
        cache_shapes.append(tuple(cache.branch_keys[0].shape))
        return output

    _synchronize(device)
    started = time.perf_counter()
    with (
        patch.object(model, "encode_planner_prompt", side_effect=audited_planner),
        patch.object(model.trunk_adapter, "forward_shared", side_effect=audited_shared),
        patch.object(model, "forward_frontier", side_effect=audited_frontier),
    ):
        result = orchestrator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            plan_nodes_override=oracle_nodes,
            plan_mask_override=oracle_mask,
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
    if len(planner_shapes) != 1 or planner_shapes[0][0] != 1:
        raise RuntimeError(f"Expected one shared planner call, got {planner_shapes}.")
    expected_calls = 1 + args.max_new_tokens
    if len(frontier_shapes) != expected_calls:
        raise RuntimeError(
            f"Expected packed prefill + decode = {expected_calls} physical calls, "
            f"got {len(frontier_shapes)}: {frontier_shapes}."
        )
    stream_count = len(config.runtime.streams)
    expected_shared_shapes = [
        (1, planner_shapes[0][1]),
        (1, frontier_shapes[0][1]),
        *[(stream_count, 1)] * args.max_new_tokens,
    ]
    if shared_shapes != expected_shared_shapes:
        raise RuntimeError(
            "The frozen lower trunk did not encode each common prefill once before "
            f"the three-way fork: expected={expected_shared_shapes}, got={shared_shapes}."
        )
    if frontier_shapes[0][0] != stream_count:
        raise RuntimeError(f"Physical prefill was not three-row packed: {frontier_shapes[0]}.")
    if any(shape != (stream_count, 1) for shape in frontier_shapes[1:]):
        raise RuntimeError(
            f"Continuation was not physically three-row packed: {frontier_shapes[1:]}."
        )
    initial_cache_tokens = cache_shapes[0][-2]
    expected_cache_tokens = [
        initial_cache_tokens + offset for offset in range(len(cache_shapes))
    ]
    actual_cache_tokens = [shape[-2] for shape in cache_shapes]
    if actual_cache_tokens != expected_cache_tokens:
        raise RuntimeError(
            "Grouped physical cache did not advance exactly one position per "
            f"synchronous decode call: {actual_cache_tokens}."
        )
    synchronized_rounds = [
        tuple(
            result.tokens_by_stream[stream][round_index]
            for stream in config.runtime.streams
        )
        for round_index in range(args.max_new_tokens)
    ]
    if any(len(round_tokens) != stream_count for round_tokens in synchronized_rounds):
        raise RuntimeError("A synchronized decode round did not emit one token per lane.")

    print("device:", device)
    print("example_id:", args.example_id)
    print("raw_records:", args.raw_records)
    print("tokenized_records:", args.tokenized_records)
    print("shared_frozen_parameters:", _parameter_count(shared_parameters))
    print("physical_branch_parameters:", _parameter_count(branch_parameters))
    print("physical_extension_parameters:", _parameter_count(extension_parameters))
    print("sidecar_parameters:", _parameter_count(sidecar_parameters))
    print("total_materialized_parameters:", _parameter_count(shared_parameters + trainable_parameters))
    print("fork_layer:", model.physical_decoder.fork_layer)
    print("physical_layer_indices:", model.physical_decoder.layer_indices)
    print("planner_dtype:", result.plan_nodes.dtype)
    print("tokens_by_stream:", result.tokens_by_stream)
    print("synchronized_three_token_rounds:", synchronized_rounds)
    print("dynamic_codes_by_stream:", result.dynamic_codes_by_stream)
    print("planner_call_shapes:", planner_shapes)
    print("shared_lower_call_shapes:", shared_shapes)
    print("frontier_call_shapes:", frontier_shapes)
    print("upper_cache_shapes:", cache_shapes)
    print("maximum_resident_set_gib:", f"{_maximum_resident_set_gib():.3f}")
    print("elapsed_seconds:", f"{elapsed:.6f}")
    print(
        "aggregate_generated_tokens_per_second:",
        f"{stream_count * args.max_new_tokens / elapsed:.6f}",
    )
    print("real physical three-decoder forward contract passed")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_audited_oracle_plan(
    *,
    raw_path: Path,
    tokenized_path: Path,
    example_id: str,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    raw_matches: list[RealPlanExample] = []
    if not raw_path.is_file():
        raise FileNotFoundError(f"Canonical v2 records do not exist: {raw_path}")
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = RealPlanExample.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{raw_path}:{line_number} violates the canonical v2 schema."
                ) from exc
            validate_real_plan_example(example)
            if example.source.source_id == example_id:
                raw_matches.append(example)
    if len(raw_matches) != 1:
        raise ValueError(
            f"Expected one canonical v2 record for {example_id!r}; "
            f"found {len(raw_matches)}."
        )
    example = raw_matches[0]
    dataset = RealPlanDataset(
        tokenized_path,
        expected_tokenizer=tokenizer_name,
        expected_tokenizer_revision=tokenizer_revision,
    )
    tokenized_matches = [
        dataset[index]
        for index in range(len(dataset))
        if dataset[index]["example_id"] == example_id
    ]
    if len(tokenized_matches) != 1:
        raise ValueError(
            f"Expected one canonical tokenized-v3 record for {example_id!r}; "
            f"found {len(tokenized_matches)}."
        )
    tokenized = tokenized_matches[0]
    if (
        tokenized["source_revision_id"] != example.source.revision_id
        or tokenized["source_model_visible_sha256"]
        != example.source.model_visible_sha256
    ):
        raise ValueError("Canonical v2 and tokenized-v3 source identities differ.")
    nodes = torch.tensor(
        tokenized["plan_semantic_targets"],
        dtype=torch.float32,
    ).unsqueeze(0)
    nodes = nodes * math.sqrt(PLAN_SEMANTIC_DIM)
    mask = torch.tensor(
        tokenized["plan_node_mask"],
        dtype=torch.bool,
    ).unsqueeze(0)
    if nodes.shape != (1, 3, 8, PLAN_SEMANTIC_DIM):
        raise ValueError(f"Canonical plan target has the wrong shape: {nodes.shape}.")
    if mask.shape != (1, 3, 8):
        raise ValueError(f"Canonical plan mask has the wrong shape: {mask.shape}.")
    return render_planner_prompt(example), nodes, mask


def _parameter_count(parameters: Sequence[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _maximum_resident_set_gib() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


if __name__ == "__main__":
    main()
