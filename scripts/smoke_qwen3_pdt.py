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
    MAX_PLAN_NODES,
    PLAN_SEMANTIC_DIM,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_EMBEDDING_REVISION,
    encode_projected_plan_texts,
)
from pdt.model import PDTModel  # noqa: E402
from pdt.runtime.orchestrator import MultiStreamOrchestrator  # noqa: E402
from pdt.trunk.physical_decoder import PhysicalFrontierCache  # noqa: E402
from model_intrinsic_parallel.training_example import TrainingExample  # noqa: E402
from model_intrinsic_parallel.wikipedia_source import (  # noqa: E402
    render_model_visible_text,
)


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
    parser.add_argument(
        "--oracle-example",
        type=Path,
        default=Path("data/model_intrinsic_parallel/examples/great_stink.json"),
    )
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires torch.cuda.is_available()")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires torch.backends.mps.is_available()")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    prompt, oracle_nodes, oracle_mask = _load_audited_oracle_plan(
        args.oracle_example
    )
    config = load_config(args.config)
    if args.trunk_profile is not None:
        apply_trunk_profile(config, args.trunk_profile)
        config.validate()
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
    frontier_shapes: list[tuple[int, int]] = []
    cache_shapes: list[tuple[int, ...]] = []
    original_planner = model.encode_planner_prompt
    original_frontier = model.forward_frontier

    def audited_planner(input_ids: torch.Tensor, attention_mask: torch.Tensor):
        planner_shapes.append((input_ids.size(0), input_ids.size(1)))
        return original_planner(input_ids, attention_mask)

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
    if frontier_shapes[0][0] != stream_count:
        raise RuntimeError(f"Physical prefill was not three-row packed: {frontier_shapes[0]}.")
    if any(shape != (stream_count, 1) for shape in frontier_shapes[1:]):
        raise RuntimeError(
            f"Continuation was not physically three-row packed: {frontier_shapes[1:]}."
        )

    print("device:", device)
    print("oracle_example:", args.oracle_example)
    print("shared_frozen_parameters:", _parameter_count(shared_parameters))
    print("physical_branch_parameters:", _parameter_count(branch_parameters))
    print("physical_extension_parameters:", _parameter_count(extension_parameters))
    print("sidecar_parameters:", _parameter_count(sidecar_parameters))
    print("total_materialized_parameters:", _parameter_count(shared_parameters + trainable_parameters))
    print("fork_layer:", model.physical_decoder.fork_layer)
    print("physical_layer_indices:", model.physical_decoder.layer_indices)
    print("planner_dtype:", result.plan_nodes.dtype)
    print("tokens_by_stream:", result.tokens_by_stream)
    print("dynamic_codes_by_stream:", result.dynamic_codes_by_stream)
    print("planner_call_shapes:", planner_shapes)
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
    path: Path,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Audited oracle example does not exist: {path}")
    example = TrainingExample.model_validate_json(path.read_text(encoding="utf-8"))
    if len(example.lanes) != 3:
        raise ValueError("The physical smoke requires exactly three audited semantic lanes.")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(
        SEMANTIC_EMBEDDING_MODEL,
        revision=SEMANTIC_EMBEDDING_REVISION,
        device="cpu",
        local_files_only=True,
    )
    fact_text = {fact.fact_id: fact.statement for fact in example.facts}
    lane_vectors: list[list[list[float]]] = []
    lane_masks: list[list[bool]] = []
    for lane in example.lanes:
        node_texts = [
            (
                f"Section: {lane.focus}. Role: {lane.selection_rationale}. "
                f"Outline objective: {node.purpose}. Owned facts: "
                + " ".join(fact_text[fact_id] for fact_id in node.owner_fact_ids)
                + "."
            )
            for node in lane.plan_nodes
        ]
        projected = encode_projected_plan_texts(embedder, node_texts)
        padding = MAX_PLAN_NODES - len(projected)
        if padding < 0:
            raise ValueError(
                f"Audited lane {lane.lane_id!r} exceeds {MAX_PLAN_NODES} plan nodes."
            )
        lane_vectors.append(
            projected + [[0.0] * PLAN_SEMANTIC_DIM for _ in range(padding)]
        )
        lane_masks.append([True] * len(projected) + [False] * padding)
    del embedder
    gc.collect()
    nodes = torch.tensor(lane_vectors, dtype=torch.float32).unsqueeze(0)
    nodes = nodes * math.sqrt(PLAN_SEMANTIC_DIM)
    mask = torch.tensor(lane_masks, dtype=torch.bool).unsqueeze(0)
    source = render_model_visible_text(
        example.source_document.headings,
        example.source_document.paragraphs,
    )
    prompt = (
        f"Source title: {example.source_document.source.title}\n\n{source}\n\n"
        f"Task: {example.question}\n\n"
        "Write three complementary long-form sections that jointly answer the task. "
        "Each section must be several connected paragraphs. Do not write a short answer, "
        "a question-answer list, or a separate final synthesis."
    )
    return prompt, nodes, mask


def _parameter_count(parameters: Sequence[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _maximum_resident_set_gib() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


if __name__ == "__main__":
    main()
