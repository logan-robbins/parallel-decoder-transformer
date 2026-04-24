"""Generate Latent Dependency Control examples without external data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for idx in range(args.num_examples):
            handle.write(json.dumps(_example(idx, rng), sort_keys=True) + "\n")


def _example(idx: int, rng: random.Random) -> dict[str, object]:
    values = rng.sample(range(35, 121), 3)
    trends = [rng.choice(["rising", "stable", "falling"]) for _ in range(3)]
    confidences = [round(rng.uniform(0.70, 0.99), 2) for _ in range(3)]
    risk_order = sorted(range(3), key=lambda i: (values[i], confidences[i]), reverse=True)
    top = risk_order[0]
    streams = []
    for i in range(3):
        level = "HIGH" if values[i] >= 90 else "MEDIUM" if values[i] >= 65 else "LOW"
        local = f"sensor=s{i + 1} value={values[i]} trend={trends[i]} confidence={confidences[i]:.2f}"
        block0 = (
            f"s{i + 1} reports {level} and {trends[i]} with confidence "
            f"{confidences[i]:.2f}."
        )
        span = f"stream_{top} has the highest risk"
        if i == top:
            block1 = f"Because {span}, stream_{i} keeps priority and asks siblings to monitor."
        else:
            block1 = f"Because {span}, stream_{i} defers priority and monitors its local channel."
        streams.append(
            {
                "stream_id": f"stream_{i}",
                "local_observation": local,
                "target_blocks": [block0, block1],
                "dependency_spans": [
                    {
                        "block_index": 1,
                        "token_span_text": span,
                        "source_stream": f"stream_{top}",
                        "source_block_index": 0,
                        "kind": "sibling_state",
                    }
                ],
            }
        )
    return {
        "example_id": f"ldc_{idx:06d}",
        "family": "latent_dependency_control",
        "split": "train",
        "k": 3,
        "shared_context": (
            "Coordinate three streams. First publish local state, then respond "
            "to sibling state after the reveal delay."
        ),
        "visibility_lag_blocks": 1,
        "stream_inputs": streams,
        "eval": {
            "permutation_invariant": False,
            "dependency_span_metric": "target_model_ce_delta",
            "nondependency_span_metric": "target_model_ce_delta",
        },
        "generator": {"type": "programmatic_oracle", "seed": rng.randrange(1_000_000_000)},
    }


if __name__ == "__main__":
    main()
