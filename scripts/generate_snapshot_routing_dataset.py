"""Generate shared-snapshot routing examples for integrated PDT runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for idx in range(args.num_examples):
            values = {f"s{i + 1}": rng.randint(35, 120) for i in range(3)}
            top_key = max(values, key=values.get)
            streams = []
            for i, key in enumerate(values):
                top_stream = f"stream_{int(top_key[1:]) - 1}"
                span = f"{top_stream} owns the highest snapshot value"
                streams.append(
                    {
                        "stream_id": f"stream_{i}",
                        "local_observation": f"focus={key}",
                        "target_blocks": [
                            f"stream_{i} routes focus to {key} with value {values[key]}.",
                            f"Because {span}, stream_{i} coordinates around {top_key}.",
                        ],
                        "dependency_spans": [
                            {
                                "block_index": 1,
                                "token_span_text": span,
                                "source_stream": top_stream,
                                "source_block_index": 0,
                                "kind": "sibling_state",
                            }
                        ],
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "example_id": f"snapshot_{idx:06d}",
                        "family": "shared_snapshot_routing",
                        "split": "train",
                        "k": 3,
                        "shared_context": json.dumps(
                            {
                                **values,
                                "task": "coordinate responses under threshold and priority rules",
                            },
                            sort_keys=True,
                        ),
                        "visibility_lag_blocks": 1,
                        "stream_inputs": streams,
                        "generator": {"type": "programmatic_oracle", "seed": args.seed},
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
