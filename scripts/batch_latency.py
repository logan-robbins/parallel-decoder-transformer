"""Per-step decode latency at batch=1 vs batch=K. The physics behind the speedup claim.

Why this exists
---------------
The whole latency argument for parallel decoding is one hardware fact: autoregressive
decode is MEMORY-BOUND. Emitting one token requires streaming every parameter from
memory, so a batch of K reads the same weights once and emits K tokens. If that
holds, K streams cost ~1x the per-step latency of one stream, and a K-way task
decomposition finishes in ~1/K the steps.

An end-to-end A/B (time the sequential run, time the parallel run, divide) CANNOT
measure this on a laptop. Sustained fp32 MPS load makes the M4 thermally throttle:
across one 8-topic run, sequential throughput drifted 10.3 -> 2.7 tok/s. Because the
sequential arm runs first and 3x longer, it absorbs more throttling, biasing the
ratio UP -- one topic reported 4.50x against a hard ceiling of 3x, which is
physically impossible and proves the instrument, not the speedup.

So measure the primitive directly, and defend it against drift:
  - ALTERNATE batch sizes trial by trial, so thermal state is shared, not confounded.
  - Discard warmup.
  - Report MEDIAN (robust to a throttle step landing mid-run) and the spread.
  - Report the derived speedup as K * (per-stream rate at batch K) / (rate at batch 1),
    which is the quantity the architecture actually buys.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT = "Tell me the history of the Second World War. Write continuous prose densely packed with specific names, dates, places and numbers. Begin immediately."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--steps", type=int, default=32, help="decode steps timed per trial")
    ap.add_argument("--trials", type=int, default=5, help="alternating repeats per batch size")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def trial(bs: int) -> float:
        """Seconds per DECODE STEP at batch size bs, prefill excluded."""
        ids = tok([prompt] * bs, return_tensors="pt", padding=True).to(args.device)
        out = model(**ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:, :].argmax(-1)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            o = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1:, :].argmax(-1)
        sync()
        return (time.perf_counter() - t0) / args.steps

    print(f"warmup ({args.warmup} trials)...", flush=True)
    for _ in range(args.warmup):
        trial(1)
        trial(max(args.batches))

    # Alternate batch sizes within each round so thermal drift hits every batch size
    # equally instead of accumulating against whichever one runs last.
    samples: Dict[int, List[float]] = {b: [] for b in args.batches}
    for t in range(args.trials):
        for b in args.batches:
            samples[b].append(trial(b))
        print(f"  round {t + 1}/{args.trials} done", flush=True)

    print("\n" + "=" * 76)
    print(
        f"PER-STEP DECODE LATENCY  ({args.dtype} on {args.device}, {args.steps} steps/trial,"
        f" {args.trials} alternating trials)"
    )
    print("=" * 76)
    base = statistics.median(samples[args.batches[0]])
    print(
        f"  {'batch':>5s} {'ms/step':>9s} {'spread':>8s} {'tok/s tot':>10s} {'vs b=1':>7s} {'speedup':>8s}"
    )
    for b in args.batches:
        med = statistics.median(samples[b])
        lo, hi = min(samples[b]), max(samples[b])
        spread = (hi - lo) / med
        # What the architecture buys: K streams each running at (1/med) steps/s, versus
        # one stream that must emit K times as many tokens at (1/base) steps/s.
        print(
            f"  {b:5d} {med * 1000:9.2f} {spread:7.1%} {b / med:10.2f} {med / base:7.2f}x"
            f" {b * base / med:7.2f}x"
        )
    print("-" * 76)
    k3 = statistics.median(samples[3]) if 3 in samples else None
    if k3:
        print(f"  K=3 costs {k3 / base:.2f}x the per-step latency of K=1 while emitting 3 tokens")
        print(
            f"  => decomposing a task 3 ways finishes in {k3 / base / 3:.2f}x the wall clock"
            f" = {3 * base / k3:.2f}x speedup"
        )
        print()
        print("  This is the ceiling for a LOSSLESS decomposition. Real speedup is lower by")
        print("  whatever the K-fold prompt prefill costs, which shrinks as segments lengthen.")


if __name__ == "__main__":
    main()
