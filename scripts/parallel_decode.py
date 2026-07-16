"""K streams decoding simultaneously from one frozen trunk. No bus. No training.

What this is
------------
The loop, in its crudest honest form:

    t=0   partition the task into K prompts        <- the "planner", as plain text
    t>0   ONE batched forward, K tokens out        <- the streams, literally parallel
          concatenate                              <- the artifact

There is no notes bus here, and that is deliberate. This is the BASELINE the bus
has to beat, and it has to exist before "the bus helps" can mean anything.

Why the prize is real
---------------------
Autoregressive decode is usually memory-bound at small batch. A packed batch-K
matmul can reuse frozen weight traffic while emitting K frontier tokens. It
does the same total arithmetic and still pays K private KV-cache reads, so the
per-step cost is not guaranteed to equal batch 1. When the task has K balanced,
weakly dependent segments, the same primitive can improve both aggregate token
throughput and complete-answer latency. This script measures the vanilla-trunk
primitive on one device; it does not measure PDT's sidecar or establish a CUDA
speed claim.

Why no bus, yet
---------------
Addressing (which segment is mine) is handled here by the prompt: the frozen
trunk follows "tell me only the middle third" at ~89% on held-out stories, so the
partition is free. What the prompt CANNOT give stream k is the arbitrary choices
its siblings made -- which window, which name variant, whether the crocodile was
already introduced. Those are the seams, and the seams are what a bus would carry.

So the output of this script is not "a story." It is a LIST OF DEFECTS at the
seams, and that list is the bus's job description. Run bandwidth_probe.py for the
same quantity in nats.
"""

from __future__ import annotations

import argparse
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"

_STYLE = " Begin immediately with the story itself. Write continuous prose, no preamble, no title, no commentary."
GENERIC = "Tell me the story of {story}." + _STYLE
PARTITION = [
    "Tell me only the first third of the story of {story}." + _STYLE,
    "Tell me only the middle third of the story of {story}." + _STYLE,
    "Tell me only the final third of the story of {story}." + _STYLE,
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--story", default="Peter Pan")
    ap.add_argument("--tokens-per-stream", type=int, default=160)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    # Left padding: batched decode appends new tokens at the right edge, so every
    # sequence's generation frontier must line up there. Right padding would have
    # the streams generating from the middle of their own pad runs.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=getattr(torch, args.dtype),
    )
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    K = len(PARTITION)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p.format(story=args.story)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in PARTITION
    ]

    # ---- K streams, ONE batched forward per step
    batch = tok(prompts, return_tensors="pt", padding=True).to(args.device)
    torch.mps.synchronize() if args.device == "mps" else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **batch,
            max_new_tokens=args.tokens_per_stream,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    torch.mps.synchronize() if args.device == "mps" else None
    t_par = time.time() - t0
    segments: List[str] = [
        tok.decode(out[i, batch["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        for i in range(K)
    ]

    # ---- the sequential control: one stream writes the whole thing.
    # Same total token count, so the comparison is like-for-like: this is the
    # work parallel decoding is trying to avoid.
    seq_prompt = tok.apply_chat_template(
        [{"role": "user", "content": GENERIC.format(story=args.story)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    seq_ids = tok(seq_prompt, return_tensors="pt").to(args.device)
    torch.mps.synchronize() if args.device == "mps" else None
    t0 = time.time()
    with torch.no_grad():
        seq_out = model.generate(
            **seq_ids,
            max_new_tokens=args.tokens_per_stream * K,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    torch.mps.synchronize() if args.device == "mps" else None
    t_seq = time.time() - t0
    seq_text = tok.decode(seq_out[0, seq_ids["input_ids"].shape[1] :], skip_special_tokens=True)

    n = args.tokens_per_stream
    print("=" * 78)
    print(f"{K} STREAMS, ONE FROZEN TRUNK, NO BUS  --  {args.story}")
    print("=" * 78)
    for i, seg in enumerate(segments):
        print(f"\n--- stream {i} ({PARTITION[i].split(' of the story')[0][14:]}) ---")
        print(seg)

    print("\n" + "=" * 78)
    print("CONCATENATED")
    print("=" * 78)
    print(" ".join(segments))

    print("\n" + "=" * 78)
    print("WALL CLOCK")
    print("=" * 78)
    print(
        f"  parallel : {t_par:6.1f}s  for {K} x {n} = {K * n} tokens  "
        f"({K * n / t_par:5.1f} tok/s aggregate, {n / t_par:4.1f} tok/s per stream)"
    )
    print(f"  sequential: {t_seq:6.1f}s  for {K * n} tokens  ({K * n / t_seq:5.1f} tok/s)")
    print(f"  ---> latency speedup: {t_seq / t_par:.2f}x   (ceiling is K = {K}x)")
    print()
    print("  The gap to Kx includes batching, K-fold prefill, private KV traffic,")
    print("  stream imbalance, and backend effects. PDT additionally pays planning,")
    print("  sidecar, synchronization, and any repair cost; measure those separately.")

    print("\n" + "=" * 78)
    print("SEQUENTIAL CONTROL (what one stream writes, for seam comparison)")
    print("=" * 78)
    print(seq_text)


if __name__ == "__main__":
    main()
