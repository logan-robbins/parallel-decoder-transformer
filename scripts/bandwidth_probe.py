"""The price of parallelism, in nats. Forward passes only, no training, $0.

The question
------------
Sequential decoding computes p(y) = prod_t p(y_t | x, y_<t). Parallel decoding
partitions y into segments s_1..s_K and asks stream k to emit s_k WITHOUT having
read s_<k. The exact quantity stream k is missing is I(s_k ; s_<k | x).

That term hides two mechanisms with different owners:

    I(s_k ; s_<k | x) = H(partition | x)               ADDRESSING -> the planner
                      + I(s_k ; s_<k | x, partition)   CONTENT    -> the bus

Conditioning on the partition is what separates them. This script measures both,
directly, by scoring the same reference text under three conditions:

    D_none = CE(s_k | generic)              stream k with nothing
    D_plan = CE(s_k | targeted_k)           stream k with the plan, no siblings
    D_seq  = CE(s_k | generic + s_<k)       the SEQUENTIAL decoder's own loss

Every quantity is operationally real. D_seq is not a hypothetical oracle: it is
literally the number an ordinary autoregressive decoder computes when it writes
s_k in order. So:

    A_k   = D_none - D_plan     nats/token the PLAN buys       (addressing)
    GAP_k = D_plan - D_seq      nats/token the BUS must close  (content)

GAP_k * |s_k| is the total budget, in nats, that a notes bus has to deliver for
stream k. That is the bandwidth of coordination, and it is the whole ballgame:

  * GAP ~ 0  -> the streams are already independent given the plan. The bus is
               unnecessary; parallel decoding is free. (Screening-off wins.)
  * GAP huge -> no narrow channel can close it. The idea is dead.
  * GAP small but > 0 -> a narrow bus is exactly the right instrument, and its
               required capacity is now a measured number rather than a guess.

Only the third outcome supports the architecture, and it is a real possibility
precisely because the trunk already knows these stories: what stream 2 lacks is
not the plot but the ARBITRARY CHOICES stream 1 made -- which window, which name
variant, whether the crocodile was introduced already. Shared priors are free;
coin flips are not. The bus carries coin flips.

Note on the sign of GAP
-----------------------
GAP can come out NEGATIVE: the plan clause ("tell me only the middle third") is
a stronger cue for s_2 than the generic prompt plus s_1, because it names the
target directly while the sequential prefix only implies it. A negative GAP is
not a bug -- it means the plan is a better conditioner than the sibling text, and
the bus has nothing to add. That is a real, reportable, falsifying outcome.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

STORIES = [
    "Peter Pan",
    "Cinderella",
    "The Three Little Pigs",
    "Little Red Riding Hood",
    "Goldilocks and the Three Bears",
    "Hansel and Gretel",
    "Jack and the Beanstalk",
    "Snow White",
    "Pinocchio",
    "The Wizard of Oz",
]

_STYLE = " Begin immediately with the story itself. Write continuous prose, no preamble, no title, no commentary."
GENERIC = "Tell me the story of {story}." + _STYLE
TARGETED = [
    "Tell me only the first third of the story of {story}." + _STYLE,
    "Tell me only the middle third of the story of {story}." + _STYLE,
    "Tell me only the final third of the story of {story}." + _STYLE,
]
K = len(TARGETED)
DENSE_NOTE_BITS = 256 * 16  # d_notes=256 transported as BF16


@dataclass
class Row:
    story: str
    k: int
    n_tokens: int
    d_none: float
    d_plan: float
    d_seq: float

    @property
    def addressing(self) -> float:
        """nats/token the plan buys over nothing."""
        return self.d_none - self.d_plan

    @property
    def gap(self) -> float:
        """nats/token the bus must close: plan-only vs. having actually read s_<k."""
        return self.d_plan - self.d_seq


class Scorer:
    def __init__(self, device: str, dtype: torch.dtype):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int) -> str:
        ids = self.tok(self.prompt(text), return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)

    @torch.no_grad()
    def ce(self, prefix: str, target: str) -> tuple[float, int]:
        """Mean CE in nats/token of `target` given `prefix`, scoring target only.

        prefix is already chat-templated and ends at the assistant generation
        point; target is appended as the assistant's own continuation. This is
        exactly the quantity an autoregressive decoder minimises, so D_seq below
        is a real decoder's loss and not a stand-in for one.

        Tokenise the CONCATENATION, never the pieces. BPE merges across the
        boundary -- ". Once" and "Once" are different token sequences -- so
        tokenising target separately and splicing would score a sequence the
        model never sees. The three conditions here have different prefixes, so
        that artifact would not cancel: it would land straight in the GAP and
        masquerade as coordination information.
        """
        n_p = self.tok(prefix, return_tensors="pt").input_ids.shape[1]
        full = self.tok(prefix + target, return_tensors="pt").input_ids.to(self.device)
        n = full.shape[1] - n_p
        if n < 2:
            raise ValueError("target too short to score")
        logits = self.model(full).logits
        # token at position i is predicted by logits at i-1; target spans [n_p, len)
        pred = logits[0, n_p - 1 : -1, :].float()
        gold = full[0, n_p:]
        return float(F.cross_entropy(pred, gold, reduction="mean")), n


def split_thirds(text: str) -> List[str]:
    sents = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if len(sents) < 2 * K:
        raise ValueError(f"telling too short: {len(sents)} sentences")
    n = len(sents)
    return [". ".join(sents[(n * i) // K : (n * (i + 1)) // K]) + "." for i in range(K)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-tokens", type=int, default=600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    s = Scorer(args.device, getattr(torch, args.dtype))
    rows: List[Row] = []

    for story in STORIES:
        # Ground truth is the trunk's OWN telling, split into thirds. In-distribution
        # by construction, and the segment boundaries are the trunk's own narrative
        # order rather than an outsider's opinion of where the story divides.
        full = s.generate(GENERIC.format(story=story), args.ref_tokens)
        try:
            thirds = split_thirds(full)
        except ValueError as e:
            print(f"  SKIP {story}: {e}", flush=True)
            continue

        gen_p = s.prompt(GENERIC.format(story=story))
        for k in range(K):
            tgt = thirds[k]
            d_none, n = s.ce(gen_p, tgt)
            d_plan, _ = s.ce(s.prompt(TARGETED[k].format(story=story)), tgt)
            # The sequential decoder: generic prompt, siblings already written.
            # For k=0 the prefix is empty, so d_seq == d_none and gap == addressing.
            prior = " ".join(thirds[:k])
            d_seq, _ = s.ce(gen_p + prior + (" " if prior else ""), tgt)
            row = Row(story, k, n, d_none, d_plan, d_seq)
            rows.append(row)
            print(
                f"  {story:32s} k={k}  n={n:4d}  D_none={d_none:5.3f}  D_plan={d_plan:5.3f}  "
                f"D_seq={d_seq:5.3f}  A={row.addressing:+6.3f}  GAP={row.gap:+6.3f}",
                flush=True,
            )

    print("\n" + "=" * 78)
    print("THE PRICE OF PARALLELISM")
    print("=" * 78)
    # k=0 is definitionally free (no siblings exist), so it is reported but excluded
    # from the bus budget -- including it would dilute the mean toward zero for a
    # reason that has nothing to do with the channel.
    for k in range(K):
        sub = [r for r in rows if r.k == k]
        if not sub:
            continue
        token_count = sum(r.n_tokens for r in sub)
        if token_count <= 0:
            raise RuntimeError(f"stream {k} produced no scored target tokens.")
        a = sum(r.addressing * r.n_tokens for r in sub) / token_count
        g = sum(r.gap * r.n_tokens for r in sub) / token_count
        nats = sum(r.gap * r.n_tokens for r in sub) / len(sub)
        tag = "  (no siblings exist; free by definition)" if k == 0 else ""
        print(
            f"  stream {k}:  addressing = {a:+.3f} nats/tok   "
            f"GAP = {g:+.3f} nats/tok   = {nats:+7.1f} nats over the segment{tag}"
        )

    dep = [r for r in rows if r.k > 0]
    if dep:
        dep_tokens = sum(r.n_tokens for r in dep)
        if dep_tokens <= 0:
            raise RuntimeError("dependent streams produced no scored target tokens.")
        g = sum(r.gap * r.n_tokens for r in dep) / dep_tokens
        total = sum(r.gap * r.n_tokens for r in dep) / len(dep)
        print("-" * 78)
        print(f"  BUS BUDGET (streams 1..{K - 1}): {g:+.4f} nats/token, {total:+.1f} nats/segment")
        print(
            f"  Implemented note transport: {DENSE_NOTE_BITS} physical bits "
            "(256 BF16 values) per producer/write."
        )
        if total > 0:
            print(
                "  -> No nats-per-note conversion is reported: the dense continuous "
                "channel has no identified effective-bit capacity model."
            )
        else:
            print(
                "  -> NEGATIVE: the plan conditions s_k BETTER than actually reading s_<k.\n"
                "     Given the plan, the siblings are redundant. The bus has nothing to carry."
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
