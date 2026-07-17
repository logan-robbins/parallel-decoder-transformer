"""Model-relative missing-prefix score gaps. Forward passes only, no training.

The question
------------
Sequential decoding conditions segment ``s_k`` on the realized earlier text
``s_<k``. Parallel decoding removes that prefix. This script scores the same
reference text with one frozen model ``q`` under three prompt conditions:

    D_none = CE(s_k | generic)              stream k with nothing
    D_plan = CE(s_k | targeted_k)           stream k with the plan, no siblings
    D_seq  = CE(s_k | generic + s_<k)       the SEQUENTIAL decoder's own loss

The telescoping score differences are:

    PLAN_GAIN_k = D_none - D_plan
    PREFIX_GAP_k = D_plan - D_seq

These are useful *model-relative conditioning diagnostics*. They are not an
information-theoretic identity, not Shannon mutual information, and not a bit
budget for the bus. For arbitrary model conditionals q_plan and q_seq,

    E[log q_seq(Y) - log q_plan(Y)]

can exceed, undershoot, or have the opposite sign from the true conditional
mutual information. A channel-capacity claim requires an explicit finite
message alphabet plus an identified estimator such as the uniform-payload
audit in ``pdt.diagnostics.information``.

Interpret this probe only as a task-screening heuristic:

* a near-zero prefix gap says this model did not benefit from the supplied
  prefix under these prompts;
* a positive gap identifies examples where realized prefix text improves this
  model's score and communication may help;
* a negative gap says the targeted plan prompt scored the reference better.

Note on the sign of PREFIX_GAP
------------------------------
PREFIX_GAP can be NEGATIVE: the plan clause ("tell me only the middle third") is
a stronger cue for s_2 than the generic prompt plus s_1. This is not a paradox;
it is direct evidence that the score difference is model- and prompt-relative.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from pdt.diagnostics.information import nominal_storage_bits

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"

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
DENSE_NOTE_BITS = nominal_storage_bits(elements=256, bits_per_element=16)


@dataclass
class Row:
    story: str
    k: int
    n_tokens: int
    d_none: float
    d_plan: float
    d_seq: float

    @property
    def plan_gain(self) -> float:
        """Model-relative nats/token gained by the targeted plan prompt."""
        return self.d_none - self.d_plan

    @property
    def prefix_gap(self) -> float:
        """Model-relative plan-only CE minus realized-prefix CE."""
        return self.d_plan - self.d_seq


class Scorer:
    def __init__(self, device: str, dtype: torch.dtype):
        self.tok = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            dtype=dtype,
        )
        nn.Module.to(self.model, torch.device(device))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
        that artifact would not cancel: it would land straight in PREFIX_GAP
        and masquerade as a conditioning effect.
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
            # For k=0 the prefix is empty, so d_seq == d_none and the two
            # telescoping differences coincide.
            prior = " ".join(thirds[:k])
            d_seq, _ = s.ce(gen_p + prior + (" " if prior else ""), tgt)
            row = Row(story, k, n, d_none, d_plan, d_seq)
            rows.append(row)
            print(
                f"  {story:32s} k={k}  n={n:4d}  D_none={d_none:5.3f}  D_plan={d_plan:5.3f}  "
                f"D_seq={d_seq:5.3f}  PLAN={row.plan_gain:+6.3f}  "
                f"PREFIX={row.prefix_gap:+6.3f}",
                flush=True,
            )

    print("\n" + "=" * 78)
    print("MODEL-RELATIVE MISSING-PREFIX SCORE GAPS")
    print("=" * 78)
    # k=0 is definitionally free (no siblings exist), so it is reported but excluded
    # from the dependent-stream summary -- including it would dilute the mean
    # toward zero for a reason that has nothing to do with the channel.
    for k in range(K):
        sub = [r for r in rows if r.k == k]
        if not sub:
            continue
        token_count = sum(r.n_tokens for r in sub)
        if token_count <= 0:
            raise RuntimeError(f"stream {k} produced no scored target tokens.")
        a = sum(r.plan_gain * r.n_tokens for r in sub) / token_count
        g = sum(r.prefix_gap * r.n_tokens for r in sub) / token_count
        nats = sum(r.prefix_gap * r.n_tokens for r in sub) / len(sub)
        tag = "  (no siblings exist; free by definition)" if k == 0 else ""
        print(
            f"  stream {k}:  plan gain = {a:+.3f} nats/tok   "
            f"prefix gap = {g:+.3f} nats/tok   = {nats:+7.1f} score-nats/segment{tag}"
        )

    dep = [r for r in rows if r.k > 0]
    if dep:
        dep_tokens = sum(r.n_tokens for r in dep)
        if dep_tokens <= 0:
            raise RuntimeError("dependent streams produced no scored target tokens.")
        g = sum(r.prefix_gap * r.n_tokens for r in dep) / dep_tokens
        total = sum(r.prefix_gap * r.n_tokens for r in dep) / len(dep)
        print("-" * 78)
        print(
            f"  PREFIX SCORE GAP (streams 1..{K - 1}): "
            f"{g:+.4f} nats/token, {total:+.1f} score-nats/segment"
        )
        print(
            f"  Implemented note transport: {DENSE_NOTE_BITS} physical bits "
            "(256 BF16 values) per producer/write."
        )
        if total > 0:
            print(
                "  -> This is not converted to delivered bits: arbitrary model CE "
                "differences are not Shannon mutual information."
            )
        else:
            print(
                "  -> NEGATIVE: under this model and these prompts, the targeted plan "
                "scores s_k better than the realized earlier text."
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
