"""Can a LATENT vector replace the text address? Objective ground truth: the years.

The question
------------
fact_recall.py shows a TEXT address ("covering only 1942 to 1943") partitions K
streams well enough that they overlap only ~10% with no communication. But a text
address costs tokens and is not differentiable. The planner in a frozen-trunk
architecture must emit a LATENT that does the same job.

So: does an activation-space vector carry the address?

    v_k = mean over TRAIN topics of [ h_L(segment_k prompt) - h_L(generic prompt) ]

injected into a stream prompted ONLY with the generic ask. If the trunk starts
emitting facts from segment k, the address is a direction in activation space and
a planner can emit it. If not, the address is not linearly available at this site
and the plan must be delivered as something the trunk can ATTEND to.

Why this measurement is trustworthy where the previous one was not
-----------------------------------------------------------------
An earlier version of this experiment used stories ("the middle third of Peter
Pan") localised by embedding similarity against the trunk's own telling. That was
fatally circular: reference and baseline were both greedy decodes of the SAME
prompt, so the baseline was a literal PREFIX of its own ground truth. A null model
emitting fixed unrelated text reproduced its headline result exactly. It is gone.

Here the ground truth is external and arithmetic: EXTRACT THE YEARS, ASK WHICH
RANGE THEY FALL IN. No embedding model, no reference generation, no similarity, no
circularity. A generation "belongs" to the segment whose date range contains the
plurality of the years it emits. That is checkable by anyone with a regex.

Controls
--------
  ceiling  : the TEXT address. Establishes the behaviour exists and is detectable.
  baseline : generic prompt, no steering. Establishes where the trunk goes unbidden.
  steered  : the LATENT address. The actual question.
A steered result is only meaningful against the CEILING; the baseline merely says
what "doing nothing" looks like.

Vectors are built from TRAIN topics and evaluated on HELD-OUT topics: an address
that transports across topics is an address, not memorised content.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_STYLE = (
    " Write continuous prose densely packed with specific names, dates, places and numbers."
    " Mention explicit years frequently. Begin immediately. No preamble, no title, no commentary."
)
GENERIC = "Tell me the history of {topic}." + _STYLE
SEGMENT = "Tell me about {topic}, covering only {lo} to {hi}." + _STYLE

# topic -> K (lo, hi) year ranges. Contiguous, non-overlapping, objective.
TRAIN_TOPICS: Dict[str, List[Tuple[int, int]]] = {
    "the Second World War": [(1939, 1941), (1942, 1943), (1944, 1945)],
    "the American Civil War": [(1861, 1862), (1863, 1863), (1864, 1865)],
    "the French Revolution": [(1789, 1791), (1792, 1794), (1795, 1799)],
    "the life of Napoleon Bonaparte": [(1769, 1799), (1800, 1811), (1812, 1821)],
    "the history of powered flight": [(1903, 1918), (1919, 1945), (1946, 1970)],
}
HELDOUT_TOPICS: Dict[str, List[Tuple[int, int]]] = {
    "the Apollo program": [(1958, 1963), (1964, 1968), (1969, 1972)],
    "the Cold War": [(1945, 1961), (1962, 1978), (1979, 1991)],
    "the Manhattan Project": [(1939, 1941), (1942, 1944), (1945, 1945)],
}
K = 3
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")


def locate(text: str, ranges: List[Tuple[int, int]]) -> Tuple[Optional[int], List[int]]:
    """Which segment does this text sit in? Plurality vote over the years it emits.

    Objective and externally checkable: no model, no embedding, no reference text.
    Years outside every range are ignored rather than forced into a bucket -- a
    generation that talks about 1806 when the ranges start at 1939 should not be
    credited to segment 0 for being numerically nearest.
    """
    votes = [0] * len(ranges)
    for y in (int(m) for m in _YEAR.findall(text)):
        for i, (lo, hi) in enumerate(ranges):
            if lo <= y <= hi:
                votes[i] += 1
                break
    return (max(range(len(votes)), key=lambda i: votes[i]) if sum(votes) else None), votes


@dataclass
class Trial:
    topic: str
    condition: str  # ceiling | baseline | steered
    layer: int
    alpha: float
    beta: float
    target: Optional[int]
    landed: Optional[int]
    votes: List[int]
    text: str


class Trunk:
    def __init__(self, device: str, dtype: torch.dtype):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.layers = self.model.model.layers

    def chat(self, t: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def residuals(self, prompt: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        cap: Dict[int, torch.Tensor] = {}

        def mk(i):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[i] = h[0, -1, :].detach().float().cpu()

            return hook

        hs = [self.layers[i].register_forward_hook(mk(i)) for i in layers]
        try:
            self.model(**self.tok(prompt, return_tensors="pt").to(self.device))
        finally:
            for h in hs:
                h.remove()
        return cap

    @torch.no_grad()
    def gen(self, prompt: str, n: int, steer: Optional[torch.Tensor] = None, layer: int = 0) -> str:
        handle = None
        if steer is not None:
            v = steer.to(self.device).to(next(self.model.parameters()).dtype)

            def hook(_m, _i, out):
                return (out[0] + v,) + out[1:] if isinstance(out, tuple) else out + v

            handle = self.layers[layer].register_forward_hook(hook)
        try:
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            o = self.model.generate(
                **ids, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id
            )
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(o[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 24, 28, 32])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t = Trunk(args.device, getattr(torch, args.dtype))
    trials: List[Trial] = []

    # ---- CEILING: does the TEXT address work? If not, nothing below is interpretable.
    print("[ceiling] text address", flush=True)
    conf = [[0] * K for _ in range(K)]
    none_ct = 0
    for topic, rs in HELDOUT_TOPICS.items():
        for k, (lo, hi) in enumerate(rs):
            txt = t.gen(t.chat(SEGMENT.format(topic=topic, lo=lo, hi=hi)), args.tokens)
            land, votes = locate(txt, rs)
            trials.append(Trial(topic, "ceiling", -1, 0, 0, k, land, votes, txt))
            if land is None:
                none_ct += 1
            else:
                conf[k][land] += 1
        print(f"  {topic}", flush=True)
    c_hit = sum(conf[k][k] for k in range(K))
    c_tot = sum(map(sum, conf))
    print(
        f"  CEILING: {c_hit}/{c_tot} on-target ({c_hit / max(c_tot, 1):.0%}), "
        f"{none_ct} unscorable  {conf}",
        flush=True,
    )

    # ---- BASELINE: where does the trunk go with no address at all?
    print("\n[baseline] generic prompt, no steering", flush=True)
    b = [0] * K
    for topic, rs in HELDOUT_TOPICS.items():
        txt = t.gen(t.chat(GENERIC.format(topic=topic)), args.tokens)
        land, votes = locate(txt, rs)
        trials.append(Trial(topic, "baseline", -1, 0, 0, None, land, votes, txt))
        if land is not None:
            b[land] += 1
        print(f"  {topic:26s} -> segment {land}  votes={votes}", flush=True)
    print(f"  BASELINE lands: {b} (expect concentration in segment 0)", flush=True)

    # ---- vectors from TRAIN topics only
    print("\n[vectors] from train topics", flush=True)
    diffs: Dict[int, Dict[int, List[torch.Tensor]]] = {
        layer: {k: [] for k in range(K)} for layer in args.layers
    }
    for topic, rs in TRAIN_TOPICS.items():
        hg = t.residuals(t.chat(GENERIC.format(topic=topic)), args.layers)
        for k, (lo, hi) in enumerate(rs):
            ht = t.residuals(t.chat(SEGMENT.format(topic=topic, lo=lo, hi=hi)), args.layers)
            for layer in args.layers:
                diffs[layer][k].append(ht[layer] - hg[layer])
    common: Dict[int, torch.Tensor] = {}
    disc: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer in args.layers:
        v = {k: torch.stack(diffs[layer][k]).mean(0) for k in range(K)}
        c = torch.stack([v[k] for k in range(K)]).mean(0)
        common[layer], disc[layer] = c, {k: v[k] - c for k in range(K)}
        cos = [
            round(float(torch.nn.functional.cosine_similarity(v[i], v[j], dim=0)), 3)
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        print(
            f"  L{layer:02d} |common|={float(c.norm()):6.1f} "
            f"|d_k|={[round(float(disc[layer][k].norm()), 1) for k in range(K)]} cos={cos}",
            flush=True,
        )

    # ---- STEERED: the latent address
    print("\n[steered] latent address (beta*common + alpha*d_k)", flush=True)
    best = (0.0, None)
    for layer in args.layers:
        for beta in args.betas:
            for alpha in args.alphas:
                cm = [[0] * K for _ in range(K)]
                nn_ = 0
                for topic, rs in HELDOUT_TOPICS.items():
                    for k in range(K):
                        vec = beta * common[layer] + alpha * disc[layer][k]
                        txt = t.gen(t.chat(GENERIC.format(topic=topic)), args.tokens, vec, layer)
                        land, votes = locate(txt, rs)
                        trials.append(
                            Trial(
                                topic,
                                "steered",
                                layer,
                                alpha,
                                beta,
                                k,
                                land,
                                votes,
                                txt,
                            )
                        )
                        if land is None:
                            nn_ += 1
                        else:
                            cm[k][land] += 1
                hit = sum(cm[k][k] for k in range(K))
                tot = sum(map(sum, cm))
                rate = hit / max(tot, 1)
                if rate > best[0]:
                    best = (rate, (layer, alpha, beta))
                print(
                    f"  L{layer:02d} a={alpha:<4g} b={beta:<4g}  {hit:2d}/{tot:<2d} {rate:5.0%}"
                    f"  unscorable={nn_}  {cm}",
                    flush=True,
                )

    print("\n" + "=" * 72)
    print(f"  CEILING (text address)  : {c_hit}/{c_tot} = {c_hit / max(c_tot, 1):.0%}")
    print(f"  BEST LATENT address     : {best[0]:.0%} at {best[1]}")
    print(f"  CHANCE                  : {1 / K:.0%}")
    print("=" * 72)
    if best[0] < 0.5 and c_hit / max(c_tot, 1) > 0.7:
        print("  The address is NOT linearly available at the last prompt token.")
        print("  A planner cannot emit it as an additive latent; the plan must be")
        print("  delivered as something the trunk can ATTEND to (KV entries / notes).")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(x) for x in trials], indent=2))
    print(f"\nwrote {len(trials)} trials -> {out}")


if __name__ == "__main__":
    main()
