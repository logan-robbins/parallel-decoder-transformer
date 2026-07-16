"""Is the "skip-ahead axis" real, or an artifact of how we measured it?

The claim under test
--------------------
cos(v1,v2) ~ 0.94 at every layer (v_k = h(segment_k) - h(generic)) is read as:
the trunk has a SKIP-AHEAD AXIS, not a K-way categorical address. That is the
strongest surviving claim in this work, so it gets attacked before it gets
published. Three independent ways it could be false:

T1. ROGUE DIMENSIONS. Timkey & van Schijndel (EMNLP 2021) show 1-3 dimensions
    dominate cosine similarity between transformer representations, and that
    those dimensions are mismatched with the ones carrying task information. If
    cos(v1,v2)=0.94 is driven by a handful of high-magnitude dims, it says
    nothing about addressing. TEST: recompute cosine after removing the top-m
    dimensions by magnitude, and after standardising each dimension.

T2. CONTRAST-BASELINE ARTIFACT. v_k = h(segment_k) - h(generic), and generic is
    ~ "start at the beginning" ~ segment_0. So v1 and v2 both contain a large
    shared "not-the-beginning" component BY CONSTRUCTION, and their high cosine
    would be a property of the SUBTRAHEND, not of the trunk. TEST: drop the
    baseline entirely -- measure pairwise distances between the raw h(segment_k).
    If |h(s1)-h(s2)| << |h(s0)-h(s1)|, middle and final really are closer
    together than first and middle, with no baseline involved.

T3. WRONG READOUT. A mean-difference direction is one linear readout among many.
    The address could be perfectly linearly decodable while the mean difference
    fails to isolate it. TEST: train a linear probe to classify k from h_L. If
    the probe succeeds where the mean-difference vector does not, the claim
    "there is no K-way address" is FALSE -- there is one, and we simply read it
    out badly. This is the test most likely to kill the claim, which is why it
    is here.

A claim that survives all three is worth publishing. One that does not, is not.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_STYLE = (
    " Write continuous prose densely packed with specific names, dates, places and numbers."
    " Mention explicit years frequently. Begin immediately. No preamble, no title, no commentary."
)
GENERIC = "Tell me the history of {topic}." + _STYLE
SEGMENT = "Tell me about {topic}, covering only {lo} to {hi}." + _STYLE

TOPICS: Dict[str, List[Tuple[int, int]]] = {
    "the Second World War": [(1939, 1941), (1942, 1943), (1944, 1945)],
    "the American Civil War": [(1861, 1862), (1863, 1863), (1864, 1865)],
    "the French Revolution": [(1789, 1791), (1792, 1794), (1795, 1799)],
    "the life of Napoleon Bonaparte": [(1769, 1799), (1800, 1811), (1812, 1821)],
    "the history of powered flight": [(1903, 1918), (1919, 1945), (1946, 1970)],
    "the Apollo program": [(1958, 1963), (1964, 1968), (1969, 1972)],
    "the Cold War": [(1945, 1961), (1962, 1978), (1979, 1991)],
    "the Manhattan Project": [(1939, 1941), (1942, 1944), (1945, 1945)],
    "the Roman Republic": [(-509, -270), (-269, -134), (-133, -27)],
    "the Industrial Revolution": [(1760, 1820), (1821, 1870), (1871, 1914)],
    "the history of the Internet": [(1969, 1983), (1984, 1994), (1995, 2005)],
    "the Space Shuttle program": [(1972, 1985), (1986, 1998), (1999, 2011)],
}
K = 3


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28, 32, 35])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def chat(t: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def resid(prompt: str) -> Dict[int, torch.Tensor]:
        cap: Dict[int, torch.Tensor] = {}

        def mk(i):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[i] = h[0, -1, :].detach().float().cpu()

            return hook

        hs = [model.model.layers[i].register_forward_hook(mk(i)) for i in args.layers]
        try:
            model(**tok(prompt, return_tensors="pt").to(args.device))
        finally:
            for h in hs:
                h.remove()
        return cap

    # collect h(generic) and h(segment_k) for every topic
    print("collecting activations...", flush=True)
    H_gen: Dict[int, List[torch.Tensor]] = {layer: [] for layer in args.layers}
    H_seg: Dict[int, Dict[int, List[torch.Tensor]]] = {
        layer: {k: [] for k in range(K)} for layer in args.layers
    }
    for topic, rs in TOPICS.items():
        g = resid(chat(GENERIC.format(topic=topic)))
        for layer in args.layers:
            H_gen[layer].append(g[layer])
        for k, (lo, hi) in enumerate(rs):
            s = resid(chat(SEGMENT.format(topic=topic, lo=lo, hi=hi)))
            for layer in args.layers:
                H_seg[layer][k].append(s[layer])
    n_topics = len(TOPICS)
    print(f"  {n_topics} topics x {K} segments + generic, {len(args.layers)} layers", flush=True)

    out: Dict[str, dict] = {}

    print("\n" + "=" * 88)
    print("T1 + T2: is cos(v1,v2) an artifact of rogue dims or of the contrast baseline?")
    print("=" * 88)
    print(
        f"  {'L':>3s} | {'cos(v1,v2)':>10s} {'no-top3':>8s} {'no-top20':>8s} {'z-scored':>9s} "
        f"| {'d(s0,s1)':>8s} {'d(s1,s2)':>8s} {'ratio':>6s}"
    )
    for layer in args.layers:
        gen = torch.stack(H_gen[layer]).mean(0)
        v = {k: torch.stack(H_seg[layer][k]).mean(0) - gen for k in range(K)}
        raw = cos(v[1], v[2])

        # T1: strip the highest-magnitude dimensions of the mean activation.
        # If a handful of rogue dims carry the similarity, cosine collapses.
        base = torch.stack([torch.stack(H_seg[layer][k]).mean(0) for k in range(K)] + [gen]).mean(0)
        order = base.abs().argsort(descending=True)

        def strip(m: int) -> float:
            keep = order[m:]
            return cos(v[1][keep], v[2][keep])

        # z-scoring across the topic population removes per-dimension scale entirely
        pop = torch.stack([h for k in range(K) for h in H_seg[layer][k]] + H_gen[layer])
        mu, sd = pop.mean(0), pop.std(0).clamp_min(1e-6)
        vz = {
            k: ((torch.stack(H_seg[layer][k]).mean(0) - mu) / sd) - ((gen - mu) / sd)
            for k in range(K)
        }

        # T2: baseline-free. Distances between RAW segment representations.
        hs = {k: torch.stack(H_seg[layer][k]).mean(0) for k in range(K)}
        d01 = float((hs[0] - hs[1]).norm())
        d12 = float((hs[1] - hs[2]).norm())
        out[f"L{layer}"] = {
            "cos_v1v2": raw,
            "cos_no_top3": strip(3),
            "cos_no_top20": strip(20),
            "cos_zscored": cos(vz[1], vz[2]),
            "d_s0s1": d01,
            "d_s1s2": d12,
        }
        print(
            f"  {layer:3d} | {raw:10.3f} {strip(3):8.3f} {strip(20):8.3f} {cos(vz[1], vz[2]):9.3f} "
            f"| {d01:8.1f} {d12:8.1f} {d12 / d01:6.2f}"
        )
    print(
        "\n  T1 verdict: if 'no-top3'/'no-top20'/'z-scored' stay high, cosine is NOT a rogue-dim artifact."
    )
    print("  T2 verdict: ratio d(s1,s2)/d(s0,s1) << 1 means middle and final really are closer")
    print("              than first and middle -- measured with NO contrast baseline at all.")

    print("\n" + "=" * 88)
    print("T3: can a LINEAR PROBE decode the segment index? (the claim-killer)")
    print("=" * 88)
    print("  Leave-one-topic-out logistic regression on h_L. If this succeeds where the")
    print("  mean-difference vector failed, a K-way address EXISTS and we read it out badly.")
    print(
        f"  {'L':>3s} | {'probe acc':>9s} {'chance':>7s} | {'0v1':>5s} {'0v2':>5s} {'1v2':>5s}  (pairwise)"
    )
    for layer in args.layers:
        X = torch.stack([h for k in range(K) for h in H_seg[layer][k]])
        y = torch.tensor([k for k in range(K) for _ in range(n_topics)])
        tid = torch.tensor([i for _ in range(K) for i in range(n_topics)])
        X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)

        def loo(mask_classes=None) -> float:
            sel = (
                torch.ones(len(y), dtype=torch.bool)
                if mask_classes is None
                else torch.tensor([int(v) in mask_classes for v in y])
            )
            Xs, ys, ts = X[sel], y[sel], tid[sel]
            classes = sorted(set(int(v) for v in ys))
            remap = {c: i for i, c in enumerate(classes)}
            ys = torch.tensor([remap[int(v)] for v in ys])
            correct = 0
            for t in range(n_topics):
                tr, te = ts != t, ts == t
                if te.sum() == 0:
                    continue
                W = torch.zeros(Xs.shape[1], len(classes), requires_grad=True)
                b = torch.zeros(len(classes), requires_grad=True)
                opt = torch.optim.LBFGS([W, b], max_iter=120, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad()
                    # L2 keeps a 2560-dim probe on ~30 points from memorising outright
                    loss = F.cross_entropy(Xs[tr] @ W + b, ys[tr]) + 1e-2 * W.pow(2).sum()
                    loss.backward()
                    return loss

                opt.step(closure)
                with torch.no_grad():
                    correct += int(((Xs[te] @ W + b).argmax(1) == ys[te]).sum())
            return correct / int(sel.sum())

        acc = loo()
        pw = {f"{a}v{b}": loo({a, b}) for a, b in combinations(range(K), 2)}
        out[f"L{layer}"].update({"probe_acc": acc, **{f"probe_{k}": v for k, v in pw.items()}})
        print(
            f"  {layer:3d} | {acc:9.1%} {1 / K:7.1%} | "
            f"{pw['0v1']:5.0%} {pw['0v2']:5.0%} {pw['1v2']:5.0%}"
        )

    print("\n  T3 verdict: if 1v2 probe accuracy is HIGH, the address IS linearly decodable and")
    print("  the 'no K-way address' claim is FALSE -- report that the mean-difference readout")
    print("  fails while a probe succeeds, which is a different (and weaker) claim.")
    print("  If 1v2 is at chance while 0v1/0v2 are high, the skip-ahead-axis claim SURVIVES.")

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {args.output}")


if __name__ == "__main__":
    main()
