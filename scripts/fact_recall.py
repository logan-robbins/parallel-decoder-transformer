"""Does a K-way parallel decomposition recall as many facts as decoding sequentially?

The metric
----------
    R_ref = fact atoms a SEQUENTIAL stream produces, given budget K*N tokens
    R_par = fact atoms K PARALLEL streams produce, given budget N tokens each

    recall   = |R_ref & R_par| / |R_ref|     did we keep what sequential got?
    coverage = |R_par| / |R_ref|             raw yield, ignoring identity
    overlap  = duplicated atoms across streams / |R_par|    <- CROWDING
    novel    = |R_par - R_ref| / |R_par|     facts sequential missed

Equal total token budget on both sides, so this is a fair fight: it asks whether
the SAME work, split K ways and run concurrently, retains the same content.

Recall collapses crowding and gaps into one number, which is why it is the right
signal. Two streams covering the same ground duplicate atoms and drop others ->
recall falls. A gap between streams drops atoms -> recall falls. There is no way
to score well except by partitioning cleanly.

Why FACTUAL, ORDERED domains
----------------------------
"The middle third of Peter Pan" is an arbitrary cut: the trunk has no crisp
representation of it, and measurement has no external ground truth. "The Second
World War, 1942-1943" is an objective address -- the trunk knows exactly what it
denotes, and whether Midway landed in the right stream is checkable.

This is not a convenience. It is the crux:

    allocating N facts to K streams ARBITRARILY costs  N*log2(K) bits
    allocating them by BOUNDARY in an ordered domain costs ~K*log2(N) bits

    N=30, K=3:  48 bits   vs   ~15 bits

Parallel decoding is cheap exactly when the task carries an order. Chronology is
what makes the partition compressible enough to fit through a narrow latent bus.
An unordered task forces a per-fact assignment and the bus cost scales with N.

What a "fact atom" is
---------------------
Dates, numbers, and proper nouns -- extracted lexically, identically from both
sides. For history this IS the fact content, and a shared extractor cannot favour
either arm. It is a proxy, deliberately crude and deliberately symmetric.

Note the PLANS below are hand-written. That is the point: this measures whether a
GOOD plan is sufficient. If parallel recall matches sequential under a good plan,
the planner is the remaining work and the architecture stands. If it does not,
no planner rescues it and the bus is load-bearing (or the idea is wrong).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_STYLE = (
    " Write continuous prose densely packed with specific names, dates, places and numbers."
    " Begin immediately. No preamble, no title, no commentary, no lists."
)

# topic -> [K segment addresses]. A DATE RANGE and nothing else.
#
# The addresses are deliberately minimal. An earlier draft used rich addresses
# ("from the invasion of Poland in 1939 through the attack on Pearl Harbor in
# December 1941") and that was a confound AND a self-inflicted theoretical wound:
#
#   - LEAKAGE: the address handed the stream the very atoms it was then scored on
#     (Poland, 1939, Pearl Harbor), which the sequential arm never received.
#     Parallel recall would be inflated by construction.
#   - BITS: if a crisp address needs a fact-laden sentence, H(partition|x) is not
#     ~3 bits but 40+, and the narrow-bus claim dies on my own prompt design.
#
# A date range is the whole address: ~3 bits to name one of a handful of spans.
# That is what a planner would have to emit, so that is what gets tested.
# Atoms occurring in ANY prompt are excluded from scoring regardless (see main).
TOPICS: Dict[str, List[str]] = {
    "the Second World War": ["1939 to 1941", "1942 to 1943", "1944 to 1945"],
    "the American Civil War": ["1861 to 1862", "the year 1863", "1864 to 1865"],
    "the Apollo program": ["1958 to 1963", "1964 to 1968", "1969 to 1972"],
    "the French Revolution": ["1789 to 1791", "1792 to 1794", "1795 to 1799"],
    "the life of Napoleon Bonaparte": ["1769 to 1799", "1800 to 1811", "1812 to 1821"],
    "the Cold War": ["1945 to 1961", "1962 to 1978", "1979 to 1991"],
    "the history of powered flight": ["1903 to 1918", "1919 to 1945", "1946 to 1970"],
    "the Manhattan Project": ["1939 to 1941", "1942 to 1944", "the year 1945"],
}

GENERIC = "Tell me the history of {topic}." + _STYLE
SEGMENT = "Tell me about {topic}, covering only {seg}." + _STYLE

# Fact atoms: 4-digit years, other numbers, and proper nouns. Applied identically
# to both arms, so the extractor cannot favour either.
#
# Three defects found by validating on hand-written WWII text, all fixed here,
# because two of them MANUFACTURED CROWDING -- a headline metric:
#   1. "Battle of Britain" and "Battle of the Bulge" both collapsed to the atom
#      "Battle" (the pattern stopped at the lowercase connective), so two
#      DIFFERENT facts registered as one shared atom across streams.
#   2. Month names were atoms, and "December" appears in nearly every segment of
#      every war -> guaranteed false overlap carrying no factual content.
#   3. A sentence-start lookbehind silently dropped every proper noun that opens a
#      sentence ("Franklin Roosevelt died..." yielded only "Roosevelt"), losing
#      real facts. Position is the wrong filter; the stoplist is the right one.
_YEAR = re.compile(r"\b1[5-9]\d{2}\b|\b20[0-2]\d\b")
_NUM = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
# Maximal run of capitalised words, allowing lowercase connectives inside, so
# multiword entities survive intact. Up to TWO consecutive connectives, because
# "Battle of the Bulge" needs "of the" and a single-connective pattern splits it
# into the bare atom "Battle" -- which then collides with every other "Battle of
# X" that splits the same way, re-creating the false-overlap bug. "and" is
# deliberately NOT a connective: it merges genuinely distinct entities
# ("Britain and France" is two facts, not one).
_PROPER = re.compile(
    r"\b[A-Z][a-z]{2,}(?:(?:\s+(?:of|the|de|von|van|di|du|la|le|el)){1,2}\s+[A-Z][a-z]{2,}"
    r"|\s+[A-Z][a-z]{2,})*"
)

_MONTHS = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
_STOP = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "They",
    "Their",
    "There",
    "Then",
    "Though",
    "When",
    "Where",
    "While",
    "With",
    "Within",
    "Without",
    "What",
    "Which",
    "After",
    "Before",
    "During",
    "Between",
    "But",
    "And",
    "For",
    "From",
    "His",
    "Her",
    "Its",
    "It",
    "In",
    "On",
    "At",
    "By",
    "As",
    "An",
    "He",
    "She",
    "Was",
    "Were",
    "Had",
    "Has",
    "Have",
    "Been",
    "Would",
    "Could",
    "Should",
    "Will",
    "Can",
    "May",
    "Might",
    "Both",
    "Each",
    "Every",
    "Many",
    "Most",
    "Some",
    "Such",
    "Than",
    "Thus",
    "Also",
    "However",
    "Meanwhile",
    "Because",
    "Although",
    "Despite",
    "Over",
    "Under",
    "Into",
    "Through",
    "Throughout",
    "Following",
    "Finally",
    "Later",
    "Soon",
    "Now",
    "Once",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "One",
} | _MONTHS


def atoms(text: str) -> Set[str]:
    out: Set[str] = set()
    out.update(_YEAR.findall(text))
    out.update(m for m in _NUM.findall(text) if len(m) > 1)
    for m in _PROPER.findall(text):
        words = m.split()
        while words and words[0] in _STOP:  # strip leading "The Battle of Britain"
            words = words[1:]
        while words and words[-1] in _STOP:
            words = words[:-1]
        if not words:
            continue
        cand = " ".join(words)
        # A bare month or stopword carries no factual content; a multiword entity
        # containing one ("Battle of the Bulge") does.
        if len(words) == 1 and (cand in _STOP or len(cand) <= 2):
            continue
        out.add(cand)
    return out


@dataclass
class Result:
    topic: str
    n_ref_atoms: int
    n_par_atoms: int
    recall: float
    coverage: float
    overlap: float
    novel: float
    per_stream_atoms: List[int]
    pairwise_overlap: List[int]
    t_seq: float
    t_par: float
    speedup: float
    # Raw text and extracted atoms are persisted so the metric can be re-scored
    # with a different extractor without re-running the model, and so any claim
    # made from these numbers can be checked against what was actually generated.
    seq_text: str
    par_texts: List[str]
    missed_atoms: List[str]
    duplicated_atoms: List[str]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens-per-stream", type=int, default=200)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"  # batched decode: every generation frontier at the right edge
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=getattr(torch, args.dtype))
    nn.Module.to(model, torch.device(args.device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def chat(text: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def gen(prompts: List[str], max_new: int) -> tuple[List[str], float]:
        b = tok(prompts, return_tensors="pt", padding=True).to(args.device)
        if args.device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        o = model.generate(
            **b, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        if args.device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        return [
            tok.decode(o[i, b["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ], dt

    results: List[Result] = []
    N = args.tokens_per_stream

    for topic, segs in TOPICS.items():
        K = len(segs)
        seq_prompt = GENERIC.format(topic=topic)
        seg_prompts = [SEGMENT.format(topic=topic, seg=s) for s in segs]

        # sequential arm: one stream, full budget
        (seq_text,), t_seq = gen([chat(seq_prompt)], N * K)
        # parallel arm: K streams, N each, ONE batched forward per step
        par_texts, t_par = gen([chat(prompt) for prompt in seg_prompts], N)

        # Neither arm gets credit for atoms it was HANDED. The segment addresses
        # contain years, and the topic name contains proper nouns; scoring those
        # would reward the prompt rather than the model, and only the parallel arm
        # receives the addresses -- so the bias runs one way. Excluding the union
        # of all prompt atoms from both sides makes it symmetric.
        given = atoms(seq_prompt)
        for prompt in seg_prompts:
            given |= atoms(prompt)

        ref = atoms(seq_text) - given
        per = [atoms(t) - given for t in par_texts]
        par = set().union(*per)

        dup = sum(len(per[i] & per[j]) for i in range(K) for j in range(i + 1, K))
        pair = [len(per[i] & per[j]) for i in range(K) for j in range(i + 1, K)]

        r = Result(
            topic=topic,
            n_ref_atoms=len(ref),
            n_par_atoms=len(par),
            recall=len(ref & par) / max(len(ref), 1),
            coverage=len(par) / max(len(ref), 1),
            overlap=dup / max(len(par), 1),
            novel=len(par - ref) / max(len(par), 1),
            per_stream_atoms=[len(p) for p in per],
            pairwise_overlap=pair,
            t_seq=t_seq,
            t_par=t_par,
            speedup=t_seq / t_par,
            seq_text=seq_text,
            par_texts=par_texts,
            missed_atoms=sorted(ref - par),
            duplicated_atoms=sorted(
                {a for i in range(K) for j in range(i + 1, K) for a in (per[i] & per[j])}
            ),
        )
        results.append(r)
        print(
            f"  {topic:32s} ref={r.n_ref_atoms:3d} par={r.n_par_atoms:3d}  "
            f"recall={r.recall:5.1%}  cover={r.coverage:5.2f}  crowd={r.overlap:5.1%}  "
            f"new={r.novel:5.1%}  {r.speedup:4.2f}x",
            flush=True,
        )

    print("\n" + "=" * 78)
    print("DID THE DECOMPOSITION KEEP WHAT SEQUENTIAL GOT?")
    print("=" * 78)
    n = len(results)
    mr = sum(r.recall for r in results) / n
    mc = sum(r.coverage for r in results) / n
    mo = sum(r.overlap for r in results) / n
    mn = sum(r.novel for r in results) / n
    ms = sum(r.speedup for r in results) / n
    print(f"  recall of sequential's facts : {mr:6.1%}   <- the number that matters")
    print(f"  coverage (raw atom yield)    : {mc:6.2f}x")
    print(f"  crowding (duplicated atoms)  : {mo:6.1%}   <- streams treading on each other")
    print(f"  novel (facts seq missed)     : {mn:6.1%}")
    print(f"  measured latency speedup     : {ms:6.2f}x  (ceiling K=3)")
    print("-" * 78)
    if mr > 0.8:
        print("  A good text plan is SUFFICIENT: parallel keeps sequential's facts.")
        print("  => the planner is the remaining work; the bus is an optimisation.")
    elif mo > 0.3:
        print("  CROWDING dominates: streams are duplicating each other's facts.")
        print("  => this is exactly what a bus is for. Its payload is now measurable.")
    else:
        print("  Recall is lost WITHOUT crowding: facts are falling into gaps between")
        print("  segments. => the partition itself is lossy, not the coordination.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nwrote {len(results)} topics -> {out}")


if __name__ == "__main__":
    main()
