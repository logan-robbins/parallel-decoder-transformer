"""Does a frozen trunk have a steerable ADDRESSING direction?

The question
------------
Parallel decoding needs stream k to start at segment k without having read
segments <k. Decompose what stream k is missing:

    I(s_k ; s_<k | x) =  H(partition | x)              <- ADDRESSING  (planner's job)
                       + I(s_k ; s_<k | x, partition)  <- CONTENT     (bus's job)

This script measures whether ADDRESSING is reachable on a FROZEN trunk. A frozen
trunk can only be *steered* to behaviour it already has; it can never be taught.
So the entire architecture rests on this being true:

    There exists a vector v_k such that injecting v_k into a stream prompted
    "tell me the story of X" makes it begin at segment k.

If no such v_k exists, the planner cannot seed anything and nothing downstream
matters. This is the cheapest possible falsification, and it needs no training.

Why held-out stories are load-bearing
-------------------------------------
Building v_2 from Peter Pan and testing it on Peter Pan proves nothing: the
vector could simply carry Peter-Pan-middle *content*. That would be an
uninteresting result dressed as a positive one.

So v_k is built as a mean over TRAIN stories and evaluated on HELD-OUT stories.
A vector that transports across stories encodes "start at segment k" and nothing
about any particular story. That is an addressing claim, and it is what the
planner would have to emit.

Method (function-vector / task-vector methodology)
-------------------------------------------------
    v_k = mean over train stories of [ h_L(targeted_k) - h_L(generic) ]

taken at the last prompt token, layer L. Inject alpha * v_k into a held-out
story's generic stream. Localise the generation against the model's OWN
unsteered telling of that story, split into thirds -- so ground truth is the
trunk's own narrative order, not an outside judge.

Success = a diagonal confusion matrix: v_k lands the generation in third k.
Baseline (no steering) = always third 1. That contrast IS the result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Stories the trunk certainly knows. Split train/heldout so the steering vector
# is never evaluated on a story it was derived from.
TRAIN_STORIES = [
    "Cinderella",
    "The Three Little Pigs",
    "Little Red Riding Hood",
    "Goldilocks and the Three Bears",
    "Hansel and Gretel",
    "Jack and the Beanstalk",
    "Snow White",
]
HELDOUT_STORIES = [
    "Peter Pan",
    "Pinocchio",
    "The Wizard of Oz",
]

# The ONLY difference between generic and targeted is the addressing clause.
# The no-preamble suffix is identical across both, so it cancels in the diff. It
# exists because "Certainly! Peter Pan is a classic children's story by J.M.
# Barrie..." is not story content: it would dominate the embedding, sit in the
# reference's first third, and make every generation localise to third 1 --
# manufacturing a correct baseline for an entirely wrong reason.
_STYLE = " Begin immediately with the story itself. Write continuous prose, no preamble, no title, no commentary."
GENERIC = "Tell me the story of {story}." + _STYLE
TARGETED = [
    "Tell me only the first third of the story of {story}." + _STYLE,
    "Tell me only the middle third of the story of {story}." + _STYLE,
    "Tell me only the final third of the story of {story}." + _STYLE,
]
K = len(TARGETED)


@dataclass
class Trial:
    story: str
    layer: int
    alpha: float  # scale on the ADDRESSING component d_k
    beta: float  # scale on the shared framing component `common`
    target: Optional[int]  # None = unsteered baseline
    predicted_third: int
    similarities: List[float]
    text: str


def build_prompt(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


class Trunk:
    """Frozen Qwen3 with residual-stream read/write hooks."""

    def __init__(self, device: str, dtype: torch.dtype):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.hidden = self.model.config.hidden_size

    @torch.no_grad()
    def last_token_residuals(self, prompt: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """h_L at the final prompt token -- where the model decides what to emit
        next -- for every requested layer, in ONE forward pass."""
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(idx: int):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[0, -1, :].detach().float().cpu()

            return hook

        handles = [self.layers[i].register_forward_hook(make_hook(i)) for i in layers]
        try:
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            self.model(**ids)
        finally:
            for h in handles:
                h.remove()
        return captured

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        steer: Optional[torch.Tensor] = None,
        layer: int = 0,
        alpha: float = 0.0,
    ) -> str:
        """Greedy decode. If steer is given, add alpha*steer to every position at
        layer L on every forward pass -- prefill and each decode step alike.
        That is what an always-on bus injection does, so it is the faithful test:
        if the strongest injection cannot move the stream, nothing weaker will."""
        handle = None
        if steer is not None and alpha != 0.0:
            vec = (steer.to(self.device) * alpha).to(next(self.model.parameters()).dtype)

            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    return (out[0] + vec,) + out[1:]
                return out + vec

            handle = self.layers[layer].register_forward_hook(hook)
        try:
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


class Localiser:
    """Where in the story does this text sit? Ground truth is the trunk's OWN
    unsteered telling, split into thirds -- no external judge, no API cost."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.refs: Dict[str, torch.Tensor] = {}

    def register(self, story: str, full_telling: str) -> List[str]:
        sents = [s.strip() for s in full_telling.replace("\n", " ").split(". ") if s.strip()]
        if len(sents) < K:
            raise ValueError(f"{story}: unsteered telling too short to split into {K} thirds")
        n = len(sents)
        thirds = [". ".join(sents[(n * i) // K : (n * (i + 1)) // K]) for i in range(K)]
        self.refs[story] = torch.tensor(self.enc.encode(thirds, normalize_embeddings=True))
        return thirds

    def locate(self, story: str, text: str) -> tuple[int, List[float]]:
        v = torch.tensor(self.enc.encode([text], normalize_embeddings=True))
        sims = (self.refs[story] @ v.T).squeeze(-1)
        return int(sims.argmax()), [round(float(s), 4) for s in sims]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28])
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 8.0, 16.0],
        help="scale on the addressing component d_k",
    )
    ap.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="scale on shared framing; 0 isolates pure addressing",
    )
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--ref-tokens", type=int, default=400)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"loading {MODEL_ID} ({args.dtype} on {args.device})", flush=True)
    trunk = Trunk(args.device, dtype)
    print(f"  {trunk.n_layers} layers, hidden={trunk.hidden}", flush=True)
    for lyr in args.layers:
        if not 0 <= lyr < trunk.n_layers:
            raise ValueError(f"layer {lyr} out of range for {trunk.n_layers}-layer trunk")

    loc = Localiser()

    # --- ground truth: the trunk's own unsteered telling of each held-out story
    print("\n[refs] eliciting unsteered tellings of held-out stories", flush=True)
    for story in HELDOUT_STORIES:
        full = trunk.generate(build_prompt(trunk.tok, GENERIC.format(story=story)), args.ref_tokens)
        thirds = loc.register(story, full)
        print(f"  {story}: {len(full)} chars -> thirds of {[len(t) for t in thirds]}", flush=True)

    # --- steering vectors: mean over TRAIN stories only. Held-out stories never
    #     touch this loop -- that is what makes a hit an ADDRESSING result rather
    #     than content injection.
    print("\n[vectors] building addressing directions from train stories", flush=True)
    diffs: Dict[int, Dict[int, List[torch.Tensor]]] = {
        lyr: {k: [] for k in range(K)} for lyr in args.layers
    }
    resid_norms: Dict[int, List[float]] = {lyr: [] for lyr in args.layers}
    for story in TRAIN_STORIES:
        h_gen = trunk.last_token_residuals(
            build_prompt(trunk.tok, GENERIC.format(story=story)), args.layers
        )
        for lyr in args.layers:
            resid_norms[lyr].append(float(h_gen[lyr].norm()))
        for k in range(K):
            h_tgt = trunk.last_token_residuals(
                build_prompt(trunk.tok, TARGETED[k].format(story=story)), args.layers
            )
            for lyr in args.layers:
                diffs[lyr][k].append(h_tgt[lyr] - h_gen[lyr])
        print(f"  {story}", flush=True)

    # The trunk's own activation magnitude at each layer. alpha is only meaningful
    # relative to this: alpha*|d_k| >> |resid| means the injection has left the
    # distribution the frozen trunk was trained on, and any "hit" is damage.
    residual_scale = {lyr: sum(v) / len(v) for lyr, v in resid_norms.items()}

    # Decompose each v_k into what the three share and what distinguishes them:
    #
    #     common = mean_k v_k        "tell only a fragment"  -- shared framing
    #     d_k    = v_k - common      "which fragment"        -- ADDRESSING
    #
    # This is the A/C split in the geometry. v_k is dominated by `common`
    # (cos(v_i,v_j) ~ 0.85 empirically), so steering with raw v_k mostly tells
    # the trunk to be terse and barely says where to start. d_k is the term the
    # planner actually has to emit, and it is small -- which is the claim: the
    # partition costs few bits. Injecting beta*common + alpha*d_k lets the two be
    # scaled independently, so a hit can be attributed to addressing rather than
    # to framing.
    common: Dict[int, torch.Tensor] = {}
    discriminative: Dict[int, Dict[int, torch.Tensor]] = {}
    for lyr in args.layers:
        v = {k: torch.stack(diffs[lyr][k]).mean(0) for k in range(K)}
        c = torch.stack([v[k] for k in range(K)]).mean(0)
        common[lyr] = c
        discriminative[lyr] = {k: v[k] - c for k in range(K)}
        resid = float(residual_scale[lyr])
        cos = [
            round(float(torch.nn.functional.cosine_similarity(v[i], v[j], dim=0)), 3)
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        print(
            f"  layer {lyr:2d}: |resid|={resid:7.1f}  |common|={float(c.norm()):5.1f}  "
            f"|d_k|={[round(float(discriminative[lyr][k].norm()), 1) for k in range(K)]}  "
            f"cos(v_i,v_j)={cos}",
            flush=True,
        )

    # --- baseline: unsteered, on the generic prompt. Should land in third 1.
    print("\n[baseline] unsteered generic prompt", flush=True)
    trials: List[Trial] = []
    for story in HELDOUT_STORIES:
        text = trunk.generate(
            build_prompt(trunk.tok, GENERIC.format(story=story)), args.max_new_tokens
        )
        pred, sims = loc.locate(story, text)
        trials.append(Trial(story, -1, 0.0, 0.0, None, pred, sims, text))
        print(f"  {story:22s} -> third {pred + 1}  sims={sims}", flush=True)

    # --- CEILING CONTROL: does the targeted PROMPT itself work?
    # Steering can only ever reproduce behaviour the trunk already exhibits when
    # asked in plain language. If direct prompting does not produce a diagonal,
    # there is no addressing behaviour to steer toward, and a null steering result
    # would indict these prompts rather than the trunk. This is the ceiling every
    # steered config below is measured against -- without it the experiment cannot
    # distinguish "steering fails" from "the target behaviour never existed".
    print("\n[ceiling] direct targeted prompts (no steering)", flush=True)
    ceiling = [[0] * K for _ in range(K)]
    for story in HELDOUT_STORIES:
        for k in range(K):
            text = trunk.generate(
                build_prompt(trunk.tok, TARGETED[k].format(story=story)), args.max_new_tokens
            )
            pred, sims = loc.locate(story, text)
            trials.append(Trial(story, -2, 0.0, 0.0, k, pred, sims, text))
            ceiling[k][pred] += 1
    c_hits = sum(ceiling[k][k] for k in range(K))
    c_total = sum(map(sum, ceiling))
    print(
        f"  prompted directly: {c_hits}/{c_total} ({c_hits / c_total:.0%})  {ceiling}", flush=True
    )
    if c_hits / c_total < 0.5:
        print(
            "  !! WARNING: the trunk does not follow the targeted prompt itself.\n"
            "     Any null steering result below is uninformative -- fix the prompts first.",
            flush=True,
        )

    # --- the test
    print("\n[steered] injecting beta*common + alpha*d_k into the generic prompt", flush=True)
    print(f"  {'cfg':>22s}  {'on-target':>9s}  confusion (rows=target k, cols=landed)", flush=True)
    for layer in args.layers:
        for beta in args.betas:
            for alpha in args.alphas:
                confusion = [[0] * K for _ in range(K)]
                for story in HELDOUT_STORIES:
                    for k in range(K):
                        vec = beta * common[layer] + alpha * discriminative[layer][k]
                        text = trunk.generate(
                            build_prompt(trunk.tok, GENERIC.format(story=story)),
                            args.max_new_tokens,
                            steer=vec,
                            layer=layer,
                            alpha=1.0,  # scaling already folded into vec
                        )
                        pred, sims = loc.locate(story, text)
                        trials.append(Trial(story, layer, alpha, beta, k, pred, sims, text))
                        confusion[k][pred] += 1
                hits = sum(confusion[k][k] for k in range(K))
                total = sum(map(sum, confusion))
                print(
                    f"  L{layer:02d} a={alpha:<4g} b={beta:<4g}  "
                    f"{hits:2d}/{total:<2d} {hits / total:5.0%}  {confusion}",
                    flush=True,
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(t) for t in trials], indent=2))
    print(f"\nwrote {len(trials)} trials -> {out}", flush=True)


if __name__ == "__main__":
    main()
