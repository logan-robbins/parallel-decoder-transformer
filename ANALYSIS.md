# PDT Research Analysis: NeurIPS 2026 Readiness Report

**Date:** 2026-02-24
**Abstract deadline (est.):** May 2026
**Status:** Not submission-ready — needs 1–2 revision cycles

---

## Executive Summary

The Parallel Decoder Transformer (PDT) paper presents a genuinely novel architecture (SNC + Dynamic Notes Bus) with solid mathematical formalism and a complete open-source codebase. The primary gap is empirical: the experiments section reports a single table with three numbers, contains two different precision values (71.6% and 77.78%) that are never reconciled, and does not include a single baseline comparison or latency measurement — despite both being the central motivation of the paper.

The good news: the evaluation infrastructure to close these gaps is already in the codebase. The tools for ROUGE/BERTScore comparison, latency benchmarking, manifest analysis, and counterfactual ablation all exist and can be run. The gap is running them and writing the results.

---

## 1. Critical Paper Errors (Submission Blockers)

These errors must be fixed before any submission. None require new experiments.

### 1.1 Precision Metric Inconsistency — 71.6% vs 77.78%

The paper reports two different precision values that are never reconciled:

| Location | Reported Value |
|---|---|
| `sections/00_abstract.tex` | **77.8%** |
| `sections/01_introduction.tex` §1.2 (Contributions item 3) | **71.6%** |
| `sections/06_experiments.tex` §6.3 figure caption | **77.8%** |
| `sections/06_experiments.tex` §6.4 Table 1 | **77.78%** |
| `sections/07_conclusion.tex` §7 | **71.6%** |

**The abstract and conclusion disagree by 6 percentage points.** This is the first thing a reviewer will notice. The distinction between the two values is not explained anywhere. Possible explanations:
- 77.78% is the final checkpoint (step 50k); 71.6% is an intermediate or stage-averaged value
- One is validation set, one is a held-out test split
- One is precision@K and the other is uncalibrated binary precision

**Action required:** Determine which value is correct and what the other represents. Update abstract, contributions section, and conclusion to report the same number with explicit qualification (e.g., "final checkpoint, step 50k, held-out validation set").

### 1.2 Appendix Placeholder Text — Submission Blocker

`sections/appendix.tex` §B.5.2 ("Practical Bounds and Validation"), lines 92–93:

```
Illustrative ranges (replace with measurements): small dim (256): L_u∈[5,20]; medium
(512): L_u∈[10,40]; large (1024): L_u∈[20,80].
```

This is a draft authoring note visible verbatim in the compiled PDF. A reviewer seeing this will desk-reject on professionalism grounds alone. Either replace with actual Lipschitz estimates from training (via `--sync-profile` or spectral norm logs) or remove the subsection entirely.

### 1.3 GPT-OSS-120B vs GPT-OSS-20B Inconsistency

`sections/appendix.tex` §A (Artifacts Summary), line 6:
> "full training and inference for GPT-OSS-**120**B"

Every other location in the paper — abstract, introduction, experiments, README — says **20B**. This is a direct contradiction. A reviewer auditing reproducibility will flag this immediately. Fix the Appendix A reference to GPT-OSS-20B.

### 1.4 Teacher Model Label Mismatch

- `README.md` line 57: teacher is **GPT-4.1**
- `sections/06_experiments.tex` §6.1, line 8: dataset "distilled from **GPT-4**"

GPT-4 and GPT-4.1 are different API models. Reviewers evaluating reproducibility will notice the inconsistency. Unify both references to the same model identifier and add a footnote with the exact API model string used (e.g., `gpt-4.1-2025-04-14`).

### 1.5 Validation Loss = 0.00 Is Unexplained

Table 1 in §6.4 reports `Validation Loss = 0.00`. This is almost certainly a display rounding artifact (the loss plateaus at ≈0.2 per §6.3), but as presented it reads as an error or a suspiciously perfect result. Fix: display as `< 0.01` or report the actual value (e.g., `0.18`) with a footnote explaining the display precision.

### 1.6 Missing Figure File

`sections/06_experiments.tex` §6.3 references `\includegraphics{figures/coverage_loss.png}`, but the `figures/` directory contains no PNG files. The compiled paper will show a broken figure box. The training curve must be regenerated or the image committed to the repo before submission.

### 1.7 GPT-OSS-20B Identity Unclear

The paper uses `GPT-OSS-20B` as if it were a well-known public model, but this identifier is not a standard HuggingFace model name. Reviewers will be unable to reproduce results without knowing the exact model source. Add a footnote in §6.1 with the HuggingFace model ID (e.g., `org/gpt-oss-20b`) or the GCS artifact path.

---

## 2. Missing Experiments for Top-Venue Submission

NeurIPS reviewers expect 3–5 results tables or figures, ablation of novel components, and comparison against named baselines. The current paper has one table with three numbers.

### P1 — CRITICAL: No Latency or Throughput Results

The entire motivation of PDT is reducing inference latency relative to sequential decoding. The abstract frames this as the core problem. The paper has **zero wall-clock measurements**.

The infrastructure exists: `scripts/run_benchmark.sh` loads PDT, runs parallel decode, runs sequential baseline, and calls `compare_seq_parallel.py` to emit timing deltas. The manifest telemetry records `timings.per_token` (wall-clock seconds per step) and the `S` (tokens/sec) speedup factor.

**Required addition:** A table comparing PDT vs. sequential baseline on:
- Tokens per second (throughput)
- Time-to-first-token
- Total latency for a representative prompt length (e.g., 512 tokens)
- Reported values of α (seconds/stride), β (rollback fraction), S (speedup)

Without this table, the paper cannot establish its primary empirical claim.

### P2 — CRITICAL: No Baseline Comparison

The paper names Skeleton-of-Thought (SoT) as the primary competing method and calls its failure mode "Coherence Drift." The paper does not include a single experiment showing PDT performing better than SoT on any metric.

Additionally, `src/parallel_decoder_transformer/baselines/token_level.py` implements Medusa, EAGLE, and Lookahead baselines that produce schema-compatible manifests. These are never reported.

**Required additions:**
- Coverage/coherence comparison against SoT on the same prompt set
- ROUGE-L and BERTScore comparison (code exists in `compare_seq_parallel.py`)
- At least one token-level baseline comparison (Medusa or Lookahead, code in `baselines/token_level.py`)

### P3 — CRITICAL: Single-Result Experiments Section

The entire experiments section (§6) contains exactly one results table (Table 1) with three numbers. This is insufficient for a NeurIPS submission. Expected for a top venue:
- 3–5 tables or figures
- Ablation study isolating individual components
- Training curve figure (already referenced but image file is missing)
- At least one comparison against an external method

**Required structure for §6:**
1. Table 1 (current): Coverage P/R at step 50k — keep, fix values
2. Figure 1: Coverage loss learning curve — regenerate and commit PNG
3. Table 2: Latency comparison (PDT vs sequential vs SoT)
4. Table 3: Quality comparison ROUGE-L/BERTScore (PDT vs sequential)
5. Table 4 (optional): Ablation — gate g=0 vs g=1.0, cf-ablate on/off

### P4 — HIGH: No Standard NLP Quality Evaluation

The paper evaluates only on internal teacher-generated ground truth (coverage precision/recall). Reviewers will ask: does parallel decoding degrade output quality? Is the frozen trunk's generation quality preserved?

**Required additions:**
- ROUGE-L comparison: PDT output vs sequential output on the same 100–500 prompts
- BERTScore comparison: same
- These metrics are computed by `scripts/compare_seq_parallel.py` directly from manifests

This can be done by running `run_benchmark.sh` on the existing 10K eval set.

### P5 — MEDIUM: No Ablation Study

PDT introduces multiple novel components: the SNC adapter, the Dynamic Notes Bus (DNB), the agreement-gated rollback mechanism, and the learned gate `g`. None are individually ablated.

**Available ablation controls (all in current CLI):**
- `--cf-ablate all`: zeros the notes window for all streams → direct gate-off comparison
- `--gate-g 0`: disables cross-lane SNC influence → SNC ablation
- `--read-lag-delta N` sweep: DNB lag ablation (Δ=0 vs Δ=4 vs Δ=8)
- `--cadence-M all=1` vs `all=4`: stride frequency ablation

A minimal ablation table with 3–4 rows (baseline, +SNC, +DNB, +rollback) would directly justify the architectural decisions in §3.

### P6 — MEDIUM: Recall Is 4.91% — Needs Stronger Justification

The paper frames 4.91% recall as a deliberate design choice ("conservatism is a desirable safety property"). At NeurIPS this framing requires much stronger justification:
- Does the rollback system fire at all at scale? What is the rollback rate on the 10K eval set?
- Is the 4.91% recall inherent to the architecture or an artifact of the training regime?
- Could recall be improved with a different classification threshold without sacrificing precision?

`scripts/analyze_infer_manifest.py` can compute rollback rate distributions from existing manifests. These statistics should appear in §6.

### P7 — LOW: No Statistical Significance

All primary results are from a single training run with no error bars. At NeurIPS, confidence intervals or multi-run variance are expected for primary metrics.

**Minimum required:** Bootstrap CIs (resample 1000×) on coverage precision/recall from the held-out validation set. This requires no new experiments, only resampling the existing evaluation results.

---

## 3. Evaluation Infrastructure Already Available

The following tools exist in the codebase and could generate paper-ready results. None are currently reported in the paper.

| Tool | Metrics Produced | Status in Paper |
|---|---|---|
| `scripts/run_benchmark.sh` | Wall-clock latency, throughput (tokens/sec) | Not reported |
| `scripts/compare_seq_parallel.py` | ROUGE-L, BERTScore, TF-IDF | Not reported |
| `scripts/analyze_infer_manifest.py` | Rollback rate, gate entropy, speedup (S), α, β | Not reported |
| `scripts/score_infer_notes.py` | NLI contradiction rate, attribute consistency | Not reported |
| `scripts/summarize_infer_manifests.py` | Aggregate metrics across manifests | Not reported |
| `src/.../baselines/token_level.py` | Medusa/EAGLE/Lookahead manifests | Not reported |
| `sim/clustered_rollback.py` | Clustered rollback simulation | Appendix E (described) |
| `--cf-ablate` flag | Notes-window ablation | Appendix G (planned, not run) |
| `--baseline sequential` flag | Sequential baseline manifest | Not reported |

The tools for a complete experiments section already exist. The gap is running them, collecting results, and writing the tables.

---

## 4. NeurIPS 2026 Readiness Scorecard

| Criterion | Status | Notes |
|---|---|---|
| Novel architecture | ✅ Strong | SNC + DNB is genuinely novel |
| Mathematical formalism | ✅ Present | §4 formalism is complete |
| Related work coverage | ✅ Adequate | Covers speculative decoding, PEFT, SSMs |
| Metric consistency | ❌ Broken | 71.6% (abstract/intro/conclusion) vs 77.78% (table/figure) |
| Baseline comparison | ❌ Missing | No experimental comparison against SoT or any other method |
| Latency/throughput results | ❌ Missing | Core motivation (latency reduction) has no supporting data |
| Standard NLP evaluation | ❌ Missing | Internal metrics only; no ROUGE/BERTScore/perplexity |
| Ablation study | ❌ Missing | No component-level ablation |
| Statistical significance | ❌ Missing | Single run, no error bars |
| Reproducibility | ⚠️ Partial | GPT-OSS-20B identity unclear; 120B/20B conflict in appendix |
| Figure availability | ❌ Missing | `figures/coverage_loss.png` referenced but file does not exist |
| Draft placeholders | ❌ Present | "replace with measurements" verbatim in Appendix B.5.2 |
| Code released | ✅ Yes | Full training + inference open-sourced |

**Verdict: Not submission-ready for NeurIPS 2026.** The architectural contribution is genuine and the formalism is solid. The paper needs an experiments section that actually demonstrates the latency claim and compares against baselines.

---

## 5. Revision Roadmap (Prioritized)

### Phase 1 — Paper Fixes (No New Experiments, 1–2 Days)

These fixes require only editing LaTeX. No code runs required.

1. **Reconcile 71.6% vs 77.78%** — determine which number is correct for which split/checkpoint. Update abstract, §1.2 contributions, and §7 conclusion to use the same number with explicit qualification ("held-out validation set, step 50k").
2. **Remove placeholder text** from Appendix B.5.2 — either fill with actual Lipschitz estimates from training logs or delete the subsection.
3. **Fix GPT-OSS-120B → 20B** in Appendix A, line 6.
4. **Unify teacher model label** — pick GPT-4.1 (README is more precise) and update §6.1 to match. Add a footnote with the exact API model string.
5. **Fix Validation Loss display** — change `0.00` in Table 1 to `< 0.01` or the actual plateau value from training logs.
6. **Identify GPT-OSS-20B** — add a footnote in §6.1 with the exact HuggingFace model ID or GCS path.

### Phase 2 — Run Existing Infrastructure (~1 Week, CPU-feasible for replay)

These steps use tools already in the codebase. Most can be run on the existing eval set.

7. **Run `scripts/run_benchmark.sh`** on the pre-generated eval set → produces latency table with α, β, S values. Add as Table 2 in §6.
8. **Run `compare_seq_parallel.py`** on sequential vs parallel manifest pair → produces ROUGE-L / BERTScore comparison. Add as Table 3 in §6.
9. **Run `analyze_infer_manifest.py`** on validation set manifests → produces rollback rate distribution, gate entropy stats. Add statistics to §6.4 prose.
10. **Regenerate `figures/coverage_loss.png`** from training logs and commit to `figures/`. Verify the LaTeX reference renders.
11. **Run `--cf-ablate all`** on 50–100 representative prompts → produces gate-off ablation data. Add as Table 4 in §6 or Appendix G.

### Phase 3 — New Experiments (~2–3 Weeks, GPU Required)

12. **SoT comparison** — run Skeleton-of-Thought on the same 100-prompt evaluation subset; compare ROUGE-L and NLI contradiction rate. This is the most important missing experiment for NeurIPS.
13. **Multi-stream ablation** — N=1 (sequential), N=2, N=3 streams; compare coverage P/R/F1 and throughput.
14. **Gate `g` ablation** — g=0 (SNC disabled), g=0.5, g=1.0; show output quality degrades without SNC. This justifies the SNC component directly.
15. **Bootstrap CIs** — resample 1000× on all primary coverage metrics. Add 95% CI to Table 1.

### Phase 4 — Optional Strengthening

16. Evaluate on WikiText-103 held-out set for perplexity to establish trunk preservation.
17. Test with a different backbone (Llama-3 8B) to demonstrate framework generality.
18. Human evaluation on 50–100 examples for factual consistency and coherence (high cost, high reviewer impact).

---

## 6. Mac-Local Verification Steps

The following can be run on Apple Silicon (M4) without GPU access. Use these to verify the analysis pipeline works before running full GPU experiments.

```bash
# 1. Verify clustered rollback simulation (Appendix E)
uv run python sim/clustered_rollback.py --rho 0.5 --L 32 --q_token 0.0033 --trials 10000

# 2. Verify CPU logit replay pipeline
uv run python scripts/logit_replay.py \
  --artifact sim/plan_decode_demo/ \
  --stream-jsonl \
  --output /tmp/replay_manifest.json

# 3. Verify manifest analysis pipeline
uv run python scripts/analyze_infer_manifest.py /tmp/replay_manifest.json

# 4. Run unit test suite
uv run pytest tests/unit/ -v

# 5. Run coverage metric unit tests specifically
uv run pytest tests/unit/test_manifest_metrics.py -v
```

---

## Appendix: Related Work Gap Analysis

The current related work section (§2) is adequate for a workshop paper but thin for NeurIPS. The following connections should be tightened or added:

**Speculative decoding (§2.1):** The paper cites Leviathan et al. (2023) and Chen et al. (2023) but does not position PDT relative to EAGLE (Li et al., 2024) or Medusa (Cai et al., 2024) — both of which are implemented in `baselines/token_level.py`. Since the codebase already has these baselines, they should be discussed and compared in related work.

**Parallel decoding taxonomy:** The paper uses "Decomposition-and-Fill" as a category name without citing prior work that uses this framing (Ning et al. 2023 uses it). Ensure the taxonomy is attributed correctly.

**PEFT methods:** The paper compares to full fine-tuning but doesn't position SNC relative to LoRA (Hu et al., 2022) or prefix tuning (Li & Liang, 2021), which are the obvious PEFT baselines. A sentence explaining why SNC is preferred over LoRA for this use case would strengthen §1.1.

**State space models:** The SSM discussion (§2.3) is present but brief. Given the DNB's similarity to linear recurrences, a more explicit connection to Mamba (Gu & Dao, 2023) would resonate with NeurIPS 2026 reviewers.
