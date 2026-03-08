# PAPER.md Mechanism Review vs Implementation

This review traces the mechanism in `docs/arxiv_submission/PAPER.md` against the code path used at training and inference.

## 1) Prompt-time planner seed (Snapshot 0)

### What this does
- At inference start, the orchestrator runs a mandatory planner path before stream generation, materializes planner token IDs, and builds plan embeddings.
- It can also derive planner notes from a provided plan contract (`plan_contract`) or per-stream plan text.
- It projects plan embeddings into notes space with `plan_notes_proj`, pools valid plan slots, normalizes, and uses this as an initial seed snapshot.

### Will it work as intended?
- **Yes, structurally.** The implementation computes planner logits/ids first, then creates plan embeddings and optional projected seed vectors before stream stepping.
- **Caveat:** if `plan_contract` mapping to stream aliases is weak/missing, seed vectors can silently drop streams (warning path), reducing initial coordination quality.

### Latent-space usage
- Good: planner logits and plan embeddings share a unified latent vocabulary (`plan_vocab_size`) and are consistency-checked against collator settings.
- Good: plan IDs are hashed canonically in data collation, preserving deterministic latent IDs across training/inference.

### Learning/data/weight update notes
- Training supervises planner slots with CE and ties them to downstream plan embeddings, so snapshot-0 is learnable under dataset plan contracts.
- N/A at inference for weight updates.

## 2) Dynamic Notes Bus (DNB) state exchange

### What this does
- Each stream owns a bus storing versioned note snapshots with lagged visibility (`lag=Δ`) and bounded capacity (`max_snapshots=K`).
- Bus can compact via FIFO or retention-aware scored eviction/merge.

### Will it work as intended?
- **Yes, for lagged communication and bounded memory.** Snapshot windows exclude freshest snapshots by lag and enforce max capacity.
- **Experiment-dependent:** retention scoring quality depends on attention-derived scores; poor attention calibration could evict useful notes.

### Latent-space usage
- Notes are stored as dense latent vectors only (no direct text exchange), matching the paper’s embeddings-only channel.

### Learning/data/weight update notes
- DNB itself is stateful runtime storage, not directly parameterized (except retention submodules); learning signals come through heads that write/read notes.

## 3) Notes window construction (who sees what, and when)

### What this does
- Per consumer stream, builder gathers producer snapshots using topology + lag, applies self-lag offset, and tracks last-seen versions to avoid re-reading stale snapshots.
- Optional self-only warmup (`sectional_self_tokens`) limits early cross-stream leakage.

### Will it work as intended?
- **Yes, mechanistically coherent.** It enforces delayed visibility and monotonic version consumption.
- **Design caveat:** `TopologyMask.producers_for` currently always returns all streams (including self), so topology is effectively fixed all-to-all.

### Latent-space usage
- Proper: vectors normalized/padded to fixed `notes_dim`, keeping SNC input contract stable.

### Learning/data/weight update notes
- This is runtime routing logic; gradients flow through SNC on consumed notes, not through window-builder control flow.

## 4) SNC read path inside trunk layers

### What this does
- Instrumented decoder layers inject stream adapter deltas + SNC residual deltas inside trunk forward pass.
- SNC computes cross-attention from hidden states (queries) to note window (keys/values), and an outer learned gate controls influence magnitude.

### Will it work as intended?
- **Likely yes.** Implementation removes cascaded gate attenuation by using `DeltaSNC` (delta-only SNC) and a single outer notes gate.
- **Stability-positive:** gate initialization and sigmoid gating let influence start small and increase with training.

### Latent-space usage
- Strong: communication stays in latent vectors; attention weights are also harvested for retention scoring.

### Learning/data/weight update notes
- Gradients flow from LM/planner/aux losses through instrumented layers into SNC and stream adapters when unfrozen by stage policy.

## 5) Provisional note emission (writes)

### What this does
- During decoding, stream accumulates attended history and periodically emits note summaries using `notes_head`, then pushes snapshots to its bus.
- Emission cadence is configurable and can include stochastic/forced cadence behavior.

### Will it work as intended?
- **Yes with caveat:** summary is currently mean over `notes_head(history)`, which is simple and may blur temporal details for long strides.

### Latent-space usage
- Proper latent write-back; no token-text sharing required for inter-stream updates.

### Learning/data/weight update notes
- `notes_loss` and `spec_loss` supervise this channel directly against teacher note tensors (MSE), including special plan-snapshot terms for sectional examples.

## 6) Coverage/ownership tracking

### What this does
- Coverage head cross-attends plan embeddings against attended hidden states and predicts per-plan-item logits.
- Plan item IDs are canonical hashes from stream+text, with masks and optional stream ownership IDs.

### Will it work as intended?
- **Mostly yes.** Multi-head + multi-scale keying is a reasonable mechanism for plan-item coverage inference.
- **Important caution:** logits are masked to `0.0` on padded plan positions. This is fine when BCE mask is correct (it is), but downstream consumers must keep respecting mask.

### Latent-space usage
- Good: explicit plan-item latent IDs and embeddings align planner space with coverage supervision space.

### Learning/data/weight update notes
- Coverage loss uses BCE-with-logits over masked targets; metrics include same-stream recall and cross-stream FP rate (useful for ownership/no-overlap behavior).

## 7) Agreement/readiness gating and rollback

### What this does
- Agreement head outputs token-level readiness scores (sigmoid).
- At emission boundaries, stride-level agreement is tracked; on stride completion, streams below threshold can be rolled back to last commit checkpoint and regenerated.

### Will it work as intended?
- **Partially aligned with paper.** It gates rollback/recovery and continuation indirectly through stride checks.
- **Mismatch vs strict global gate:** implementation rolls back failing streams individually; paper describes a simple global `min_k r_k > gamma` gate over active streams. This is a selective policy variant rather than exact base rule.

### Latent-space usage
- Uses attended latent states for readiness prediction; no textual cross-stream arbitration required.

### Learning/data/weight update notes
- Agreement BCE is trained using labels from dataset or auto-derived stability targets from pre/post planner logits over commit regions.
- This is a pragmatic proxy for “continuation sufficiency,” but its fidelity depends on whether KL/argmax stability correlates with real coherence outcomes.

## 8) Training dataset structure and sectional learning

### What this does
- Dataset requires rich metadata and carries stream IDs, planner IDs, notes tensors, snapshots, plan catalog, coverage targets, and optional agreement labels.
- Collator builds sectional label masks so each stream’s LM labels can be restricted to its assigned segment.

### Will it work as intended?
- **Yes for stream specialization** when metadata includes valid segment boundaries.
- **Risk:** if segmentation metadata is missing/noisy, sectional mask falls back to full labels (or may degrade supervision specificity), weakening ownership separation.

### Latent-space usage
- Strong canonicalization via hashed plan entries ensures stable latent IDs for planner/coverage.

### Learning/data/weight update notes
- This structure supports planner, notes, speculation, LM, coverage, and agreement supervision in one batch contract.

## 9) Weight updates and freeze/unfreeze correctness

### What this does
- Optimizer is built from `iter_trainable_parameters()` and only includes params with `requires_grad=True`.
- Stage policies can freeze/unfreeze named modules; main loop applies backward, grad clipping, optimizer step, scheduler step, zero grad.

### Will it work as intended?
- **Yes, generally correct.** Trainable parameter filtering and staged requires_grad toggles are implemented.
- **Operational caveat:** changing `requires_grad` does not rebuild optimizer param groups; newly unfrozen params are still present if originally included. This is usually fine here because optimizer starts with all iter_trainable params and stage policy toggles grad flow.

### Latent-space usage
- N/A.

### Learning/data/weight update notes
- Loss composition closely matches paper objective family: planner CE, notes/spec MSE, LM CE/KD, coverage BCE, agreement BCE, plus optional stability/retention/etc.

## 10) End-to-end viability for parallel decoding and stream communication

### Why this can be viable
- **Internal shared state exists:** DNB + SNC provide continuous latent read/write communication across streams.
- **Coordination checkpoint exists:** stride-level agreement + rollback enforce a synchronization discipline rather than uncontrolled free-running generation.
- **Ownership grounding exists:** coverage over canonical plan items gives a concrete mechanism for section ownership and overlap control.
- **Frozen-trunk compatible:** instrumentation injects sidecar behavior without full trunk finetuning.

### Main technical risks to monitor experimentally
1. **Agreement target mismatch risk:** auto-derived agreement labels (stability proxy) may not perfectly encode true “safe continuation.”
2. **Information bottleneck risk:** mean-pooled note summaries may under-represent nuanced dependencies.
3. **Topology rigidity risk:** currently all-to-all producer exposure by design; dependency-aware sparse graphs are future work.
4. **Rollback churn risk:** if threshold/cadence poorly tuned, streams can oscillate around commit boundaries.

## Overall assessment
- The implementation is **substantially consistent** with the paper’s core mechanism: planner-seeded latent workspace, lagged inter-stream latent communication, SNC conditioning, coverage ownership tracking, and agreement-mediated rollback/continuation.
- It is a **credible approach** for coherence-preserving parallel decoding because communication is explicit, synchronized, and learnable in latent space.
- The largest open question is not mechanism correctness but **training signal fidelity for readiness/coherence** (agreement labels and evaluation protocol must verify that readiness predicts coherent continuation).

## 11) What is the most important contribution?

If we prioritize the contribution stack, the most important piece is **not just the planner**, but the **planner-seeded synchronization protocol**:

1. **Planner at `t=0` seeds a shared latent contract** (who likely owns what),
2. **SNC + DNB provide ongoing latent communication** during decoding,
3. **Agreement/rollback enforce synchronized continuation** instead of unconstrained divergence.

So the plan seed is necessary, but it is only one-third of the key contribution. The real novelty is that PDT turns decomposition + communication + continuation control into one internal decode protocol.

### Is the plan effectively “S1 covers A/B, S2 covers C/D”?

Yes, that is a good operational interpretation for current implementation and dataset structure:
- planner slots and canonical plan items induce latent ownership priors,
- coverage tracks whether each stream is staying on assigned plan items,
- agreement decides whether the present shared state is good enough to continue.

That makes PDT particularly natural for **retrieval-structured prompts** (e.g., “tell me about the Civil War”) where decomposition is topical/sectional and inter-stream coherence matters.

## 12) Inference-time token retrieval/return contract (what should API return?)

Today the orchestrator decodes stream-wise internally, but for product/API behavior we should make the return policy explicit. There are two viable serving modes:

### Mode A — Live per-stream streaming
- Return partial text events independently for each stream as tokens arrive.
- Pros: lowest latency to first output; good for developer debugging.
- Cons: users see speculative text before stride-level agreement has validated readiness; can expose unstable trajectories.

### Mode B — Stride-commit streaming (**recommended default**)
- Buffer tokens per stream during a stride; release only after stride commit gate passes.
- Return payload as structured per-stream blocks, e.g.:
  - `stride_id`
  - `stream_id`
  - `committed_text_block`
  - optional `coverage/ownership telemetry`
- Pros: aligns with PDT semantics (decode → summarize → agree → commit), reduces visible incoherence and rollback artifacts.
- Cons: slightly higher user-visible latency than raw token streaming.

### Merge policy for final response
- Keep two outputs:
  1. **Structured multi-stream artifact** (for observability and research),
  2. **Merged user answer** (ordered by planner ownership/section contract).

This dual output gives both product simplicity and mechanism transparency.

## 13) Recommended next-task list (implementation + paper)

### P0 — Clarify and lock mechanism semantics
1. **Document planner contract semantics** in code/docs: explicit statement that planner seed expresses ownership priors per stream/plan item.
2. **Align paper with implementation gate policy**: either (a) implement strict global `min_k r_k > gamma`, or (b) update paper to describe selective rollback policy as the primary rule.
3. **Define serving contract**: choose Mode B as default and add a config switch for Mode A.

### P1 — Inference API and telemetry updates
4. Add an explicit orchestrator output schema:
   - `committed_blocks_by_stream`
   - `provisional_blocks_by_stream` (optional)
   - `stride_commit_events`
   - `rollback_events`.
5. Add per-stride commit IDs so clients can reconcile retries/rollbacks deterministically.
6. Add a merge module that composes final answer from committed stream blocks + planner ownership order.

### P1 — Coherence-critical training/eval upgrades
7. Replace/augment auto-derived agreement targets with **continuation-sufficiency labels** from rollout outcomes (did next stride require rollback? contradiction delta?).
8. Add evaluation split focused on retrieval-style prompts (historical topics, multi-facet knowledge prompts) and report:
   - cross-stream contradiction rate,
   - ownership collision rate,
   - rollback rate and recovery success.
9. Add ablations:
   - no planner seed,
   - no DNB/SNC,
   - no agreement gate,
   to isolate which component contributes most to coherence.

### P2 — Architecture refinements
10. Upgrade note summarization from simple mean pooling to a small learned summarizer (e.g., attention pooling over stride history).
11. Add dependency-aware topology masks (beyond fixed all-to-all) so streams only consume relevant producers.
12. Explore adaptive stride/cadence by agreement confidence to reduce rollback churn.

### P2 — Paper positioning improvements
13. In `PAPER.md`, add a subsection explicitly framing the strongest near-term use case as **parallelized knowledge-structured responses** rather than unconstrained creative generation.
14. Add an explicit inference contract diagram: provisional tokens, stride commit, buffered release, final merge.
15. State that coherence comes from latent communication + synchronized commit, not from raw token sharing.
