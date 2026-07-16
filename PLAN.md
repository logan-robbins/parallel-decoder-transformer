# PDT Research Plan: Latent Coordination Under Real Dependency

Status date: 2026-07-16.

This is the canonical plan. The mechanism-first rebuild, revision-pinned
Instruct trunk, exact chat prompts, and privileged-context KD path are aligned
and contract-tested. The 32/1k trained causal gates remain empirical work.

## Living Task Status

1. [done] Reframe the thesis from generic parallel decoding to learned latent coordination under dependency.
2. [done] Incorporate the useful part of `docs/Logan.txt`: the value proposition is K parallel streams with a compressed, reveal-delayed channel versus blind parallelism, full-text exchange, or full-KV sharing.
3. [done] Discard weak claims: "no prior paper occupies this point" as an absolute statement, "one forward pass" wording that hides K continuations, and one-block expository citations as a proof of informational necessity.
4. [done] Remove hash-era planner/notes supervision from code. 2026-04-24: removed hash datasets, `NotesHead`, `PlanEmbedding`, planner CE targets, teacher-note MSE, and spec-note MSE.
5. [done] Build a controlled dependency benchmark where sibling state is unavailable except through the bus. 2026-04-24: added programmatic LDC generation, retokenization, structural validation, and local-model CE audit tooling.
6. [done] Implement differentiable block rollout so LM loss reaches SNC, notes gates, speculation writes, plan seeding, and the planner codebook. 2026-04-24: trainer now runs planner-seeded block rollout with in-graph speculation writes and span-local CE reporting.
7. [partial] Add ablations and baselines that can falsify the claim. The trainer now runs aligned baseline, gate-zero, sibling norm-scramble, and targeted-write mutation rollouts with token-weighted metrics. An exact parameter-matched self-only attention module exists; its training runner and the remaining comparison runners remain.
8. [partial] Rewrite the paper around mechanism-first evidence, with natural-language and sensor/signal tasks as downstream validation. `THEORY.md` now records the implementation boundary; empirical results remain pending.
9. [done] Lock the July 2026 starting recipe: frozen `Qwen3-4B-Instruct-2507`, same-trunk full-prefix context distillation, exact-entropy synthetic data first, and one rented H100 SXM 80GB for the scale gate.
10. [done] Align the canonical YAML, revision, adapter, and `qwen3-instruct-temporal-chat-v2` tokenization path to `Qwen3-4B-Instruct-2507`.
11. [done] Feed same-trunk privileged per-block teacher logits into dependency-masked forward KL at temperature 2; raw hidden-state MSE is absent.
12. [todo] Pass the 32-example overfit gate and 1k-example H100 gate before launching the 20k synthetic corpus.
13. [todo] After synthetic and planner gates pass, generate and validate the 50k HotpotQA/OASST1 natural-transfer corpus with `Qwen3-30B-A3B-Instruct-2507` offline.
14. [done] Use one strict atomic checkpoint format for training, resume, inference, and ablation, including global-step/stage policy restoration.
15. [done] Validate the real pinned checkpoint before GPU rental. A local MPS round proved the 4,022,468,096-parameter frozen trunk is disjoint from exactly 401,325,095 trainable phi parameters and exercised FP32 sidecar/BF16 trunk boundaries, instrumented caches, and all K streams.
16. [done] Publish the complete local implementation and documentation worktree, excluding ignored model artifacts and heavyweight generated datasets. 2026-07-16: formatting, lint, types, and 208 tests passed before staging the complete source/documentation worktree.

## First-Principles Thesis

The real problem is not "can a model emit several strings in parallel?" That already exists in many forms. The problem is:

> Can K locally causal streams from one frozen trunk exchange enough learned latent state through a narrow, delayed bus that they no longer need sequential full-text execution once each stream has computed or observed the state siblings need?

This is the primitive needed later for signal/sensor settings. A future input may be a snapshot such as `{s1: 100, s2: 85, s3: 93}`. The goal is not to make one monolithic decoder reason serially about every channel. The goal is for a frozen trunk plus small trainable sidecar to route work into K streams, let each stream form a local state, publish a compressed note, and let sibling outputs condition on those notes in the next block. The paper must prove that this mechanism can exist before claiming robotics, sensor fusion, or small-model deployment.

The central mechanism claim is conditional and falsifiable:

> If a target span depends on sibling latent state that is not available in the receiving stream's local history, then a trained PDT should reduce that span's LM cross-entropy through the bus; zeroing or scrambling the bus should selectively damage that span, while a parameter-matched self-only replacement should not close the gap.

This is stronger than "streams become different." It is a causal, span-local claim about information flow.

## Claims We Can Defend

1. **Causal latent coordination.** Mutating stream j's published note at block b changes stream i's logits at block b + Delta on annotated dependency spans.
2. **Load-bearing bottleneck.** Gate-zero, norm-scramble, and source-swap ablations increase dependency-span CE much more than nondependency CE.
3. **Not just extra capacity.** A parameter-matched module that reads only the receiver's own history does not recover the bus-enabled dependency-span performance.
4. **Bandwidth tradeoff.** PDT sits between blind parallel generation and full-token/full-KV collaboration: less coherent than unlimited communication in some regimes, but much lower communication bandwidth and lower serial latency.
5. **Planner integration.** A VQ prompt planner can seed stream roles without hash targets or per-stream user hints, but this is a second claim. It must not be used to hide whether the bus itself works.

## Claims We Must Not Make Yet

1. **Do not claim one output cost for K outputs.** PDT uses one shared prompt prefill plus synchronized/batched K-stream continuation. It avoids an external decomposition call and serial full-text rounds; it does not make K continuations free.
2. **Do not claim strict informational necessity when all facts are deterministic from a fully shared prompt.** If every stream can read all input facts, a sufficiently capable receiver can compute sibling facts itself. That can still be a useful benchmark, but it is not a clean proof that the bus was necessary.
3. **Do not use one-block targets with Delta=1 as evidence.** If sibling notes become visible only at block 1, a one-block target gives the receiver no chance to use sibling information.
4. **Do not lead with sensors as an empirical contribution.** Sensors motivate the architecture. The initial paper earns that direction only after proving text/synthetic latent coordination.
5. **Do not keep hash targets.** SHA-256 note embeddings and hashed planner IDs create artificial supervision and contaminate the bottleneck claim.
6. **Do not treat a no-planner diagnostic as PDT.** The submitted mechanism requires planner-produced snapshot-0 notes on the bus. A fixed-anchor or stream-local-input run is only a unit test for SNC and DNB information flow.

## Disposition Of `docs/Logan.txt`

Relevant and kept:

- The comparison axis is right: K parallel streams, compressed reveal-delayed latent coordination, lower latency than sequential interdependent execution, more coherent than blind parallel streams.
- The bottleneck distinction from Hogwild/full-KV sharing is valuable.

Discarded or narrowed:

- "No prior paper occupies exactly this point" is too broad. Recent work on token-level concurrent reasoning, native parallel generation, and adaptive parallel reasoning is close enough that reviewers will object. The defensible novelty is the combination of frozen causal trunk, trainable sidecar, VQ planner, low-bandwidth learned latent bus, reveal delay, and ablation-proven dependency-span information flow.
- "Single-forward parallel" language is dangerous. Use "shared prefill plus synchronized K-stream continuation."
- Do not describe close baselines as using a single transformer forward unless the paper explicitly says that. Group Think uses concurrent token generation through batched/interleaved inference and modified attention masks; APR uses parent and child inference threads with spawn/join; Multiverse uses a Map/Process/Reduce generation system with dynamic switching between sequential and parallel generation. These are parallel inference methods, not evidence that everyone has already solved one-pass K-output generation.

## Positioning

| Method | Parallelism | Coordination channel | Difference PDT must emphasize |
|---|---|---|---|
| Skeleton-of-Thought | External skeleton, then parallel workers | None between workers | External decomposition; blind workers cannot resolve downstream sibling-state dependencies. |
| Mixture-of-Agents | Sequential or layered agent rounds | Full text | Quality through full-text exchange; latency scales with rounds. |
| Hogwild! Inference | Concurrent generation | Shared KV/full attention memory | High-bandwidth text-equivalent state, no learned compressed bottleneck. |
| Group Think | Concurrent token-level agents | Token-level cross-thread visibility | Close baseline; PDT must show compressed latent notes can substitute for full-token visibility on dependency spans. |
| Multiverse | Native parallel generation with merge/reduce structure | Model-internal parallel branches | Close systems neighbor; PDT differs by explicit K persistent streams and measured low-bandwidth bus. |
| Adaptive Parallel Reasoning | Learned branching and serialized/parallel reasoning | Reasoning graph / branch outputs | Focuses on adaptive reasoning structure; PDT focuses on persistent stream coordination through a latent bus. |
| SPRINT / Medusa / EAGLE / speculative decoding | Parallel drafts or branches | Verification/draft machinery | Usually produces one committed output stream, not K coordinated outputs. |
| Diffusion/non-autoregressive LMs | Parallel token positions | Iterative denoising state | Parallelizes positions in one sequence; PDT coordinates streams. |
| LatentMAS | Multi-agent latent collaboration | Latent agent state | Adjacent latent-collaboration work; PDT needs bottleneck, delay, frozen trunk, and span-local ablations. |

The paper's novelty statement should be:

> PDT tests whether a frozen causal decoder can be augmented with a learned low-bandwidth latent bus so multiple persistent streams can coordinate under delayed, compressed sibling-state visibility. The contribution is not that parallelism exists; it is the causal demonstration that the compressed bus carries dependency information that blind, self-only, and full-text-free baselines lack.

The submitted PDT system must include this path:

```text
shared input -> frozen trunk prompt encode -> VQ planner -> plan_notes_proj
             -> snapshot-0 notes on Dynamic Notes Bus
             -> SNC reads visible notes in K stream continuations
             -> speculation head publishes later notes
```

If `planner -> plan_notes_proj -> bus` is absent, the experiment may still test SNC as a component, but it is not the main PDT claim.

## Experimental Ladder

The old plan tried to prove planner discovery, bus necessity, natural text quality, and latency in one benchmark. That is too entangled. We will separate them.

### A. Diagnostic Mechanism Benchmark: Latent Dependency Control

Purpose: prove the bus works before adding planner complexity.

This benchmark may use stream-local observations and fixed stream anchors. That is acceptable because it is the cleanest analogue of sensor channels and creates a real information barrier. It is a diagnostic: it can prove SNC/DNB information flow, but it cannot be the submitted PDT system unless the planner-produced snapshot-0 path is reintroduced.

Each example contains:

- A shared task header visible to all streams.
- K stream-local observation sequences, one per stream and block. Stream k sees
  only its own block-m observation, revealed at the boundary before block m;
  future private registers are absent from its prompt and cache.
- At least two target blocks per stream.
- Block 0 requires each stream to compute or summarize its local latent state.
- Block 1+ contains dependency spans whose correct tokens depend on sibling block-0 state.
- Randomized per-example values so dependency spans cannot be guessed from stream ID, topic template, or pretrained world knowledge.

Example family:

- Stream-local inputs are small signal records: sensor value, trend, confidence, local anomaly flag, or object-region claim.
- Block 0 output establishes each stream's local state.
- Block 1 output selects an action, warning, or coordination message that depends on another stream's state, such as "defer to stream_2 because it has the highest risk" or "avoid region B because stream_1 reports occupancy."

This benchmark is the first NeurIPS evidence for causal latent coordination. The main-system evidence must repeat the key ablations with planner-produced snapshot-0 notes.

### B. Planner Benchmark: Single-Prompt Decomposition

Purpose: prove internal planning and non-overlap.

Each example has one shared prompt and K unordered target segments. No per-stream private prompts, no role hints, no hash targets. Evaluation is permutation-invariant: the generated set of streams is matched to the target set by Hungarian alignment or semantic coverage, not by fixed `stream_0 = first label`.

This benchmark measures:

- Planner codebook utilization and slot entropy.
- Segment coverage.
- Segment overlap/redundancy.
- Stream specialization.
- Latency versus external-decomposition baselines.

It does not, by itself, prove strict bus necessity if every target fact is inferable from the shared prompt.

### C. Main Integrated Benchmark: Shared Snapshot Routing

Purpose: bridge the two claims and match the long-term sensor/signal use case. This is the first benchmark that should be treated as full PDT.

Input is one shared structured snapshot, for example:

```json
{"s1": 100, "s2": 85, "s3": 93, "task": "coordinate responses under threshold and collision rules"}
```

The planner must seed K streams with roles or channel focus through `planner -> plan_notes_proj -> snapshot-0 bus writes`. Each stream computes a local state, publishes a note, and later emits an output that depends on sibling notes. The important distinction from the mechanism benchmark is that the raw snapshot is globally available at prompt time; therefore success here is evidence of useful routing and compression, not strict information-theoretic necessity. The strictest bus proof still comes from Latent Dependency Control, but the submitted PDT claim requires this integrated path.

### D. Natural-Language Transfer/Demo

Purpose: make the result legible to NLP reviewers.

Use structured expository or comparative tasks only after A-C work:

- Multi-section answers with cross-references.
- Collaborative triage summaries.
- Multi-aspect analysis.

These tasks demonstrate that the mechanism survives outside synthetic signals. They are not the first proof because most cross-references are recoverable from world knowledge or the shared prompt.

## Data Contract

One canonical schema should support all benchmark families.

```json
{
  "example_id": "ldc_000001",
  "family": "latent_dependency_control",
  "split": "train",
  "k": 3,
  "shared_context": "Coordinate the three streams. Publish local state first, then respond to sibling state.",
  "visibility_lag_blocks": 1,
  "stream_inputs": [
    {
      "stream_id": "stream_0",
      "block_observations": [
        {"block_index": 0, "text": "sensor=s1 value=100 trend=rising confidence=0.91"},
        {"block_index": 1, "text": "sensor=s1 value=104 trend=rising confidence=0.93"}
      ],
      "target_blocks": [
        "s1 reports HIGH and rising with confidence 0.91.",
        "Because stream_2 is lower but stable, stream_0 keeps priority and requests monitoring from stream_1."
      ],
      "dependency_spans": [
        {
          "block_index": 1,
          "token_span_text": "stream_2 is lower but stable",
          "source_stream": "stream_2",
          "source_block_index": 0,
          "kind": "sibling_state"
        }
      ]
    }
  ],
  "eval": {
    "permutation_invariant": false,
    "dependency_span_metric": "target_model_ce_delta",
    "nondependency_span_metric": "target_model_ce_delta"
  },
  "generator": {
    "type": "programmatic_oracle",
    "seed": 123
  }
}
```

Rules:

- `target_blocks` must contain at least two blocks when `visibility_lag_blocks = 1`.
- Dependency spans must occur only after the source note is visible.
- For mechanism examples, sibling dependency values must be randomized and absent from the receiver's local observation.
- For single-prompt planner examples, stream assignment must be randomized or evaluated permutation-invariantly. Fixed label ordering is not allowed as evidence of discovery.
- There are no `notes_teacher`, `notes_student`, `planner_ids`, hash buckets, or hash-derived embeddings.
- Oracle metadata is allowed for validation and evaluation, not as planner or notes supervision.

## Validation Contract

The dataset validator must compute target-model logprob audits, not only LLM-judge labels.

For each dependency span:

1. Run receiver with local information only.
2. Run receiver with full sibling state visible through an oracle/full-context condition.
3. Measure token-level CE on dependency spans and nondependency spans using the same tokenizer/model family used for PDT evaluation.
4. Keep examples where full visibility substantially improves dependency spans while nondependency spans remain predictable from local/shared context.

Initial thresholds for dataset admission:

- Mean dependency-span CE gap, local-only minus full-visibility: at least 1.5 nats/token.
- Mean nondependency-span CE gap: below 0.3 nats/token.
- At least 80% of examples have a positive dependency-span gap.

These are admission thresholds, not final paper claims. They protect us from training on examples where the bus is unnecessary.

## Architecture Contract

Keep the core PDT structure:

- Frozen `Qwen/Qwen3-4B-Instruct-2507` trunk for the main paper implementation.
- Trainable phi sidecar.
- Per-stream adapters for stream identity and local specialization.
- Speculative Note Conditioning (SNC) cross-attention in selected decoder layers.
- Dynamic Notes Bus with `d_notes = 256`, block size `tau = 32`, and reveal delay `Delta = 1`.
- Fixed addressed `2K` SNC window: K prompt anchors followed by the latest
  eligible dynamic write from each producer under last-write-wins replacement.
  For K=3 the canonical window is `(B, 6, 256)`.
- Speculation head as the only learned writer to the bus.
- VQ planner and `plan_notes_proj` are required for the submitted PDT system, producing snapshot-0 notes on the bus before K stream continuations begin.

Required corrections:

- Remove `NotesHead`; it is a supervised side writer and not part of the real mechanism.
- Remove hashed teacher notes and hashed plan IDs.
- Require `planner -> plan_notes_proj -> Dynamic Notes Bus` for all non-diagnostic training and evaluation.
- Do not detach bus writes during training unless an ablation explicitly requires it.
- In training, the receiving stream's LM loss must backprop through SNC into visible sibling notes, the sibling speculation head, and any upstream trainable sidecar modules on that path.
- Keep the frozen trunk frozen. If full fine-tuning is used, it is a baseline, not PDT.

## Training Plan

### Stage 0: Diagnostic Mechanism Sanity With Fixed Anchors

Goal: prove the bus path can carry dependency information before adding planner complexity.

Setup:

- Use Latent Dependency Control.
- Use fixed learned stream anchors or stream-local inputs.
- Train stream adapters, SNC, notes gates, and speculation head.
- Active loss: LM CE on all target blocks, reported separately for dependency and nondependency spans.

Gate to pass:

- Nonzero gradients on SNC projections, notes gates, speculation head, and stream adapters.
- Bus mutation changes receiver block-1 logits.
- Gate-zero hurts dependency spans more than nondependency spans on a 32-example dry run.

Status of this stage:

- Diagnostic only.
- Not sufficient for the paper's main PDT claim.
- Must be followed by planner-produced snapshot-0 notes before any headline result.

### Stage 1: VQ Planner Integration

Goal: replace fixed anchors with prompt-time VQ slot vectors and make planner-to-bus mandatory.

Setup:

- Planner emits per-slot pre-quantization vectors `u`.
- Learned codebook `C` quantizes to `z_q` with straight-through estimator.
- `plan_notes_proj` maps planner slots to stream snapshot-0 notes.
- Snapshot-0 notes are pushed onto the Dynamic Notes Bus and are the only initial stream-role signal in full PDT.
- Use commitment/codebook losses only as stabilizers. Do not claim a VQ-only warmup discovers semantics; semantics come from downstream LM loss.

Additional regularization:

- Codebook usage entropy.
- Slot diversity loss.
- Optional EMA codebook update if collapse occurs.

Gate to pass:

- Codebook uses more than a trivial number of entries.
- Planner slots differ across streams on nondegenerate prompts.
- Removing planner snapshot-0 degrades assignment/coverage on planner benchmark.

### Stage 2: Integrated Block Rollout

Goal: match inference during training.

Training step:

1. Encode shared context once.
2. Initialize K stream states.
3. For each block, run each stream with its current visible notes window.
4. Before block m>0, consume the canonical private-observation chat transition
   under block m's frozen notes context without assigning LM loss to it.
5. Compute LM CE per stream/block/span.
6. At block end, run speculation head and publish notes.
7. Continue to the next block with reveal delay.
8. Backprop through the whole block graph.

The current trainer implements this differentiable rollout with exact tau
blocks, fixed 2K windows, in-graph sibling writes, and one differentiable
incremental KV cache per stream. New notes affect only newly consumed tokens;
earlier states are never re-encoded. Structured runtime requires the same
per-stream block-transition rows and consumes each next observation only after
all sibling writes publish, under that next block's frozen Delta=1 window.

### Stage 3: Late Mechanism Training; Commit Control Deferred

Goal: add production-facing control after the core information-flow claim works.

The canonical model does not instantiate coverage/agreement heads and runtime
does not fabricate commit scores or rollback events. The fourth schedule
interval continues training the causal mechanism; commit/agreement control is
a later, separately validated architecture stage.

### Stage 4: Distillation And Natural Transfer

Goal: improve quality and language fluency without changing the mechanism.

Use two teachers with non-overlapping responsibilities:

- Functional teacher: the same frozen `Qwen3-4B-Instruct-2507` trunk with the
  complete serialized prefix and every sidecar context disabled. Apply
  forward token KL at temperature 2 on dependency spans while retaining hard
  CE on all target tokens.
- Response teacher: `Qwen3-30B-A3B-Instruct-2507`, loaded offline only to
  generate validated three-stream HotpotQA/OASST1 targets. Train on its
  sequences; do not align its MoE hidden states with the dense 4B trunk.

After teacher-forced context KD passes, add student-prefix on-policy examples
scored by the functional teacher. Final ablations must still show that the bus
carries dependency information rather than merely reproducing teacher text.

## Loss Contract

Active loss family after hash removal:

```text
L_total =
    L_LM_CE
  + lambda_kd * L_KD_LM
  + beta_commit * L_vq_commit
  + beta_codebook * L_vq_codebook
  + lambda_usage * L_codebook_usage
  + 0.1 * L_stream_classifier
```

Deleted losses:

- `L_notes` against hashed teacher notes.
- `L_spec` against hashed teacher/speculative notes.
- `L_plan` cross-entropy against hashed planner IDs.

Reporting requirement:

- Always report `L_LM_CE_dependency` and `L_LM_CE_nondependency` separately.
- A uniform loss gap under bus ablation is weak evidence. A dependency-concentrated gap is the desired signature.

## Ablations

These are pre-registered. Any failure is a real null result, not a tuning inconvenience.

1. **Gate zero.** Force SNC contribution to zero. Dependency-span CE should increase.
2. **Norm scramble.** Preserve note norms but randomize directions. Tests whether content, not magnitude, matters.
3. **Source swap.** Swap sibling notes between examples or streams. Receiver output should follow the swapped note on dependency spans.
4. **Bus mutation.** Directly perturb one source stream's block-0 note and measure receiver block-1 logit changes.
5. **Parameter-matched self-only replacement.** Replace SNC with same-parameter attention over receiver's own prior hidden states. If this closes the gap, the bus claim fails.
6. **Full-token text bus.** Give streams sibling text summaries. This is an upper-bound communication baseline, not a competitor PDT must beat on quality.
7. **Full-KV/concurrent attention baseline.** Approximate Hogwild/Group Think style high-bandwidth sharing where feasible.
8. **No planner snapshot.** For integrated runs, remove snapshot-0 plan notes to test planner usefulness.
9. **Frozen random notes.** Same bandwidth, no learned writer. Tests whether learned note content matters.

## Baselines

Core mechanism baselines:

- Blind parallel: K streams, same local setup, no bus.
- Sequential oracle: stream dependencies resolved by full previous sibling text/state.
- Full-text bus: sibling state serialized into text between blocks.
- Parameter-matched self-only module.
- Full-KV/full-token concurrent sharing where implementable.

Planner/natural-language baselines:

- Single-stream full answer.
- Skeleton-of-Thought with external decomposition.
- Mixture-of-Agents with sequential full-text rounds.
- Group Think / token-level concurrent reasoning where available.
- Multiverse/APR-style parallel reasoning baselines where reproducible enough for fair comparison.
- Full fine-tune single-stream baseline as an upper-bound quality reference.

Latency reporting:

- Report wall-clock on fixed hardware.
- Report prompt prefill count, continuation forward count, synchronization blocks, and communication bandwidth.
- Include external decomposition latency for SoT and full-text round latency for MoA.
- Do not hide K-stream continuation cost.

## Metrics

Primary mechanism metrics:

- Dependency-span CE delta: ablated minus normal.
- Nondependency-span CE delta: ablated minus normal.
- Dependency selectivity ratio: dependency delta / max(nondependency delta, epsilon).
- Bus mutation effect: KL or logit delta on annotated receiver spans.
- Physical note bandwidth: `K * d_notes * bytes * blocks`, compared with text
  tokens and KV bytes. A current note is dense BF16: `256 * 2 = 512` bytes or
  4096 bits per producer/write. The 18-bit synthetic payload therefore has an
  efficiency ceiling of `18/4096 = 0.00439453125`; planner VQ is not note VQ.
- Effective information: measured causal CE/KL change, reported separately
  from physical transport capacity.
- Gate opening: mean sigmoid notes gate by layer.
- Gradient path health: nonzero gradients on SNC, notes gates, speculation head, `plan_notes_proj`, and planner codebook when active.

Planner metrics:

- Codebook unique entries.
- Per-slot entropy.
- Stream assignment diversity.
- Segment coverage and overlap.
- Permutation-invariant target alignment score.

Quality/transfer metrics:

- Cross-reference recall.
- Contradiction rate.
- Segment redundancy.
- Task success for signal/sensor rules.
- Held-out family dependency-span CE gap.

## Implementation Phases

### Phase 1: Scrub Hash Era

Files to remove:

- `src/pdt/datasets/hashing.py`
- `src/pdt/datasets/plan_catalog.py`
- `src/pdt/sidecar/heads/notes.py`
- `scripts/download_corpus.sh`

Files to update:

- Remove hashing imports and outputs from `src/pdt/datasets/retokenize.py`.
- Remove `teacher_notes`, `student_notes`, `planner_targets`, and `planner_mask` from `src/pdt/training/dataset.py`.
- Remove notes/spec/planner CE branches from `src/pdt/training/losses.py`.
- Remove `NotesHeadConfig`, `TeacherCacheConfig`, `plan_hash_salt`, `weights.notes`, and `weights.spec` from `src/pdt/config/schemas.py`.
- Remove `NotesHead` and `PlanEmbedding` assumptions from `src/pdt/model.py` once planner owns the codebook.

Verification:

- Add `test_no_hashing.py`.
- `rg "hash|planner_ids|notes_teacher|notes_student|NotesHead|TeacherCache" src/pdt tests` must find no runtime/training dependency except migration notes or explicit negative tests.

### Phase 2: Data And Validators

New files:

- `scripts/generate_dependency_dataset.py`
- `scripts/validate_dependency_dataset.py`
- `scripts/generate_snapshot_routing_dataset.py`
- `scripts/retokenize_corpus.py` rewritten for the new schema.

Requirements:

- Generate Latent Dependency Control programmatically first.
- Enforce at least two blocks for Delta=1.
- Reveal private randomized registers one block at a time. Reject any legacy
  record that places the complete future register log in the initial prompt.
- Emit dependency span token indices after retokenization.
- Emit local-only and full-visibility CE audit reports.
- Fail fast if audit thresholds are not met.

### Phase 3: Differentiable Block Rollout Trainer

Rewrite:

- `src/pdt/training/trainer.py`
- `src/pdt/training/dataset.py`
- `src/pdt/training/losses.py`
- `src/pdt/runtime/window.py` if training needs a batchable window builder.

Requirements:

- Training and inference must share the same block/lag semantics.
- Bus writes stay in graph.
- Span masks survive batching.
- Per-stream local observations are supported for the mechanism benchmark.
- Single shared prompt mode is supported for planner benchmark.

Verification tests:

- `test_block_rollout_semantics.py`
- `test_functional_distillation.py`
- `test_runtime_bus_window.py`
- `test_dependency_data_contracts.py`

### Phase 4: VQ Planner

Rewrite:

- `src/pdt/sidecar/heads/planner.py`
- `src/pdt/sidecar/heads/plan_notes_proj.py`
- `src/pdt/sidecar/plan_embedding.py` removed or collapsed into planner codebook.
- `src/pdt/diagnostics/codebook.py`

Requirements:

- Planner returns indices, quantized vectors, pre-quantization vectors, and VQ losses.
- Straight-through quantization is explicit.
- Codebook collapse diagnostics are logged every eval interval.
- No external slot ID targets.

Verification tests:

- `test_vq_planner.py`
- `test_coordination_calibration.py`

### Phase 5: Ablation CLI And Baselines

Status: partial. Gate-zero, norm-scramble, anchor swap, donor-only source swap,
and addressed bus mutation primitives exist. The generation CLI exposes the
applicable interventions, and paired token-weighted causal accumulation is
implemented. `src/pdt/baselines/self_only.py` provides an SNC-parameter-matched
receiver-history-only control with strict causal ownership checks. The
teacher-forced rollout still needs to invoke these paths, and the blind,
sequential-oracle, full-text, full-KV, single-stream, and full-finetune runners
do not yet exist.

Update:

- `src/pdt/cli/ablate.py`
- `src/pdt/runtime/counterfactuals.py`

New directory:

- `baselines/`

Required runners:

- `blind_parallel.py`
- `sequential_oracle.py`
- `full_text_bus.py`
- `self_only_replacement.py`
- `skeleton_of_thought.py`
- `mixture_of_agents.py`
- `single_stream.py`
- `full_finetune.py`

Optional if reproducible:

- `group_think_like.py`
- `hogwild_like.py`
- `multiverse_like.py`

### Phase 6: Paper Rewrite

Rewrite:

- Abstract and introduction around causal latent coordination.
- Related work table with closer 2025 neighbors.
- Architecture section with precise shared-prefill + K-continuation wording.
- Training section with no hash losses.
- Dataset section led by Latent Dependency Control.
- Evaluation section led by dependency-span ablations.
- Application section labels sensor/signal deployment as motivation and future transfer, not the first empirical claim.

Remove:

- "Existence proof" language that is too vague.
- "One forward pass" claims that imply false compute savings.
- Sectional-independence apology from the old paper; replace with the new dependency validator contract.

### Phase 7: Scale Gate

Do not run large training until all are true:

- 32-example Latent Dependency Control dry run passes gradient, mutation, and gate-zero checks.
- 1k-example run shows stable dependency-span gap and nondependency selectivity.
- Parameter-matched self-only replacement fails to close the dependency gap.
- VQ planner does not collapse on the planner benchmark.
- README and paper draft reflect the actual implemented system.

CUDA allocation:

- One H100 SXM 80GB, BF16, PyTorch SDPA, and batch size 1. The canonical
  incremental differentiable rollout requires reusable KV caches, so Hugging
  Face gradient checkpointing is disabled rather than allowed to silently
  suppress `past_key_values`.
- At least 200GB persistent storage for the environment, model revisions,
  datasets, telemetry, and checkpoints.
- First rental is the 32/1k gate only. Measure peak VRAM and examples/second;
  do not begin the configured 50k optimizer-step job without a measured
  runtime estimate.
- The current configuration has exactly 401,325,095 trainable parameters
  (approximately 401.3M), so 48GB devices are outside the canonical plan.

## Dry-Run Acceptance Criteria

On 32 examples:

- `test_gradient_flow.py` passes for SNC q/k/v/o, notes gates, speculation head, `plan_notes_proj`, and planner codebook when planner is active.
- Bus mutation produces measurable receiver logit change on dependency spans.
- Gate-zero dependency CE delta is positive and at least 3x nondependency CE delta.
- Codebook uses at least 20 entries when planner is active.
- No hash-era fields are present in runtime/training batches.

On 1k examples:

- Mean dependency-span gate-zero CE delta >= 0.75 nats/token.
- Mean nondependency-span gate-zero CE delta <= 0.25 nats/token.
- Source-swap moves receiver predictions toward swapped source state.
- Parameter-matched self-only replacement recovers less than 50% of the bus gain.
- Full-text bus performs at least as well as PDT on dependency spans, validating that the task truly benefits from communication.

## README State

After implementing any phase, update `README.md` with:

- Current phase status.
- Exact data generation commands.
- Exact training commands.
- Exact ablation commands.
- Known null/failure conditions.

The README is for future AI/LLM operators. It should be operational and specific, not promotional.
