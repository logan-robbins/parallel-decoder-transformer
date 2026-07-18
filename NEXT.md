# PDT Next Steps Handoff

Status date: 2026-04-24.

This file is the continuation plan after the hash-era scrub, canonical
dependency dataset loader, VQ planner, and differentiable block rollout were
implemented. An AI coding agent should start here, not by re-reading old
paper drafts.

## Current Implemented State

The repository now targets this PDT path:

```text
shared context -> frozen Qwen3 trunk prompt encode -> VQ planner
               -> plan_notes_proj -> snapshot-0 Dynamic Notes Bus writes
               -> SNC reads visible notes in K stream block rollout
               -> SpeculationHead writes block-end notes for later blocks
```

Implemented and smoke-tested:

1. Hash-era planner and note supervision were removed from runtime/training.
   Deleted modules include `src/pdt/datasets/hashing.py`,
   `src/pdt/datasets/plan_catalog.py`, `src/pdt/sidecar/heads/notes.py`,
   `src/pdt/sidecar/plan_embedding.py`, and `scripts/download_corpus.sh`.
2. `PlannerHead` owns a learned VQ codebook and returns logits, indices,
   pre-quantized vectors, straight-through quantized vectors, and VQ losses.
3. `PlanNotesProjection` consumes quantized planner slot vectors directly.
4. `PDTCollator` and `PDTDependencyDataset` load the canonical no-hash schema
   and reject removed fields such as `planner_ids`, `notes_teacher`, and
   `notes_student`.
5. `compute_pdt_losses` reports `lm_ce_dependency` and
   `lm_ce_nondependency` separately. Removed losses are planner CE, notes MSE,
   and speculation-note MSE.
6. `PDTTrainer._train_step` now performs teacher-forced block rollout with
   planner-seeded notes and in-graph speculation writes. Receiver LM CE can
   backpropagate through SNC into visible sibling notes and upstream sidecar
   modules.
7. Dataset scripts exist for synthetic Latent Dependency Control, shared
   snapshot routing, retokenization, and CE audit.
8. `README.md` reflects the current commands and Apple Silicon / CUDA split.

Validation already run locally on Mac M4:

```bash
uv run pytest tests/smoke/ -q
# 19 passed

uv run scripts/ablate.py --help

uv run scripts/generate_dependency_dataset.py \
  --output /tmp/pdt_ldc_smoke.jsonl \
  --num-examples 2
```

Do not assume any full Qwen3-4B training or CE audit has been run yet.

## Immediate Next Milestone

Build the 32-example dry run required before scale training. This is the next
real gate.

1. Generate a small Latent Dependency Control dataset.
2. Retokenize with a local tokenizer path only. Do not download model weights
   on the Mac unless the user explicitly approves it.
3. Run CE audit on an NVIDIA/CUDA host or against an already-local model path.
4. Run a 32-example PDT dry training pass.
5. Prove these diagnostics:
   - Nonzero gradients on SNC q/k/v/o, outer notes gates, SpeculationHead,
     stream adapters, `plan_notes_proj`, and planner codebook when planner is
     active.
   - Bus mutation changes receiver block-1 logits on annotated dependency
     spans.
   - Gate-zero hurts dependency-span CE at least 3x more than nondependency CE.
   - Planner codebook uses at least 20 entries in the dry run.
   - Runtime/training batches contain no hash-era fields.

Suggested commands once local tokenizer/model paths exist:

```bash
uv run scripts/generate_dependency_dataset.py \
  --output data/datasets/ldc/train_32.jsonl \
  --num-examples 32 \
  --seed 123

uv run scripts/retokenize_corpus.py \
  --input data/datasets/ldc/train_32.jsonl \
  --output data/processed/latent_dependency_control/train_32.jsonl \
  --tokenizer /path/to/local/Qwen3-4B-Base

uv run scripts/validate_dependency_dataset.py \
  --input data/processed/latent_dependency_control/train_32.jsonl \
  --model /path/to/local/Qwen3-4B-Base \
  --output-report data/processed/latent_dependency_control/audit_32.json
```

The audit script uses `local_files_only=True`; if the tokenizer/model is
missing, stop and ask the user before any large download.

## Tests To Add Next

Add focused tests before changing training behavior again:

1. `test_gradient_flow.py`
   - Assert nonzero gradients for SNC projections, notes gates,
     SpeculationHead, stream adapters, `plan_notes_proj`, and planner codebook
     on a tiny model.
2. `test_bus_mutation_propagation.py`
   - Perturb a source stream block-0 note and assert receiver block-1 logits
     change on dependency positions.
3. `test_dependency_span_masks.py`
   - Verify retokenized dependency spans map to the intended block/token
     indices and nondependency masks exclude them.
4. `test_training_matches_runtime_lag.py`
   - Compare training helper lag semantics against runtime `NotesWindowBuilder`
     for snapshot-0 and block-end writes.
5. `test_slot_specialization.py`
   - Check planner slots/codebook selections differ on nondegenerate prompts.
6. `test_single_prompt_planner.py`
   - Add the planner benchmark shape/evaluation contract.
7. `test_planner_to_bus_required.py`
   - For integrated runs, removing snapshot-0 planner notes must degrade
     assignment/coverage or fail the run explicitly.

## Phase 5: Ablations And Baselines

Update `src/pdt/cli/ablate.py` and `src/pdt/runtime/counterfactuals.py` so
ablation output is span-local, not only text-level.

Required ablations:

1. Gate zero: force SNC contribution to zero.
2. Norm scramble: preserve note norms and randomize directions.
3. Source swap: swap sibling notes between streams or examples.
4. Bus mutation: directly perturb one source stream note and measure receiver
   dependency-span logit/CE change.
5. Parameter-matched self-only replacement: replace SNC with attention over
   receiver history only.
6. Full-token text bus: serialize sibling state between blocks as an upper
   communication bound.
7. Full-KV/full-token concurrent sharing where feasible.
8. No planner snapshot for integrated runs.
9. Frozen random notes with the same bandwidth.

Create `baselines/` with these runners:

1. `blind_parallel.py`
2. `sequential_oracle.py`
3. `full_text_bus.py`
4. `self_only_replacement.py`
5. `skeleton_of_thought.py`
6. `mixture_of_agents.py`
7. `single_stream.py`
8. `full_finetune.py`

Optional only if reproducible enough for fair comparison:

1. `group_think_like.py`
2. `hogwild_like.py`
3. `multiverse_like.py`

Primary metrics to report:

1. Dependency-span CE delta: ablated minus normal.
2. Nondependency-span CE delta: ablated minus normal.
3. Dependency selectivity ratio:
   `dependency_delta / max(nondependency_delta, epsilon)`.
4. Bus mutation KL or logit delta on annotated receiver spans.
5. Note bandwidth: `K * d_notes * bytes * blocks`, compared with text tokens
   and KV bytes.
6. Mean sigmoid notes gate by layer.
7. Gradient path health.

## Phase 6: Paper Rewrite

Rewrite the paper only after the 32-example and 1k-example mechanism gates
produce real results.

Paper direction:

1. Lead with causal latent coordination under dependency, not generic parallel
   decoding.
2. Use the novelty statement:
   PDT tests whether a frozen causal decoder can be augmented with a learned
   low-bandwidth latent bus so multiple persistent streams can coordinate
   under delayed, compressed sibling-state visibility.
3. State compute honestly:
   shared prefill plus synchronized K-stream continuation, not one free
   forward pass for K outputs.
4. Lead evaluation with dependency-span ablations.
5. Treat sensors/signal applications as motivation and downstream transfer,
   not as the first empirical claim.

Remove from the paper:

1. Any claim that no prior paper occupies the general space.
2. Any "one forward pass" wording that hides K continuations.
3. Any strict information-necessity claim when all facts are inferable from a
   globally shared prompt.
4. Old sectional-independence framing from the previous paper.

## Phase 7: Scale Gate

Do not run large training until all are true:

1. 32-example Latent Dependency Control dry run passes gradient, mutation, and
   gate-zero checks.
2. 1k-example run shows stable dependency-span gap and nondependency
   selectivity.
3. Parameter-matched self-only replacement fails to close the dependency gap.
4. VQ planner does not collapse on the planner benchmark.
5. `README.md` and the paper draft describe the actual implemented system.

1k-example acceptance criteria:

1. Mean dependency-span gate-zero CE delta >= 0.75 nats/token.
2. Mean nondependency-span gate-zero CE delta <= 0.25 nats/token.
3. Source-swap moves receiver predictions toward swapped source state.
4. Parameter-matched self-only replacement recovers less than 50% of the bus
   gain.
5. Full-text bus performs at least as well as PDT on dependency spans,
   validating that the task benefits from communication.

## Hardware Boundary

Mac M4 Silicon is appropriate for:

1. Code edits.
2. Synthetic dataset generation.
3. Retokenization if tokenizer files already exist locally.
4. Tiny smoke tests.
5. Shape, masking, config, and gradient-path tests.

Use NVIDIA/CUDA for:

1. Full PDT training with Qwen3-4B.
2. CE audits using Qwen3-4B over nontrivial datasets.
3. Ablations on trained checkpoints.
4. Latency and throughput claims.

If any step requires missing model weights, large downloads, or cloud/GPU
spend, stop and ask the user first.
