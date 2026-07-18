# PDT Research Plan: Latent Coordination Under Real Dependency

Status date: 2026-07-17.

This is the living implementation record. The canonical architecture is now
the source-grounded continuous-plan system in `07_16.md`; older synthetic
dependency work remains historical diagnostic context. The admissible
historical-source, teacher-data, split, retokenization, packed bus/self-only,
architecture-screen, automatic-screen, and blinded-human-audit paths are
implemented. Candidate review, real-plan Batch data, H100 optimization, and
every empirical result remain pending.

## Living Task Status

1. [done] Reframe the thesis from generic parallel decoding to learned latent coordination under dependency.
2. [done] Incorporate the useful part of `docs/Logan.txt`: the value proposition is K parallel streams with a compressed, reveal-delayed channel versus blind parallelism, full-text exchange, or full-KV sharing.
3. [done] Discard weak claims: "no prior paper occupies this point" as an absolute statement, "one forward pass" wording that hides K continuations, and one-block expository citations as a proof of informational necessity.
4. [done] Remove hash-era planner/notes supervision from code. 2026-04-24: removed hash datasets, `NotesHead`, `PlanEmbedding`, planner CE targets, teacher-note MSE, and spec-note MSE.
5. [done] Build a controlled dependency benchmark where sibling state is unavailable except through the bus. 2026-04-24: added programmatic LDC generation, retokenization, and structural validation. 2026-07-16: replaced the pooled teacher audit with document-paired blind and causal full-information quality controls.
6. [done] Implement differentiable block rollout so LM loss reaches SNC, notes gates, speculation writes, plan seeding, and the planner codebook. 2026-04-24: trainer now runs planner-seeded block rollout with in-graph speculation writes and span-local CE reporting.
7. [partial] Add ablations and baselines that can falsify the claim. The trainer runs aligned baseline, gate-zero, sibling norm-scramble, and exact-source targeted-write mutation rollouts. The independently trainable, checkpoint-isolated self-only condition uses the same 16-block history capacity, exposes only receiver-owned states, and now runs through the same packed three-lane free-generation and human-audit path as the bus checkpoint. Bus versus self-only is compared by a paired per-document bootstrap of the bus advantage; the unsupported recovery-fraction threshold is gone. Frozen-trunk blind and causal sequential full-information quality controls now retain paired document CEs and align strictly with PDT telemetry. Separately trained full-text, full-KV, single-stream, and full-finetune runners remain.
8. [partial] Rewrite the paper around mechanism-first evidence, with natural-language and sensor/signal tasks as downstream validation. `THEORY.md` now records the implementation boundary; empirical results remain pending.
9. [done] Lock the July 2026 starting recipe: frozen `Qwen3-4B-Instruct-2507`, same-trunk full-prefix context distillation, exact-entropy synthetic data first, and one rented H100 SXM 80GB for the scale gate.
10. [done] Align the canonical YAML, revision, adapter, and `qwen3-instruct-temporal-chat-v2` tokenization path to `Qwen3-4B-Instruct-2507`.
11. [done] Feed same-trunk privileged per-block teacher logits into dependency-masked forward KL at temperature 2; raw hidden-state MSE is absent.
12. [in progress] Pass the 32-document long-form overfit gate and matched self-only control before launching the larger synthetic corpus. Immutable raw train/validation/null sets and pinned 4B retokenizations now exist locally.
13. [in progress] Build the missing planner-supervision corpus before claiming the planner gate. The first corpus is restricted to reliable historical events and processes, not arbitrary Wikipedia pages. Ground every example in immutable, pinned-revision source material whose complete admissible prose is 3,000-7,000 Qwen tokens. The student-visible source is an allowlist of article title, hierarchical content headings, and ordinary prose paragraphs only. Citation relations and source metadata remain auditable but never enter the rendered student prompt. After cited-fact feasibility passes, use one joint Batch API teacher request to emit a shared expository prompt, exactly three unordered complementary stream plans, three several-paragraph stream targets, explicit cross-stream dependency links, a coverage map, and source-span provenance. Preserve paragraph and discourse structure; do not convert the task into short QA or independent per-stream responses. Validate locally, group related topics before splitting, randomly permute stream order, and train/evaluate decomposition with permutation-invariant matching so fixed labels cannot masquerade as learned planning.
14. [done] Use one strict atomic checkpoint format for training, resume, inference, and ablation, including global-step/stage policy restoration. Format v4 records the continuous-plan architecture and `coordination_source`, so bus and self-only checkpoints cannot cross-load.
15. [done] Validate the real pinned checkpoint before GPU rental. The original full-width MPS round proved the frozen/trainable partition and exercised FP32 sidecar/BF16 trunk boundaries, packed cache, addressed notes, and all streams. The current low-rank profiles supersede its parameter count with exact 4B/14B contracts of 156,484,647/305,374,247 trainable scalars.
16. [done] Publish the complete local implementation and documentation worktree, excluding ignored model artifacts and heavyweight generated datasets. 2026-07-16: the current real-plan implementation passes Ruff, mypy across 84 source files, and all 198 tests. These are plumbing results only.
17. [done] Re-audit the thesis at the information-theory and GPU-kernel boundary. 2026-07-16: separated three necessary gates—low residual conditional information, width in the output dependency DAG, and a physically packed K-frontier executor that reuses frozen weights.
18. [done] Correct measurement language that treats arbitrary model-relative cross-entropy differences as Shannon mutual information. 2026-07-16: paired dependency CE remains a causal utility metric; finite-bit claims now require the known-entropy variational audit and an explicit finite message.
19. [done] Add a machine-checkable decode roofline/work-span model for the pinned Qwen3-4B trunk, current recurrent sidecar, GQA KV traffic, packed versus sequential stream calls, and full-KV versus fixed-note communication. The local MPS primitive measured 2.81x batch-3 aggregate throughput at short context; this is not an end-to-end PDT result.
20. [done] Replace per-stream Python continuation calls with one packed K-stream decode path, including batch-addressed stream adapters, explicit note metadata, KV state, and structured block transitions. One frontier-owned HF cache has physical shape `(K, heads, P, head_dim)`; a `(K, P)` validity mask and per-row logical `position_ids` preserve unequal prompt/transition histories while every trunk invocation advances all K rows together. The real 4B MPS audit observed one `(3,18)` prefill and 33 `(3,1)` continuation calls through a delayed-note read. This proves physical packing, not CUDA speedup.
21. [done] Implement an explicit dynamic-note rate before the 32/1k training gate. The single canonical writer now uses four 256-entry product codebooks: 32 bits per write. The bus accepts only four indices and reconstructs the float note from the shared codebook; straight-through losses, marginal-entropy anti-collapse pressure, and per-codebook telemetry are wired into training. The 18-bit relay occupies at most `18/32 = 0.5625` of configured capacity.
22. [partial] Earn the remaining empirical gates. 2026-07-16: the fresh two-update bus probe passed on an H100 80GB in 14.12 seconds at 30.44GB peak reserved VRAM, with finite nonzero gradients in every active mechanism group and a verified format-v3 checkpoint. The obsolete 512-update short-register run produced a positive dependency effect and measurable mutation but is now only historical mechanism evidence. The evaluator now retains paired document effects, lag-resolved intervals, and source-aligned mutation KL; new long-form bus/self-only results remain.
23. [done] Replace the short relay experiment with the canonical long-form document program. Each example has three continuous 1,024-token prose streams, 32 synchronized 32-token blocks, sixteen exact cross-section constraints at lags 1/4/8/16, sixteen within-document local controls, immutable private source packets, exact 18-bit dependency payloads, and a surface-matched self-owned null twin. Generation and retokenization reject short-horizon, leaked, unresolvable, or tokenizer-misaligned records.
24. [done] Make the architectural extension trunk-width generic. Pinned dense-Qwen3 profiles materialize hidden width, decoder depth, twelve equal-depth instrumentation sites, and profile-specific tokenized data. SNC uses a fixed 512-wide eight-head communication core; the planner, projection, and classifier use fixed 512-wide bottlenecks. The same implementation has exact parameter contracts for 4B and 14B.
25. [in progress] Promote dense Qwen3-14B to the serious single-H100 long-form rung after a fail-fast memory probe. Qwen3-4B remains the mechanism/debug rung. Do not substitute the 30B-A3B MoE merely for nominal parameter count: its expert topology and frozen low-precision path are a separate architectural variable to test only if the dense result shows an active-capacity limitation.
26. [done] Replace ratio gates and phase-confused codebook gates. Evaluation now reports paired per-document dependency and nondependency effects, their difference-in-differences, deterministic bootstrap intervals, lag-resolved effects, exact-future-use mutation KL, and actual head/layer parameter norms and gate openings. The evidence gate requires enough documents plus positive lower confidence bounds for dependency effect, selectivity, and mutation. Codebook statistics report their observable ceiling and exact collapse only; utilization is diagnostic, not causal evidence.
27. [partial] Regenerate and verify long-form fixtures. Immutable 32-document train, validation, and null raw sets plus complete pinned-4B retokenizations are present and pass 3,072-block stress validation. All three processed splits were regenerated after adding per-receiver causal full-information prompts. Pinned-14B retokenization, a true 4B train-set overfit evaluation, the matched self-only run, and the 14B memory/optimizer probe remain.
28. [partial] Add checkpoint-aligned long-form quality bounds. Retokenization now materializes and validates a receiver-explicit causal full-information text prompt for every block. One frozen-trunk scorer retains paired per-document CE for the blind local-history lower control and sequential full-information upper control; one strict comparison aligns those controls with profile-labeled PDT normal-rollout telemetry. The H100 train-split control completed all 6,144 scoring tasks over 32 documents: causal full information reduced dependency CE by 2.0218 nats with 95% CI `[1.9581, 2.0836]`, while the dependency-minus-nondependency selectivity lower bound was 2.2621. This establishes frozen-4B oracle headroom for the controlled data, not a PDT result. Null-split scoring and checkpoint-aligned PDT comparisons remain.
29. [done] Remove avoidable H100 work from functional distillation. The three privileged receiver targets at each synchronization block are one left-padded frozen-trunk frontier call, reducing the unpruned maximum from 96 to 32 calls per document. Pinned Qwen3 projects only the target-plus-one logits, and tests prove exact causal-position agreement under unequal padded target lengths. Dataset load also rejects missing or duplicate document identity before model construction.
30. [done] Make functional KD dependency-sparse end to end. Local-control blocks launch no teacher call, so the exact long-form schedule uses sixteen packed teacher frontiers per document. The trainer retains only 379–403 dependency vocabulary rows on the current 4B train split instead of 3,072 dense target rows. Sparse row count/order is checked against boolean indexing of the student rollout, and tests prove the same temperature-two KL with detached teacher distributions.
31. [in progress] Make the full 32-block differentiable rollout fit one H100 without shortening the causal horizon or truncating gradients. The first post-sparse 4B optimizer probe exhausted 79.16 GiB inside masked SDPA. FlexAttention produced an illegal CUDA access; native GQA with the materialized mask and a block-boundary KV truncation trial also exhausted memory and were rejected. Fused lower-right causal GQA plus pinned and unpinned saved-tensor offload held HBM low but crossed the container's hidden 251 GB host-memory cgroup because it copied overlapping cached prefixes quadratically; saved-tensor offload is rejected.
32. [in progress] Run the immediate one-terminal-layer plumbing rung with the complete 1,024-token streams. This preserves the learned 32-bit writer, delayed addressed bus, SNC, stream adapter, exact-token loss, and all 32 synchronization blocks while keeping frozen recurrent KV state outside the trainable graph. Require a two-update gradient-and-parameter-movement probe, then move to real source-grounded data. A terminal-layer null is a diagnostic of that restricted path only; it cannot falsify the recurrent twelve-layer architecture or model-intrinsic parallel generation.
33. [in progress] Correct the attribution hazards exposed by the independent possibility audit before interpreting a negative result. The 64-update terminal run left its inner, outer, and adapter gates bit-for-bit at `sigmoid(-4)` and produced a null held-out dependency delta of `0.0000057` nats; replacement-layer casting had silently converted trainable gates to BF16, whose spacing at -4 exceeds the AdamW update. Keep frozen Qwen weights BF16 but promote all trainable per-layer phi to FP32, use an optimization-neutral open-gate positive control, and make the CUDA probe verify parameter movement. In parallel, score blind versus causally full-information frozen-4B controls. A terminal-layer, quantized, teacher-forced failure cannot reject the recurrent twelve-layer architecture, semantic coordination, free generation, or a larger trunk.
34. [done] Add a real semantic decomposition objective to the planner rather than assuming downstream LM loss and round-robin slot ownership will discover it. The planner predicts three unordered eight-node continuous outlines; exact six-way matching aligns them to frozen semantic teacher targets. Every teacher lane axis is randomly rebound to D1-D3 during collation.
35. [done] Implement the 2026-07-16 real-data architecture in `07_16.md`. The canonical path now has persistent continuous plan memory, one shared plan-conditioned adapter, one packed three-row training/inference frontier, exact per-lane fact ownership and hard-negative supervision, real self-only control, exact dependency-token causal ablation, and stage-aware oracle/learned/zero/swap free-generation evaluation.
36. [in progress] Generate the real source-grounded corpus and earn the empirical gates. The rejected cleaned-Wikipedia sampler has been deleted. Tasks 37-41 now pass locally. On 2026-07-17, two immutable eight-category acquisition screens measured real transfer and rejection behavior before scaling candidate review. The broad famous-event screen rejected all eight, primarily because complete cleaned bodies contained 8,259-22,777 Qwen tokens. A wikitext-size-targeted mid-scale screen produced the first strict eligible source, `Baltimore railroad strike of 1877`; seven other rows failed explicit size, section, assessment, reference-resolution, or scholarly-source gates. The source is now fully materialized from the pinned revision as a 6,474-token `pdt-historical-source-v1` record with 14 sections, 61 substantive paragraphs, 55 reference records, exact content hashes, family assignment, and split. Because exact eight-category balance was impossible, no accepted manifest was published. Filter failures now atomically retain `failure.json`, `rejections.json`, and any integrity-bound `eligible_sources.jsonl`, fixing the prior path that told the operator to inspect rejection evidence and then discarded it. A verifier rejects any eligible-pool mutation or accepted-manifest contamination. These screens are not the 128 accepted-example pilot and cannot satisfy the data gate by themselves. The next full action remains a reviewed candidate catalog large enough to yield 128 accepted strict-schema examples, followed by the 2,048-example architecture corpus only after every accepted pilot record is manually inspected. After validation and pinned retokenization, run the H100 two-update optimizer probe, oracle executor gate, planner gate, joint bus training, self-only control, and held-out long-form generation evaluation. Do not infer a result from local plumbing tests. Do not provision a second H100 unless a measured dense-14B fit/sharding requirement establishes that it is necessary.
37. [done] Replace the Hugging Face Wikipedia sampler with one canonical pinned-revision Wikimedia ingress. Retain immutable raw wikitext, rendered semantic HTML, page/revision identity, PageAssessments metadata, and reference metadata. Parse the rendered DOM with an allowlist; do not recover prose with regular expressions. Keep only the title, hierarchical content headings, and ordinary paragraph nodes. Inline link display text remains prose. Exclude every table, list, infobox, template, figure, caption, gallery, map, coordinate, citation marker, footnote body, URL, navigation element, category, authority-control block, hatnote, pronunciation fragment, math/code block, and edit artifact. Exclude the complete References, Notes, Citations, Sources, Bibliography, Further reading, External links, See also, Gallery, and equivalent section subtrees. Unexpected structure is ignored, never serialized into model-visible text.
38. [done] Implement the strict historical-event eligibility manifest. Accept only English main-namespace, non-redirect, non-disambiguation historical events or processes that ended at least 25 years before the pinned dump, are attached to a history-oriented WikiProject with relevant `FA`, `GA`, `A`, or `B` quality. Reject lists, timelines, chronologies, indexes, year/decade pages, outlines, current events, biographies as the primary topic, and maintenance-tagged pages. Require the complete cleaned body, without truncation, to contain 3,000-7,000 pinned-Qwen tokens, at least six substantive sections and at least twelve substantive paragraphs. Acquisition now reads maintenance transclusions from Parsoid metadata rather than assuming an empty set or regex-scanning prose.
39. [done] Make citations a hard local evidence gate rather than model-visible text. Require at least 30 inline reference occurrences, 15 distinct cited works, eight identifiable books/journals/archives/institutional publications, citations on at least 70% of substantive paragraphs, and no single work responsible for more than 25% of occurrences. Reject `citation needed`, `unreferenced`, `more citations needed`, `POV`, `disputed`, `original research`, `cleanup`, `hoax`, or equivalent maintenance templates. Extend source provenance so every paragraph records reference IDs outside its text and every accepted atomic fact maps its exact normalized quote to at least one local reference ID. Require 18-48 cited facts distributed across at least twelve paragraphs and four sections before joint generation.
40. [done] Prevent split leakage and sampling collapse. Cluster near-duplicate and closely related event/article families before assigning train, validation, or test; an entire family belongs to exactly one split. Balance the 128 accepted pilot examples across wars/battles, revolutions/transitions, treaties/crises, social movements/reforms, exploration/migration, scientific/industrial developments, disasters/reconstruction, and cultural/institutional transformations. Store immutable accepted/rejected manifests with deterministic reason codes. Validated teacher examples are atomically repartitioned by the inherited family split with their own hash manifest. Search more of the dump when yield is low; never lower the quality contract to fill a quota.
41. [done] Prove source-renderer cleanliness before any paid call. Golden and adversarial fixtures show exact heading hierarchy, prose, paragraph/reference mapping, maintenance metadata, and exclusion of forbidden DOM classes/sections. Model-visible rendering rejects markup, reference markers, bibliography text, list/table cells, URLs, navigation text, and truncated bodies. Schema validation fails on uncited facts, incomplete articles, unexpected fields, and cross-split topic families. The published contracts are `pdt-historical-source-v1`, `pdt-real-plan-v2`, and `pdt-real-plan-tokenized-v3`; Batch construction requires the exact SHA-256 of a validator-produced accepted manifest.
42. [pending] Execute the paid data sequence only after task 41. Run the cheap fact stage on candidate historical articles; locally enforce cited-fact and three-way decomposition feasibility; submit the more expensive joint three-lane request only for qualifying candidates. Produce 128 accepted final examples and manually inspect every one. Freeze the schema, renderer, teacher prompt, rejection report, and pilot audit before requesting the 2,048-example architecture corpus. The 20,000-example corpus remains forbidden until the architecture gate shows value.
43. [done] Replace hard-coded single-configuration testing with a registered architecture-screen compiler. The checked-in one-factor screen covers 27 variants across trunk scale, instrumented depth, notes/SNC/adapter/planner widths and depths, product-VQ shape, residual gates, optimizer values, and bus/self-only source, with three fixed independent seeds and 81 validated, hashed configs. Compilation loads no model and claims no scientific result.
44. [done] Replace automatic similarity as final fact evidence with a two-layer evaluation. A pinned local-only NLI model provides a disjoint-calibration automatic screen. The final lane-exact gate requires a blinded queue covering every condition/lane/fact and hard negative, two complete independent annotations, distinct third-party adjudication of every disagreement/uncertain item, exact generated evidence quotes, Cohen kappa, paired document bootstraps, and lane-swap moved-versus-unmoved address scoring from the same judgments.
45. [done] Remove remaining shortcut/proxy execution paths. Deleted the obsolete lexical fact-recall, addressing, geometry, bandwidth, blind parallel-decode, and laptop latency scripts; generation ablation now captures raw text/tokens/codes without bag-of-token cosine or ROUGE gates. Real-plan retokenization and generation evaluation publish atomically and require locally cached pinned model assets. Local tests use explicit contract doubles only to validate plumbing and never assert an empirical PDT result.
46. [done] Audit the data that physically exists and trace whether the current planner can learn an unordered, semantically natural three-way decomposition. There are zero `pdt-real-plan-v2` or `pdt-real-plan-tokenized-v3` examples; the Baltimore artifact is one real source packet, not training supervision. The 32-example synthetic long-form corpus hard-codes `historical evidence`, `risk analysis`, and `practical recommendations` and injects marker codewords, while sampled legacy `pdt_10k` rows are explicitly sectionally independent. Neither is admissible. The architecture is physically capable: 24 independently initialized learned queries form three eight-node slots, exact six-way matching aligns the unordered set, every teacher axis and target lane is rebound together, and matched plans condition three independent packed KV rows through parameter-shared adapters. This proves a gradient path, not learned decomposition.
47. [pending] Strengthen the real teacher-data contract before any paid joint batch. Current validation guarantees three long sections, one owner per fact, sibling references, and delayed dependencies, but it does not by itself establish that the selected three-way partition is natural, non-template, or reasonably balanced. Add exact per-plan minimum unique ownership and fact-importance coverage, reject dominant or vestigial lanes, require an explicit teacher decomposition rationale and competing candidate partitions, and manually judge naturalness and complementarity for every 128-example pilot record. Treat BGE/JL plan targets as distillation coordinates only, never as evidence that a decomposition is natural.
48. [done] Build and inspect one complete training example from the immutable pinned Wikipedia revision for `Baltimore railroad strike of 1877`. The new `model_intrinsic_parallel` path imports no existing project dataset code and consumes only the raw Wikimedia record. Its allowlist retains 13 content headings and 52 ordinary paragraphs, removes complete excluded sections and dangling prose that only introduces an excluded list or quotation, and binds every retained citation to raw Parsoid reference metadata. The corrected manually curated example contains 24 exact cited facts across 22 paragraphs and nine sections, eight owned facts per semantic lane, sparse backward references of zero/one/two per presented lane, three delayed dependencies, all six uniform semantic-to-D1/D2/D3 bindings, and actual Qwen target tensors of 765, 777, and 771 tokens including EOS. Inspection exposed and corrected both historical-claim leakage and a data-design error that had forced every lane to reference both siblings, including later material. Every target paragraph now declares its complete allowed fact-support set, compilation requires exact equality with OWNER/REFERENCE realizations, presentation order is explicit, and references are optional but must be backward and delayed. The artifact remains explicitly marked manual and inspection-only; it is neither an empirical result nor a substitute for a scalable teacher-generated corpus.
49. [done] Manually teach and audit several additional real historical examples with different narrative structures. Added featured-article examples for `Blackwater Fire of 1937`, `Battle of Sluys`, and `Great Stink`, covering disaster causality, medieval naval strategy, and civic infrastructure policy. The examples contain 18 cited facts each, six owners per lane, four long-form paragraphs per lane, actual Qwen targets of 705–812 tokens, and zero, one, and two delayed notes respectively. All facts resolve to exact quotes from cited paragraphs spanning at least twelve paragraphs and four top-level sections. Each compiled record stores the teacher's source-quality, claim-coverage, and decomposition audit. The source parser now excludes combined reference headings such as `Notes, citations and sources`; the compiler requires explicit presentation order, permits zero notes, rejects forward or same-block notes, validates representative audited references against selected evidence paragraphs, and retains all six randomized semantic-to-physical bindings. These records validate the data contract only and make no empirical model claim.

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

The canonical `long-form-private-stream-v1` record contains three section
streams (`historical evidence`, `risk analysis`, `practical recommendations`),
32 private observations and 32 target blocks per stream, and exact dependency
metadata for every cross-section constraint. Each dependency records target
block, source stream, source block, lag, payload text, payload codewords, and
exact bits. The document metadata fixes 32 tokens per block, 1,024 tokens per
stream, lags 1/4/8/16, and a 16-block history. `question_answering` is false.

Rules:

- Every stream must contain exactly 32 blocks of exactly 32 pinned-tokenizer tokens.
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

Dataset admission uses paired per-document local-minus-privileged CE effects.
The dependency effect and dependency-minus-nondependency selectivity lower 95%
bootstrap bounds must both be positive over at least 32 documents. There is no
fixed nats threshold or effect ratio.

## Architecture Contract

Keep the core PDT structure:

- Frozen dense Qwen3 trunk selected from one revision-pinned 4B or 14B profile.
- Trainable phi sidecar.
- Per-stream adapters for stream identity and local specialization.
- Speculative Note Conditioning (SNC) cross-attention in selected decoder layers.
- Dynamic Notes Bus with `d_notes = 256`, block size `tau = 32`, and reveal delay `Delta = 1`.
- Fixed addressed history window: K prompt anchors followed by sixteen exact
  per-producer write versions ordered by age. For K=3 the canonical window is
  `(B, 51, 256)` and can expose lags 1 through 16 without version loss.
- Fixed-width structural extensions: 512-wide SNC attention, 512-wide planner
  and classifier bottlenecks, and per-stream bottleneck adapters inside twelve
  equal-depth decoder layers.
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
blocks, versioned 16-block windows, in-graph sibling writes, and one differentiable
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

- Functional teacher: the same selected frozen dense Qwen3 trunk with the
  complete serialized prefix and every sidecar context disabled. Apply
  forward token KL at temperature 2 on dependency spans while retaining hard
  CE on all target tokens.
- Response teacher: use only if needed to generate source-grounded long-form
  section continuations from encyclopedic/report prose. Do not introduce QA,
  short instruction-response targets, or hidden-state alignment to a different
  MoE topology.

After teacher-forced context KD passes, add student-prefix on-policy examples
scored by the functional teacher. Final ablations must still show that the bus
carries dependency information rather than merely reproducing teacher text.

## Loss Contract

Active loss family after hash removal:

```text
L_total =
    L_LM_CE
  + lambda_kd * L_KD_LM
  + beta_plan_commit * L_planner_vq_commit
  + beta_plan_codebook * L_planner_vq_codebook
  + beta_note_commit * L_dynamic_vq_commit
  + beta_note_codebook * L_dynamic_vq_codebook
  + lambda_plan_usage * L_planner_codebook_usage
  + lambda_note_usage * L_dynamic_codebook_usage
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
4. **Bus mutation.** Cycle one transmitted product-code index in one source stream's block-0 note and measure receiver block-1 logit changes. This guarantees a different valid 32-bit message; a pre-quantization float perturbation is not an admissible substitute because it can quantize back to the original tuple.
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
- Dependency selectivity difference: dependency delta minus nondependency delta.
- Bus mutation effect: KL or logit delta on annotated receiver spans.
- Logical note bandwidth: `K * 32 bits * blocks`, compared with text tokens
  and KV bytes. Each write is four indices into four 256-entry codebooks. The
  bus-side 256-dimensional decoded tensor is checkpoint-shared local state,
  not transmitted payload. The 18-bit synthetic source/channel ratio is
  `18/32 = 0.5625`.
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
and addressed bus mutation primitives exist. The teacher-forced trainer runs
the applicable paired interventions. Its independently initialized self-only
condition replaces every SNC reader at construction, uses only receiver-owned
prompt/block tails under `Delta=1`, emits condition-labeled telemetry, and is
compared by `scripts/compare_self_only.py`. Frozen-trunk blind and receiver-
explicit sequential full-information quality controls are implemented through
one scorer and one strict PDT comparison. Separately trained full-text,
full-KV, single-stream, and full-finetune runners do not yet exist.

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

- 32-document long-form dry run passes gradient, mutation, and paired gate-zero checks.
- A larger validation run shows positive lower confidence bounds overall and at each lag.
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
- The low-rank 4B and 14B profiles have exactly 156,484,647 and 305,374,247
  trainable parameters respectively. Memory is established by the optimizer
  probe rather than inferred from parameter count alone.

## Dry-Run Acceptance Criteria

On 32 examples:

- `test_gradient_flow.py` passes for SNC q/k/v/o, notes gates, speculation head, `plan_notes_proj`, and planner codebook when planner is active.
- Bus mutation produces measurable receiver logit change on dependency spans.
- Gate-zero dependency and dependency-minus-nondependency lower 95% confidence bounds are positive.
- Exact-source mutation KL has a positive lower 95% confidence bound at annotated future uses.
- Planner and dynamic codebooks report observable ceilings, effective entries, and exact collapse without serving as causal gates.
- No hash-era fields are present in runtime/training batches.

On 1k examples:

- Dependency and selectivity lower confidence bounds remain positive on held-out long-form documents and at lags 1/4/8/16.
- Source-swap moves receiver predictions toward swapped source state.
- Paired bus-minus-self-only dependency advantage has a positive lower confidence bound.
- Full-text bus performs at least as well as PDT on dependency spans, validating that the task truly benefits from communication.

## README State

After implementing any phase, update `README.md` with:

- Current phase status.
- Exact data generation commands.
- Exact training commands.
- Exact ablation commands.
- Known null/failure conditions.

The README is for future AI/LLM operators. It should be operational and specific, not promotional.
