# Parallel Decoder Transformer: Planner-Seeded Latent Coordination for Synchronized Parallel Generation

Logan Robbins  
Independent Researcher  
`ljrweb@gmail.com`

## Abstract

Autoregressive language models can often identify parallel subproblems, but standard decoding exposes only a single left-to-right output interface. External orchestration methods can launch multiple prompts concurrently, yet they provide no model-internal state through which those generations can synchronize, resolve ownership, or wait for missing information. We present the **Parallel Decoder Transformer (PDT)**, a frozen-trunk architecture that augments a decoder with a planner-seeded latent workspace and a synchronized multi-stream output protocol. Before any stream emits tokens, a mandatory prompt-time planner predicts fixed latent plan slots and projects them as snapshot 0 on an embeddings-only Dynamic Notes Bus. During decoding, each stream reads the visible notes window through Speculative Note Conditioning (SNC), emits provisional token blocks and latent summaries, and advances only when agreement logic determines that the current shared state is sufficient for continued parallel generation. Coverage heads track plan-item ownership, while rollback handles incoherent or premature commits. PDT therefore shifts parallel task decomposition from an external prompting strategy to a model-internal coordination mechanism over the output interface of a frozen language model.

# Introduction

Large language models frequently encounter prompts whose natural solution is not a single uninterrupted chain, but a set of partially independent sections, subquestions, or arguments. A model may internally recognize that decomposition, yet standard autoregressive decoding exposes only one causal output stream. The consequence is structural rather than merely computational: even when multiple sections could in principle be developed concurrently, the model must serialize them through a single textual channel.

External decomposition methods partially relieve that constraint by prompting for an outline and then spawning multiple generations in parallel [@ning2023skeleton; @PSLM2024]. Those methods can improve throughput and encourage modularity, but they do not create model-internal shared state. Once the work is split across separate calls, no stream can directly know whether a sibling stream has already established a key fact, claimed ownership of a section, or left a dependency unresolved. We refer to the resulting failure mode as **coherence drift**: parallel branches remain semantically related, but their local continuations can become redundant, contradictory, or prematurely specific.

This paper introduces the **Parallel Decoder Transformer (PDT)**, a decoder-only architecture that moves both decomposition and synchronization inside the model. PDT begins with a mandatory prompt-time latent planner that seeds a shared snapshot-0 workspace before any stream emits tokens. Parallel streams then decode against that workspace through **Speculative Note Conditioning (SNC)**, write provisional latent summaries at synchronized block boundaries, and use coverage plus agreement logic to decide whether the current multi-stream state may be committed and extended.

The key claim of PDT is not that parallel streams are always faster, but that a decoder can be given a mechanism for **synchronized parallel generation**: streams may emit concurrently, but they only commit and continue when shared latent state is sufficient for cross-stream consistency.

Under that interpretation, PDT is not just a decoding heuristic. It is a model-internal coordination protocol over the output interface of a frozen language model. The contribution is therefore not merely parallelism, but a way for a decoder to decompose a task into parallel streams, exchange latent state, and advance the parallel frontier only when the shared workspace supports safe continuation.

## Contributions

1. **Planner-seeded multi-stream generation protocol.** A mandatory prompt-time planner maps the prompt into fixed latent plan slots and initializes a shared snapshot-0 workspace before any stream emits tokens.
2. **Embeddings-only coordination bus for synchronized continuation.** Parallel streams read a lagged latent workspace during token emission, emit provisional summaries at block boundaries, and continue only when agreement deems the shared state sufficient for the next block.
3. **Ownership-aware commit control.** Coverage, ownership, and agreement jointly determine whether provisional content should be committed, withheld, or regenerated, enabling coordinated parallel generation without raw-text inter-stream exchange.
4. **Frozen-trunk realization.** The full coordination stack attaches to a frozen decoder through lightweight sidecar modules, preserving the base model while adding planner, bus, and synchronization behavior.

# Related Work

## Token-Level Acceleration Versus Coordination

Standard autoregressive decoding generates tokens sequentially as $p(x_t \mid x_{<t})$. Token-level acceleration methods such as Speculative Decoding [@leviathan2023fast], Blockwise Parallel Decoding [@Stern2018], Medusa [@cai2024medusa], EAGLE [@EAGLE2024], Lookahead Decoding [@Li2024lookahead], and related analyses of speculative trade-offs [@Yan2024; @survey2025] improve local efficiency by predicting or verifying multiple future tokens. More recent work also studies concurrent attention and interleaved planning mechanisms for faster generation [@Rodionov2025; @Xiao2025sprint].

These methods are important reference points, but they solve a different problem. They accelerate a single causal stream or speculate over future tokens within one stream. PDT is not primarily a faster decoding scheme; it is a model-internal synchronization mechanism for deciding when multiple partial generations may safely continue in parallel.

## External Prompt-Level Decomposition

Prompt-level approaches such as Skeleton-of-Thought (SoT) [@ning2023skeleton] and PSLM [@PSLM2024] ask a model for a decomposition and then launch multiple prompted generations. Their central advantage is semantic modularity: different calls can elaborate different parts of a solution. Their central limitation is where coordination lives. Shared state remains outside the model and is mediated by prompt text, API orchestration, or post-hoc merging. As a result, these systems can decompose work but cannot provide an internal continuation rule for when sibling streams have supplied enough information for safe further generation.

PDT is closest in motivation to this family of methods, but differs in mechanism. The decomposition prior, the shared workspace, and the continuation gate all live inside one decoder rather than in an external controller.

## Shared Workspaces, Blackboard Systems, and Latent Communication

A growing literature coordinates multiple model calls or agents through shared workspaces, graph structures, or latent communication channels. GPTSwarm [@Zhuge2024GPTSwarm], Mixture-of-Agents [@Wang2025MoA], blackboard-style systems [@Han2025Blackboard; @Salemi2025Blackboard], and topology-adaptive agent systems such as AMAS [@Leong2025AMAS] all point toward coordination as a systems problem rather than a single-prompt problem. Earlier work on shared neural workspaces [@Goyal2021] provides a conceptual precedent for a global coordination substrate, while LatentMAS [@Zou2025LatentMAS] demonstrates the value of latent rather than text-mediated communication.

PDT shares that intuition but places the workspace inside a single frozen decoder and binds it directly to the output interface. Its streams are not separate prompted agents. They are coordinated generation branches of one model, each with its own local cache but a common latent synchronization workspace.

## Parameter-Efficient Adaptation and Distillation

Because modifying all parameters of a large decoder is computationally expensive, PDT follows the parameter-efficient adaptation tradition of LoRA [@hu2021lora] and adapters [@houlsby2019parameter]. Distillation and learning-with-privileged-information frameworks [@LopezPaz2015; @Vapnik2015] further motivate training auxiliary coordination modules with teacher signals that may be unavailable or undesirable at inference time.

These techniques matter here not only for efficiency, but because they make PDT a realistic architectural extension of a frozen model. The planner, synchronization workspace, and commit-control heads can be added without rewriting the underlying language model.

# Architecture

The Parallel Decoder Transformer turns one frozen decoder into $K$ coordinated output streams. Its central design decision is that decomposition and cross-stream state are **model-internal**. The prompt is planned once, the resulting latent plan initializes a shared workspace, and each stream decodes against that workspace rather than against a separate prompt string.

## Frozen Trunk Topology

We initialize PDT from a pre-trained decoder-only backbone parameterized by $\theta_{\text{pre}}$ and freeze all weights in $\theta_{\text{pre}}$. Trainable parameters $\phi$ are introduced as a lightweight coordination stack. The full parameter set is
$$
\Theta = \theta_{\text{pre}} \cup \phi.
$$

The trainable subset $\phi$ contains four components:

- **Stream adapters**, inserted into selected transformer blocks to provide stream-specific conditioning.
- **SNC backends**, implemented as cross-attention layers that read from the shared latent workspace.
- **Planner and notes modules**, including a fixed-slot planner head, a shared plan-embedding matrix, and a projection into the notes space used by the bus.
- **Auxiliary control heads**, including note-emission, coverage, agreement, and stream-classification heads.

All streams share the same frozen trunk parameters, but each stream maintains its own KV cache, adapter state, and decode position. The base language model therefore remains canonical, while coordination is expressed through sidecar modules rather than through changes to pre-trained weights.

## Prompt-Time Latent Planner

Let $\mathbf{H}_x \in \mathbb{R}^{T \times d}$ denote the hidden states of the prompt under the frozen trunk. PDT performs a mandatory planning pass before any output tokens are generated. The planner predicts logits over $S$ latent plan slots and a shared plan vocabulary of size $V_p$:
$$
\mathbf{\Pi} = \mathrm{PlannerHead}(\mathbf{H}_x) \in \mathbb{R}^{S \times V_p}, \qquad
z_i = \arg\max_{a} \mathbf{\Pi}_{i,a}.
$$
In the current implementation, $S = 16$ and $V_p = 65{,}536$.

The latent slot identifiers $z_{1:S}$ are not language-model tokens. They index a shared latent plan vocabulary that is also used by the plan embedding matrix, the canonical plan catalog, and coverage targets. Active slots are re-embedded and projected into notes space:
$$
\mathbf{e}_i = E_{\text{plan}}[z_i], \qquad
\mathbf{n}^{\text{plan}}_0 =
\mathrm{Norm}\!\left(
\frac{1}{|M|}\sum_{i \in M} W_{\text{plan}} \mathbf{e}_i
\right),
$$
where $M$ denotes the active planner slots.

The resulting vector $\mathbf{n}^{\text{plan}}_0$ is broadcast as **snapshot 0** on the Dynamic Notes Bus before any stream emits tokens. The planner does not merely assign semantic subtopics. It initializes the shared coordination state against which subsequent continuation decisions are made. Snapshot 0 therefore serves both as a decomposition prior and as the first synchronization contract among streams.

In PDT, parallel generation does not begin from independent empty states. It begins from a common latent commitment structure.

## Dynamic Notes Bus as Synchronization Workspace

The **Dynamic Notes Bus** is PDT's shared latent workspace. It is an embeddings-only, versioned store of planner and stream summaries. Textual plans and rendered notes may be used for dataset construction, observability, or supervision, but inference-time coordination uses only note embeddings.

We describe the runtime at two timescales:

- $\tau$: the number of tokens in each provisional block emitted per stream between synchronization decisions,
- $\Delta$: the reveal delay between a note being emitted and becoming visible to sibling streams,
- $H$: the rollback or commit horizon over which recently generated provisional content may be withheld or regenerated.

Let $\widehat{\mathbf{n}}^{(j)}_u$ denote the provisional note emitted by stream $j$ at synchronization round $u$, and let $B$ be the workspace budget. The visible notes window for stream $k$ at round $v$ is
$$
\mathcal{W}^{(k)}_v =
\mathrm{Window}\!\left(
\left\{\mathbf{n}^{\text{plan}}_0\right\}
\cup
\left\{\widehat{\mathbf{n}}^{(j)}_u : j \in [K],\; u \le v-\Delta\right\},
B
\right).
$$

The Dynamic Notes Bus is therefore not just an auxiliary memory. It is the synchronization workspace that determines whether provisional multi-stream generation may be committed and extended.

## Synchronized Block Emission Protocol

PDT runs in synchronized rounds rather than as unconstrained free-running streams. At synchronization round $v$:

1. The planner-seeded bus exposes the visible notes window $\mathcal{W}^{(k)}_v$ to each active stream $k$.
2. Each stream emits a provisional block of $\tau$ tokens conditioned on its local cache and the visible workspace:
   $$
   \widehat{\mathbf{y}}^{(k)}_v =
   \left(\hat y^{(k)}_{v,1}, \ldots, \hat y^{(k)}_{v,\tau}\right).
   $$
3. At the end of the block, stream $k$ emits a provisional latent note $\widehat{\mathbf{n}}^{(k)}_v$ summarizing what it has established, which plan items it believes it owns, and which dependencies remain unresolved.
4. Coverage and agreement heads evaluate whether the provisional block is consistent with the shared plan and whether sibling state is sufficient for further continuation.
5. If agreement passes, the block is committed and the new note snapshot becomes visible after delay $\Delta$.
6. If agreement fails, the system withholds commit, stalls a subset of streams, or rolls back and regenerates within horizon $H$ using fresher shared context.

This protocol is the core mechanism of PDT. Streams are allowed to generate in parallel only up to a provisional block boundary; continuation beyond that boundary is gated by agreement over the shared latent workspace.

## Speculative Note Conditioning Cross-Attention

SNC is the read mechanism over the visible workspace. During token emission inside round $v$, stream $k$ conditions on $\mathcal{W}^{(k)}_v$ at every decode step. Let $\mathbf{H}^{(k)}_{v,u}$ denote the hidden state for token position $u \in \{1,\ldots,\tau\}$ in round $v$. The SNC projections are
$$
\mathbf{Q}^{(k)}_{v,u} = \mathbf{H}^{(k)}_{v,u} \mathbf{W}_Q, \qquad
\mathbf{K}^{(k)}_v = \mathcal{W}^{(k)}_v \mathbf{W}_K, \qquad
\mathbf{V}^{(k)}_v = \mathcal{W}^{(k)}_v \mathbf{W}_V,
$$
with learned parameters $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \phi$.

The resulting SNC context vector $\mathbf{C}^{(k)}_{v,u}$ is injected through a trust-gated residual:
$$
\widetilde{\mathbf{H}}^{(k)}_{v,u} =
\mathbf{H}^{(k)}_{v,u}
+
\lambda_l \cdot \mathbf{C}^{(k)}_{v,u} \mathbf{W}_O,
$$
where $\lambda_l \in [0,1]$ is a learned residual gate for the instrumented layer. The gate preserves the magnitude statistics of the frozen trunk early in training and allows coordination to become stronger only when the auxiliary path is reliable.

SNC provides continuous low-bandwidth conditioning during token emission, but synchronization decisions are made only at block boundaries. PDT therefore separates two timescales: **token-time latent conditioning** and **block-time continuation control**.

## Provisional Writes, Ownership, and Commit Readiness

At the end of each provisional block, stream $k$ writes a latent summary back to the bus:
$$
\widehat{\mathbf{n}}^{(k)}_v =
\mathrm{SpeculationHead}\!\left(
\widetilde{\mathbf{H}}^{(k)}_{v,\tau}
\right).
$$
That note is intended to summarize the stream's newly established content, its unresolved dependencies, and its current ownership claims.

Coverage is predicted against the planner-seeded plan catalog:
$$
\mathbf{c}^{(k)}_v =
\mathrm{CoverageHead}\!\left(
\widetilde{\mathbf{H}}^{(k)}_{v,\tau},
E_{\text{plan}}(z_{1:S}),
m_{\text{plan}}
\right),
$$
where $m_{\text{plan}}$ masks inactive or padded plan items.

Because coverage targets are stream-qualified canonical plan items, they track more than semantic relevance. They also track ownership and non-overlap. A provisional block is not ready to commit merely because it is fluent; it must also preserve ownership consistency with respect to the planner-seeded plan catalog.

## Agreement-Gated Commit and Rollback Control

Agreement in PDT is not only a rollback trigger after a bad block. It is the permission signal for the next parallel block. For each active stream $k$ at round $v$, PDT predicts a **readiness score**
$$
r^{(k)}_v =
\mathrm{AgreeHead}\!\left(
\widetilde{\mathbf{H}}^{(k)}_{v,\tau},
\mathcal{W}^{(k)}_v,
\mathbf{c}^{(k)}_v,
\widehat{\mathbf{n}}^{(k)}_v
\right),
$$
which estimates whether stream $k$ has enough synchronized information about sibling streams to safely commit its current block and continue.

The base continuation rule is a global gate over active streams:
$$
A_v =
\mathbf{1}\!\left[
\min_{k \in \mathcal{A}_v} r^{(k)}_v > \gamma
\right],
$$
where $\mathcal{A}_v$ is the set of active streams at round $v$ and $\gamma$ is a learned or tuned threshold.

If $A_v = 1$, PDT commits the provisional blocks $\widehat{\mathbf{y}}^{(k)}_v$ and schedules the new notes $\widehat{\mathbf{n}}^{(k)}_v$ to become visible after delay $\Delta$. If $A_v = 0$, PDT may roll back only the streams whose readiness falls below threshold and let stable streams keep their current commit point. In that selective policy, the global gate acts as a synchronization checkpoint while rollback is applied per stream.

In PDT, agreement is the gate that decides whether the system may advance the parallel frontier.

A richer alternative would replace the scalar gate with pairwise or graph-structured compatibility scores between streams. We view that extension as important future work, but the scalar readiness formulation plus selective rollback is sufficient to define the current synchronized continuation protocol.

## Inference-Time Coordination Loop

The full inference procedure is:

1. Encode the prompt with the frozen trunk.
2. Run the planner to predict latent plan slots $z_{1:S}$.
3. Project the latent plan into notes space and publish snapshot 0 on the Dynamic Notes Bus.
4. For synchronization rounds $v = 1, 2, \ldots$ until all streams terminate:
   1. Expose the visible notes window $\mathcal{W}^{(k)}_v$ to each active stream.
   2. Let each active stream emit $\tau$ provisional tokens using its local cache plus SNC conditioning.
   3. Let each active stream emit a provisional latent note $\widehat{\mathbf{n}}^{(k)}_v$.
   4. Compute coverage $\mathbf{c}^{(k)}_v$ and readiness $r^{(k)}_v$ for each active stream.
   5. If $A_v = 1$, commit the current block and schedule the new notes for publication after delay $\Delta$.
   6. Otherwise, stall, selectively withhold, or roll back within horizon $H$, then regenerate with updated visible context.
5. Merge committed outputs according to planner ownership, section order, and stream-completion state.

The inference loop makes the architectural claim explicit: PDT is a decode $\rightarrow$ summarize $\rightarrow$ agree $\rightarrow$ commit $\rightarrow$ continue protocol.

## Inference Serving Contract (Stride-Commit Default)

PDT can expose outputs in two serving modes:

1. **Live stream mode.** Emit per-stream text as soon as tokens are sampled.
2. **Stride-commit mode (default).** Buffer per-stream provisional tokens within each stride and release them only after stride-level agreement/rollback resolution.

The default in our implementation is stride-commit mode because it best matches the synchronization semantics above. A practical API returns both (i) structured per-stream commit artifacts and (ii) a merged user-facing answer.

```
Prompt
  └─> planner seed (snapshot 0)
        └─> per-stream provisional decode (stride v)
              └─> latent note summaries + readiness
                    └─> commit/rollback decision
                          ├─ commit: release committed_text_block(stream, stride)
                          └─ rollback: regenerate failing streams
                                   ↓
                           final merged answer
```

Coherence in PDT comes from latent communication and synchronized commit decisions, not from direct raw-token sharing across streams.

## Near-Term Use Case: Parallelized Knowledge-Structured Responses

The strongest near-term product setting is prompts with explicit topical substructure (e.g., historical overviews, multi-facet knowledge synthesis, and sectioned explanatory answers). In these cases, planner-seeded ownership priors, latent note exchange, and stride-commit synchronization naturally map onto section-oriented generation and reduce cross-stream contradiction compared to unconstrained live token streaming.

## Parameter-Efficient Curriculum

Training a coordination mechanism on a frozen trunk is unstable if all modules are enabled at once. PDT therefore uses a staged curriculum:

- **Stage 0 (planner pretraining).** Train the planner head, plan embedding pathway, and initial notes projection against latent planner targets and the snapshot-0 plan contract.
- **Stage 1 (stream bootstrap).** Unfreeze stream adapters and SNC cross-attention so streams learn stream-specific conditioning against fixed teacher workspaces.
- **Stage 2 (bus enablement).** Train note-emission modules so streams begin writing learned latent summaries into the versioned bus while the planner seed remains stable.
- **Stage 3 (commit control).** Train coverage and agreement heads so ownership consistency and continuation sufficiency become part of the generation policy.

The accompanying dataset and supervision contract reflect the same runtime assumptions. Snapshot 0 is explicitly materialized from the planner contract, canonical plan items are hashed into the shared latent vocabulary, and language-model supervision is masked so each stream is trained only on its assigned section.

# Conclusion

PDT reframes parallel generation as a synchronization problem inside the decoder. Its central contribution is not simply to decompose tasks, but to let a model emit multiple provisional streams, exchange latent summaries, and decide when those streams have enough shared state to continue in parallel without collapsing back to text-mediated orchestration.

The architecture's key pieces serve one unified protocol. A mandatory prompt-time planner initializes a shared latent commitment structure; SNC provides continuous token-time access to the visible workspace; coverage tracks ownership and overlap with respect to the planner-seeded catalog; and agreement gates block-level commit, withholding, rollback, and continuation. Together they define a planner-seeded, model-internal coordination mechanism over the output interface of a frozen language model.

PDT therefore shifts the question from “How can multiple prompts be run at once?” to “How can one decoder maintain synchronized multi-stream state while generating?” That shift is the conceptual center of the method and the primary claim of this paper.

## Future Directions

1. **Agreement as continuation sufficiency.** The most immediate evaluation target is whether readiness scores actually predict safe continuation, not merely whether they detect bad commits after the fact.
2. **Dependency-aware synchronization.** Scalar readiness can be generalized to pairwise or graph-structured compatibility so streams gate continuation only on the siblings they depend on.
3. **Adaptive block size $\tau$.** Some streams may be able to advance farther than others when their dependencies are weak; adaptive block sizing would relax the current fixed-round protocol.
4. **Ownership-aware merge policies.** Final composition can itself become a learned function of planner slots, coverage state, and stream completion rather than a fixed section-order merge.
5. **Scaling planner capacity.** Richer ownership semantics, adaptive slot counts, and larger numbers of streams may allow the synchronization workspace to represent more complex task structures.

# Appendix A: Training Objectives

Let $\Pi^S$ denote the student planner logits and $y_{\text{plan}}$ the latent planner-slot targets. The planner objective is cross-entropy over latent slots:
$$
\mathcal{L}_{\text{plan}} =
\mathrm{CE}\!\left(\Pi^S, y_{\text{plan}}\right),
$$
with padded slots ignored by the label mask.

Let $\widehat{\mathbf{N}}^{\text{notes}}$ be the notes-head prediction, $\widehat{\mathbf{N}}^{\text{spec}}$ the speculative note prediction, and $\mathbf{N}^T$ the teacher notes tensor. The note-alignment losses are
$$
\mathcal{L}_{\text{notes}} =
\mathrm{MSE}\!\left(\widehat{\mathbf{N}}^{\text{notes}}, \mathbf{N}^T\right),
\qquad
\mathcal{L}_{\text{spec}} =
\mathrm{MSE}\!\left(\widehat{\mathbf{N}}^{\text{spec}}, \mathbf{N}^T\right).
$$

Once token-level supervision is active, the language-model losses are
$$
\mathcal{L}_{\text{LM-CE}} =
\mathrm{CE}\!\left(\texttt{lm\_logits}, \texttt{labels}\right)_{m_{\text{labels}}},
\qquad
\mathcal{L}_{\text{KD-LM}} =
\mathrm{KL}\!\left(p^S_{\text{LM}} \,\Vert\, p^T_{\text{LM}}\right)_{m_{\text{LM}}}.
$$

Coverage supervision remains binary cross-entropy with logits over canonical plan items:
$$
\mathcal{L}_{\text{cov}} =
\mathrm{BCEWithLogits}\!\left(\mathbf{c}, y_{\text{cov}}\right)_{m_{\text{cov}}},
$$
where $\mathbf{c}$ are coverage logits, $y_{\text{cov}}$ are binary ownership-aware coverage targets, and $m_{\text{cov}}$ is the coverage mask.

Agreement supervision is reframed as **commit-readiness supervision**:
$$
\mathcal{L}_{\text{ready}} =
\mathrm{BCE}\!\left(\mathbf{r}, y_{\text{ready}}\right)_{m_{\text{ready}}},
$$
where $\mathbf{r}$ are predicted readiness scores and $y_{\text{ready}}$ indicates whether a provisional block had sufficient sibling information for safe commitment and continuation. When rollback annotations are available, premature or incoherent blocks are labeled as not ready rather than merely as post-hoc errors.

The total training loss is
$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{plan}}
+ \mathcal{L}_{\text{notes}}
+ 0.5\,\mathcal{L}_{\text{spec}}
+ \mathcal{L}_{\text{LM-CE}}
+ \lambda_{\text{KD}}\,\mathcal{L}_{\text{KD-LM}}
+ \lambda_{\text{cov}}\,\mathcal{L}_{\text{cov}}
+ \lambda_{\text{ready}}\,\mathcal{L}_{\text{ready}}.
$$

This objective matches the revised mechanism: the planner seeds the initial contract, note losses supervise the shared workspace, coverage supervises ownership consistency, and readiness supervises whether provisional parallel generation may safely continue.
