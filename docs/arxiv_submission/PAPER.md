# Parallel Decoder Transformer: Planner-First Latent Coordination for Model-Internal Parallel Decoding

Logan Robbins
Independent Researcher
`ljrweb@gmail.com`

## Abstract
Autoregressive decoding in large language models remains sequential even when a task naturally decomposes into parallel sections. External decomposition methods can launch multiple prompts at once, but they do not create a shared model-internal state that keeps those sections non-overlapping and coherent. We present the **Parallel Decoder Transformer (PDT)**, a frozen-trunk architecture centered on **planner-first latent coordination**. Before any stream emits tokens, a mandatory prompt-time planner pools prompt hidden states, predicts a fixed set of latent plan IDs over a shared plan vocabulary, and uses shared plan embeddings plus a projection layer to seed snapshot 0 on an embeddings-only Dynamic Notes Bus. During decoding, streams read the visible lagged notes window at every step, while SNC, note-emission, coverage, and agreement modules consume and update that workspace through cadence and rollback logic. This paper therefore focuses on the coordination stack and implementation contract itself: fixed latent planner slots, a shared latent vocabulary, snapshot 0 derived from the plan contract, and an embeddings-only bus that lets parallel streams coordinate without exchanging raw text.

# Introduction

A foundational bottleneck in large language model (LLM) inference is the sequential, left-to-right nature of autoregressive decoding. While causal masking is fundamental to this paradigm, it forces a trade-off between latency and reasoning depth: complex tasks require long chains of thought, but long chains incur prohibitive latency.

Recent research has attempted to bypass this bottleneck through ``around-the-model'' parallelization. Techniques like Skeleton-of-Thought (SoT) [@ning2023skeleton] decompose tasks into sub-questions and prompt the model to answer them in parallel. While promising, these methods treat the model as a black box, leading to a critical failure mode we term **Coherence Drift**: because parallel streams cannot communicate, they often hallucinate conflicting facts or redundant content.

We address this by moving both planning and coordination *inside* the model. We introduce the **Parallel Decoder Transformer (PDT)**, a decoder-only architecture whose central mechanism is a mandatory prompt-time latent planning pass followed by a shared embeddings-only coordination bus. The planner seeds a shared latent state at $t=0$, and downstream streams condition on that state through SNC rather than waiting for a textual plan string to be written into the prompt or passed between streams. The resulting coordination policy is implemented by a stack rather than a single head: the planner provides the initial latent code, while plan projection, note emission, SNC consumption, and rollback heads maintain the shared state during decoding.

## The Efficiency Imperative
Prior attempts at architectural modification often require expensive full-model fine-tuning, which is computationally prohibitive for modern 70B+ parameter models. A key design constraint of PDT is therefore that the base language model's weights remain frozen while coordination is added through sidecar modules.

We present a **Parameter-Efficient** design where the massive transformer ``trunk'' remains frozen and coordination is delegated to lightweight *Stream Adapters*, planner modules, and SNC heads. This approach allows PDT to be deployed as a ``sidecar'' to existing open-weights models without changing the core causal decoder.

## Our Contributions
Our contributions are as follows:
- **Planner-first latent coordination stack.** PDT performs a prompt-time planner pass that predicts fixed latent plan slots over a shared plan vocabulary, re-embeds those slots into notes space, and broadcasts the resulting seed as snapshot 0 before any stream emits tokens; later modules then consume, refresh, and police that workspace during decoding.
- **Versioned embedding bus semantics.** SNC reads lagged note embeddings every decode step, while cadence and agreement logic control when new snapshots are written or when rollback is triggered. Textual plans and notes remain auxiliary views for data construction, scoring, and debugging rather than the representation carried across streams.
- **Implementation-aligned dataset and training contract.** The dataset writes a real snapshot-0 plan seed, hashes canonical plan items into the planner vocabulary, and can mask LM supervision so each stream trains only on its own section.

Because the coordination stack has been materially revised to match this planner-first latent contract, previously reported quantitative results are omitted from this version of the paper. Section \ref{sec:experiments} therefore focuses on the evaluation scope and the metrics that matter for the revised system, including the distinction between stack-level mechanism claims and head-level attribution claims.

The rest of the paper is organized as follows. Section \ref{sec:related} reviews related work in parallel decoding, shared workspaces, and latent coordination. Section \ref{sec:architecture} details the frozen-trunk architecture and planner-first curriculum. Section \ref{sec:snc} gives an implementation-grounded formalism for the latent planner and Dynamic Notes Bus runtime. Section \ref{sec:experiments} defines the current evaluation scope for the revised coordination stack.

# Related Work

## Parallel Decoding and External Decomposition
Standard autoregressive decoding generates tokens sequentially, $p(x_t | x_{<t})$. Parallelization efforts typically fall into two categories: *token-level* speculation and *prompt-level* decomposition.

Token-level methods like Speculative Decoding [@leviathan2023fast], Blockwise Parallel Decoding [@Stern2018], Medusa [@cai2024medusa], EAGLE [@EAGLE2024], and Lookahead Decoding [@Li2024lookahead] use auxiliary heads or draft models to predict multiple future tokens, which are then verified by the main model. Yan et al. [@Yan2024] provide a systematic analysis of the trade-offs in speculative decoding approaches. Recent work has also explored concurrent attention mechanisms [@Rodionov2025] and interleaved planning [@Xiao2025sprint] for parallel generation. While efficient, these methods are limited to local syntactic acceleration and cannot plan globally. Recent surveys [@survey2025] provide comprehensive overviews of these token-level approaches.

Prompt-level methods like Skeleton-of-Thought (SoT) [@ning2023skeleton] and PSLM [@PSLM2024] use external orchestration to query the model for an outline, then trigger parallel API calls for each section. While SoT achieves high semantic parallelism, it suffers from *coherence drift* as parallel calls lack shared memory.

PDT is closest in motivation to prompt-level decomposition, but differs in where coordination lives: the planner and shared state are internal model modules rather than external controllers passing text among independent calls.

## Shared Workspaces and Multi-Agent Coordination
Several recent systems improve reasoning by coordinating multiple agents or model calls through explicit communication structures. GPTSwarm [@Zhuge2024GPTSwarm] optimizes agent graphs, Mixture-of-Agents [@Wang2025MoA] aggregates layered agent outputs, and blackboard-style systems [@Han2025Blackboard; @Salemi2025Blackboard] revive a shared workspace for LLM coordination. AMAS [@Leong2025AMAS] further argues that communication topology itself should adapt to the input. Earlier work on shared neural workspaces [@Goyal2021] provides a conceptual precedent for a global coordination substrate.

PDT shares the workspace intuition but moves the mechanism inside a single decoder. Its streams are not separate prompted agents; they are coordinated passes of one frozen trunk coupled by a versioned embeddings bus seeded by a prompt-time planner.

## Latent Communication
Recent multi-agent work has also begun replacing natural-language exchange with continuous internal states. LatentMAS [@Zou2025LatentMAS] shows that a shared latent working memory can improve efficiency and reasoning quality relative to text-mediated collaboration. PDT follows the same broad shift away from text transport, but with a different objective: fixed planner slots and an embeddings-only Dynamic Notes Bus for coordinating parallel sections inside one frozen language model.

## Parameter-Efficient Coordination
Updating all parameters of Large Language Models is computationally prohibitive. Techniques like LoRA [@hu2021lora] and Adapters [@houlsby2019parameter] inject trainable rank-decomposition matrices or bottleneck layers into frozen transformers. Knowledge distillation approaches [@LopezPaz2015] and learning with privileged information [@Vapnik2015] provide theoretical foundations for training student models with access to teacher signals unavailable at inference.

Our work uses PEFT not only for task adaptation but to attach a planner, bus reader, and coordination heads to a frozen backbone while preserving one canonical coordination path.

# Architecture and Curriculum

The Parallel Decoder Transformer (PDT) is a decoder-only architecture that turns one frozen backbone into multiple coordinated generation streams. The key design decision is that decomposition and cross-stream state are model-internal: the prompt is planned once, the resulting latent plan seeds a shared bus, and streams decode against that bus rather than against separate prompt strings. A second constraint is **Parameter Efficiency**: we assume the underlying Large Language Model (the ``trunk'') is too large to fine-tune.

## The Frozen Trunk Topology
We initialize PDT with a pre-trained GPT-OSS backbone parameterized by $\theta_{\text{pre}}$. We freeze all weights in $\theta_{\text{pre}}$. To enable task-specific behavior, we inject a lightweight set of trainable parameters $\phi$.
The total parameters $\Theta = \theta_{\text{pre}} \cup \phi$. The trainable set $\phi$ consists of:
- **Stream Adapters:** Bottleneck MLPs inserted into selected transformer blocks to inject stream-specific conditioning.
- **SNC Backends:** Cross-attention layers (Section \ref{sec:snc}) that read from the shared Dynamic Notes Bus.
- **Planner and Notes Modules:** A fixed-slot `planner_head`, a shared `plan_embedding`, and a projection layer that maps latent plan IDs into the notes space used by the bus.
- **Auxiliary Heads:** Lightweight heads for note prediction, speculative note emission, coverage prediction, agreement scoring, and stream classification.

## Prompt-Time Latent Planner

PDT uses a mandatory planner pass before token decoding begins. The frozen trunk first encodes the prompt into hidden states $\mathbf{H}_x \in \mathbb{R}^{T \times d}$. The planner then pools those prompt states and predicts a fixed set of latent plan slots:
$$
    \mathbf{P} = \mathrm{PlannerHead}(\mathbf{H}_x) \in \mathbb{R}^{S \times V_p}, \qquad
    \hat{z}_i = \arg\max_{v} \mathbf{P}_{i,v},
$$
where the current implementation uses $S=16$ planner slots and a shared latent plan vocabulary of size $V_p = 65{,}536$.

These slot IDs are *not* LM tokens. They live in the same latent vocabulary that also indexes `plan_embedding`, canonical plan catalog hashes, and coverage targets. Each active slot is re-embedded and projected into notes space:
$$
    \mathbf{e}_i = E_{\text{plan}}[\hat{z}_i], \qquad
    \mathbf{n}^{\text{plan}}_0 = \mathrm{Pool}\!\left(W_{\text{plan}} \mathbf{e}_{1:S}\right).
$$
The pooled plan seed $\mathbf{n}^{\text{plan}}_0$ is broadcast to every stream as snapshot 0 on the Dynamic Notes Bus. In the canonical trained path described in this paper, that seed is produced by the model planner itself. The repository also contains contract-injection and text-seeding hooks for diagnostics and ablations, but they are not part of the core mechanism claim.

The planner head therefore solves the initial latent code-selection problem, not the entire coordination problem by itself. In the runtime path, `plan_embedding` and `plan_notes_proj` convert those latent IDs into the shared snapshot-0 seed, SNC consumes the visible workspace on each decode step, `notes_head` writes refreshed summaries back to the bus, and coverage/agreement heads regulate overlap and rollback. We intentionally avoid the stronger claim that the planner head alone internalizes the full coordination policy.

## Parallel Streams and Dynamic Notes Bus

Instead of a single causal stream, PDT maintains $K$ parallel streams. All streams share the same frozen trunk parameters $\theta_{\text{pre}}$ but maintain distinct KV-caches, adapter states, and decode positions. Coordination is mediated by the **Dynamic Notes Bus**, a versioned snapshot store populated with fixed-dimensional note embeddings.

The bus carries embeddings only. Textual plans, note renderings, and coverage strings are auxiliary artifacts used for dataset construction, observability, and scoring. They are not the representation consumed by sibling streams during decoding. At decode step $t$, every stream reads the currently visible lagged notes window; cadence and agreement logic determine whether a new summary is written back to the bus or whether a rollback event is triggered. The runtime uses an all-to-all broadcast topology over snapshot windows, so a stream can condition on the latest visible summaries from its siblings without exchanging raw text.

Snapshot 0 is part of the data and training contract rather than an inference-only convenience. The dataset materializes `versioned_notes[0]` from `plan_contract`, and training preserves that initial snapshot through the first stride so the planner seed remains stable during early decode steps.
Because canonical plan items are stream-qualified, snapshot 0 also encodes an initial ownership structure. Later coverage and rollback heads therefore judge not just whether content was produced, but whether the shared latent plan is being executed by the intended stream without unnecessary overlap.

## Parameter-Efficient Curriculum

Training a parallel coordination mechanism on a frozen trunk is unstable if attempted end-to-end. We therefore use a staged curriculum that progressively enables the planner, bus, and rollback machinery.
- **Stage 0 (Planner Pretrain):** We train `planner_head`, `notes_head`, and `plan_notes_proj` while the trunk remains frozen. Supervision comes from latent `planner_ids` and the snapshot-0 plan contract, not from natural-language planner strings. This stage teaches the initial latent code and its projection into notes space, but not the entire downstream coordination circuit.
- **Stage 1 (Stream Bootstrap):** We unfreeze stream adapters and SNC cross-attention so streams learn stream-specific conditioning against fixed dataset snapshots while teacher notes remain the supervision target.
- **Stage 2 (Notes Bus Enable):** We unfreeze `speculation_head`. Streams begin emitting learned note embeddings into the versioned bus while the planner seed remains fixed at snapshot 0.
- **Stage 3 (Rollback Training):** We unfreeze `coverage_head` and `agreement_head`. Coverage is predicted against canonical plan catalog items, and agreement supervision controls rollback behavior over the commit horizon.
- **Stage 4 (Stability Supervision):** The configuration also contains an optional trunk-unfreezing stage with additional stability supervision, but the current paper does not report quantitative results for that path.

This curriculum keeps the planner-first path canonical throughout training: latent planning seeds the bus first, stream adapters and SNC consume that seed, note-emission heads refresh the workspace, and later stages add item-level coverage and rollback control without changing the representational contract. The learned behavior is therefore distributed across the stack even though the planner initializes the shared latent state.

# Planner-Seeded Latent Coordination

To enable parallel coordination without modifying the pre-trained weights of the base language model (the ``trunk''), we use a planner-seeded latent workspace together with **Speculative Note Conditioning (SNC)**. In the current implementation, SNC is best understood as the read/write mechanism over that shared embedding-space state rather than as a textual note-passing protocol.

Formally, let the frozen trunk be parameterized by $\theta_{\text{pre}}$. We freeze these parameters to preserve the general reasoning capabilities of the base model. We introduce a lightweight set of trainable parameters $\phi$, consisting of Stream Adapters, Note Heads, and the SNC Cross-Attention mechanism.

## Planner-Seeded Note State
Let $\mathbf{H}_x \in \mathbb{R}^{T \times d}$ denote the prompt hidden states produced by the frozen trunk. The planner predicts logits over $S$ latent plan slots and a shared plan vocabulary of size $V_p$:
$$
    \mathbf{\Pi} = \mathrm{PlannerHead}(\mathbf{H}_x) \in \mathbb{R}^{S \times V_p}, \qquad
    z_i = \arg\max_v \mathbf{\Pi}_{i,v}.
$$
In the reported configuration, $S=16$ and $V_p=65{,}536$. If an external `plan_contract` is supplied, the same canonical plan catalog is hashed into this vocabulary and padded to the same $S$ slots. The planner therefore always exposes a fixed-width latent state.

The latent slot IDs are re-embedded and projected into notes space,
$$
    \mathbf{e}_i = E_{\text{plan}}[z_i], \qquad
    \mathbf{n}^{\text{plan}}_0 = \mathrm{Norm}\!\left(\frac{1}{|M|} \sum_{i \in M} W_{\text{plan}} \mathbf{e}_i\right),
$$
where $M$ denotes the active planner slots. The pooled seed $\mathbf{n}^{\text{plan}}_0$ is then broadcast as snapshot 0 on the Dynamic Notes Bus for every stream before decoding begins.
This fixed latent seed couples task decomposition, initial stream ownership, and later coverage supervision before any stream writes free-running content.

## Frozen Trunk and Stream Adapters
Let $\mathbf{H}^{(k)}_l \in \mathbb{R}^{T \times d}$ denote the hidden states of stream $k$ at layer $l$. In a standard Transformer, $\mathbf{H}^{(k)}_{l+1} = \text{Block}_{\theta}(\mathbf{H}^{(k)}_l)$. In PDT, we inject stream-specific conditioning via **Stream Adapters**.

The adapter for stream $k$, denoted as $A_{\phi}^{(k)}$, is a bottleneck MLP with residual connection:
$$
    \mathbf{H}^{(k)}_{l, \text{adapt}} = \mathbf{H}^{(k)}_l + \mathbf{W}_{\text{up}} \cdot \sigma(\mathbf{W}_{\text{down}} \cdot \text{LayerNorm}(\mathbf{H}^{(k)}_l))
$$
where $\sigma$ is the activation function (GELU). Crucially, these adapters allow the frozen trunk to process identical tokens differently depending on the stream index, breaking the symmetry of parallel decoding. They therefore let multiple streams condition on the same planner seed while still specializing into different sections of the output.

## SNC Cross-Attention Mechanism

The core coordination primitive is the Speculative Note Conditioning layer. Unlike standard decoding where context is strictly local (autoregressive), SNC allows stream $k$ to attend to a lagged window of note embeddings assembled from versioned snapshots across sibling streams.

Let $\mathcal{W}^{(k)}_t \in \mathbb{R}^{M_t \times d_{\text{note}}}$ denote the notes window visible to stream $k$ at decode step $t$. The window is built from the most recent lagged snapshots under reveal delay $\Delta$ and a fixed snapshot budget. Reads happen at every decode step; cadence affects when new snapshots are written, not whether the window is consulted. In other words, every generated token is conditioned on a planner-seeded latent workspace even when no new write occurs at that step.

### Query-Key-Value Construction.
The query $\mathbf{Q}^{(k)}$ is derived from the current stream's hidden state. The keys $\mathbf{K}^{(j)}$ and values $\mathbf{V}^{(j)}$ are projected from the sibling notes:
$$
    \mathbf{Q}^{(k)} &= \mathbf{H}^{(k)}_{l} \mathbf{W}_Q \\
    \mathbf{K}^{(k)}_t &= \mathcal{W}^{(k)}_t \mathbf{W}_K, \quad \mathbf{V}^{(k)}_t = \mathcal{W}^{(k)}_t \mathbf{W}_V
$$
where $\mathbf{W}_{\{Q,K,V\}} \in \phi$ are learned projections.

### Trust-Gated Residual Injection.
A critical challenge in adding attention to a frozen model is preserving the signal magnitude distribution. Unconstrained injection can destabilize the pre-trained features. In the active instrumented path, PDT therefore uses a single learned residual gate on the SNC delta before it is added back to the hidden state.

The context vector $\mathbf{C}^{(k)}_t$ is computed via standard scaled dot-product attention over the visible notes window. The final update to the trunk's hidden state is:
$$
    \tilde{\mathbf{H}}^{(k)}_{l,t} = \mathbf{H}^{(k)}_{l,t} + \lambda_l \cdot \mathbf{C}^{(k)}_t \mathbf{W}_O
$$
where $\lambda_l \in [0, 1]$ is the learned residual gate for the instrumented layer. This lets the model retain its pre-trained behavior early in training while gradually incorporating cross-stream information through a single trainable control surface rather than a stacked gate product.

### Writes and Coverage.
When cadence logic schedules an update, the speculation head emits a new note embedding
$$
    \widehat{\mathbf{n}}^{(k)}_{v+1} = \mathrm{SpeculationHead}\!\left(\tilde{\mathbf{H}}^{(k)}_{l,t}\right),
$$
which is appended to the versioned bus with stride and version metadata. Coverage is computed separately as
$$
    \mathbf{c}^{(k)}_t = \mathrm{CoverageHead}\!\left(\tilde{\mathbf{H}}^{(k)}_{l,t}, E_{\text{plan}}(z), m_{\text{plan}}\right),
$$
where the plan embeddings come from the same latent vocabulary used by the planner. These coverage logits supervise plan-item completion and can be logged with snapshots, but they are not the representation consumed by sibling streams.
Because the coverage targets are stream-qualified canonical plan items, they act as a diagnostic for whether the shared latent plan is being executed coherently and without redundant cross-stream claims.

## Agreement and Rollback Control

While the SNC attention mechanism allows information flow, it does not guarantee correctness. A sibling stream may still write a misleading summary. To manage this, we introduce the **Agreement Head**, a scalar classifier trained to estimate whether the latest committed token block should be trusted.

For a hidden state $\mathbf{h}_t$, the Agreement Head predicts a trust score $s_t \in [0, 1]$:
$$
    s_t = \sigma(\mathbf{w}_{\text{agree}}^T \cdot \text{Dropout}(\mathbf{h}_t) + b_{\text{agree}})
$$
During inference, this score serves as the rollback trigger. If $s_t < \tau$, the system identifies a coherence failure inside the current commit horizon and can regenerate that segment with fresher note context.

This separation of concerns---planner for initial decomposition, SNC for information flow, agreement for control flow, and coverage for plan-item diagnostics---matches the current implementation. In this paper, coverage and agreement outputs are treated as internal coordination signals and diagnostics; they do not by themselves establish full serial equivalence or benchmark leadership on answer quality.

# Systems Implementation

Implementing planner-first latent coordination turns the latent contract into a systems problem. The current code path is organized around fixed-width snapshot tensors, lagged reads, and a frozen-trunk sidecar runtime.

## Versioned Snapshot Bus
The runtime does not pass variable-length note text between streams. Instead, both training and inference operate on fixed-dimensional snapshot tensors plus metadata. The dataset pipeline materializes `versioned_notes` with snapshot 0 sourced from `plan_contract`; the collator then assembles teacher and student snapshot blocks with note embeddings, masks, stride metadata, version IDs, stream ownership, and optional coverage metadata.

At runtime, each stream reads a lagged notes window at every decode step. Reveal delay $\Delta$ determines which snapshots are visible, while cadence controls only when a new speculative note embedding is appended. Coverage metadata can be logged alongside a snapshot, but the attended payload is always the note embedding itself. This distinction matters: the Dynamic Notes Bus is an embeddings-only transport, not a text transport.
Because snapshot 0 is derived from the plan contract and preserved through the first stride, the runtime begins from a shared plan-conditioned workspace instead of waiting for streams to negotiate a plan through generated text.

## Frozen-Trunk Sidecar Runtime
The current implementation attaches planner, SNC, note-emission, coverage, and agreement modules as sidecar components around a frozen backbone rather than routing coordination through full-trunk adaptation. This keeps the coordination pathway explicit in code: prompt hidden states feed the planner, the planner seeds snapshot 0, streams read lagged snapshot windows through SNC, and cadence/agreement logic govern later writes and rollback.

One learned control surface matters in the active SNC path: each instrumented layer applies a learned residual gate to the SNC delta before it is added back to the hidden state. The repository also exposes inference-time ablation knobs such as logit blending and diagnostic bus interventions, but those are debugging and analysis utilities rather than part of the paper's claimed trained mechanism.

# Evaluation Scope

The coordination module described in this paper was materially rewritten to match the planner-first latent contract in Sections \ref{sec:architecture} and \ref{sec:snc}. For that reason, quantitative results from earlier module versions are omitted here: they no longer represent the current system. This version of the pre-print is therefore an architecture and implementation paper rather than a settled benchmark paper.

## What Should Be Evaluated
The revised system should be evaluated as a coordination mechanism, not merely as a text generator. The relevant question is whether a mandatory prompt-time planner and embeddings-only Dynamic Notes Bus help parallel streams maintain section ownership, avoid redundant content, and remain mutually consistent while decoding.

## Evaluation Protocol for the Revised Stack
The current dataset and training pipeline expose the right supervision contract for that question. Each example carries a teacher plan, per-stream `section_contract` and `notes_contract` fields, a `plan_contract` materialized as `versioned_notes[0]`, fixed-width `planner_ids`, and canonical `plan_item_ids` derived from the same stream-qualified plan catalog. This contract enables evaluation along at least four axes:
- planner-slot accuracy against canonical latent planner IDs;
- plan-item coverage and ownership consistency against the canonical catalog;
- cross-stream overlap and contradiction rates in generated outputs;
- latency and throughput relative to serial decoding and external decomposition baselines.

## Current Scope
This pre-print does not report benchmark numbers for those quantities yet. In particular, it does not claim validated answer-quality gains, latency gains, exact serial equivalence, or superiority over external multi-agent baselines for the revised coordination stack. It also does not claim that the `planner_head` alone carries the revised coordination behavior: head-level attribution requires dedicated ablations and intervention studies beyond the current mechanism paper. Quantitative reevaluation on the current implementation is deferred to a future revision.

# Conclusion

In this work, we introduced the **Parallel Decoder Transformer (PDT)**, a planner-first architecture that moves task decomposition and coordination inside the decoding stack. The central result is a mechanism rather than a benchmark claim: a mandatory prompt-time latent planner seeds an embeddings-only Dynamic Notes Bus before any stream emits tokens, and later stream coordination is mediated through that shared latent workspace by a broader stack of plan projection, SNC, note-emission, coverage, and agreement modules.

The current implementation aligns the planner, plan embeddings, plan catalog hashing, and coverage targets to one shared latent vocabulary. Snapshot 0 is a real dataset artifact derived from the plan contract, the bus carries embeddings rather than text, and SNC reads lagged snapshot windows every decode step while cadence governs writes. The paper therefore establishes a coherent contract across dataset generation, training supervision, and inference runtime for planner-plus-latent-bus coordination, while keeping the attribution claim appropriately stack-level rather than assigning the full behavior to the planner head alone.

## Future Directions
Three directions matter most from here:
- **Direct overlap and coherence evaluation:** The next priority is to measure contradiction rate, semantic redundancy, and section ownership directly, rather than relying only on plan-item coverage as a proxy.
- **Adaptive communication schedules:** The current system reads the visible notes window every step and gates only writes. Future work should test sparse or topology-adaptive read schedules without abandoning the planner-seeded latent workspace.
- **Planning capacity and scaling:** The current planner uses a fixed 16-slot latent layout. Future work should test adaptive slot counts, richer plan-item ownership, and larger stream counts while preserving the shared latent vocabulary contract.

The current pre-print therefore makes a focused claim: planner-plus-latent-bus coordination is implemented end-to-end in the current codebase and specified as a consistent training/runtime contract, while quantitative reevaluation of the revised module remains future work.

# Appendix

## Implementation-Grounded Scope

This appendix retains only details re-audited against the current codebase and the narrower mechanism-focused scope of this version of the pre-print. In particular, we removed stale claims about textual planner supervision, F1-style coverage loss, and broad theoretical guarantees that are not directly supported by the present implementation or experiments.

## Fixed Latent Planning Contract

The planner uses a fixed-width latent interface. The current configuration sets
$$
S = 16 \qquad \text{and} \qquad V_p = 65{,}536,
$$
where $S$ is the number of planner slots and $V_p$ is the shared latent plan vocabulary. The model config enforces
$$
`planner_head.vocab_size`
=
`plan_vocab_size`
=
`collator.plan_hash_buckets`,
$$
and also requires the collator slot count to match the planner slot count.

Canonical plan catalog entries are built from stream-qualified plan structure, including `notes_contract`, per-stream summaries, and serialized `section_contract` metadata when present. Each catalog entry is hashed into the shared latent vocabulary. After padding, planner supervision has shape $[B, S]$:
$$
    \mathbf{z}_{\text{plan}} = \mathrm{Pad}_{S}\!\left(\mathrm{Hash}\!\left(\mathcal{C}_{\text{plan}}\right)\right).
$$
These latent IDs are the `planner_ids` used by the planner loss. Separately, the unpadded canonical catalog defines `plan_item_ids` and stream ownership metadata for coverage supervision.

## Training Objectives

Let $\Pi^S$ denote student planner logits, $\Pi^T$ teacher planner logits, and $y_{\text{plan}}$ the latent `planner_ids`. The planner objective is cross-entropy over latent slots:
$$
    \mathcal{L}_{\text{plan}} =
    \mathrm{CE}\!\left(\Pi^S, y_{\text{plan}}\right),
$$
with padded slots ignored via the label pad index.

Let $\widehat{\mathbf{N}}^{\text{notes}}$ be the notes-head prediction, $\widehat{\mathbf{N}}^{\text{spec}}$ the speculative note prediction, and $\mathbf{N}^T$ the teacher note tensor. The baseline note-alignment losses are
$$
    \mathcal{L}_{\text{notes}} = \mathrm{MSE}\!\left(\widehat{\mathbf{N}}^{\text{notes}}, \mathbf{N}^T\right),
    \qquad
    \mathcal{L}_{\text{spec}} = \mathrm{MSE}\!\left(\widehat{\mathbf{N}}^{\text{spec}}, \mathbf{N}^T\right).
$$

When teacher snapshot metadata exposes snapshot 0 under the active sectional mask, the trainer adds snapshot-specific alignment terms on those positions:
$$
    \mathcal{L}_{\text{snap}} =
    \mathrm{MSE}\!\left(\widehat{\mathbf{N}}^{\text{notes}}_{v=0}, \mathbf{N}^T_{v=0}\right),
    \qquad
    \mathcal{L}_{\text{snap-spec}} =
    \mathrm{MSE}\!\left(\widehat{\mathbf{N}}^{\text{spec}}_{v=0}, \mathbf{N}^T_{v=0}\right).
$$
The trainer also aligns a pooled projection from the latent plan embeddings to the teacher snapshot-0 notes with an additional MSE term $\mathcal{L}_{\text{proj}}$.

Planner distillation and planner stability are masked KL terms:
$$
    \mathcal{L}_{\text{KD-plan}} =
    \mathrm{KL}\!\left(p^S_{\text{plan}} \Vert p^T_{\text{plan}}\right)_{m_{\text{plan}}},
    \qquad
    \mathcal{L}_{\text{stab-plan}} =
    \mathrm{KL}\!\left(p^S_{\text{plan}} \Vert p^{\text{pre}}_{\text{plan}}\right)_{m_{\text{plan}}},
$$
where $m_{\text{plan}}$ is the planner-slot mask and $p^{\text{pre}}_{\text{plan}}$ is the pre-update planner distribution recorded for stability logging.

Once token-level LM supervision is active, the trainer adds masked LM cross-entropy plus teacher-KL and pre/post stability terms on token logits:
$$
    \mathcal{L}_{\text{LM-CE}} =
    \mathrm{CE}\!\left(`lm_logits`, `labels`\right)_{m_{\text{labels}}},
    \qquad
    \mathcal{L}_{\text{KD-LM}} =
    \mathrm{KL}\!\left(p^S_{\text{LM}} \Vert p^T_{\text{LM}}\right)_{m_{\text{LM}}},
$$
with a corresponding masked stability term outside the commit horizon.

Coverage supervision is binary cross-entropy with logits over canonical plan items:
$$
    \mathcal{L}_{\text{cov}} =
    \mathrm{BCEWithLogits}\!\left(\mathbf{c}, y_{\text{cov}}\right)_{m_{\text{cov}}},
$$
where $\mathbf{c}$ are coverage logits, $y_{\text{cov}}$ are binary coverage targets, and $m_{\text{cov}}$ is the coverage mask. This is a direct code-path correction from earlier drafts: the current trainer does *not* optimize $1-\mathrm{F1}$ as the coverage loss.

Agreement supervision is also binary cross-entropy:
$$
    \mathcal{L}_{\text{agree}} =
    \mathrm{BCE}\!\left(\mathbf{a}, y_{\text{agree}}\right)_{m_{\text{agree}}},
$$
where agreement labels are pooled over committed token positions.

The trainer also contains optional, config-gated terms for usage penalty, NLI contradiction margin, redundancy, retention, inter-head speculative KL, and stream classification. Because these branches are optional, they should be described as implementation hooks unless a reported run explicitly enables and ablates them.

The total training loss can therefore be summarized as
$$
    \mathcal{L}_{\text{total}} &=
    \mathcal{L}_{\text{plan}}
    + \mathcal{L}_{\text{notes}}
    + 0.5 \mathcal{L}_{\text{spec}}
    + \mathcal{L}_{\text{snap}}
    + 0.5 \mathcal{L}_{\text{snap-spec}}
    + \mathcal{L}_{\text{proj}}  \\
    &\quad
    + \lambda_{\text{KD}}\!\left(\mathcal{L}_{\text{KD-plan}} + \mathcal{L}_{\text{KD-LM}}\right)
    + \lambda_{\text{stab}}\!\left(\mathcal{L}_{\text{stab-plan}} + \mathcal{L}_{\text{stab-LM}}\right)
    + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}}
    + \lambda_{\text{agree}} \mathcal{L}_{\text{agree}}
    + \mathcal{L}_{\text{optional}},
$$
where $\mathcal{L}_{\text{optional}}$ abbreviates the config-gated auxiliary terms above.

## Snapshot 0 and Bus Semantics

Snapshot 0 is part of the dataset and runtime contract. The notes-generation pipeline derives initial notes from the plan contract, writes them to `versioned_notes[0]` with source `plan_contract`, and then appends later procedural or teacher snapshots. The KD export path preserves these versioned notes and hashes the canonical plan catalog into latent planner IDs.

During training, the collator assembles snapshot tensors with masks, version IDs, stride metadata, and stream ownership. The trainer can freeze snapshot 0 through the first stride so the planner commitment is not immediately overwritten. In the canonical path studied in this paper, inference runs the planner head on the prompt to obtain the initial latent plan code. The repository also includes contract-based initialization hooks for diagnostics and replay, but the paper's mechanism claim concerns the learned planner path. After initialization, `plan_notes_proj`, `notes_head`, SNC, and the rollback heads maintain the shared workspace.

Most importantly, the bus carries embeddings only. Textual notes and plan strings are auxiliary artifacts for observability, scoring, and dataset construction. The repository's manual override hooks are diagnostic only and fall outside the paper's mechanism claim. Coverage vectors may be logged with a snapshot for telemetry or retained for supervision, but sibling streams consume note embeddings through SNC.

## Reported Claims and Non-Claims

The current pre-print supports four concrete claims:
- PDT implements planner-first latent coordination with fixed planner slots, a shared latent vocabulary, and an embeddings-only Dynamic Notes Bus.
- The dataset, training loop, and inference runtime all preserve snapshot 0 as a real planning artifact.
- The mechanism claim is stack-level: the planner head initializes latent plan IDs, while plan projection, note emission, SNC, and rollback heads jointly implement the runtime coordination path.
- This version of the paper intentionally omits quantitative claims from earlier module versions because the coordination stack has been materially revised and requires reevaluation.

The current evidence therefore does *not* justify broader claims of validated answer-quality gains, latency gains, exact serial equivalence, hierarchical multi-stream scaling, the effectiveness of every optional regularizer, or the claim that the `planner_head` alone internalizes the full coordination policy. Those remain future empirical questions.

## References

- **cai2024medusa**. T.~Cai et~al. Medusa: Simple llm inference acceleration framework with multiple decoding heads. {\em arXiv preprint arXiv:2401.10774}, 2024.

- **Goyal2021**. A.~Goyal et~al. Coordination among neural modules through a shared workspace. {\em arXiv preprint arXiv:2103.01197}, 2021.

- **Han2025Blackboard**. Bochen Han and Songmao Zhang. Exploring advanced {LLM} multi-agent systems based on blackboard architecture. {\em arXiv preprint arXiv:2507.01701}, 2025.

- **houlsby2019parameter**. Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morri, Andrea De~Coro, Sergei Vassilvitskii, Ariel Fisher, and Deep Ganguli. Parameter-efficient transfer learning for nlp. In {\em International Conference on Machine Learning}, 2019.

- **hu2021lora**. Edward~J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu~Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In {\em International Conference on Learning Representations}, 2022.

- **Leong2025AMAS**. Hui~Yi Leong, Yuheng Li, Yuqing Wu, Wenwen Ouyang, Wei Zhu, and Jiechao Gao. {AMAS}: Adaptively determining communication topology for {LLM}-based multi-agent system. {\em arXiv preprint arXiv:2510.01617}, 2025.

- **leviathan2023fast**. Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In {\em International Conference on Machine Learning}, 2023.

- **Li2024lookahead**. Y.~Li, F.~Wei, C.~Zhang, and H.~Zhang. Break the sequential dependency of llm inference using lookahead decoding. In {\em Proceedings of ICML 2024}, 2024. arXiv:2402.02057.

- **PSLM2024**. X.~Liu et~al. Pslm: Parallel generation of text and speech with llms. In {\em Findings of EMNLP 2024}, 2024.

- **LopezPaz2015**. D.~Lopez-Paz, L.~Bottou, B.~Schölkopf, and V.~Vapnik. Unifying distillation and privileged information. In {\em ICLR Workshop}, 2016.

- **ning2023skeleton**. X.~Ning, Z.~Lin, H.~Yang, and Y.~Wang. Skeleton-of-thought: Prompting llms for efficient parallel generation. {\em arXiv preprint arXiv:2307.15337}, 2023.

- **Rodionov2025**. A.~Rodionov et~al. Hogwild! inference: Parallel llm generation via concurrent attention. {\em arXiv preprint arXiv:2504.06261}, 2025.

- **Salemi2025Blackboard**. Alireza Salemi, Mihir Parmar, Palash Goyal, Yiwen Song, Jinsung Yoon, Hamed Zamani, Tomas Pfister, and Hamid Palangi. {LLM}-based multi-agent blackboard system for information discovery in data science. {\em arXiv preprint arXiv:2510.01285}, 2025.

- **Stern2018**. M.~Stern, N.~Shazeer, and J.~Uszkoreit. Blockwise parallel decoding for deep autoregressive models. {\em arXiv preprint arXiv:1811.03115}, 2018.

- **EAGLE2024**. Z.~Sun et~al. Eagle: Speculative sampling requires rethinking feature uncertainty. {\em arXiv preprint arXiv:2401.15077}, 2024.

- **Vapnik2015**. V.~Vapnik. Learning using privileged information: Similarity control and knowledge transfer. {\em Journal of Machine Learning Research}, 16, 2015.

- **Wang2025MoA**. Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce~Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. In {\em International Conference on Learning Representations}, 2025.

- **Xiao2025sprint**. G.~Xiao et~al. Sprint: Enabling interleaved planning and parallelized execution in large reasoning models. {\em arXiv preprint arXiv:2506.05745}, 2025.

- **Yan2024**. M.~Yan et~al. Decoding speculative decoding. {\em arXiv preprint arXiv:2402.01528}, 2024.

- **survey2025**. L.~Zhang, L.~Fang, C.~Duan, M.~He, L.~Pan, P.~Xiao, S.~Huang, Y.~Zhai, X.~Hu, P.~S. Yu, and A.~Liu. A survey on parallel text generation: From parallel decoding to diffusion language models. {\em arXiv preprint arXiv:2508.08712}, 2025.

- **Zhuge2024GPTSwarm**. Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and J{\"u}rgen Schmidhuber. {GPTSwarm}: Language agents as optimizable graphs. In {\em Proceedings of the 41st International Conference on Machine Learning}, volume 235 of {\em Proceedings of Machine Learning Research}, pages 62743--62767, 2024.

- **Zou2025LatentMAS**. Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke~Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. Latent collaboration in multi-agent systems. {\em arXiv preprint arXiv:2511.20639}, 2025.
