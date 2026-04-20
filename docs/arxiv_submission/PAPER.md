# Parallel Decoder Transformer: Concept-Space Co-Referencing via a Shared Latent Workspace

Logan Robbins  
Independent Researcher  
`ljrweb@gmail.com`

## Abstract

A single frozen decoder can be extended with trainable sidecar modules so that it exposes $K$ coordinated output streams, each of which reads from and writes to a shared latent workspace during generation. We present the **Parallel Decoder Transformer (PDT)**: a frozen Qwen3-4B Base trunk augmented with a planner-seeded Dynamic Notes Bus, per-layer Speculative Note Conditioning (SNC) cross-attention, and coverage + readiness heads over the shared workspace. The central claim is neither task decomposition nor inference-speed: it is **concept-space co-referencing**. Each stream's trajectory is shaped by continuous low-bandwidth awareness of *where siblings are operating in concept-space*, mediated only by a learned embeddings-only channel. We formalize the mechanism as a stream-level residual-add SNC pathway with an outer gated residual on every instrumented decoder layer, state the existence-proof experiment and its falsifiability contract precisely, and report the pre-registered ablation protocol (Interventions A/B/C) that governs the paper's positive or null conclusion. The same mechanism generalizes -- unchanged -- to concurrent sensor streams, modality-specific encoders, and parallel reasoning strategies where the partition is given rather than inferred.

---

## 1. Introduction

Large language models frequently face prompts whose natural solution is a set of partially independent sections, subquestions, or arguments. A model may internally recognize the decomposition, but standard autoregressive decoding exposes only a single causal output stream. External decomposition methods [@ning2023skeleton; @PSLM2024] partially relieve that constraint by prompting for an outline and then spawning multiple generations in parallel. Those methods can encourage modularity, but they do not create model-internal shared state: once the work is split across separate calls, no stream can directly know whether a sibling has already established a key fact or is operating in a region of concept-space that requires complementary (rather than redundant) content.

We re-frame the problem from **synchronization** to **concept-space co-referencing**. A single frozen decoder can be given an internal mechanism by which $K$ inputs produce $K$ outputs that coordinate somewhere in between -- and change their own trajectories based on sibling state -- via a shared latent workspace. The workspace is the thesis. The architecture we present is its instantiation.

The contribution is a mechanism, not a speedup. We do not claim PDT decodes faster than a baseline; we claim the base decoder's output interface can be re-shaped so that $K$ streams share a low-bandwidth concept-level channel without needing a high-bandwidth text channel. We demonstrate the mechanism on a same-prompt, $K$-stream sectional-generation setting (the existence-proof domain), and argue by analogy that the same mechanism extends directly to domains where the $K$ partition is given (multi-modal fusion, robotic sensor channels, ensemble reasoning).

## 2. Thesis

Let $\theta_{\text{pre}}$ be the frozen parameters of a pre-trained decoder and let $\phi$ be a trainable sidecar. We extend $\theta_{\text{pre}}$ with $\phi$ such that the composite model exposes $K$ coordinated output branches. Each branch is a locally causal generation stream; the shared latent workspace (the Dynamic Notes Bus) carries per-block summaries between streams. The thesis is:

> A single frozen decoder can be trained -- through $\phi$ alone -- to expose $K$ output streams whose next-token distributions are a non-trivial, prompt-conditional function of sibling state in a learned concept-space workspace, with that dependence mediated specifically by the SNC pathway rather than by any high-bandwidth text channel.

The paper's empirical contribution is the pre-registered ablation protocol that tests this thesis as a falsifiable claim. See §6.

## 3. Related Work

We organize prior work by the **shape of the output interface a single model exposes**. On that axis, the existing landscape populates three points; PDT introduces a fourth.

### 3.1 One stream, one next token

The default output interface of a causal language model is a single stream emitting $p(x_t \mid x_{<t})$. PDT retains the single-model setting and the frozen pre-trained decoder but replaces the single-stream assumption with $K$ coexisting causal streams sharing a latent workspace.

### 3.2 One stream plus lookahead verification

Speculative Decoding [@leviathan2023fast], Blockwise Parallel Decoding [@Stern2018], Medusa [@cai2024medusa], Hydra [@Ankner2024Hydra], EAGLE [@EAGLE2024], SpecDec++ [@Huang2024SpecDecPP], Lookahead Decoding [@Li2024lookahead], Jacobi parallel decoding [@Santilli2023Jacobi], and CLLMs [@Kou2024CLLM] keep a single committed causal stream and add drafted future tokens accepted or rejected by the same model [@Yan2024; @survey2025]. Even a perfectly accepted Medusa or EAGLE draft leaves a single output sequence. PDT's $K$ threads persist and coordinate across synchronized rounds.

### 3.3 One stream plus parallel positions

Non-Autoregressive Transformers [@Gu2018NAT], Mask-Predict [@Ghazvininejad2019MaskPredict], the Glancing Transformer [@Qian2021GLAT], and diffusion-based language models [@Li2022DiffusionLM; @Lou2024SEDD] parallelize *positions* within one output. The output interface remains a single sequence.

### 3.4 K causal streams sharing a latent workspace

The closest points to PDT are *Hogwild!* Inference [@Rodionov2025] -- concurrent generations of one model attending to each other's KV caches -- and SPRINT [@Xiao2025sprint], which interleaves a planner and a parallel executor within a single reasoning model. Conceptual precedents include shared neural workspaces [@Goyal2021] and the Consciousness Prior [@Bengio2017ConsciousnessPrior]. PDT adds three properties that prior methods on this axis do not combine: a *mandatory prompt-time latent planner* that seeds a shared snapshot 0 *per stream* before any token is emitted; an *embeddings-only Dynamic Notes Bus* with a reveal delay $\Delta$; and an *agreement-gated commit protocol* that decides per block whether the parallel frontier may advance.

### 3.5 External multi-call orchestration

Skeleton-of-Thought [@ning2023skeleton], PSLM [@PSLM2024], GPTSwarm [@Zhuge2024GPTSwarm], Mixture-of-Agents [@Wang2025MoA], blackboard-style multi-agent systems [@Han2025Blackboard; @Salemi2025Blackboard], topology-adaptive AMAS [@Leong2025AMAS], and LatentMAS [@Zou2025LatentMAS] compose multiple calls externally. PDT modifies the output interface of a *single* model so that coordination is a latent workspace *inside* the decoder.

## 4. Architecture

PDT turns one frozen decoder into $K$ coordinated output streams. Its central design decision is that decomposition and cross-stream state are **model-internal**.

### 4.1 Frozen trunk topology

We initialize PDT from Qwen3-4B Base (36 layers, $d_{\text{model}} = 2560$, GQA 32 query heads / 8 KV heads) and freeze all parameters. Trainable parameters $\phi$ are introduced as a lightweight coordination stack:

$$
\Theta = \theta_{\text{pre}} \cup \phi.
$$

The trainable subset $\phi$ follows the parameter-efficient adaptation tradition of LoRA [@hu2021lora] and adapters [@houlsby2019parameter]. It contains four components:

- **Per-layer stream adapters**, inserted into every 3rd instrumented trunk layer as a per-stream bottleneck delta with a learned outer gate.
- **Per-layer SNC cross-attention**, reading from the shared workspace on the same 12 instrumented layers with its own independent $W_Q, W_K, W_V, W_O$ projections (symmetric multi-head, independent of the trunk's GQA).
- **Planner and notes modules**: a fixed-slot planner head, a shared plan-embedding matrix, a per-stream snapshot-0 projection, a notes head, and a speculation head.
- **Auxiliary control heads**: coverage, agreement (readiness), and stream-classification heads.

**Symmetry breaking at round 1.** Although every stream receives the same prompt tokens, stream identity $k$ enters the forward pass through three compounded mechanisms: (i) each instrumented decoder layer hosts a dedicated bottleneck adapter with independently-initialized weights, routed by stream identifier; (ii) snapshot 0 is *per-stream seeded* -- each stream's slot of snapshot 0 is a projection of only the plan items the planner assigned to that stream, under a disjoint-ownership invariant; and (iii) each stream maintains its own Dynamic Notes Bus window and KV cache.

### 4.2 Prompt-time latent planner

Let $\mathbf{H}_x \in \mathbb{R}^{T \times d}$ denote the frozen-trunk hidden states of the prompt. The planner performs a mandatory planning pass before any output tokens:

$$
\mathbf{\Pi} = \mathrm{PlannerHead}(\mathrm{MaskedMeanPool}(\mathbf{H}_x)) \in \mathbb{R}^{S \times V_p}, \qquad z_i = \arg\max_{a} \mathbf{\Pi}_{i,a}.
$$

In our configuration $S = 16$ and $V_p = 8{,}192$. We reduced $V_p$ from $65{,}536$ (v1) to $8{,}192$ to bring the planner-head parameter count from ~3B to ~336M at $d_{\text{model}} = 2560$; Stage 0 training is gated on codebook-utilization diagnostics (§5.2) so brittleness or collapse surfaces before scaling back up.

Active slots are re-embedded and projected per-stream into notes space:

$$
\mathbf{e}_i = E_{\text{plan}}[z_i], \qquad
\mathbf{n}^{\text{plan},(k)}_0 = \mathrm{LayerNorm}\!\left(W_{\text{plan}} \cdot \mathrm{pool}_{i \in \mathrm{own}(k)} \mathbf{e}_i\right),
$$

where $\mathrm{own}(k)$ is the disjoint subset of planner slots owned by stream $k$. The resulting $\mathbf{n}^{\text{plan},(k)}_0$ is published *per stream* as snapshot 0 on that stream's Dynamic Notes Bus. This per-stream seeding is the first symmetry-breaking mechanism and was the most significant missing piece in the v1 implementation.

### 4.3 Dynamic Notes Bus as shared latent workspace

The **Dynamic Notes Bus** is PDT's shared latent workspace: an embeddings-only, versioned FIFO of per-stream snapshots, one bus per stream. Textual plans and rendered notes may be used for dataset construction, observability, or supervision, but inference-time coordination uses only note embeddings.

We describe the runtime at two timescales:

- $\tau$: tokens per provisional block between synchronization decisions ($\tau = 32$),
- $\Delta$: reveal delay between emission and visibility to siblings ($\Delta = 1$).

Let $\widehat{\mathbf{n}}^{(j)}_u$ be the provisional note emitted by stream $j$ at round $u$. The visible notes window for stream $k$ at round $v$ is

$$
\mathcal{W}^{(k)}_v = \mathrm{Window}\!\left(\{\mathbf{n}^{\text{plan},(k)}_0\} \cup \{\widehat{\mathbf{n}}^{(j)}_u : j \in [K],\; u \le v-\Delta\}\right).
$$

### 4.4 Speculative Note Conditioning

SNC is the read mechanism over the visible workspace. At each decode step $u$ inside round $v$, stream $k$ conditions on $\mathcal{W}^{(k)}_v$. The SNC projections are

$$
\mathbf{Q}^{(k)}_{v,u} = \mathbf{H}^{(k)}_{v,u} W_Q, \qquad
\mathbf{K}^{(k)}_v = \mathcal{W}^{(k)}_v W_K, \qquad
\mathbf{V}^{(k)}_v = \mathcal{W}^{(k)}_v W_V,
$$

with learned parameters $W_Q, W_K, W_V \in \phi$ (independent of trunk weights, so Qwen3's GQA does not collide with SNC's symmetric multi-head layout). Each instrumented layer applies the resulting context through a trust-gated residual:

$$
\widetilde{\mathbf{H}}^{(k)}_{v,u} = \mathbf{H}^{(k)}_{v,u} + \sigma(\lambda_l) \cdot \mathrm{SNC}_l(\mathbf{H}^{(k)}_{v,u}, \mathcal{W}^{(k)}_v) \cdot W_O,
$$

where $\lambda_l$ is a per-layer pre-sigmoid gate parameter. We initialize $\lambda_l \leftarrow -4.0$ so that $\sigma(\lambda_l) \approx 0.018$ at step 0, preserving the trunk's magnitude statistics. The SNC output projection $W_O$ is additionally zero-initialized so the delta is exactly zero at step 0 regardless of the gate.

### 4.5 Provisional writes, coverage, readiness

At the end of each provisional block, stream $k$ writes a latent summary back to its own bus:

$$
\widehat{\mathbf{n}}^{(k)}_v = \mathrm{SpeculationHead}(\widetilde{\mathbf{H}}^{(k)}_{v,\tau}).
$$

Coverage is predicted against the planner-seeded plan catalog:

$$
\mathbf{c}^{(k)}_v = \mathrm{CoverageHead}(\widetilde{\mathbf{H}}^{(k)}_{v,\tau}, E_{\text{plan}}(z_{1:S}), m_{\text{plan}}),
$$

with $m_{\text{plan}}$ masking inactive slots.

### 4.6 Agreement-gated commit

Per the paper contract, the readiness score for stream $k$ at round $v$ is a function of four inputs:

$$
r^{(k)}_v = \mathrm{AgreeHead}\!\left(\widetilde{\mathbf{H}}^{(k)}_{v,\tau},\ \mathcal{W}^{(k)}_v,\ \mathbf{c}^{(k)}_v,\ \widehat{\mathbf{n}}^{(k)}_v\right).
$$

The implementation attends over the notes window with the block-end hidden as a query, encodes coverage via a compact statistic (mean, std, min, max, fraction-above-threshold), projects the stream's own provisional note, concatenates all four features, and scores. The threshold $\gamma$ is tuned from a held-out ROC sweep, not learned.

The global continuation rule is

$$
A_v = \mathbf{1}\!\left[\min_{k \in \mathcal{A}_v} r^{(k)}_v > \gamma\right].
$$

If $A_v = 1$, PDT commits the provisional blocks and schedules the new notes for visibility after $\Delta$. If $A_v = 0$, PDT rolls back only the streams whose readiness falls below threshold.

## 5. Staged Training Curriculum

Training a coordination mechanism on a frozen trunk is unstable if all modules are enabled at once. PDT uses a four-stage curriculum:

| Stage | Name                  | Trainable                                              | Purpose                                        |
| ----: | :-------------------- | :----------------------------------------------------- | :--------------------------------------------- |
|     0 | Planner pretrain      | planner head, plan embedding, plan_notes_proj, notes head | latent plan decomposition + snapshot-0 contract |
|     1 | Stream bootstrap      | per-layer adapters, per-layer SNC, stream classifier   | stream-specific conditioning                   |
|     2 | Bus enablement        | speculation head                                       | learned latent summaries on the bus            |
|     3 | Commit control        | coverage head, agreement head                          | ownership + continuation sufficiency           |

### 5.1 Training objectives (see Appendix A)

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{plan}} + \mathcal{L}_{\text{notes}} + 0.5\,\mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{LM-CE}} + \lambda_{\text{KD}}\,\mathcal{L}_{\text{KD-LM}} + \lambda_{\text{cov}}\,\mathcal{L}_{\text{cov}} + \lambda_{\text{ready}}\,\mathcal{L}_{\text{ready}}.
$$

Stage gating is explicit in the training loop: LM CE/KD activate at Stage 2+; coverage and readiness BCE activate at Stage 3+.

### 5.2 Codebook-utilization diagnostics (Stage 0 gate)

Since $V_p = 8{,}192$ is mid-granularity rather than conservative, Stage 0 logs:

- **unique entries used**: fraction of $V_p$ selected at least once over the epoch.
- **per-slot entropy (bits)**: Shannon entropy of each slot's selection distribution.
- **pairwise anchor cosine**: mean cosine similarity between per-stream snapshot-0 anchors on the same prompt (near-identical $\Rightarrow$ no differentiation possible).
- **usage histogram**: top-$k$ entries.

The Stage 0 $\to$ Stage 1 gate is $\texttt{unique\_entries} \ge 1000$ AND $\max_s \texttt{entropy}(s) \ge 2$ bits. Failure to meet either triggers entropy regularization before scaling.

## 6. Falsifiable Existence-Proof Experiment

We state the coordination claim precisely. With the SNC-conditioned hidden

$$
\widetilde{\mathbf{H}}^{(k)}_{v,u} = \mathbf{H}^{(k)}_{v,u} + \sigma(\lambda_l) \cdot \mathrm{softmax}\!\left(\tfrac{Q^{(k)}_{v,u} K^{(k)\top}_v}{\sqrt{d}}\right) V^{(k)}_v W_O,
$$

the coordination claim is

$$
\exists\, k \ne j,\ v,\ u \le v-\Delta:\quad \frac{\partial\, p(\hat y^{(k)}_{v,t})}{\partial\, \widehat{\mathbf{n}}^{(j)}_{u}} \ne \mathbf{0},
$$

and this partial derivative vanishes whenever $\lambda_l \equiv -\infty$ at every instrumented layer.

### 6.1 Primary hypothesis -- cross-stream differentiation

K streams with trained SNC produce outputs with measurably higher cross-stream differentiation than K streams with ablated SNC, where differentiation is measured by:

- pairwise cosine distance between per-stream output embeddings,
- cross-stream ROUGE-L (lower = more differentiated),
- Jensen-Shannon divergence of entity distributions,
- cross-stream contradiction rate,
- coverage overlap against the planner-seeded catalog.

### 6.2 Pre-registered interventions

- **Intervention A -- SNC gate ablation.** Force $\sigma(\lambda_l) \leftarrow 0$ at every instrumented layer (implemented by the runtime's ``snc_force_gate=False`` override, which collapses the SNC delta to exactly zero regardless of the trained gate).
- **Intervention B -- Norm-matched sibling-write scramble.** Replace sibling notes $\widehat{\mathbf{n}}^{(j \ne k)}_u$ with Gaussian vectors rescaled to the empirical per-note norm. Keep $\sigma(\lambda_l)$ as trained. This isolates informational content from attention-softmax numerics.
- **Intervention C -- Anchor swap.** Replace stream $k$'s snapshot-0 anchor with one drawn from a *different prompt entirely*. Tests whether SNC interprets sibling position prompt-conditionally rather than as generic noise.

### 6.3 Pre-registered substantiation

An absolute delta of at least 3 points in coverage overlap or cross-stream contradiction rate between the trained-gate baseline and Intervention A or B -- on the sectional-knowledge evaluation suite -- rejects the null that SNC is decorative under the current training contract.

### 6.4 Pre-registered null handling

If Interventions A, B, and C all produce null results, the paper reports that coordination under the current training contract is produced by per-stream adapters and planner seeds alone, SNC is superfluous at this configuration, and the *informational-necessity* training contract (sibling-citing targets) is a prerequisite for a positive result. We do not pivot the architecture; we report the null honestly and name the next experimental regime.

## 7. Beyond Text: The Long-Horizon Frame

The same mechanism extends directly to three K-unrelated-inputs domains where the partition is given rather than inferred by a planner.

**Multi-modal fusion.** K modality-specific encoders (text, vision, audio, depth) produce concurrent streams whose outputs must refer to each other in concept-space. The fusion problem is classically solved with cross-attention over concatenated modality tokens; that approach scales poorly and does not give each modality stream a persistent ability to sense what the others are currently representing. PDT's workspace gives each modality stream continuous low-bandwidth access to sibling concept-state during its own generation. Modality identity provides built-in symmetry breaking, so the planner may become optional or vestigial.

**Sensor fusion in robotics.** K sensor channels (joint encoders, IMUs, vision, force/torque) produce concurrent streams whose outputs drive K motor channels. A left arm reaching for object A does not need the right arm's full trajectory; it needs *what region of workspace the right arm is occupying* so it can avoid collision, balance the body, or coordinate a handoff. The mechanism is a candidate substrate for when compute, latency, and energy constraints for deployed robotics relax; the text demonstration exists so the mechanism's properties are characterized in a testable domain before deployment to one where testing is harder.

**Ensemble reasoning.** K reasoning strategies (chain-of-thought, tool-use, retrieval-grounded, program-of-thought) run concurrently on the same problem. Each stream develops its line of reasoning while reading sibling concept-state, so strategies can notice when siblings have found a promising lead, diverge to cover unexplored regions, or converge on an agreed answer. Distinct from self-consistency, which runs $K$ independent rollouts and votes after the fact -- here the streams are not independent and may adapt during generation.

All three share the same underlying primitive: *low-bandwidth, concept-level awareness between K concurrent producers that do not share a high-bandwidth channel*. The text demonstration is the existence proof; these three are the reason the existence proof is worth doing.

## 8. Conclusion

PDT re-frames parallel generation as a concept-space co-referencing problem inside the decoder. Its central contribution is not decomposition and not speedup: it is the mechanism by which $K$ streams of a single frozen decoder read from and write to a shared latent workspace such that each stream's trajectory is shaped by low-bandwidth awareness of where siblings are operating in concept-space. A mandatory prompt-time planner seeds the workspace per stream; SNC provides continuous token-time access; coverage tracks ownership; agreement gates commit.

The paper's empirical contribution is the pre-registered ablation protocol stated in §6. Either outcome -- positive or null -- produces a publishable result; what matters is that the mechanism has been tested rather than assumed.

---

## Appendix A: Training Objectives

Let $\Pi^S$ denote the student planner logits and $y_{\text{plan}}$ the latent planner-slot targets. The planner objective is cross-entropy over latent slots:

$$
\mathcal{L}_{\text{plan}} = \mathrm{CE}(\Pi^S, y_{\text{plan}})
$$

with padded slots ignored.

Let $\widehat{\mathbf{N}}^{\text{notes}}$ be the notes-head prediction, $\widehat{\mathbf{N}}^{\text{spec}}$ the speculative note prediction, and $\mathbf{N}^T \in \mathbb{R}^{K \times d_{\text{notes}}}$ the teacher notes tensor. Teacher notes are produced at dataset-export time by prompting a teacher LLM *per stream* with only that stream's own text slice (the `TRUE_NOTES_PROMPT` schema), then deterministically reducing each textual entry to a $d_{\text{notes}}$-dimensional unit vector via a SHA-256 byte-parity sign hash ($d_{\text{notes}} = 256$). Because each teacher call sees only its own stream's slice, $\mathbf{N}^T$ contains no teacher-generated cross-stream information by construction. The note-alignment losses are

$$
\mathcal{L}_{\text{notes}} = \mathrm{MSE}(\widehat{\mathbf{N}}^{\text{notes}}, \mathbf{N}^T), \qquad
\mathcal{L}_{\text{spec}} = \mathrm{MSE}(\widehat{\mathbf{N}}^{\text{spec}}, \mathbf{N}^T).
$$

The language-model losses (active at Stage 2+) are

$$
\mathcal{L}_{\text{LM-CE}} = \mathrm{CE}(\texttt{lm\_logits}, \texttt{labels})_{m_{\text{labels}}}, \qquad
\mathcal{L}_{\text{KD-LM}} = \mathrm{KL}(p^S_{\text{LM}} \Vert p^T_{\text{LM}})_{m_{\text{LM}}}.
$$

Coverage supervision is binary cross-entropy with logits over canonical plan items:

$$
\mathcal{L}_{\text{cov}} = \mathrm{BCEWithLogits}(\mathbf{c}, y_{\text{cov}})_{m_{\text{cov}}}.
$$

Agreement supervision is commit-readiness:

$$
\mathcal{L}_{\text{ready}} = \mathrm{BCE}(\mathbf{r}, y_{\text{ready}})_{m_{\text{ready}}}.
$$

In the current dataset the readiness target $y_{\text{ready}}$ is synthesized at dataset-export time via a trailing-flip prior ($p \in [0.15, 0.25]$), not observed from a live rollback oracle. Replacing this procedural oracle with observed rollback annotations from a live inference run is named future work.

**SNC role under sectional independence.** By construction, plan items are assigned to disjoint stream-owned slots and the training corpus enforces *sectional independence* at the plan level. The gradient pressure on the SNC pathway therefore does not come from informational necessity; it comes from *ownership consistency* -- the coverage and readiness heads are stream-qualified, so SNC must transmit enough information about sibling ownership and dependency state to keep those supervisory signals well-calibrated across streams. Whether SNC can be made *strictly necessary* by relaxing sectional independence (constructing a task subset in which stream $k$'s target cites facts only sibling streams establish) is named future work.

The total training loss is

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{plan}} + \mathcal{L}_{\text{notes}} + 0.5\,\mathcal{L}_{\text{spec}} + \mathcal{L}_{\text{LM-CE}} + \lambda_{\text{KD}}\,\mathcal{L}_{\text{KD-LM}} + \lambda_{\text{cov}}\,\mathcal{L}_{\text{cov}} + \lambda_{\text{ready}}\,\mathcal{L}_{\text{ready}},
$$

with $\lambda_{\text{KD}} = 2.0$, $\lambda_{\text{cov}} = 1.0$, $\lambda_{\text{ready}} = 1.0$ in the canonical configuration.
