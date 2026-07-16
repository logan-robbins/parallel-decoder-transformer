# The right abstraction: one model, one plan, three causal arms

Your system is not “three models in parallel,” and it is not “one model guessing three future tokens.”

It is a **single generative process whose output topology is changed from a chain into a three-column causal lattice**:

[
\begin{matrix}
y_{1,1} & y_{2,1} & y_{3,1}\
y_{1,2} & y_{2,2} & y_{3,2}\
y_{1,3} & y_{2,3} & y_{3,3}\
\vdots & \vdots & \vdots
\end{matrix}
]

Each row is one wall-clock decoding round. Each column is a stateful arm with:

* its own causal history and KV cache;
* a distinct assignment;
* a learned understanding of what the other two arms are responsible for;
* access to a common latent map;
* access to a lagged, dynamically updated coordination bus;
* a physically separate output slot.

The final answer is a deterministic serialization of the columns:

[
Y=\mathcal S_\pi(Y_1,Y_2,Y_3)
]

where (\pi) says, for example, “slot 1 comes first, then slot 2, then slot 3.”

The critical architectural change is therefore not the number of vocabulary heads. It is the **causal graph of generation**.

A working name for the architecture below is a **Forked Causal-Lattice Decoder**.

## Implementation status: July 15, 2026

This document contains both the long-horizon causal-lattice theory and the
smaller falsifiable PDT system now implemented for the first experiment. They
must not be conflated.

The implemented system is a block-synchronized three-stream decoder built on
the frozen, revision-pinned `Qwen/Qwen3-4B-Instruct-2507` trunk. It has a VQ
prompt planner, addressed prompt anchors, per-layer Shared Notes
Cross-Attention, stream adapters, block-end speculation writes, exact
`tau=32`, reveal delay `Delta=1`, and a fixed `2K` last-write-wins notes window.
Training uses exact chat prompts, incremental teacher-forced stream state, hard
CE on every active target token, and dependency-only same-trunk functional KD
at `T=2`. The diagnostic corpus reveals one private register per stream/block
through an exact multi-turn transition; future observations never appear in the
initial prompt. Runtime uses one prompt prefill per stream, consumes each
generated token exactly once, and synchronously publishes each round's boundary
writes.

The typed bus, posterior-plan encoder, copied upper arms, joint token selector,
`HOLD`, sufficiency controller, same-tick arbitration, and learned
coverage/agreement commit policy below are target architecture, not present-day
code. No trained causal-gate, latency, quality, VRAM, or throughput result is
claimed in this document.

---

# 1. The fundamental information problem

A serial language model factors an answer as:

[
p(Y\mid X)=\prod_{g=1}^{N}
p(y_g\mid X,y_{<g})
]

Every token gets the complete realized prefix. That prefix contains two different kinds of information:

1. **Information predictable from the prompt**, such as the topic, requested structure, likely entities, and answer format.
2. **Information created during generation**, such as a chosen name, a particular notation, an intermediate calculation, or the exact wording used earlier.

Your three-arm model cannot give a later span its real earlier prefix because that prefix does not exist yet. Therefore, for every token generated “ahead,” the system must answer:

> Which parts of the missing prefix actually matter for this token, and where will those parts come from?

There are only four legitimate answers:

1. The information was committed in the pre-fork latent map.
2. It arrived through the bus during an earlier parallel round.
3. It is resolved jointly with the other arms during the current round.
4. The arm must wait.

Anything else is an unsupported guess.

## Missing-prefix information

Suppose local token (y_{i,t}) in arm (i) corresponds to global serialized position (g=\pi(i,t)).

Let (U_{i,t}) be the portion of the normal serial prefix that this arm has not observed. The amount of missing information that matters is:

[
\mathcal I_{i,t}
=

I\left(
y_{i,t};
U_{i,t}
\mid
X,
C_i,
M_t,
y_{i,<t}
\right)
]

where:

* (C_i) is the arm’s static contract;
* (M_t) is the shared bus before round (t);
* (y_{i,<t}) is that arm’s own prefix.

Interpretation:

* If (\mathcal I_{i,t}\approx 0), the arm can emit safely.
* If it is small, a small latent summary may be enough.
* If it is large, the arm must receive more information, resolve the dependency jointly, or emit `HOLD`.

That quantity—not token confidence by itself—is the real definition of parallelizability.

---

# 2. The latent map is common information, not merely an outline

A topic-level plan is insufficient. The pre-fork map must contain all decisions that are simultaneously:

* relevant to more than one arm;
* not safely derivable independently;
* needed before the relevant sibling output has been generated.

For example, suppose all three arms mention an invented scientist. If each independently chooses the name, they may choose three names. The arbitrary shared choice must happen before the fork.

A useful generative decomposition is:

[
Z\sim p_\phi(Z\mid X)
]

[
Y_i=f_i(X,Z,U_i)
]

where:

* (Z) contains globally shared choices;
* (U_i) is private randomness belonging only to arm (i);
* the (U_i) variables are independent across arms.

This gives a hard architectural rule:

> Every stochastic decision that must agree across arms must be sampled into (Z), or written to the bus before another arm depends on it.

The map should therefore contain things such as:

* semantic units to be produced;
* ownership of each unit;
* dependency relations;
* canonical entity identities and surface names;
* terminology and notation;
* answer stance or conclusion;
* discourse relations;
* formatting conventions;
* entry and exit conditions for every arm;
* approximate length and progress budgets;
* explicit negative assignments: what each arm must not cover.

## The minimum-common-code objective

Let (Y_1,Y_2,Y_3) denote the completed streams. Their conditional total correlation is:

[
\operatorname{TC}(Y_{1:3}\mid X,Z)
=

## \sum_{i=1}^{3}H(Y_i\mid X,Z)

H(Y_1,Y_2,Y_3\mid X,Z)
]

This measures how much cross-arm dependence remains after providing the plan.

The ideal planner solves approximately:

[
\min_Z I(Z;Y\mid X)
]

subject to:

[
\operatorname{TC}(Y_{1:3}\mid X,Z)\leq\epsilon
]

In words:

> Find the smallest shared latent code that makes the arms nearly conditionally independent.

If (Z) is too small, the arms conflict.

If (Z) is enormous, the planner secretly encodes the whole answer and merely moves serial generation into a latent representation. That might still work computationally if produced in fixed depth, but it is no longer the intended high-level planning mechanism.

The useful region is between those extremes.

---

# 3. The planner is really a causal-graph compiler

The module before the fork should not simply emit three vectors labeled “part one,” “part two,” and “part three.”

It should compile the prompt into a parallel execution graph.

Let:

[
G=(V,E)
]

where each node in (V) is a semantic decision or content unit, and an edge:

[
u\rightarrow v
]

means that (v) requires information established by (u).

The planner then predicts:

[
A\in[0,1]^{|V|\times 3}
]

where (A_{m,i}) is the ownership of semantic unit (m) by arm (i), along with a schedule or readiness condition for every node.

A valid parallel schedule has to ensure that every dependency edge is handled in one of three ways:

1. The source node is completed in an earlier parallel round.
2. The relevant result is precomputed into the global map.
3. The dependent node waits.

That makes the planning objective something like:

[
\min_{\pi,A,Z}
\quad
\max_i \operatorname{Work}(Y_i)
+
\lambda_{\text{cut}}
\sum_{u\rightarrow v}
w_{uv}\mathbf 1[A(u)\neq A(v)]
+
\lambda_Z R(Z)
+
\lambda_H N_{\text{hold}}
]

subject to every required unit being assigned exactly once.

The terms represent:

* **load balance:** avoid one 25-token stream and two 3-token streams;
* **dependency cut cost:** avoid splitting tightly coupled units;
* **plan rate:** do not put the entire answer in (Z);
* **stall cost:** avoid schedules that constantly wait.

This is closer to a compiler partitioning a dependency graph across processors than to an ordinary language-model planning head.

---

# 4. The map should be structured latent state

A single flat vector is a weak representation for this. It entangles ownership, entities, dependencies, formatting, and progress.

Use a set of latent slots:

[
Q=[q_1,\ldots,q_M],
\qquad q_m\in\mathbb R^d
]

with associated heads:

[
A_{m,:}=\operatorname{softmax}(W_Aq_m)
]

[
D_{mn}
======

\sigma\left(
(W_Qq_m)^\top(W_Kq_n)
\right)
]

[
T_m=\operatorname{softmax}(W_Tq_m)
]

where:

* (A) predicts ownership;
* (D) predicts dependencies;
* (T) predicts slot type, such as entity, proposition, boundary, style, or formatting.

The planner can be a fixed-depth set transformer:

[
Q^{r+1}
=

\operatorname{PlanBlock}
\left(
Q^r,
\operatorname{CrossAttn}(Q^r,H_X)
\right)
]

where:

[
H_X=F_\theta(X)
]

is the frozen model’s prompt representation.

This is important: **the planner itself should not generate a textual plan autoregressively**. If it takes 30 serial token steps to create a plan that saves 20 decoding steps, nothing has been gained.

The map should be produced in a fixed number of neural layers, with all latent slots predicted in parallel.

## Static map versus dynamic realization

Separate two concepts:

[
Z^{\text{intent}}
]

and:

[
M_t^{\text{realized}}
]

The static map says what should happen. The dynamic bus says what has actually happened.

The difference:

[
E_t=Z^{\text{intent}}-M_t^{\text{realized}}
]

is the basis of your residual control system.

That distinction is more useful than treating the notes bus as a continuously rewritten plan.

---

# 5. The physical model topology

Three final vocabulary projections are not enough.

If all three heads see one hidden state, they do not possess separate causal trajectories. Over multiple rounds, each arm requires its own:

* local token prefix;
* local position counter;
* KV cache;
* progress state;
* assignment-conditioned transformations.

A suitable topology is:

[
\text{prompt}
\rightarrow
\text{frozen shared representation}
\rightarrow
\text{map compiler}
\rightarrow
\text{fork neck}
\rightarrow
\begin{cases}
\text{arm 1}\
\text{arm 2}\
\text{arm 3}
\end{cases}
\rightarrow
\text{shared vocabulary geometry}
]

## Shared lower trunk, specialized upper arms

Split the transformer at layer (b):

[
F_\theta
=

F^{\text{hi}}*\theta
\circ
F^{\text{lo}}*\theta
]

At parallel round (t), arm (i) first gets a generic feature:

[
u_{i,t}
=

F^{\text{lo}}*\theta
\left(
y*{i,t-1},
KV^{\text{lo}}_{i,t}
\right)
]

The fork neck produces dynamic role conditioning:

[
\Lambda_{i,t}
=

\Gamma_\phi
\left(
C_i,
Z,
M_t,
p_{i,t}
\right)
]

where (p_{i,t}) is progress through the arm’s assignment.

Then:

[
h_{i,t}
=

F^{\text{hi}}*{\theta+\Delta_i}
\left(
u*{i,t},
KV^{\text{hi}}*{i,t};
\Lambda*{i,t}
\right)
]

Here:

* (\theta) remains frozen;
* (\Delta_i) is an arm-specific adapter or LoRA path;
* (\Lambda_{i,t}) is dynamic map-and-bus conditioning;
* each arm has a separate KV cache.

The vocabulary head can remain shared:

[
\ell_{i,t}
=

W_{\text{vocab}}h_{i,t}
]

This keeps all three arms in the same lexical geometry.

## Where should the fork occur?

If the fork happens only at the vocabulary head, it is too late. The arms can relabel essentially the same representation, but they cannot develop deeply different reasoning trajectories.

If the whole model is copied three times, cost and memory approach three complete decoders.

A practical starting point is:

* share approximately the lower 70–80% of layers;
* specialize the upper 20–30% through arm-specific adapters or copied top blocks;
* inject map and bus state at the fork and repeatedly inside the specialized portion.

The exact point is empirical. The key requirement is that enough nonlinear depth remains after the fork for role conditioning to alter the computation, not merely the final token ranking.

---

# 6. The sidecar should impute missing causal state

The arm does not truly need all absent text. It needs the effect that the absent text would have had on its next-token distribution.

That gives a more exact interpretation of the sidecar.

Let (U_{i,t}) be the missing serial prefix. Somewhere inside an ordinary transformer, that prefix induces layer-specific attention keys, values, and hidden-state changes.

The sidecar should approximate a sufficient statistic of those effects:

[
\left(
\widetilde K^{\text{missing}}*{i,\ell,t},
\widetilde V^{\text{missing}}*{i,\ell,t}
\right)
=

\Gamma_\ell(Z,M_t,C_i,p_{i,t})
]

The arm attends to these dynamic virtual ancestors:

[
\operatorname{Attn}
\left(
Q_{i,\ell,t},
[
K^{\text{local}};
\widetilde K^{\text{missing}}
],
[
V^{\text{local}};
\widetilde V^{\text{missing}}
]
\right)
]

The point is not simply that prefix KV injection is convenient. The deeper claim is:

> The sidecar is a learned causal-state imputer for the prefix that does not yet physically exist.

For an ideal imputed state (S^*_{i,t}):

[
y_{i,t}
\perp
U_{i,t}
\mid
X,y_{i,<t},S^*_{i,t}
]

The planner and bus jointly approximate (S^*_{i,t}).

That is the most precise function-level description of the sidecar.

---

# 7. “Think like one arm of three” requires a team model

Giving each arm an integer ID is not enough. The arm must model:

* its own responsibility;
* sibling responsibilities;
* sibling progress;
* content it must leave untouched;
* dependencies it is waiting on;
* what later serialization will make its output mean.

Each contract should therefore look conceptually like:

[
C_i=
[
C_i^{\text{owned}},
C_i^{\text{forbidden}},
C_i^{\text{entry}},
C_i^{\text{exit}},
C_i^{\text{dependencies}},
C_i^{\text{length}},
C_i^{\text{position}}
]
]

The arm should also maintain a belief about the other arms:

[
\widehat B_{i,t}^{(-i)}
=

\operatorname{SiblingPredictor}*i(h*{i,t},M_t,C_{1:3})
]

After the other arms write to the bus, compare that belief with what actually happened:

[
r^{\text{team}}_{i,t}
=

\left|
\widehat M^{(-i)}_{i,t+1}
-------------------------

M^{(-i)}_{t+1}
\right|
]

This residual trains a literal internal version of:

> I am one component of a three-component computation; what did I expect the other components to establish?

Useful training perturbations include:

* randomly swapping lane IDs while preserving contracts;
* changing sibling assignments while holding the local assignment fixed;
* dropping one sibling;
* slowing a sibling with artificial `HOLD` steps;
* duplicating assignments and training the ownership residual to detect the collision;
* removing a semantic unit from all assignments and training coverage detection.

The output should follow the contract, not merely learned stereotypes such as “arm 1 always writes the introduction.”

---

# 8. The bus must be typed and permissioned

A single recurrent vector updated by all three arms is likely to become unstable or degenerate. It also gives no guarantee that an arm will not overwrite something important.

Use a typed bus:

[
M_t=
[
M^{\text{plan}},
M^{\text{symbols}},
M^{\text{coverage}},
M^{\text{dependencies}},
M^{\text{boundaries}},
M^{\text{uncertainty}},
M^{1},
M^{2},
M^{3}
]
]

Different slot classes should have different update laws.

### Plan slots

Read-only after the fork.

### Symbol slots

Write-once or versioned commitments for exact shared choices:

* entity name;
* acronym;
* variable notation;
* units;
* heading convention.

A purely continuous embedding may not preserve exact spelling reliably. A robust implementation can use a hybrid latent-symbol slot containing an embedding plus an internal token or subword identifier. It remains internal coordination; it simply acknowledges that some shared information is inherently discrete.

### Coverage slots

Monotonic state such as:

[
\text{unclaimed}
\rightarrow
\text{active by arm }i
\rightarrow
\text{completed by arm }i
]

These should not be freely overwritten.

### Dependency slots

Boolean or probabilistic readiness flags.

### Boundary slots

The realized entry and exit state of each stream.

### Uncertainty slots

Predicted need-to-know scores, contradictions, and requests to wait.

## Bus update

Each arm emits a compact write:

[
w_{i,t}
=

W_i
\left(
h_{i,t},
e(y_{i,t}),
C_i,
p_{i,t}
\right)
]

For slot (s):

[
M_{t+1,s}
=

(1-g_{s,t})M_{t,s}
+
g_{s,t}
\sum_i
\alpha_{i,s,t}W_sw_{i,t}
]

where the write gates and permissions depend on slot type and arm ownership.

Crucially, all arms read (M_t), generate simultaneously, and write (M_{t+1}). No arm reads another arm’s new write during the same ordinary forward phase. That one-round delay prevents a circular computational dependency.

---

# 9. One complete decoding round

At wall-clock round (t), the system performs the following computation.

## 9.1 Read

Known:

[
X,\quad Z,\quad C_i,\quad M_t,\quad y_{i,<t},\quad KV_{i,t}
]

Needed:

A lane-specific approximation of the missing causal prefix.

Transformation:

[
r_{i,t}
=

\operatorname{BusRead}*i(M_t,C_i,p*{i,t})
]

[
\Lambda_{i,t}
=

\Gamma_i(Z,C_i,r_{i,t})
]

## 9.2 Parallel arm forward

All three arms execute in one packed or batched forward operation:

[
h_{i,t}
=

\operatorname{Arm}*i
\left(
y*{i,t-1},
KV_{i,t},
\Lambda_{i,t}
\right)
]

Each produces:

[
\ell_{i,t}
\quad
\text{token logits}
]

[
w_{i,t}^{\text{pre}}
\quad
\text{provisional bus message}
]

[
\rho_{i,t}
\quad
\text{residual and risk predictions}
]

[
q_{i,t}^{\text{hold}},
q_{i,t}^{\text{eos}}
\quad
\text{control probabilities}
]

## 9.3 Same-tick joint arbitration

Independent arm sampling cannot coordinate arbitrary same-round choices. But making arm 2 wait for arm 1 would destroy simultaneity.

Instead, each arm supplies a small candidate set:

[
\mathcal A_{i,t}
=

\operatorname{TopR}(\ell_{i,t})
\cup
{\texttt{HOLD},\texttt{EOS}}
]

A lightweight joint selector scores candidate triples:

[
a_t^*
=====

\arg\max_{a_i\in\mathcal A_{i,t}}
\left[
\sum_i \ell_{i,t}[a_i]
+
J_\omega(a_1,a_2,a_3,S_t)
\right]
]

A pairwise version of (J) is:

[
J_\omega
=

\sum_{i<j}
e(a_i)^\top
W_{ij}(M_t)
e(a_j)
+
\Phi_{\text{plan}}(a_{1:3},Z,C_{1:3})
]

With four token candidates per arm, there are only:

[
4^3=64
]

triples to evaluate before adding `HOLD` and `EOS`.

This gives same-round coupling without imposing a serial order among the arms.

The selector can prevent things such as:

* two streams both beginning with “First”;
* incompatible punctuation at planned joins;
* contradictory surface forms selected in the same round;
* all streams choosing language associated with the same plan unit.

It does not replace the map. It handles residual local coupling that was not worth encoding globally.

## 9.4 Commit gate

Before external emission, evaluate whether the tuple is safe:

[
c_{i,t}
=

\operatorname{CommitGate}
\left(
a_t^*,
\rho_{1:3,t},
M_t,
C_{1:3}
\right)
]

An unsafe arm emits `HOLD` internally and produces no token in its visible slot.

The external interface can run with a one-token commit buffer:

1. Compute candidate tuple.
2. Run joint and residual checks.
3. Release committed tokens.
4. Advance.

This still streams token by token while preventing already displayed tokens from requiring immediate retraction.

## 9.5 Bus write

After selection:

[
w_{i,t}
=

\operatorname{MessageCompressor}
\left(
h_{i,t},
a_{i,t}^*,
C_i,
p_{i,t}
\right)
]

[
M_{t+1}
=

\operatorname{BusUpdate}
\left(
M_t,
w_{1,t},
w_{2,t},
w_{3,t}
\right)
]

The selected token becomes the next input token for that arm, and its KV cache advances.

---

# 10. The probabilistic model is autoregressive over rows, not over text positions

The full model becomes:

[
p(Y,Z\mid X)
=

p_\phi(Z\mid X)
\prod_{t=1}^{T}
p_\psi
\left(
y_{1,t},
y_{2,t},
y_{3,t}
\mid
X,Z,Y_{:,<t}
\right)
]

This is still causal, but the causal unit is a row of up to three tokens.

A useful factorization is:

[
p_\psi(y_{1,t},y_{2,t},y_{3,t}\mid S_t)
\propto
\left[
\prod_i p_i(y_{i,t}\mid S_t)
\right]
\exp J_\omega(y_{1,t},y_{2,t},y_{3,t},S_t)
]

The arm distributions do most of the work. The joint energy represents residual same-row dependence.

The same-row total correlation is:

[
\operatorname{TC}_t
=

## \sum_i H(y_{i,t}\mid S_t)

H(y_{1,t},y_{2,t},y_{3,t}\mid S_t)
]

The planner and bus should drive this down. The joint selector handles what remains.

This is a cleaner mathematical target than saying the arms should merely “coordinate.”

---

# 11. Residual checks should form a closed-loop controller

The sidecar should not be passive context. It should be a reference trajectory and feedback controller.

Let:

[
z_i^*(p_{i,t})
]

be the intended latent state for arm (i) at progress (p_{i,t}).

An observer projects the actual arm state back into plan space:

[
\widehat z_{i,t}
=

O_i(h_{i,t})
]

The latent control error is:

[
e_{i,t}
=

## z_i^*(p_{i,t})

\widehat z_{i,t}
]

A corrective signal is generated:

[
u_{i,t+1}
=

C_\omega(e_{i,t},M_t,C_i)
]

and injected into the next round or into later layers of the current branch:

[
h'_{i,t+1}
=

h_{i,t+1}
+
g_{i,t+1}u_{i,t+1}
]

This is a closed-loop system:

[
\text{desired plan}
\rightarrow
\text{arm behavior}
\rightarrow
\text{observed realization}
\rightarrow
\text{residual}
\rightarrow
\text{correction}
]

## The most important residual: information sufficiency

During training, the frozen serial model can provide an oracle distribution:

[
p_{\text{serial}}
\left(
\cdot
\mid
X,Y_{<g}
\right)
]

The parallel arm produces:

[
p_{\text{arm}}
\left(
\cdot
\mid
X,Z,M_t,Y_{i,<t}
\right)
]

Define:

[
d_{i,t}
=

D_{\mathrm{KL}}
\left(
p_{\text{serial}}
;|;
p_{\text{arm}}
\right)
]

Train a residual head to predict (d_{i,t}) without seeing the missing prefix:

[
\widehat d_{i,t}
=

R_{\text{need}}
(h_{i,t},M_t,C_i)
]

At inference:

* low (\widehat d): emit;
* moderate (\widehat d): strengthen bus conditioning or use the joint selector;
* high (\widehat d): `HOLD`;
* persistently high (\widehat d): collapse this dependency back to serial execution.

This turns “Do I know enough?” into a supervised quantity.

## Other useful residuals

**Ownership residual**

[
r^{\text{owner}}_{i,t}
=

1-
\sum_{m:A_{m,i}=1}
p(m\mid h_{i,t})
]

**Coverage residual**

Difference between intended and completed plan slots.

**Team-prediction residual**

Difference between what the arm expected siblings to establish and what their messages actually report.

**Symbol residual**

Mismatch between canonical symbol slots and local lexical choices.

**Boundary residual**

Mismatch between realized exit state and planned exit invariant.

**Bus surprise**

[
r^{\text{bus}}_{i,t}
=

\left|
\widehat M_{i,t+1}
------------------

M_{t+1}
\right|
]

These should affect the next computation, not merely be logged after generation.

---

# 12. Training from a frozen serial model

The central training problem is that natural language answers are not uniquely determined. The planner cannot learn a single averaged map for many incompatible valid answers.

## Lock the trunk before defining the teacher

The first implementation uses
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
at revision `cdbee75f17c01a7cc42f958dc650907174af0554`, not the base
checkpoint. The schema, YAML, loader, tokenizer preflight, and checkpoint
identity all lock that exact pair. It is a 4.0B, 36-layer causal decoder with the same
Qwen3 layer topology already instrumented by this repository. It is also a
non-thinking model, so it does not add a hidden serial chain-of-thought to a
system whose purpose is to shorten the serial critical path. The model card
lists Apache-2.0 licensing, 32 query heads, 8 KV heads, and a native 262,144
token context.

This choice is deliberately narrower than "use the newest 4B model":

* `Qwen3-4B-Base` is appropriate for pretraining or broad fine-tuning, but a
  frozen trunk plus a coordination sidecar should inherit instruction
  following and fluent answer formation rather than relearn them.
* `Qwen3-4B-Thinking-2507` adds long internal serial generation and therefore
  confounds the latency thesis.
* `Qwen3.5-4B-Base` uses a different hybrid stack of Gated DeltaNet and sparse
  full-attention layers with a much larger vocabulary. It is not a drop-in
  trunk for the current `Qwen3DecoderLayer` instrumentation, and changing both
  the coordination mechanism and the trunk architecture in the first causal
  experiment would make a null result uninterpretable. Its
  [official model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) confirms
  that architectural break.
* `Qwen3-1.7B` makes a negative result ambiguous because the frozen trunk may
  be the bottleneck. `Qwen3-8B` increases activation and sidecar cost before
  the mechanism has earned that scale. Four billion parameters is the first
  defensible capability/compute point.

## Two teachers with different jobs

"Teacher model" must not refer to one undifferentiated system. PDT needs two
teachers at different stages.

**Privileged-context functional teacher.** Use the exact same frozen
`Qwen3-4B-Instruct-2507` checkpoint, with the sidecar disabled and the complete
serialized prefix made visible. The student is the same frozen checkpoint with
only its local prefix, plan, and causally visible bus notes. This is context
distillation: the teacher has information that the student must learn to
replace with a compact state. The setup is closely aligned with
[Learning by Distilling Context](https://arxiv.org/abs/2209.15189), which
trains a model to retain behavior available under richer context when that
context is removed.

This path is implemented. Retokenization stores one complete privileged prompt
per block under schema `qwen3-instruct-temporal-chat-v2`, using
`add_generation_prompt=True` and `enable_thinking=False`. Each prompt contains
all private observations and all completed prior blocks. The trainer clears
every sidecar runtime context, runs the identical frozen trunk, detaches its
logits, and applies forward KL only on annotated dependency tokens at exactly
`T=2`; hard CE remains active on every target token. There is no hidden-state
matching objective.

**Offline response teacher.** After the causal mechanism passes, use
[`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
to generate structured natural-language targets. It is Apache-2.0, non-thinking,
and has 30.5B total parameters with 3.3B activated per token. It should provide
completed three-part answers and explicit source/ownership metadata, not online
hidden-state targets. This is sequence-level distillation in the sense of
[Kim and Rush](https://arxiv.org/abs/1606.07947): train on validated sequences
sampled from a stronger teacher.

Do not match hidden states from the 30B teacher. Its MoE representation is not
identifiable with the dense 4B trunk, and the planner could learn an arbitrary
projection that has nothing to do with missing-prefix sufficiency. The larger
teacher supplies language targets only; the same-trunk teacher supplies the
causal functional target.

Use a posterior-plan/prior-plan setup.

## Posterior map encoder

During training, let a teacher map encoder see the completed target:

[
q_\eta(Z\mid X,Y,\pi)
]

It extracts the shared commitments required to reconstruct the three streams.

## Inference-time planner

The real planner sees only the prompt:

[
p_\phi(Z\mid X)
]

Train it to match the posterior plan distribution:

[
D_{\mathrm{KL}}
\left(
q_\eta(Z\mid X,Y,\pi)
;|;
p_\phi(Z\mid X)
\right)
]

For an open-ended prompt, (p_\phi) should be stochastic. It samples one globally consistent plan before branching. Otherwise it may average together incompatible plans.

## Overall objective

A useful objective is:

[
\begin{aligned}
\mathcal L
={}&
\mathcal L_{\text{token}}
+
\lambda_{\text{serial}}
\mathcal L_{\text{serial-KL}}
+
\beta
D_{\mathrm{KL}}(q_\eta|p_\phi)
\
&+
\lambda_{\text{state}}
\mathcal L_{\text{hidden-state}}
+
\lambda_{\text{owner}}
\mathcal L_{\text{ownership}}
+
\lambda_{\text{coverage}}
\mathcal L_{\text{coverage}}
\
&+
\lambda_{\text{dependency}}
\mathcal L_{\text{dependency}}
+
\lambda_{\text{team}}
\mathcal L_{\text{sibling-prediction}}
+
\lambda_{\text{res}}
\mathcal L_{\text{residual-calibration}}
\
&+
\lambda_{\text{latency}}
\mathcal L_{\text{rounds}}
\end{aligned}
]

## Preventing latent answer smuggling

The posterior map sees (Y), so without restrictions it can encode the answer verbatim.

Use several constraints:

* limited number of map slots;
* vector quantization or categorical plan codes;
* noise and dropout;
* rate penalty through the posterior/prior KL;
* typed write heads;
* random paraphrase targets sharing the same semantic map;
* a probe measuring how accurately exact target text can be decoded from (Z);
* a requirement that one (Z) support multiple valid lexical realizations.

The plan should retain cross-arm commitments while discarding arm-private wording.

---

# 13. A concrete curriculum

## Stage 1: Learn target partitions

Split completed answers into three contiguous or section-level spans.

Initially, use easy partitions:

* fixed-format outputs;
* three requested sections;
* enumerations;
* deterministic sequences;
* multi-facet factual answers.

Later, estimate dependency edges between sentences or semantic units and partition them by balancing length against dependency cuts.

One way to estimate an edge (u\rightarrow v) is to remove or compress (u) from the serial prefix and measure the KL change in the teacher distribution at (v).

## Stage 2: Oracle-map reconstruction

Use (q_\eta(Z\mid X,Y)) to create a high-quality map.

Train the arm adapters and state imputer to reconstruct their assigned spans while seeing only:

* the prompt;
* their own previous tokens;
* the oracle map;
* causally legal bus messages.

This tests whether the proposed sidecar has enough representational capacity.

## Stage 3: Serial-state distillation

For every annotated dependency token, run two views of the same frozen trunk:

* the teacher sees the full causally serialized prefix, including the sibling
  state that would have existed in ordinary serial decoding;
* the student sees only the shared prompt, its local prefix, its contract, and
  the notes legal under the configured reveal delay.

The primary functional target is the teacher's next-token distribution:

[
\mathcal L_{\text{context-KD}}
=

T^2
D_{\mathrm{KL}}
\left(
p_{\text{full-prefix}}^T
\;\|\;
p_{\text{PDT}}^T
\right)
]

Start with (T=2). Apply this KL on dependency spans and retain ordinary hard
cross-entropy on every target token. Do not activate raw hidden-state MSE in
the first CUDA run. Serial and parallel tokens have different positions and
different available ancestors, so exact hidden-state identity is not a
well-posed requirement; next-token behavior is the relevant sufficient
statistic.

Only after teacher-forced context KD passes the gate should free-running
prefixes enter the distillation distribution. At that point, serialize the
student's committed rows, ask the same privileged-context teacher to score the
student-generated continuation, and train on those mistakes. This follows the
motivation of
[Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649):
student-generated sequences reduce the train/inference prefix mismatch.

## Stage 4: Train the prompt-only map compiler

Replace the posterior map with (p_\phi(Z\mid X)), first gradually and then completely.

Use stochastic map sampling for open-ended prompts.

## Stage 5: Enable learned bus writes

Start with teacher summaries or oracle coverage state. Gradually replace them with messages produced by the arms.

Never provide future sibling information during training. Otherwise the model learns a noncausal bus that cannot exist at inference.

## Stage 6: Free-running three-slot generation

Run all arms from their own predictions.

Optimize:

* final answer quality;
* contradiction rate;
* omission;
* duplication;
* number of parallel rounds;
* number of `HOLD` events;
* repair frequency;
* residual calibration.

The larger 30B response teacher is still offline in this stage. Loading it in
the student training process would consume memory without providing the
same-prefix functional comparison that the mechanism needs.

---

# 14. The interface is part of the model

The output should not pretend to be one ordinary token stream.

The serving API should emit events like:

```text
{
  slot: 2,
  local_position: 14,
  token: "...",
  committed: true,
  plan_region: 7
}
```

The UI displays three independently growing regions.

For the alphabet:

```text
round 1:  A  |  K  |  U
round 2:  B  |  L  |  V
round 3:  C  |  M  |  W
round 4:  D  |  N  |  X
round 5:  E  |  O  |  Y
round 6:  F  |  P  |  Z
round 7:  G  |  Q  |  EOS
round 8:  H  |  R  |  HOLD
round 9:  I  |  S  |  HOLD
round 10: J  |  T  |  HOLD
```

The final serializer concatenates:

```text
ABCDEFGHIJ + KLMNOPQRST + UVWXYZ
```

The model completes 26 output tokens in 10 decoding rounds, plus the fixed-depth planning computation.

`HOLD` is not shown to the user. It is a scheduler action.

## Immutable versus provisional streaming

Externally displayed text should generally be immutable. Use one of:

* joint arbitration before release;
* a one-token commit buffer;
* a very short hidden provisional horizon;
* visible revision semantics in the interface.

The first two preserve the ordinary expectation that streamed text will not be rewritten.

---

# 15. Where the alphabet is easy and prose becomes hard

For the alphabet, the map can encode:

[
C_1=(A,J,10,\text{successor rule})
]

[
C_2=(K,T,10,\text{successor rule})
]

[
C_3=(U,Z,6,\text{successor rule})
]

Once those contracts exist, essentially no cross-arm communication is needed. The task has:

* low uncertainty;
* an obvious partition;
* no arbitrary lexical choices;
* no later computation depending on an earlier result.

For a story:

> Alice entered the room. She saw Bob. He handed her the documents.

A three-sentence split requires the global map to commit before the fork:

* protagonist entity ID;
* surface name “Alice”;
* gender or pronoun agreement;
* Bob’s existence and name;
* the document object;
* the temporal relation among events;
* the fact that “he” in arm 3 refers to Bob.

If the first arm is allowed to decide between “Alice,” “Dr. Chen,” and “the detective” privately, the later arms cannot begin safely.

The planner must either make that lexical decision globally or the later arms must wait for a symbol-bus update.

For a calculation, the limit is sharper:

> Arm 1 computes a value. Arm 2 explains the implications of that value.

If the value is not predictable before arm 1 computes it, arm 2 cannot genuinely skip ahead. The correct causal compiler will schedule arm 2 to `HOLD`.

The model must learn not only how to parallelize, but also when parallelization is invalid.

---

# 16. Information-theoretic limits

## Bus bandwidth

If the bus can communicate only (B) effective bits per round, it cannot remove more than (B) bits of uncertainty from a sibling’s next decision.

A continuous vector does not eliminate this limit in practice. Noise, dimensionality, regularization, precision, and training determine its effective capacity.

For the implemented first system, physical capacity is concrete: a dynamic
note is a dense 256-dimensional BF16 vector, hence 512 bytes or 4096 transmitted
bits per producer/write. The exact-entropy relay carries 18 fresh payload bits
per dependency block, so even perfect delivery has physical efficiency

[
\eta_{\max}=\frac{18}{4096}=0.00439453125.
]

The VQ codebook quantizes planner slots only. It does not quantize dynamic
notes, and no current 32-bit product-VQ note transport exists. Effective
information must be measured separately through dependency-token CE and KL;
the dense vector's nominal bit count is not evidence that those bits are used.

If a stream needs a high-entropy result produced by another stream, the architecture must:

* allocate more bus capacity;
* precompute it in the plan;
* delay the consumer;
* or accept error.

## Sequential critical path

The planner creates a dependency DAG. Let (L_{\text{crit}}) be its longest dependency chain and (W) its total work.

Even with unlimited arms:

[
T_{\min}\geq L_{\text{crit}}
]

With three arms:

[
T_{\min}
\geq
\max\left(
L_{\text{crit}},
\frac{W}{3}
\right)
]

The architecture can exploit width in the dependency graph. It cannot parallelize a true causal chain merely by relabeling its positions.

## The planning no-free-lunch condition

The plan must move shared decisions earlier without reproducing the full serial computation.

A useful architecture has:

[
\text{fixed-depth plan cost}
\ll
\text{serial steps eliminated}
]

If the latent planner internally performs the same sequential reasoning needed for the full answer, the design may still change the output interface, but it will not deliver meaningful skip-ahead acceleration.

---

# 17. Runtime speed condition

Let:

* (N=\sum_i n_i) be total visible output tokens;
* (c_1) be the cost of one ordinary serial decode step;
* (c_3) be the cost of one three-arm packed decode round;
* (T_{\text{plan}}) be planning cost;
* (h) be additional hold rounds;
* (r) be repair rounds.

Then:

[
T_{\text{serial}}
\approx
Nc_1
]

[
T_{\text{parallel}}
\approx
T_{\text{plan}}
+
\left(
\max_i n_i+h+r
\right)c_3
]

and:

[
\operatorname{Speedup}
=

\frac{Nc_1}
{
T_{\text{plan}}
+
(\max_i n_i+h+r)c_3
}
]

For three balanced streams:

[
\max_i n_i\approx \frac N3
]

but the speedup is not automatically (3\times), because (c_3) may exceed (c_1). The system benefit depends on whether three simultaneous query tokens can reuse weight movement and hardware capacity efficiently enough to offset the larger KV and compute workload.

This must be measured as wall-clock latency, not inferred from token counts.

---

# 18. Failure modes that matter

## Plan collapse

All three contracts encode approximately the same obvious introductory content.

Treatment:

* exact-once ownership loss;
* negative contracts;
* plan-slot assignment constraints;
* duplication residuals;
* counterfactual sibling assignments.

## Latent answer copying

The posterior map stores nearly the entire target.

Treatment:

* rate limiting;
* quantization;
* paraphrase invariance;
* exact-text leakage probes;
* fixed-depth prompt-only prior.

## Bus echo

The three arms repeatedly copy the same bus summary and converge toward identical output.

Treatment:

* typed write permissions;
* private versus shared slots;
* ownership-conditioned read masks;
* no free-form all-to-all token attention.

## False parallelism

A dependent arm guesses rather than waiting.

Treatment:

* serial-KL sufficiency target;
* calibrated `HOLD` head;
* explicit dependency supervision;
* adversarial examples where the prerequisite result is unpredictable.

## Semantic clock mismatch

Token 12 in arm 1 and token 12 in arm 2 need not represent comparable progress.

Treatment:

* semantic progress cursors rather than only local token positions;
* plan-slot completion heads;
* asynchronous `HOLD`;
* bus updates indexed by semantic state as well as round.

## Instability from recurrent feedback

A bus error alters all arms, which then amplify the error in their next writes.

Treatment:

* one-round-lagged writes;
* bounded update gates;
* write-once commitments;
* trust gates initialized near zero;
* bus dropout during training;
* spectral or norm constraints on the updater.

## Irreversible streaming error

A problem is detected after a token was displayed.

Treatment:

* same-tick joint arbitration;
* short internal commit buffer;
* strong sufficiency calibration;
* repair inside the affected slot rather than rewriting all three.

---

# 19. A falsifiable first implementation

I would not begin with unrestricted essays. The first system should isolate whether future causal state can actually be predicted.

## Model

The first causal experiment is now fixed:

* Frozen `Qwen/Qwen3-4B-Instruct-2507` decoder trunk.
* Three persistent streams with separate local histories and KV caches.
* The current 12 instrumented Qwen3 layers, each with SNC and three
  stream-specific bottleneck adapters.
* Shared frozen vocabulary head.
* Sixteen VQ planner slots, `d_notes = 256`, block size `tau = 32`, and reveal
  delay `Delta = 1`.
* One fixed addressed `2K` notes window at every SNC read: K immutable prompt
  anchors followed by the latest eligible dynamic write from each producer.
  Dynamic slots use last-write-wins replacement rather than accumulating an
  ever-growing FIFO. With K=3 the window shape is `(B, 6, 256)`.
* Planner-produced snapshot 0 followed by one learned speculation write per
  stream per block.
* Teacher-forced block rollout first; free-running rows only after the
  dependency-selective ablation gate passes.

A local parameter audit of the current configuration reports exactly
401,325,095 trainable parameters: 133,704,707 in canonical sidecar heads and
22,301,699 in each of 12 instrumented layers. That is substantial adapter
training, not a tiny probe.
The FP32 standalone sidecar heads and BF16 trunk-resident SNC/adapters cross
dtype boundaries explicitly, while token CE and functional KD reduce in FP32.
A local real-checkpoint MPS round verified 4,022,468,096 frozen base parameters,
the exact trainable count above, disjoint parameter ownership, finite planner
logits, live KV caches, and one emitted token from every stream. This is a
forward contract, not training evidence.
The frozen 4B trunk still avoids training a language model from scratch, but
the optimizer and activation budget justify an 80GB CUDA device.

The top-4 tuple selector, external commit buffer, adaptive `HOLD`, typed bus
expansion, copied upper blocks, posterior-plan encoder, sufficiency controller,
and token-row arbitration remain later architecture stages. Coverage and
agreement source modules remain standalone scaffolding, but the canonical
Sidecar does not instantiate, checkpoint, or optimize them. Runtime publishes
notes without computing agreement or rollback and exposes no corresponding
result fields. Structured runtime requires the same exact per-block private
transition rows as training and consumes them only after synchronous prior-block
publication. These features must not be described as implemented evidence.

## Dataset ladder

**Gate A: exact-entropy synthetic dependency.** Use the repository's
programmatic cross-stream register relay. Each stream receives a private
register refreshed every block and must reproduce its neighbor's delayed
register. With 64 codewords and three slots, each dependency block contains
exactly 18 fresh cross-stream bits. Generate 32 examples for overfit/gradient
tests, 1,000 for the scale gate, then 20,000 train, 2,000 validation, and 2,000
rho-zero null examples with disjoint seeds.

Temporal visibility is part of the record contract. The initial private prompt
contains only block 0's register. Before each later block, a canonical Qwen
multi-turn transition closes the prior assistant turn, reveals exactly that
block's new private register, and opens the next assistant turn. Teacher block
m sees observations only through m and completed targets only before m. A
legacy full eight-block private log is rejected because one dense note could
otherwise front-load every future payload and defeat the `Delta=1` timing test.

This corpus, not history or generic world knowledge, is the first training
data. If a fact is present in pretraining or visible to every stream, the
receiver can solve the task independently and a good score does not identify
the bus as the cause. Random private state removes that ambiguity.

**Gate B: prompt-only snapshot routing.** Use programmatically generated shared
snapshots with randomized labels and an oracle segment/dependency graph. This
is the first planner test. The stream-local diagnostic does not become the
headline PDT result until `planner -> plan_notes_proj -> snapshot 0` is
load-bearing under a single shared prompt.

**Gate C: grounded natural transfer.** Only after Gates A and B pass, build a
50,000-example natural corpus from two attributable sources:

* 40,000 HotpotQA training examples, preserving the supplied passages,
  question, answer, and supporting-fact IDs. HotpotQA supplies natural
  multi-hop questions with sentence-level evidence and is released under
  CC BY-SA 4.0 ([dataset site](https://hotpotqa.github.io/)).
* Up to 10,000 English root prompts from OpenAssistant OASST1 for open-ended
  instruction and prose diversity. OASST1 is released under Apache-2.0
  ([dataset license](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE)).

Have the 30B offline response teacher emit a validated record containing one
complete answer, exactly three ordered stream targets, ownership metadata,
dependency edges, and citations back to source IDs. Reject records with answer
mismatch, missing supporting facts, duplicated ownership, empty streams, or
serialization mismatch. Retain source IDs and license metadata in every row.

World knowledge and history are therefore transfer content, not causal proof.
The frozen trunk already contains broad knowledge; the sidecar data should
teach partitioning, delayed communication, coverage, and safe serialization.

## Measurements

The formulas below are the evaluation target. They have not been measured on a
trained PDT checkpoint. Local smoke tests establish tensor, cache, timing,
checkpoint, and metric-aggregation contracts only; they do not establish
latency, contradiction, duplication, `HOLD`/repair behavior, plan-bandwidth
curves, or end-to-end causal CE improvements.

The decisive metrics are:

[
\text{quality at equal wall-clock latency}
]

[
\text{maximum stream length}
]

[
\text{cross-stream contradiction rate}
]

[
\text{duplication and omission}
]

[
\text{HOLD and repair frequency}
]

[
\text{serial-to-parallel KL}
]

[
\text{sufficiency-residual calibration}
]

[
\text{bits or slots in the plan}
]

The strongest diagnostic curve would be:

> Final quality and real latency as a function of plan bandwidth.

That would show whether a compact common code genuinely removes sequential dependence or whether the plan must approach answer-sized capacity.

---

# 20. The literature boundary

Your narrower claim is defensible: I did not find a demonstrated system combining the complete architecture above.

There are now nearby systems, so I would avoid the absolute statement that no related parallel model exists.

A March 2026 arXiv proposal called **Parallel Decoder Transformer** describes a frozen trunk, prompt-time planner, latent workspace, stream adapters, and synchronized block commits. It is unusually close in vocabulary and motivation, although it works through provisional blocks and presents core continuation-sufficiency evaluation as future work rather than implementing the token-by-token causal-lattice mechanism derived here. ([arXiv][1])

The May 2026 **Multi-Stream LLM** work is a real multi-stream implementation: it predicts one token in each stream per forward pass using stream-specific positions and a cross-stream causal mask. Its streams are principally an I/O and role format. It does not use a pre-fork semantic dependency compiler, branch-specific upper arms, a typed latent realization bus, or serial-KL information-sufficiency control. ([arXiv][2])

**Multiverse** implements model-directed Map, Process, and Reduce stages and reports parallel inference, but its topology is parallel reasoning branches followed by synthesis, rather than three continuously visible output arms coordinated token by token through an internal shared latent state. ([arXiv][3])

The exact differentiating bundle is:

1. A fixed-depth latent compiler that predicts a semantic dependency graph.
2. Global stochastic commitments sampled before the fork.
3. A physical fork into three stateful, role-fine-tuned causal arms.
4. Separate KV caches and native output slots.
5. Dynamic per-layer imputation of missing causal-prefix effects.
6. A typed, permissioned, per-token latent bus.
7. Same-tick joint token arbitration.
8. A learned information-sufficiency residual trained against the serial model.
9. `HOLD` as a first-class causal action.
10. Direct token-by-token streaming from all three slots without a mandatory reduce phase.

That complete object is substantially different from multiple prediction heads, externally launched agents, branching reasoning traces, or blockwise speculative decoding.

# 21. July 2026 execution decision

The research decision is no longer open-ended. Start in this order.

1. **Completed: align the frozen trunk and prompts.** YAML, schema, adapter,
   local preflight, and checkpoint identity pin
   `Qwen/Qwen3-4B-Instruct-2507` at revision
   `cdbee75f17c01a7cc42f958dc650907174af0554`. Canonical planner, private-stream,
   and per-block privileged prompts preserve the Instruct chat template.
2. **Completed: implement the same-trunk functional teacher.** The teacher sees
   the full serialized block prefix with every sidecar context disabled;
   dependency-only forward KL uses `T=2`, alongside hard CE on all active
   tokens.
3. **Completed locally: lock timing, windows, and persistence contracts.**
   Retokenized blocks are exactly 32 tokens; runtime and training use
   `Delta=1`, fixed `2K` addressed LWW windows, synchronous writes, and strict
   versioned checkpoints. Paired causal metrics aggregate raw token sums and
   counts rather than batch means.
4. **Pass the 32-example gate locally or in a short CUDA session.** Require
   nonzero gradients, measurable source-note mutation, and dependency-selective
   gate-zero damage. An overfit run that cannot move the randomized payload is
   an architecture bug, not a data-scaling problem.
5. **Rent one H100 SXM 80GB for the 1,000-example gate.** The current
   [Runpod pricing page](https://www.runpod.io/pricing) lists this device at
   $2.99/hour as of July 15, 2026. Use at least 200GB persistent storage,
   BF16, PyTorch SDPA, batch size 1, differentiable incremental KV caches, and
   the repository's gradient accumulation. Hugging Face gradient checkpointing
   is intentionally disabled because it suppresses reusable caches in training
   mode. Measure peak allocated VRAM and examples/second
   before estimating the 20,000-example run. Do not launch the configured
   50,000 optimizer steps as an unmeasured first job.
6. **Run the synthetic training set only after the 1,000-example ablations
   pass.** Train on 20,000 exact-entropy examples, evaluate the held-out and
   rho-zero twins, and stop if self-only attention recovers at least half of
   the bus gain.
7. **Add planner routing, then natural transfer.** The 30B-A3B teacher is
   loaded only to generate the validated HotpotQA/OASST1 corpus, then unloaded.
   It never shares VRAM with PDT training and never replaces the same-trunk
   full-prefix functional teacher.

The Mac remains the code, smoke-test, tokenizer, and frozen-inference host.
Scale training belongs on CUDA. Multi-GPU training is not the next step: one
80GB device is sufficient to test the claim, while DDP would replicate the
model and would not fix per-rank memory or causal-graph errors.

# Bottom line

The deepest version of your idea is not:

> “Predict three parts of an answer.”

It is:

> **Compile a prompt into a low-bandwidth common causal state, fork one model into three stateful local causal processes, and continuously ensure that the common state is sufficient to replace the serial prefixes that do not yet exist.**

The plan moves shared uncertainty before the fork.

The arms resolve private uncertainty after the fork.

The bus communicates newly realized shared information.

The joint selector resolves residual same-round coupling.

The residual controller determines whether an arm genuinely knows enough to emit.

And `HOLD` acknowledges the irreducible cases where the answer contains real sequential causality.

That gives you a precise experimental thesis:

[
\boxed{
\text{Can a compact, fixed-depth latent map plus a low-bandwidth dynamic bus}
}
]

[
\boxed{
\text{reduce the missing-prefix information of three concurrent causal arms}
}
]

[
\boxed{
\text{enough that their maximum stream length falls faster than coordination and repair costs rise?}
}
]

That—not merely whether three streams can produce coherent text—is the core test of genuine skip-ahead generation.

[1]: https://arxiv.org/html/2512.10054v2 "https://arxiv.org/html/2512.10054v2"
[2]: https://arxiv.org/html/2605.12460v1 "https://arxiv.org/html/2605.12460v1"
[3]: https://arxiv.org/abs/2506.09991 "https://arxiv.org/abs/2506.09991"
