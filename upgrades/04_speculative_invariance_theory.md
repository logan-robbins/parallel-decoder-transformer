# Speculative Invariance: A Theoretical Framework for the Lag-Correction Mechanism

**Document:** `upgrades/04_speculative_invariance_theory.md`
**Status:** Working draft for NeurIPS submission
**Date:** 2026-03-04

---

## Table of Contents

1. Overview and Scope
2. System Formalization
3. Literature Review and Theoretical Anchors
4. Definition: Speculative Invariance
5. Component Analysis I — Bounded Influence via Gated Residual (SNC)
6. Component Analysis II — Lag Causality in the Dynamic Notes Bus
7. Component Analysis III — Rollback Correctness of the Agreement Gate
8. Component Analysis IV — spec_kl as a Distributional Proximity Guarantee
9. Component Analysis V — Stability Loss as a Parameter Drift Bound
10. End-to-End Composition
11. What CAN and CANNOT Be Proven
12. Empirical Validation Plan
13. Paper Presentation Strategy
14. Notation Reference

---

## 1. Overview and Scope

The Parallel Decoder Transformer (PDT) introduces a Dynamic Notes Bus (DNB) that routes compressed semantic context between simultaneously decoding streams. Stream $i$ writes a notes snapshot to the bus at stride boundaries; stream $j$ reads lagged snapshots from the bus through a Shared Notes Cross-Attention (SNC) layer applied after the frozen GPT-OSS-20B trunk.

The central claim the paper needs to defend is **speculative invariance**: at inference time, the trunk's output distribution should remain well-calibrated regardless of whether the notes entering SNC come from (a) teacher-generated embeddings, (b) self-speculated embeddings from the SpeculationHead, or (c) a zero vector (no notes). This invariance is not merely desirable — it is the condition under which the agreement-gated rollback system is **safe**: if the model is invariant, then the rollback can correct a bad speculative step without inducing cascading distribution shift in the non-rolled-back stream.

This document formalizes that claim, component by component, identifies what is provable, and proposes a paper presentation strategy that is honest about the gap between formal guarantees and empirical evidence.

---

## 2. System Formalization

### 2.1 Notation Reference

| Symbol | Definition | Source |
|--------|-----------|--------|
| $\mathcal{V}$ | Vocabulary, $|\mathcal{V}|$ tokens | — |
| $d$ | Trunk hidden size (4096 for GPT-OSS-20B) | `hidden_size` |
| $d_n$ | Notes embedding dimension (2048) | `notes_dim` |
| $\mathbf{h} \in \mathbb{R}^{T \times d}$ | Trunk output hidden states for sequence length $T$ | `snc_cross_attn.py:69` |
| $\mathbf{n} \in \mathbb{R}^{K \times d_n}$ | Notes tensor on the bus ($K$ snapshots) | `dnb_bus.py:25` |
| $\mathbf{n}^* \in \mathbb{R}^{K \times d_n}$ | Teacher-generated notes (ground truth) | training |
| $\hat{\mathbf{n}} \in \mathbb{R}^{K \times d_n}$ | Speculated notes from SpeculationHead | `speculation.py:26` |
| $\mathbf{0}$ | Zero notes (counterfactual/ablated) | `trainer.py:2077` |
| $g \in [0,1]$ | Learned scalar SNC gate, $g = \sigma(\gamma)$ | `snc_cross_attn.py:95` |
| $\delta \in \mathbb{Z}_{\geq 0}$ | Lag parameter | `dnb_bus.py:69` |
| $B$ | Stride (tokens per commitment window) | `CurriculumConfig.B` |
| $L$ | Commit horizon (max rollback tokens) | `CurriculumConfig.L` |
| $\tau$ | Agreement threshold | `agreement_threshold` |
| $\text{SNC}(\mathbf{h}, \mathbf{n})$ | SNC output, shape $\mathbb{R}^{T \times d}$ | `snc_cross_attn.py:120` |
| $f_\theta$ | Full model forward pass | `parallel_decoder_transformer.py` |
| $p_\theta(\cdot \mid x, \mathbf{n})$ | Model logit distribution given context $x$ and notes $\mathbf{n}$ | — |
| $\sigma(\cdot)$ | Sigmoid function | — |
| $\sigma_\text{max}(W)$ | Spectral norm (largest singular value) of matrix $W$ | `torch.nn.utils.parametrizations.spectral_norm` |

### 2.2 The SNC Forward Pass

From `snc_cross_attn.py`, lines 61–120, the SNC layer computes:

```
Q = W_Q h,  K = W_K n,  V = W_V n
A = softmax(QK^T / sqrt(d_head))
context = A V
projected = W_O context
output = h + sigma(gamma) * projected
```

where `gamma` is a scalar learned parameter initialized to -5.0 (so initial gate $g = \sigma(-5) \approx 0.0067$). The output formula is the critical expression:

$$\text{SNC}(\mathbf{h}, \mathbf{n}) = \mathbf{h} + \sigma(\gamma) \cdot W_O \cdot \text{Attn}(W_Q \mathbf{h}, W_K \mathbf{n}, W_V \mathbf{n})$$

### 2.3 The DNB Lag Mechanism

From `dnb_bus.py`, lines 61–76, when `snapshot()` is called:

```python
cut_index = max(0, len(snapshots) - effective_lag)
window = snapshots[:cut_index]
```

This means stream $j$ at time $t$ reads snapshots pushed by stream $i$ up to but not including the $\delta$ most recent pushes. If stream $i$ pushes a snapshot at stride $s$, that snapshot becomes readable by stream $j$ only after $\delta$ additional pushes. In wall-clock terms with stride $B$: the information visible to stream $j$ is at least $\delta \cdot B$ tokens behind stream $i$'s current position.

### 2.4 The Agreement Gate and Rollback

From `orchestrator.py` lines 87–99 and 471–474:

```python
score = float(agreement_tensor.detach().mean().item())
triggered = score < self.threshold     # rollback when score BELOW threshold
```

And from `_perform_rollback` (lines 1325–1352), when triggered:
1. `state.rollback()` removes up to $L$ tokens from the stream's KV cache
2. The trunk is re-run with the current (corrected) notes window
3. A new notes embedding is pushed to the bus

The agreement head (`agreement.py`) is a linear scorer followed by sigmoid: $a(\mathbf{h}) = \sigma(W_a \mathbf{h} + b_a) \in [0, 1]$. Rollback triggers when the mean agreement score over the sequence drops below $\tau$.

### 2.5 The spec_kl Loss

From `trainer.py` lines 2422–2512, `_interhead_spec_kl` computes pairwise symmetric KL divergence between all pairs of speculated notes embeddings in a batch (after treating each $d_n$-dimensional vector as a softmax distribution scaled by temperature). The loss is:

$$\mathcal{L}_{\text{spec\_kl}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} w_{ij} \cdot \text{sym-KL}(\hat{\mathbf{n}}_i, \hat{\mathbf{n}}_j)$$

where $\mathcal{P}$ is the set of active note pairs, $w_{ij} = \min(\text{cov}_i, \text{cov}_j)$ weights pairs by coverage overlap, and sym-KL is the symmetric KL divergence.

**Observation:** This loss penalizes speculative notes for being mutually inconsistent. It is an intra-batch consistency term, not a direct alignment to teacher notes. The alignment to teacher notes comes from `notes_loss` (MSE between NotesHead and teacher notes) and `spec_loss` (MSE between SpeculationHead and teacher notes).

### 2.6 The Usage Penalty (Counterfactual Penalty)

From `trainer.py` lines 2074–2092:

```python
masked_outputs = model(hidden_states,
    notes=zeros_like(notes),
    notes_mask=zeros_like(notes_mask), ...)
masked_loss = cross_entropy(masked_logits, planner_ids)
delta = masked_loss - planner_loss     # positive = notes help
usage_loss = relu(margin - delta)      # penalize if notes help by less than margin
```

This counterfactual pass measures $\Delta\mathcal{L} = \mathcal{L}(\theta; \mathbf{0}) - \mathcal{L}(\theta; \mathbf{n})$. The penalty is $\max(0, m - \Delta\mathcal{L})$ where $m$ is `usage_margin`. This ensures the model actually uses notes rather than ignoring them — it penalizes scenarios where zeroing the notes does not increase loss by at least $m$.

### 2.7 The Stability Loss

From `trainer.py` lines 1974–1979:

```python
stability_loss = masked_kl(
    planner_logits_student,          # post-update logits
    pre_update_logits.detach(),      # pre-update logits (stop-grad)
    stability_mask,                  # non-commit tokens
)
```

where `stability_mask = planner_mask & (~commit_mask)`. This KL is computed between the model's distribution before and after a parameter update, restricted to tokens that are NOT at commit boundaries (i.e., tokens that should be stable).

---

## 3. Literature Review and Theoretical Anchors

### 3.1 Speculative Decoding (Leviathan et al., 2023; Chen et al., 2023)

**Leviathan, Y., Kalman, M., and Matias, Y.** "Fast Inference from Transformers via Speculative Decoding." ICML 2023. arXiv:2211.17192.

The canonical speculative decoding result: a small draft model proposes $\gamma$ tokens, which are verified in parallel by the target model. The acceptance criterion uses modified rejection sampling to guarantee **exact distributional equivalence** — the accepted tokens are identically distributed to those that would be sampled from the target alone. The key theoretical contribution is that the output distribution is preserved exactly regardless of draft quality, because the rejection/correction step absorbs any discrepancy.

**Chen, C., et al.** "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318 (2023).

Extends the Leviathan framework with a different acceptance criterion (direct probability ratio comparison) and proves the same distributional guarantee under weaker conditions on the draft model.

**Relevance to PDT:** The PDT's speculative notes mechanism is conceptually distinct from speculative decoding: instead of speculating tokens (which must be accepted/rejected by the target model), the system speculates *conditioning context* (notes), which then influences the target model's distribution. There is no rejection step that can guarantee exact distributional preservation. The spec_kl loss is the PDT's substitute: instead of a hard rejection gate, it uses a soft training pressure toward consistency. This is a **fundamental difference** that must be acknowledged in the paper. The PDT does not claim the distributional guarantee of Leviathan et al. — it claims the weaker property that training minimizes divergence between the speculated and teacher-conditioned distributions.

**Reference for context:**

> "Unlike speculative decoding [Leviathan et al., 2023; Chen et al., 2023] where a rejection sampling step ensures exact distributional equivalence, our architecture speculates conditioning context rather than token sequences. The absence of a hard rejection gate means we cannot claim the same lossless guarantee. Instead, we characterize the weaker but practically meaningful property we call *speculative invariance*."

### 3.2 Robustness Certificates for Neural Networks

**Cohen, J., Rosenfeld, E., and Kolter, J.Z.** "Certified Adversarial Robustness via Randomized Smoothing." ICML 2019.

Provides $\ell_2$ certified radius guarantees for classifiers under input perturbation. The key insight is that Lipschitz continuity of the classifier bounds output change as a function of input perturbation magnitude.

**Virmaux, A., and Scaman, K.** "Lipschitz Regularity of Deep Neural Networks: Analysis and Efficient Estimation." NeurIPS 2018.

Shows that the Lipschitz constant of a deep neural network can be bounded by the product of spectral norms of weight matrices: $\text{Lip}(f) \leq \prod_\ell \sigma_\text{max}(W_\ell)$.

**Miyato, T., et al.** "Spectral Normalization for Generative Adversarial Networks." ICLR 2018.

Introduces spectral normalization as a training tool to constrain spectral norms of weight matrices, enabling control over Lipschitz constants.

**Relevance to PDT:** The SNC module optionally applies spectral normalization to all four projection matrices ($W_Q, W_K, W_V, W_O$). When enabled, this directly constrains $\sigma_\text{max}(W_Q), \ldots, \sigma_\text{max}(W_O) \leq 1$. This is the entry point for a Lipschitz-based bound on output divergence.

### 3.3 Knowledge Distillation Theory

**Hinton, G., Vinyals, O., and Dean, J.** "Distilling the Knowledge in a Neural Network." arXiv:1503.02531 (2015).

The foundational KD framework: softened cross-entropy between student and teacher distributions. Temperature scaling $T$ controls the softness.

**Phuong, M., and Lampert, C.** "Towards Understanding Knowledge Distillation." ICML 2019.

Proves convergence of KD under linear student/teacher assumption: minimizing KL divergence to teacher distribution causes the student to match the teacher's decision boundary in the limit of zero training loss.

**Tang, J., et al.** "Understanding the Role of Importance Weighting for Deep Learning." arXiv:2011.14696 (2020).

Shows that KD with temperature acts as importance weighting, emphasizing high-entropy (uncertain) teacher predictions where the teacher's soft labels carry information.

**Relevance to PDT:** The spec_kl loss is not a student-teacher KL — it is an intra-batch consistency loss between different speculative notes. The student-teacher KL for notes is `notes_loss` (MSE between NotesHead and teacher notes) and `spec_loss` (MSE between SpeculationHead and teacher notes). Therefore, the spec_kl loss enforces consistency among speculations but does not directly bound the distance between speculated and teacher notes. This is an important distinction: the paper should not claim spec_kl provides a teacher-alignment guarantee.

### 3.4 PAC-Bayes Bounds for Multi-Task Learning

**McAllester, D.A.** "PAC-Bayesian Model Averaging." COLT 1999. Subsequently refined in many works.

**Maurer, A.** "Bounds for Linear Multi-Task Learning." Journal of Machine Learning Research 7 (2006).

Multi-task learning bounds show that when a model shares representations across $T$ tasks, the generalization gap shrinks proportionally to $O(1/\sqrt{T})$ relative to single-task bounds. The shared representation amortizes representation learning cost.

**Yin, M., et al.** "Meta-Learning without Memorization." ICLR 2020.

Studies when auxiliary losses (like coverage prediction, agreement prediction) help or hurt the primary task. Auxiliary losses that are well-calibrated with the primary task reduce effective hypothesis class size.

**Relevance to PDT:** The PDT's training objective is a sum of nine losses (kd, stab, use, cov, nli, red, spec_kl, stream, agree). This is formally a multi-task learning problem. PAC-Bayes bounds in the multi-task setting would suggest that: (a) if the auxiliary tasks (agreement prediction, coverage prediction) share features with the primary task (planner KD), then their inclusion improves generalization bounds; (b) if auxiliary tasks are orthogonal or contradictory, they could worsen bounds. The paper can invoke this literature qualitatively without claiming a quantitative bound.

### 3.5 Self-Correction in LLMs

**Huang, J., et al.** "Large Language Models Cannot Self-Correct Reasoning Yet." arXiv:2310.01848 (2023).

Shows that LLMs attempting to correct their own outputs without external feedback often introduce errors rather than fix them. The key insight: self-correction requires an oracle signal that distinguishes errors from correct outputs.

**Madaan, A., et al.** "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023.

When the model is given a structured feedback signal (not just its own outputs), self-refinement is effective. The feedback signal acts as an external oracle.

**Akyurek, A.F., et al.** "RL4F: Generating Natural Language Feedback with Reinforcement Learning for Rewriting." ACL 2023.

Shows that learned feedback signals (trained binary classifiers on output quality) can guide correction effectively.

**Relevance to PDT:** The PDT's rollback+correction mechanism is not pure self-correction — the agreement head provides an *externally trained* binary signal that approximates an oracle signal. The training labels for agreement come from comparing pre-update and post-update logits (lines 2590–2629 in trainer.py), which means the agreement head is trained to predict KL-based stability. This is closer to the "structured feedback" regime of Self-Refine than the "no feedback" regime of Huang et al. The paper should position the agreement gate as a learned quality estimator, not a pure self-correction mechanism.

### 3.6 Binary Classifier Calibration Theory

**Platt, J.** "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." 1999.

**Guo, C., et al.** "On Calibration of Modern Neural Networks." ICML 2017.

Modern neural networks are often overconfident. Temperature scaling post-training is the most reliable calibration method. Well-calibrated classifiers satisfy: $P(\hat{y} = 1 \mid f(x) = p) = p$ for all $p$.

**Naeini, M.P., et al.** "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." AAAI 2015.

Provides the Expected Calibration Error (ECE) metric: $\text{ECE} = \sum_m \frac{|B_m|}{n} \left|\overline{\text{acc}}(B_m) - \overline{\text{conf}}(B_m)\right|$.

**Relevance to PDT:** The agreement head outputs $a(\mathbf{h}) = \sigma(W_a \mathbf{h}) \in [0,1]$. The threshold $\tau$ at which rollback is triggered is determined by ROC analysis over training (lines 2687–2748 in trainer.py). The paper can frame the agreement threshold selection as **empirical calibration via ROC curve**, which has a rigorous statistical interpretation: the selected threshold is the one maximizing $\text{precision} \cdot \text{recall}$ tradeoff on the training distribution. This is not a formal calibration guarantee but is a principled selection mechanism.

---

## 4. Definition: Speculative Invariance

### Definition 4.1 (Speculative Invariance, Formal)

Let $f_\theta: \mathcal{X} \times \mathcal{N} \to \Delta^{|\mathcal{V}|-1}$ denote the full model mapping input context $x \in \mathcal{X}$ and notes $\mathbf{n} \in \mathcal{N}$ to a distribution over the vocabulary. Let $\mathbf{n}^*$ denote teacher notes, $\hat{\mathbf{n}}$ denote speculated notes (from SpeculationHead), and $\mathbf{0}$ denote zero notes.

We say $f_\theta$ is **$\epsilon$-speculatively invariant** if for all $x \in \mathcal{X}$:

$$\max\left\{ D_{\text{KL}}(f_\theta(x, \mathbf{n}^*) \| f_\theta(x, \hat{\mathbf{n}})),\ D_{\text{KL}}(f_\theta(x, \mathbf{n}^*) \| f_\theta(x, \mathbf{0})) \right\} \leq \epsilon$$

We call $\epsilon$ the **invariance gap**.

### Remark 4.1

Zero-gap invariance ($\epsilon = 0$) holds trivially if $g = 0$ (SNC gate closed), but this defeats the purpose of the architecture. Meaningful invariance requires $g > 0$ while bounding the sensitivity to notes content. The goal of training is to push $\epsilon$ toward a small value while keeping $g$ large enough that notes actually improve performance.

### Definition 4.2 (Rollback Correctness)

The rollback mechanism is **$(\alpha, \beta)$-correct** if, when the agreement gate triggers a rollback:
- With probability at least $1 - \alpha$, the triggered rollback corresponds to a genuinely incorrect speculative step (true positive rate $\geq 1 - \alpha$)
- With probability at most $\beta$, the agreement gate fails to trigger on a genuinely incorrect speculative step (false negative rate $\leq \beta$)

### Definition 4.3 (Lag Causality)

The DNB with lag $\delta$ provides **$\delta$-causal separation**: stream $j$ at decode step $t$ can only observe notes from stream $i$ that were committed at step $\leq t - \delta \cdot B$, where $B$ is the stride. This is a strict **causal barrier** — no information from stream $i$'s current speculative state can influence stream $j$'s current computation.

---

## 5. Component Analysis I — Bounded Influence via Gated Residual (SNC)

### 5.1 Setup

From `snc_cross_attn.py`, the SNC output is:

$$\text{SNC}(\mathbf{h}, \mathbf{n}) = \mathbf{h} + g \cdot W_O \cdot \text{Attn}(W_Q \mathbf{h}, W_K \mathbf{n}, W_V \mathbf{n})$$

where $g = \sigma(\gamma)$ is the learned scalar gate.

### 5.2 Proposition 1 (Bounded Output Perturbation under Notes Perturbation)

**Proposition 1.** Let $\mathbf{n}$ and $\mathbf{n}'$ be two notes tensors (e.g., teacher vs. speculated) with $K$ snapshots. Under the Lipschitz bound from spectral normalization:

$$\|\text{SNC}(\mathbf{h}, \mathbf{n}) - \text{SNC}(\mathbf{h}, \mathbf{n}')\|_2 \leq g \cdot \sigma_\text{max}(W_O) \cdot \sigma_\text{max}(W_V) \cdot \|\mathbf{n} - \mathbf{n}'\|_F \cdot C_{\text{attn}}$$

where $C_{\text{attn}}$ is a bound on the Lipschitz constant of the attention operation.

**Proof Sketch.** The difference in SNC outputs is:

$$\Delta = g \cdot W_O \left[\text{Attn}(W_Q\mathbf{h}, W_K\mathbf{n}, W_V\mathbf{n}) - \text{Attn}(W_Q\mathbf{h}, W_K\mathbf{n}', W_V\mathbf{n}')\right]$$

By submultiplicativity of operator norms:

$$\|\Delta\|_2 \leq g \cdot \sigma_\text{max}(W_O) \cdot \|\text{Attn}(\cdot, W_K\mathbf{n}, W_V\mathbf{n}) - \text{Attn}(\cdot, W_K\mathbf{n}', W_V\mathbf{n}')\|_F$$

The attention mechanism maps notes through $W_K$ and $W_V$, then applies softmax-weighted averaging. Let $\mathbf{V} = W_V \mathbf{n}$ and $\mathbf{V}' = W_V \mathbf{n}'$. The attention output difference can be decomposed into:
- Perturbation via changed $\mathbf{V}$: contributes $\sigma_\text{max}(W_V) \|\mathbf{n} - \mathbf{n}'\|_F$ scaled by the attention weights
- Perturbation via changed attention weights (from changed $\mathbf{K}$): contributes a second-order term involving $\sigma_\text{max}(W_K)$ and the gradient of softmax

The softmax attention is not globally Lipschitz — it can amplify perturbations in keys when attention is near-uniform and contract them when attention is peaked. A valid bound requires either (a) assuming bounded key perturbations and using the Lipschitz constant of softmax in bounded domains, or (b) accepting a looser bound via the triangle inequality:

$$\|\Delta\|_2 \leq g \cdot \sigma_\text{max}(W_O) \cdot (\sigma_\text{max}(W_V) \cdot \|\mathbf{n} - \mathbf{n}'\|_F + \sigma_\text{max}(W_K) \cdot \|\mathbf{n} - \mathbf{n}'\|_F \cdot \|\mathbf{V}'\|_F \cdot L_{\text{sm}})$$

where $L_{\text{sm}}$ is the Lipschitz constant of the softmax-weighted averaging operation in the key space. $\square$

### 5.3 Corollary 1 (Zero-Notes Perturbation Bound)

When comparing $\mathbf{n}$ to $\mathbf{0}$ (the counterfactual forward pass in usage_loss), the perturbation is $\|\mathbf{n} - \mathbf{0}\|_F = \|\mathbf{n}\|_F$. Then:

$$\|\text{SNC}(\mathbf{h}, \mathbf{n}) - \text{SNC}(\mathbf{h}, \mathbf{0})\|_2 \leq g \cdot \sigma_\text{max}(W_O) \cdot \sigma_\text{max}(W_V) \cdot \|\mathbf{n}\|_F \cdot C_{\text{attn}}$$

When $\mathbf{n} = \mathbf{0}$, `SNC(h, 0) = h` (trivially, since $W_V \mathbf{0} = \mathbf{0}$, so $\text{context} = \mathbf{0}$, and $\text{projected} = \mathbf{0}$). This is verified by `snc_cross_attn.py` line 71–72:

```python
if notes_len == 0:
    return hidden_states
```

Therefore the SNC layer with zero notes **exactly** recovers the trunk output. The invariance gap for zero notes is not bounded above by Proposition 1 — it is **exactly zero** at the SNC layer level, by construction. However, this does not mean the full model output is unchanged, because the planner head, notes head, and speculation head all depend on the same hidden states, and those hidden states DO change when notes are non-zero (the SNC adds a residual). So the argument applies at the SNC output level; the full output distribution changes by however much the planner head is sensitive to SNC's residual.

### 5.4 Spectral Norm Constraint

When `spectral_norm=True` in `SharedNotesCrossAttentionConfig`, PyTorch's `spectral_norm` is applied to all four projection matrices. This constrains $\sigma_\text{max}(W_Q), \sigma_\text{max}(W_K), \sigma_\text{max}(W_V), \sigma_\text{max}(W_O) \leq 1$ after each forward pass (with `n_power_iterations` iterations of the power method). The default `n_power_iterations=1` is an approximation — exact spectral normalization requires infinite iterations, but one iteration is standard practice (Miyato et al., 2018).

Under this constraint:

$$\|\text{SNC}(\mathbf{h}, \mathbf{n}) - \text{SNC}(\mathbf{h}, \mathbf{n}')\|_2 \leq g \cdot \|\mathbf{n} - \mathbf{n}'\|_F \cdot C_{\text{attn}}$$

The gate $g = \sigma(\gamma)$ is the multiplicative scalar. Since $g \in (0, 1)$, the SNC output perturbation is strictly bounded above by $\|\mathbf{n} - \mathbf{n}'\|_F \cdot C_{\text{attn}}$, and as $g \to 0$ the bound vanishes. The gate thus provides a **training-time control knob** on the sensitivity of the model to notes perturbation.

### 5.5 Important Caveat: $C_{\text{attn}}$ is Not Globally Bounded

The softmax attention mechanism is **not globally Lipschitz** in its key inputs. In the extreme case where two keys are identical and a small perturbation breaks the tie, the attention weights can change drastically (from 0.5/0.5 to near 1.0/0.0), leading to a large change in the context vector. For the bound to be useful, one must either:

1. **Restrict to bounded key space:** If $\|\mathbf{n}\|_\infty \leq R$, then $\|W_K \mathbf{n}\|_\infty \leq R$ (under spectral norm), and the softmax attention Lipschitz constant in this ball is bounded by $O(1/\sqrt{d_\text{head}})$ (from the scaled dot-product normalization by $\sqrt{d_\text{head}}$).

2. **Use concentration arguments:** If the attention weights are sufficiently concentrated (peaked), perturbations in $\mathbf{K}$ cause small relative changes in the argmax. This is the practical regime during inference.

3. **Accept a vacuous bound in the worst case:** The bound holds for all $\mathbf{n}, \mathbf{n}'$ in a restricted domain but not globally.

**For the paper:** Claim the bound holds in the operating regime (bounded note magnitudes, concentrated attention) and report the empirical value of $C_\text{attn}$ measured via the Lipschitz probe already present in the codebase (`LipschitzMonitorConfig` in `config.py`, lines 168–176).

---

## 6. Component Analysis II — Lag Causality in the Dynamic Notes Bus

### 6.1 Formalization

Let $S = \{s_1, \ldots, s_N\}$ be the set of streams. For stream $s_i$, let $v_i^{(t)}$ denote the snapshot version pushed at decode step $t$ (in units of strides). For stream $s_j$ reading from stream $s_i$ with lag $\delta$:

**Theorem 6.1 (Lag Causality, Formal).** With `DynamicNotesBus.lag = delta`, stream $s_j$ at step $t$ observes only notes from stream $s_i$ that were committed at steps $\leq t - \delta$. Equivalently:

$$\mathbf{n}_j^{(t)} \perp \{w_i^{(t)}, w_i^{(t-1)}, \ldots, w_i^{(t-\delta+1)}\}$$

where $w_i^{(t)}$ denotes stream $i$'s write operations at step $t$. That is, the notes visible to stream $j$ at step $t$ are causally independent of stream $i$'s computations in the $\delta$ most recent strides.

**Proof.** By inspection of `dnb_bus.py` lines 69–73:

```python
cut_index = max(0, len(snapshots) - effective_lag)
window = snapshots[:cut_index]
```

If $\delta = 1$ and the bus currently holds snapshots at versions $[v_1, v_2, \ldots, v_m]$ in order, then `cut_index = m - 1`, and stream $j$ sees only $[v_1, \ldots, v_{m-1}]$ — the most recent push ($v_m$) is invisible. Since pushes happen at stride boundaries and the version counter increments strictly monotonically, the most recent $\delta$ pushes are always excluded from the visible window. $\square$

### 6.2 Why Lag Causality Matters for Rollback Safety

**Proposition 6.2 (Rollback Safety under Lag Causality).** If stream $j$ triggers a rollback at step $t$, the rollback reverts at most $L$ tokens (the commit horizon). Under lag $\delta \geq 1$:

1. The notes visible to stream $j$ during the rolled-back generation were committed at steps $\leq t - \delta$, hence they reflect stream $i$'s state at a prior stride boundary.

2. After rollback and re-generation, stream $j$ re-reads the same lagged notes (the lag window does not change during rollback), because the bus is **not modified during rollback** — only stream $j$'s KV cache and token sequence are modified.

3. Therefore, stream $i$'s future reads from stream $j$ will observe the **corrected** notes from stream $j$ only after stream $j$'s next push, which occurs at the next stride boundary after re-generation.

**Corollary 6.3.** Rollback in stream $j$ cannot cause immediate changes to stream $i$'s conditioning, because stream $i$ reads from stream $j$ with lag $\delta$. The correction propagates with at least $\delta$ stride delay. This prevents **rollback cascades**: a rollback in one stream cannot instantaneously destabilize another stream.

### 6.3 The Lag-Correctness Tradeoff

The lag $\delta$ controls a fundamental tradeoff:
- **Small $\delta$** (fast propagation): Stream $j$ sees more recent notes from stream $i$, which may be more informative but also more likely to be speculative/wrong.
- **Large $\delta$** (safe propagation): Stream $j$ sees older, more stable notes, but misses recent information.

The system uses `lag=1` by default (`DynamicNotesBusConfig.lag = 1`). This means streams always work with at least one stride of delay — never reading each other's most current (potentially unverified) speculation.

**Formal statement of the tradeoff.** Let $\text{freshness}(\delta) = 1 - \delta/K$ be a measure of information freshness (fraction of available snapshots visible). Let $\text{stability}(\delta) = P(\text{visible notes} | \text{no rollback at lag boundary})$ be the probability that the visible notes were not rolled back. Then:

- $\text{freshness}(\delta)$ is decreasing in $\delta$
- $\text{stability}(\delta)$ is increasing in $\delta$ (fewer rollbacks affect the visible window)
- The product $\text{freshness}(\delta) \cdot \text{stability}(\delta)$ has an interior maximum at some $\delta^*$ that depends on rollback rate

**For the paper:** The lag parameter is a hyperparameter that trades off information freshness against conditioning stability. Empirically, measure rollback rate as a function of $\delta$ and report the stability-freshness tradeoff curve.

---

## 7. Component Analysis III — Rollback Correctness of the Agreement Gate

### 7.1 The Agreement Head as a Binary Hypothesis Test

The agreement head computes $a(\mathbf{h}) = \sigma(W_a \mathbf{h} + b_a) \in [0,1]$. During inference, the agreement score at stride boundaries is the mean of $a(\mathbf{h})$ over the sequence. Rollback triggers when this mean falls below threshold $\tau$.

**Hypothesis Test Formulation.** Let $H_0$: "the speculative notes were correct (no rollback needed)" and $H_1$: "the speculative notes were incorrect (rollback needed)."

- **Type I error (false positive):** Rollback triggered when not needed. Rate: $\alpha(\tau) = P(\bar{a}(\mathbf{h}) < \tau \mid H_0)$
- **Type II error (false negative):** No rollback when rollback was needed. Rate: $\beta(\tau) = P(\bar{a}(\mathbf{h}) \geq \tau \mid H_1)$

The threshold $\tau$ selected by `_maybe_recalibrate_agreement_threshold` (trainer.py lines 2714–2748) maximizes precision subject to recall constraints over the empirical ROC curve. This is equivalent to minimizing the $F_1$-weighted combination of $\alpha$ and $\beta$ on the training distribution.

### 7.2 Training Labels for Agreement

From `_derive_agreement_targets` (trainer.py lines 2590–2629), the training label for agreement is:

$$y_\text{agree} = \mathbb{1}[\text{argmax}(p_\text{pre}) = \text{argmax}(p_\text{post})] \cdot \mathbb{1}[\text{sym-KL}(p_\text{pre}, p_\text{post}) \leq \tau_\text{kl}]$$

where $p_\text{pre}$ are the model logits before a parameter update and $p_\text{post}$ are the logits after the update (both detached from the computational graph). This label definition encodes "the model's prediction did not change meaningfully after incorporating updated notes."

**Observation:** The training labels for agreement are derived from parameter updates during training, not from inference-time rollback events. There is a **distribution shift** between:
- Training: agreement labels reflect stability of predictions under parameter updates
- Inference: agreement head is used to detect stability of predictions under speculative notes substitution

This distribution shift is the key limitation of the agreement head's guarantees. The head is trained to detect one type of instability (parameter-induced) and deployed to detect another (notes-induced). The claim that these are correlated is an **empirical hypothesis**, not a theorem.

### 7.3 Proposition 7.1 (Error Rate Characterization)

**Proposition 7.1.** Let $\tau^*$ be the threshold selected by ROC calibration on training data with $n$ samples. Under i.i.d. assumptions between training and test distributions, the Type I and Type II error rates at test time satisfy:

$$\hat{\alpha}(\tau^*) = \alpha_\text{train}(\tau^*) \pm O\left(\sqrt{\frac{\log(1/\delta')}{n}}\right)$$
$$\hat{\beta}(\tau^*) = \beta_\text{train}(\tau^*) \pm O\left(\sqrt{\frac{\log(1/\delta')}{n}}\right)$$

with probability $1 - \delta'$, by standard uniform convergence arguments (Hoeffding's inequality applied to the bounded random variable $a(\mathbf{h}) \in [0,1]$).

**Proof.** This follows from Hoeffding's inequality: for bounded random variables in $[0,1]$, the empirical mean concentrates around the true mean at rate $O(\sqrt{\log(1/\delta')/n})$. The ROC curve constructed from $n$ samples is a consistent estimator of the true ROC curve. $\square$

**Limitation:** The $O(\sqrt{1/n})$ rate requires $n$ to be the number of commitment boundaries evaluated during training, which is much smaller than the total number of tokens. In practice, agreement labels are generated only at commit boundaries (where `commit_mask = True`), and the sample count may be small relative to what is needed for tight concentration bounds.

### 7.4 Rollback Cascade Bounds

**Proposition 7.2 (Rollback Cascade Depth).** Under lag $\delta \geq 1$, a rollback in stream $j$ can cause a cascade rollback in stream $i$ only if:
1. Stream $i$ reads from stream $j$ (in an all-to-all topology, all streams read each other)
2. The corrected notes from stream $j$ arrive at stream $i$ within one lag window
3. The corrected notes cause stream $i$'s agreement score to drop below $\tau$

The expected cascade depth $D$ satisfies:
$$E[D] \leq \frac{\alpha(\tau)}{1 - \alpha(\tau)}$$

assuming rollbacks are independent Bernoulli trials with Type I error rate $\alpha(\tau)$.

**Proof Sketch.** A cascade occurs only if a correct stream receives bad (changed) notes that cause spurious rollback. The probability of this at each depth is at most $\alpha(\tau)$ (the false positive rate). The expected cascade depth follows from geometric series. $\square$

**For the paper:** If the empirical Type I error rate is $\alpha \approx 0.05$ and rollbacks are approximately independent, the expected cascade depth is $\approx 0.053$, meaning cascades are extremely rare. Report empirical cascade rates.

---

## 8. Component Analysis IV — spec_kl as a Distributional Proximity Guarantee

### 8.1 What spec_kl Actually Computes

From `_interhead_spec_kl` (trainer.py lines 2422–2512), the spec_kl loss is the coverage-weighted mean pairwise symmetric KL divergence between speculative notes embeddings within a batch. Treating each $d_n$-dimensional speculative embedding $\hat{\mathbf{n}}_i$ as a distribution $q_i = \text{softmax}(\hat{\mathbf{n}}_i / T)$:

$$\mathcal{L}_{\text{spec\_kl}} = \frac{\sum_{(i,j)\in\mathcal{P}} w_{ij} \cdot \text{sym-KL}(q_i \| q_j)}{\sum_{(i,j)\in\mathcal{P}} w_{ij}}$$

### 8.2 What spec_kl Does NOT Guarantee

Minimizing $\mathcal{L}_{\text{spec\_kl}} \to 0$ means that all speculative embeddings within a batch converge to the **same distribution**. This is a **within-sample consistency** guarantee, not a **teacher-alignment** guarantee.

Specifically: if all $\hat{\mathbf{n}}_i$ converge to some common distribution $q^*$, spec_kl = 0 regardless of whether $q^*$ is close to or far from the teacher distribution $q^{\text{teacher}} = \text{softmax}(\mathbf{n}^* / T)$.

The teacher alignment is enforced separately by:
- `spec_loss = MSE(SpeculationHead(h), n^*)` — direct regression to teacher notes
- `notes_loss = MSE(NotesHead(h), n^*)` — direct regression for notes head

### 8.3 Proposition 8.1 (spec_kl as Variance Reduction)

**Proposition 8.1.** Minimizing $\mathcal{L}_{\text{spec\_kl}}$ is equivalent to minimizing the variance of the speculative notes distribution across planning contexts. Formally, if spec_kl = 0, then all speculative notes within a batch are identical (up to numerical precision at temperature $T$).

**Proof.** Sym-KL$(q_i \| q_j) = 0$ iff $q_i = q_j$ almost everywhere. If all pairwise sym-KLs are zero, all distributions are identical. $\square$

### 8.4 Composition with MSE Losses

**Proposition 8.2 (Joint Minimization).** If both $\mathcal{L}_{\text{spec}} = \mathbb{E}[\|\hat{\mathbf{n}} - \mathbf{n}^*\|^2_2] \to 0$ and $\mathcal{L}_{\text{spec\_kl}} \to 0$, then the speculative notes concentrate around the teacher notes: $\hat{\mathbf{n}}_i \to \mathbf{n}^*$ for all $i$ in the batch.

**Proof.** From $\mathcal{L}_\text{spec} \to 0$: by Markov's inequality, $\hat{\mathbf{n}} \to \mathbf{n}^*$ in probability. From $\mathcal{L}_\text{spec\_kl} \to 0$: all $\hat{\mathbf{n}}_i$ converge to the same point. Combining: they must converge to $\mathbf{n}^*$. $\square$

**Caveat:** This joint minimization can only be achieved simultaneously if the SpeculationHead is sufficiently expressive and the teacher notes are consistent across the distribution. In practice, the losses are weighted ($w_\text{spec} = 0.5$, $w_\text{spec\_kl} = 0.1$ from `LossWeights`), meaning the MSE alignment dominates. The spec_kl acts as a regularizer that prevents degenerate solutions where the speculation head overfits to specific teacher notes but fails to generalize.

### 8.5 Implication for Speculative Invariance

The spec_kl loss contributes to speculative invariance only indirectly, by encouraging the speculative notes distribution to be consistent across contexts. Direct evidence for speculative invariance requires measuring $D_\text{KL}(f(x, \mathbf{n}^*) \| f(x, \hat{\mathbf{n}}))$ directly, which is what the **rollback_kl** metric (trainer.py lines 2182–2183) measures at commit boundaries:

```python
rollback_kl_value = _masked_kl(planner_logits_student, teacher_logits, rollback_mask)
```

This is the most direct empirical measurement of speculative invariance in the existing codebase.

---

## 9. Component Analysis V — Stability Loss as a Parameter Drift Bound

### 9.1 The Stability Loss Mechanism

From trainer.py lines 1974–1979, the stability loss is:

$$\mathcal{L}_\text{stab} = D_\text{KL}(p_\theta(x, \mathbf{n}) \| p_{\theta_0}(x, \mathbf{n}))_{\text{stability\_mask}}$$

where $\theta_0$ is the parameter vector before the update (pre_update_logits, computed with `torch.no_grad()`) and $\theta$ is the current parameter vector.

The mask is `stability_mask = planner_mask & (~commit_mask)`: tokens that are being planned but NOT at rollback commit boundaries. The idea is that commit-boundary tokens are being actively corrected (it is acceptable for them to shift); non-commit tokens should remain stable under the update.

### 9.2 Proposition 9.1 (Stability Loss as KL Drift Bound)

**Proposition 9.1.** If $\mathcal{L}_\text{stab} \leq \epsilon_\text{stab}$ at training step $t$, then the per-token KL divergence between the model's distribution before and after the parameter update is bounded by $\epsilon_\text{stab}$ on average over the stability-masked tokens:

$$\frac{1}{|\mathcal{M}_s|} \sum_{t \in \mathcal{M}_s} D_\text{KL}(p_\theta(\cdot \mid x_{1:t}) \| p_{\theta_0}(\cdot \mid x_{1:t})) \leq \epsilon_\text{stab}$$

where $\mathcal{M}_s$ is the stability mask set.

**Proof.** This is immediate from the definition of $\mathcal{L}_\text{stab}$ as a masked mean KL. $\square$

### 9.3 Connection to Proximal Policy Optimization

The stability loss has a structural analogy to the PPO-clip objective (Schulman et al., 2017): PPO constrains the ratio $\pi_\theta(a|s) / \pi_{\theta_0}(a|s)$ to avoid large policy updates. The stability loss constrains the KL divergence $D_\text{KL}(\pi_\theta \| \pi_{\theta_0})$ directly. In the TRPO literature (Schulman et al., 2015), this KL constraint is known to bound the policy improvement step and prevent catastrophic forgetting.

The PDT's stability loss is a **soft KL constraint** — it penalizes large KL rather than hard-constraining it. This means the model can still make large updates, but pays an increasing penalty.

### 9.4 What Stability Loss Cannot Guarantee

The stability loss constrains **consecutive** distribution shift (between update $t$ and $t+1$). It does not bound:
- **Cumulative drift:** The sum $\sum_{t=0}^T \mathcal{L}_\text{stab}^{(t)}$ bounds cumulative drift, but individual steps can be large if $\epsilon_\text{stab}^{(t)}$ is large early in training.
- **Notes-induced shift:** The stability loss is computed with the same notes in both pre and post passes (the notes used during the standard forward pass). It does not directly measure how much the distribution shifts when notes are changed.

---

## 10. End-to-End Composition

### 10.1 The Lag-Correction Invariance Chain

The core theoretical claim can be framed as a chain of three properties:

**Step 1 — Bounded Notes Perturbation:**
$$\|\text{SNC}(\mathbf{h}, \mathbf{n}^*) - \text{SNC}(\mathbf{h}, \hat{\mathbf{n}})\|_2 \leq g \cdot \Lambda \cdot \|\mathbf{n}^* - \hat{\mathbf{n}}\|_F$$

where $\Lambda = \sigma_\text{max}(W_O) \cdot \sigma_\text{max}(W_V) \cdot C_\text{attn}$ (bounded when spectral norm is enabled).

**Step 2 — Notes Alignment by Training:**
$$\|\mathbf{n}^* - \hat{\mathbf{n}}\|_F \leq \sqrt{2\mathcal{L}_\text{spec}}$$

by definition of MSE loss.

**Step 3 — Output Distribution Sensitivity (via Planner Head):**
$$D_\text{KL}(p(x, \mathbf{n}^*) \| p(x, \hat{\mathbf{n}})) \leq C_\text{head} \cdot \|\text{SNC}(\mathbf{h}, \mathbf{n}^*) - \text{SNC}(\mathbf{h}, \hat{\mathbf{n}})\|_2$$

where $C_\text{head}$ is a Lipschitz constant of the planner head composed with log-softmax.

**Composing:** Under the above conditions,

$$D_\text{KL}(f(x, \mathbf{n}^*) \| f(x, \hat{\mathbf{n}})) \leq g \cdot \Lambda \cdot C_\text{head} \cdot \sqrt{2\mathcal{L}_\text{spec}}$$

This is the central composable bound. As training progresses and $\mathcal{L}_\text{spec} \to 0$, the output divergence also vanishes. When $g \to 0$, the bound vanishes trivially.

### 10.2 The Full Lag-Correction Narrative

Putting all five components together, the theoretical narrative for the paper is:

1. **Gate limits influence** (Proposition 1): The SNC gate $g = \sigma(\gamma)$, initialized near zero, controls the maximum influence notes can have on the hidden states. Under spectral norm constraints, this influence is $O(g \cdot \|\delta\mathbf{n}\|)$.

2. **Lag limits cascade** (Theorem 6.1, Proposition 6.2): The lag parameter $\delta$ ensures that stream $j$ never directly observes stream $i$'s current speculative state — it observes a lagged, committed version. Rollbacks cannot instantaneously cascade because the corrected notes propagate with at least $\delta$ stride delay.

3. **Training closes the invariance gap** (Proposition 8.2): Joint minimization of $\mathcal{L}_\text{spec}$ and $\mathcal{L}_\text{spec\_kl}$ drives $\hat{\mathbf{n}} \to \mathbf{n}^*$, closing the gap between speculated and teacher notes.

4. **Agreement gating corrects errors** (Proposition 7.1): The agreement head provides a calibrated binary test for speculative correctness. When triggered, rollback reverts to the last commit boundary and re-generates with correct conditioning.

5. **Stability loss prevents drift** (Proposition 9.1): The KL-based stability loss prevents large distribution shifts during parameter updates at non-commit tokens, providing a per-update drift bound.

### 10.3 The Composited End-to-End Bound (Informal)

Under the following conditions:
- Spectral norm enabled on SNC ($\Lambda \leq 1$)
- Training converged so $\mathcal{L}_\text{spec} \leq \epsilon_1$, $\mathcal{L}_\text{stab} \leq \epsilon_2$
- Agreement gate calibrated with Type I rate $\alpha(\tau)$
- Lag $\delta \geq 1$

The system satisfies an approximate speculative invariance property:

$$\text{Pr}\left[ D_\text{KL}(f(x, \mathbf{n}^*) \| f(x, \hat{\mathbf{n}})) > g \cdot C_\text{head} \cdot \sqrt{2\epsilon_1} \right] \leq \beta(\tau)$$

where $\beta(\tau)$ is the Type II error rate of the agreement gate (missed rollbacks). In words: the output KL is bounded above with high probability, with the exceptional cases caught by the rollback system (at cost Type I error rate $\alpha(\tau)$).

This is the strongest defensible claim. It is **not** a uniform bound (coverage is probabilistic, controlled by $\beta(\tau)$), and it requires the training conditions (small $\epsilon_1$) to hold.

---

## 11. What CAN and CANNOT Be Proven

### 11.1 What CAN Be Proven (Rigorously)

| Claim | Proof Status | Section |
|-------|-------------|---------|
| SNC output difference is bounded by $g \cdot \Lambda \cdot \|\delta\mathbf{n}\|$ (under spectral norm + bounded attention) | Proposition 1 — tight under bounded attention assumption | 5.2 |
| Zero-notes SNC output exactly equals trunk output (no notes) | Direct code inspection | 5.3 |
| Lag $\delta$ prevents reads of the $\delta$ most recent pushes (causal barrier) | Theorem 6.1 — exact from code | 6.1 |
| Rollback cannot immediately cascade to other streams (lag isolation) | Proposition 6.2 — exact from code | 6.2 |
| spec_kl = 0 implies all speculative notes converge to same distribution | Proposition 8.1 — trivial | 8.3 |
| Joint spec_kl + spec MSE = 0 implies speculative notes converge to teacher notes | Proposition 8.2 — straightforward | 8.4 |
| Stability loss bounds per-update token-level KL drift (in expectation over stability mask) | Proposition 9.1 — direct | 9.2 |
| Agreement Type I/II rates concentrate around training estimates (under i.i.d.) | Proposition 7.1 — Hoeffding | 7.3 |

### 11.2 What CANNOT Be Proven (Honest Limitations)

| Limitation | Reason |
|------------|--------|
| Full end-to-end $\epsilon$-speculative invariance bound | Requires Lipschitz constant of planner head (not available), bounded attention Lipschitz constant (not globally bounded), and zero training loss (unreachable) |
| Agreement head works at inference time (distribution shift) | Training labels are parameter-update stability; inference use is notes-substitution stability; these correlate empirically but there is no formal bridge |
| Rollback correctness generalizes to out-of-distribution inputs | The agreement head is a neural network with no formal generalization guarantees beyond standard PAC bounds |
| spec_kl provides teacher-alignment guarantee | spec_kl is a within-batch consistency loss, not a teacher-alignment loss |
| Stability loss prevents cumulative drift over training | Each step's loss is independently bounded; cumulative bound requires additional assumptions |
| The cascade depth bound (Proposition 7.2) is tight | Rollbacks are not independent; the geometric series argument assumes independence |

---

## 12. Empirical Validation Plan

The following experiments would provide the strongest empirical support for the theoretical claims.

### 12.1 Experiment 1: Speculative Invariance Measurement

**What to measure:** `rollback_kl` metric already logged by the trainer (trainer.py lines 2182–2183). This is $D_\text{KL}(p_\theta(x, \mathbf{n}^*) \| p_\theta(x, \hat{\mathbf{n}}))$ at commit boundaries.

**What to report:** KL divergence as a function of training stage. Hypothesis: rollback_kl decreases monotonically as training progresses through curriculum stages.

**Confounding factors:** `rollback_mask` selects only commit-boundary tokens, not all tokens. The KL at non-commit tokens is not directly measured.

**Additional experiment:** Run inference with teacher notes vs. speculated notes vs. zero notes. Measure token-level KL divergence and top-1 disagreement rate. This is the direct measure of $\epsilon$-invariance.

### 12.2 Experiment 2: Gate Value vs. Invariance Gap

**What to measure:** The SNC gate value $g = \sigma(\gamma)$ throughout training, alongside `rollback_kl`.

**Hypothesis:** As $g$ increases (gate opens), the system can tolerate larger notes perturbations only if the notes alignment (spec_loss) also improves. There should be a correlation: high $g$ + low spec_loss = low invariance gap; high $g$ + high spec_loss = high invariance gap.

**Design:** Plot a 2D scatter of ($g$, spec_loss) colored by rollback_kl. This directly validates the composed bound in Section 10.1.

### 12.3 Experiment 3: Spectral Norm and Lipschitz Probe

**What to measure:** Use the existing `LipschitzMonitorConfig` (config.py lines 168–176) to measure the empirical Lipschitz constant of the SNC + planner head composition. Probe with random perturbations $\|\delta\mathbf{n}\| = \epsilon$ for varying $\epsilon$.

**What to report:** Empirical Lipschitz constant $L_\text{emp} = \|\delta\text{output}\| / \|\delta\mathbf{n}\|$. Compare with theoretical bound $g \cdot \Lambda \cdot C_\text{head}$. If the theoretical bound is loose, tighten it by reporting the empirical value with confidence intervals.

### 12.4 Experiment 4: Lag Parameter Ablation

**What to measure:** Train models with $\delta \in \{0, 1, 2, 4\}$ and compare:
- Rollback rate (how often agreement gate triggers)
- Cascade depth (how many streams roll back following a single rollback)
- Generation quality (e.g., plan coverage, NLI scores)

**Hypothesis:** $\delta = 1$ (default) provides the best tradeoff. $\delta = 0$ (no lag) leads to higher cascade rates and unstable training. $\delta > 1$ leads to lower rollback rates (safer) but worse generation quality (less fresh information).

### 12.5 Experiment 5: Agreement Head Calibration

**What to measure:** Use the existing ROC curve infrastructure (`write_agreement_threshold`, trainer.py lines 508–527) to report calibration curves.

**What to report:**
- ECE (Expected Calibration Error) of the agreement head scores
- Reliability diagram (confidence vs. accuracy)
- PR curve (precision-recall at all thresholds)
- Cross-domain transfer: train on one domain, evaluate calibration on another

**Hypothesis:** The agreement head is well-calibrated on the training distribution but shows calibration degradation on out-of-distribution inputs, consistent with the distribution shift limitation identified in Section 7.2.

### 12.6 Experiment 6: Stability Loss and Drift

**What to measure:** The existing `stability_kl` and `stability_error_rate` metrics (trainer.py). Plot these over training steps alongside `rollback_kl`.

**Hypothesis:** `stability_kl` (KL between pre- and post-update logits) decreases as training progresses, consistent with the model learning to be stable at non-commit tokens. Stability_error_rate (token-level argmax disagreement) should also decrease.

**Connection to theory:** `stability_kl` is the empirical value of $\mathcal{L}_\text{stab}$, providing direct evidence for Proposition 9.1.

---

## 13. Paper Presentation Strategy

### 13.1 Section Structure for the Theory Section

**Recommended structure (for a NeurIPS Theory/Analysis section, ~2 pages main text + 3 pages appendix):**

**Main text:**
```
Section 4: Theoretical Analysis

4.1 Bounded Notes Influence
  - Define speculative invariance (Definition 4.1)
  - Proposition 1 (gated residual bound, informal statement)
  - Corollary 1 (zero-notes exactly recovers trunk)

4.2 Lag Causality and Rollback Safety
  - Theorem 6.1 (lag causality, formal)
  - Proposition 6.2 (no immediate cascade)
  - The lag-correctness tradeoff (informal)

4.3 Convergence of Speculative Notes
  - Proposition 8.2 (joint minimization converges to teacher)
  - Connection to rollback_kl metric

4.4 Rollback Correctness
  - Definition 4.2
  - Proposition 7.1 (calibrated error rates, informal)
  - Honest statement of distribution shift limitation
```

**Appendix:**
```
Appendix A: Full proofs of Propositions 1, 8.1, 8.2, 9.1
Appendix B: Composition of bounds (Section 10)
Appendix C: Attention Lipschitz analysis
Appendix D: Extended empirical validation
```

### 13.2 What to Claim vs. Conjecture

**CLAIM (provably):**
- The lag parameter provides strict causal separation between streams (Theorem 6.1)
- The SNC gate bounds the maximum influence of notes perturbations on hidden states (Proposition 1, under spectral norm and bounded attention assumption)
- Joint minimization of spec_kl and spec_loss drives speculative notes toward teacher notes (Proposition 8.2)
- The stability loss bounds per-update KL drift at stability-masked tokens (Proposition 9.1)

**CONJECTURE (empirically supported):**
- The agreement head correctly identifies speculative failures at inference time (supported by rollback_kl curves)
- The composed system achieves approximate speculative invariance in practice (supported by Experiments 1–3)
- The lag-correctness tradeoff has an interior optimum at $\delta = 1$ for the PDT architecture (supported by Experiment 4)

**EXPLICITLY ACKNOWLEDGE:**
- No formal end-to-end uniform speculative invariance guarantee exists due to (a) non-global Lipschitz of attention, (b) distribution shift in agreement head labels, (c) non-zero training loss
- The spec_kl loss does not provide teacher-alignment guarantee (it is a within-batch consistency term)
- Rollback cascades, while bounded in expectation, are not ruled out with certainty

### 13.3 Framing for NeurIPS Reviewers

NeurIPS 2025/2026 reviewers in the ML systems/efficiency track will likely:
1. Ask whether the speculative invariance guarantee is as strong as speculative decoding's exact distribution preservation (Leviathan et al.)
2. Question the distribution shift between training and inference for the agreement head
3. Ask for ablation evidence that each theoretical component (gate, lag, spec_kl) individually contributes

**Recommended framing:**
> "Our architecture makes a fundamental architectural choice that distinguishes it from speculative decoding: we speculate *conditioning context* rather than *tokens*. This means we cannot apply the rejection sampling argument of Leviathan et al. (2023) to guarantee exact distributional preservation. Instead, we provide a weaker but practically meaningful characterization: under spectral norm constraints on SNC and training convergence of the spec_kl + spec_loss objectives, the output KL divergence between teacher-conditioned and speculation-conditioned inference is bounded by a quantity that shrinks with training loss. The lag mechanism independently guarantees that errors cannot cascade instantaneously across streams. Together, these properties define *speculative invariance* — not a hard guarantee, but a measurable, trainable property with theoretical support."

---

## 14. Notation Reference (Complete)

| Symbol | Meaning | Codebase Location |
|--------|---------|-------------------|
| $d$ | Hidden dimension (4096) | `ParallelDecoderModelConfig.hidden_size` |
| $d_n$ | Notes dimension (2048) | `ParallelDecoderModelConfig.notes_dim` |
| $K$ | Number of snapshots in notes window | `DynamicNotesBusConfig.max_snapshots` |
| $B$ | Stride (tokens per commitment window) | `CurriculumConfig.B` |
| $L$ | Commit horizon (max rollback depth) | `CurriculumConfig.L` |
| $\delta$ | Lag parameter | `DynamicNotesBusConfig.lag` |
| $M$ | Note emission cadence | `emission_cadence_M_by_stream` |
| $g = \sigma(\gamma)$ | SNC scalar gate | `snc_cross_attn.py:59,95` |
| $\gamma_0 = -5.0$ | SNC gate initialization | `SharedNotesCrossAttentionConfig.gating_init` |
| $\tau$ | Agreement threshold | `InferenceConfig.agreement_threshold_tau` |
| $\tau_\text{kl}$ | KL threshold for agreement label generation | `TrainingConfig.agreement_threshold` |
| $W_Q, W_K, W_V, W_O$ | SNC projection matrices | `snc_cross_attn.py:34-37` |
| $\sigma_\text{max}(W)$ | Spectral norm of $W$ | `snc_cross_attn.py:38-58` |
| $\mathbf{n}^*$ | Teacher notes | training labels |
| $\hat{\mathbf{n}}$ | Speculated notes (SpeculationHead output) | `speculation.py:26-29` |
| $\mathbf{0}$ | Zero notes (usage penalty counterfactual) | `trainer.py:2077-2078` |
| $p_\theta(x, \mathbf{n})$ | Model distribution given context and notes | forward pass |
| $\mathcal{L}_\text{spec\_kl}$ | Inter-head spec KL loss | `trainer.py:2422-2512, weight 0.1` |
| $\mathcal{L}_\text{spec}$ | Speculation head MSE loss | `trainer.py`, weight 0.5 |
| $\mathcal{L}_\text{stab}$ | Stability loss (pre/post update KL) | `trainer.py:1975-1979, weight 0.1` |
| $\mathcal{L}_\text{use}$ | Usage penalty (counterfactual delta) | `trainer.py:2092, weight 0.0 default` |
| $a(\mathbf{h})$ | Agreement score | `agreement.py:24-27` |
| $\alpha(\tau)$ | Type I error rate at threshold $\tau$ | empirical ROC |
| $\beta(\tau)$ | Type II error rate at threshold $\tau$ | empirical ROC |
| $C_\text{attn}$ | Lipschitz constant of attention | bounded in practice |
| $C_\text{head}$ | Lipschitz constant of planner head | measurable empirically |
| $\Lambda$ | Combined SNC Lipschitz bound | $\sigma_\text{max}(W_O) \cdot \sigma_\text{max}(W_V) \cdot C_\text{attn}$ |
| $\epsilon_1$ | Training spec_loss value | metric: `spec_loss` |
| $\epsilon_2$ | Training stability loss value | metric: `stability_loss` |
| rollback_kl | KL at commit boundaries (empirical $\epsilon$) | `trainer.py:2260` |
| stability_kl | KL pre/post update (empirical $\epsilon_2$) | `trainer.py:2261` |

---

**End of Document.**

*This document represents the strongest theoretical characterization that can be built from the existing architecture without claiming guarantees that the system does not provide. The central contribution — the lag-correction mechanism — is supported by a chain of five component analyses, each rigorously bounded where possible and honestly limited where not.*

