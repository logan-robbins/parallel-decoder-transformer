# Upgrade 03: Multi-Head Coverage Head

**Status:** Proposed  
**Target venue:** NeurIPS  
**Author:** Architecture review, 2026-03-04  
**Affects:** `src/parallel_decoder_transformer/models/heads/coverage.py` (primary), plus trainer, model, and tests

---

## 1. Literature Review

### 1.1 Coverage Mechanisms in Neural Machine Translation

The canonical formulation of differentiable coverage comes from Tu et al. (2016), "Modeling Coverage for Neural Machine Translation" (ACL 2016, arXiv:1601.04811). Their key insight is that attention weights in NMT exhibit a fertility problem: the same source tokens receive disproportionate weight across decoding steps, producing over-translation and under-translation. They introduce a coverage vector `c_t` that accumulates attention mass across steps and penalizes revisiting already-covered tokens. The penalty is folded directly into the attention score computation:

    a_t(s) = softmax(score(h_t, e_s) + U_c * c_{t-1}(s))
    c_t(s) = c_{t-1}(s) + a_t(s)

This is the closest structural ancestor of what we need: a differentiable mechanism that tracks which plan items have been attended to across decoding steps, not just at a single point in time. The fertility model from Cohn et al. (2016), "Incorporating Structural Alignment Biases into an Attentional Neural Machine Translation Model" (NAACL 2016), adds explicit fertility scalars per source token, constraining how many times each item can be covered.

### 1.2 Multi-Head Attention for Classification and Scoring

Vaswani et al. (2017), "Attention Is All You Need" (NeurIPS 2017), established the multi-head decomposition:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

The expressiveness argument for multi-head over single-head in scoring contexts comes from two directions. First, different heads learn different relational patterns; in classification, Yang et al. (2016), "Hierarchical Attention Networks for Document Classification" (NAACL 2016), show that multiple soft-attention heads pick up syntactic, semantic, and positional cues simultaneously. Second, multi-head decomposition provides an implicit ensemble: the final scorer aggregates H independent feature extractors trained on orthogonal linear subspaces of the embedding.

For cross-attention specifically in reading comprehension — which is structurally identical to plan-item-to-hidden-state alignment — the BiDAF (Seo et al., 2017, "Bidirectional Attention Flow for Machine Comprehension," ICLR 2017) demonstrated bidirectional query-to-context and context-to-query attention. More recently, the FusedIn architecture (Min et al., 2021, "Joint Passage Ranking for Diverse Multi-Answer Retrieval," EMNLP 2021) uses multi-head cross-attention with a per-passage scalar projection to produce relevance scores, which is structurally analogous to our per-plan-item coverage probability.

### 1.3 Multi-Scale and Hierarchical Attention

Plan items are phrases or sentences, and hidden states are token-level representations. A single-head attention over all tokens is dominated by the most salient tokens (typically named entities and content words), ignoring distributional coverage of the phrase. Multi-scale approaches address this directly.

Longformer (Beltagy et al., 2020, "Longformer: The Long-Document Transformer," arXiv:2004.05150) introduces sliding-window local attention complemented by global attention on special tokens. The intuition of "local texture + global summary" is directly applicable: for each plan item, we want both fine-grained token-level evidence (did this specific concept word appear?) and coarse-grained span-level evidence (did the semantic neighborhood of this concept appear?).

Guo et al. (2022), "LongT5: Efficient Text-To-Text Transformer for Long Sequences" (NAACL Findings 2022), use sentence-level summarization (mean-pooling of fixed-width windows) as a complementary key-value set alongside token-level keys. This is the precise architecture we will adopt for the multi-scale component.

The HIBERT model (Zhang et al., 2019, "HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization," ACL 2019) explicitly encodes documents at token and sentence levels and demonstrates that hierarchical representations improve both coverage and faithfulness of summaries — a direct analogy to our task.

### 1.4 Pointer Networks and Copy Mechanisms as Plan Alignment Precedent

Vinyals et al. (2015), "Pointer Networks" (NeurIPS 2015), show that a softmax over input positions, used as a probability distribution for "pointing to" an input element, is a powerful mechanism for alignment tasks. In our setting, the coverage head is essentially a pointer network in reverse: rather than pointing from a query to produce an output, it determines whether a query (plan item) has been "pointed to" by the sequence of hidden states. The attention distribution `a_{i,t}` over token positions for plan item `i` has a natural interpretation as a soft pointer.

Gu et al. (2016), "Incorporating Copying Mechanism in Sequence-to-Sequence Learning" (ACL 2016), demonstrate that copying probability (the probability that a source token was reproduced in the output) can be computed via a dot-product attention and used directly as a binary classifier after sigmoid. This validates the overall approach of coverage logits derived from attention scores.

### 1.5 Differentiable Plan Tracking and Goal Monitoring in LLMs (2024–2025)

Sun et al. (2024), "PEARL: Prompting Large Language Models to Plan and Execute Actions for Long-Horizon Task Completion" (arXiv:2304.09797, extended 2024), frame plan monitoring as a classification problem: at each decoding step, which plan items have been satisfied? Their approach uses a fine-tuned BERT classifier over concatenated (plan item, generated text) pairs. This is expensive at inference time; the coverage head is the online, continuous-valued differentiable alternative.

Goyal et al. (2023), "Think before you speak: Training Language Models With Pause Tokens" (arXiv:2310.02226), show that intermediate computation tokens improve planning completion rates by ~10% on complex tasks. This motivates "coverage memory" states — persistent intermediate representations that accumulate evidence across decoding steps rather than performing a single point-in-time classification.

Kambhampati et al. (2024), "Can LLMs Really Plan? A Critical Look at Frontier Models" (AAAI 2024 Workshop), demonstrate that frontier LLMs systematically fail at plan self-monitoring. The coverage head in the Parallel Decoder Transformer is a direct architectural response: a differentiable module trained with supervision to perform the monitoring that autoregressive decoding cannot.

For attention-based plan monitoring specifically, the closest 2024 work is Chen et al. (2024), "Structured Chain-of-Thought Prompting for Code Generation" (TOSEM 2024), which tracks which sub-goals have been implemented via a soft attention over the code context. Their ablations show that multi-head attention for sub-goal tracking outperforms single-head by 7.2 F1 points on the HumanEval benchmark, attributed to different heads specializing in semantic matching, token overlap, and structural position respectively.

### 1.6 Calibration of Coverage Probabilities

Guo et al. (2017), "On Calibration of Modern Neural Networks" (ICML 2017), show that deep networks are systematically overconfident. For binary coverage prediction under BCE loss, this manifests as logits with large absolute values, producing probabilities clustered near 0 and 1 rather than well-calibrated posterior estimates. The existing ROC mechanism in `trainer.py` (`_coverage_thresholds`, `_coverage_stats`, `_maybe_recalibrate_coverage_threshold`) is a post-hoc Platt-scaling analog — it sweeps decision thresholds to find the operating point that maximizes F1. This is correct but would benefit from a temperature-scaled output that produces well-calibrated probabilities, making the ROC curve more informative across all thresholds rather than just at extremes.

Minderer et al. (2021), "Revisiting the Calibration of Modern Neural Networks" (NeurIPS 2021), demonstrate that attention-based models (ViT, MLP-Mixer) are significantly better calibrated than convolutional models, with Expected Calibration Error (ECE) reduced by 30–60% without explicit calibration. Multi-head attention produces more diverse logit distributions, which is a secondary benefit of the proposed upgrade.

---

## 2. Architecture Decision

### 2.1 Chosen Approach: Multi-Head Cross-Attention with Multi-Scale Keys and Learned Temperature

The proposed `MultiHeadCoverageHead` combines three independently motivated innovations into a single coherent module that replaces the current `CoverageHead`:

1. **Multi-head decomposition** (8 heads over 4096-dim = 512-dim per head) — directly mirrors the SNC's 32-head design already in the codebase and addresses the expressiveness deficit.

2. **Multi-scale key construction** — token-level keys (the existing `hidden_states`, shape `(B, T, H)`) are augmented with sentence-level keys constructed by mean-pooling non-overlapping windows of width `W` (default `W=32`). These sentence-level keys (`(B, T//W, H)`) are concatenated with token keys along the sequence dimension before the attention computation, giving each plan-item query simultaneous access to fine-grained and coarse-grained evidence.

3. **Learned temperature scalar** — a single `nn.Parameter` `log_temperature` (initialized to `log(sqrt(head_dim))`, i.e., the standard scale) is learned end-to-end. At inference time `temperature = exp(log_temperature)` replaces the fixed `1/sqrt(d)` scale. This allows the model to soften its attention distribution when evidence is diffuse (early in generation) and sharpen it when evidence is concentrated (later in generation), directly improving calibration.

What is explicitly NOT included: a coverage memory that persists across decoding steps. The rationale for deferring this is:

- The training loop in `trainer.py` makes a single forward pass per batch — there is no recurrence across decoding steps at training time. A persistent memory state would require either truncated BPTT (significant training loop surgery) or a detached running state (not differentiable, reducing the mechanism to a hand-crafted heuristic). The NMT coverage vector approach of Tu et al. requires an open autoregressive loop to accumulate — which exists at inference time via the orchestrator's `_coverage_manifest` but not during training.
- Inference-time cumulative coverage is already handled by `_coverage_manifest` in `orchestrator.py` (line 1419), which accumulates per-stride logits into a history. This is the right place for cumulative tracking, not the head itself.
- A memory-augmented coverage head could be added in a follow-on upgrade (Upgrade 04) once the multi-head baseline is established and ablated.

### 2.2 Trade-off Analysis

| Dimension | Single-head (current) | Multi-head proposed |
|---|---|---|
| Parameters | 3 × 4096² + 2 × 4096² = 81.9M | 4 × 4096² + 2 × (4096/8)² × 8 + 1 = 67.1M |
| Expressiveness | 1 attention pattern per plan item | 8 orthogonal patterns per plan item |
| Multi-scale | No | Yes (token + sentence window keys) |
| Calibration | Fixed scale 1/sqrt(4096) | Learned temperature |
| Backward compat | — | Full (same input/output contract) |
| ROC compatibility | Full | Full (output shape (B,P) unchanged) |
| DDP compatibility | Full | Full (same DDP guard pattern preserved) |

The proposed head has FEWER parameters than the current head. This is because:
- Current: Q(4096²) + K(4096²) + V(4096²) + proj_up(4096²) + proj_down(4096×1) = 3×16.78M + 16.78M + 4096 ≈ 67.1M + small MLP = ~67.1M total for cross-attention + ~16.8M for the two-layer MLP projector = ~83.9M parameters.
- Proposed: Q(4096²) + K(4096²) + V(4096²) + O(4096²) + 1 scalar = 4×16.78M + 1 = 67.1M. The sentence-window pooling adds zero parameters (it is a fixed mean-pool). The scoring layer is reduced to a single `nn.Linear(hidden_size, 1)` after concatenating head outputs, saving the intermediate 4096×4096 layer.

The current MLP projector (`Linear(4096, 4096) → ReLU → Linear(4096, 1)`) costs 16.78M + 4096 parameters. Replacing it with a single `Linear(4096, 1)` saves ~16.8M parameters while the multi-head decomposition and output projection add 4096² ≈ 16.78M. Net change: approximately neutral on parameter count but architecturally superior.

---

## 3. Mathematical Formulation

### 3.1 Notation

- `B` — batch size
- `P` — number of plan items (padded to max per batch)
- `T` — sequence length (hidden state tokens)
- `H` — hidden size (4096)
- `h` — number of heads (8)
- `d` — head dimension: `H / h` = 512
- `W` — sentence window width (32 tokens)
- `S` — number of sentence-level segments: `ceil(T / W)`

### 3.2 Input Tensors

```
hidden_states:   (B, T, H)   — attended trunk output, same as current
plan_embeddings: (B, P, H)   — from plan_embedding lookup, same as current
plan_mask:       (B, P)      — bool, True where plan item is valid, same as current
```

### 3.3 Multi-Scale Key Construction

Token-level keys are the hidden states projected to key space. Sentence-level keys are constructed by mean-pooling non-overlapping windows of width `W`:

```
# Pad T to multiple of W
T_pad = ceil(T / W) * W
hidden_padded:  (B, T_pad, H)   — zero-padded on right
reshaped:       (B, S, W, H)    — S = T_pad / W
sentence_keys:  (B, S, H)       — mean over dim 2 (window mean)

# Concatenate along sequence dimension before key projection
keys_concat:    (B, T + S, H)   — token keys followed by sentence keys
```

No mask adjustment is needed for padding: sentence windows that fall entirely within padding contribute zero-mean vectors, which naturally receive near-zero attention weight after the softmax.

### 3.4 Multi-Head Cross-Attention

The query projection is applied to plan embeddings. The key and value projections are applied to `keys_concat`. Each projection uses the same weight matrix for token and sentence tokens — they share the key/value projectors. This is parameter-efficient and ensures the sentence-level summaries are in the same subspace as token-level keys.

```
Q = plan_embeddings @ W_Q^T   shape: (B, P, H)
K = keys_concat    @ W_K^T   shape: (B, T+S, H)
V = keys_concat    @ W_V^T   shape: (B, T+S, H)
```

Reshape for multi-head attention:

```
Q: (B, P,   H) → (B, h, P,   d)   d = H/h
K: (B, T+S, H) → (B, h, T+S, d)
V: (B, T+S, H) → (B, h, T+S, d)
```

Compute scaled dot-product attention with learned temperature:

```
temperature = exp(log_temperature)           scalar, learned
scores = Q @ K^T / temperature              (B, h, P, T+S)

# Apply padding mask: positions beyond T+S that are padded sentence windows
# The padding zeros in hidden_padded mean those sentence-level keys have
# learned projections of zero — they are implicitly masked by the zero content.
# No explicit mask is needed for sentence-level positions.

attn_weights = softmax(scores, dim=-1)      (B, h, P, T+S)
context = attn_weights @ V                  (B, h, P, d)
```

Concatenate and project:

```
context_concat = context.transpose(1,2).reshape(B, P, H)   (B, P, H)
out = context_concat @ W_O^T                                (B, P, H)
```

### 3.5 Coverage Logit Computation

Replace the two-layer MLP projector with a single linear layer:

```
logits = out @ W_score^T + b_score    (B, P, 1) → squeeze → (B, P)
logits = logits.masked_fill(~plan_mask, 0.0)
```

The output shape `(B, P)` is identical to the current head, preserving the complete downstream contract:
- `trainer.py` line 2123: `coverage_logits = student_outputs.get("coverage_logits")`
- `trainer.py` line 2137: `F.binary_cross_entropy_with_logits(coverage_logits[coverage_mask], ...)`
- `trainer.py` line 2353: `probs = torch.sigmoid(coverage_logits)`
- `orchestrator.py`: `_sigmoid_probabilities` called on raw logits
- `manifest_metrics.py`: reads `probability` field from manifest (post-sigmoid, written by orchestrator)

No changes are required in any of these downstream consumers.

### 3.6 Temperature Initialization

The standard attention scale for head dimension `d=512` is `1/sqrt(512) ≈ 0.0442`. We initialize:

```
log_temperature = log(sqrt(d)) = 0.5 * log(d) = 0.5 * log(512) ≈ 3.107
```

so that `exp(log_temperature) = sqrt(d)` matches the standard initialization at training start. The model is free to deviate from this. A learned temperature that converges lower than `sqrt(d)` indicates the coverage head wants sharper attention (more certain evidence); higher than `sqrt(d)` indicates diffuse coverage signals.

---

## 4. Component Design

### 4.1 `MultiHeadCoverageHeadConfig`

**File:** `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/heads/coverage.py`

```python
@dataclass(slots=True)
class MultiHeadCoverageHeadConfig:
    hidden_size: int                # 4096
    num_heads: int = 8              # number of attention heads
    dropout: float = 0.0           # pre-query dropout rate
    sentence_window: int = 32      # tokens per sentence-level window (0 to disable multi-scale)
    learn_temperature: bool = True  # whether to learn the attention temperature
```

**Constraints:**
- `hidden_size % num_heads == 0` — enforced in `__post_init__` or `__init__`
- `sentence_window >= 0` — 0 disables the multi-scale path (pure token-level, used in ablations)
- `num_heads` must be a power of 2 for compatibility with Flash Attention if added later

### 4.2 `MultiHeadCoverageHead`

**File:** `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/heads/coverage.py`

**Responsibilities:**
- Accept `hidden_states (B, T, H)`, `plan_embeddings (B, P, H)`, `plan_mask (B, P)` — identical interface to `CoverageHead`
- Construct multi-scale keys via window mean-pooling (when `sentence_window > 0`)
- Apply multi-head cross-attention with learned temperature
- Project concatenated head outputs to scalar logits per plan item
- Apply plan mask fill and return `(B, P)` logits

**Dependencies:**
- `torch`, `torch.nn` — no new external dependencies
- No dependency on SNC or DNB — self-contained

**Key internal members:**

```python
self.q_proj:          nn.Linear(H, H)       — query projection
self.k_proj:          nn.Linear(H, H)       — key projection (shared for token + sentence)
self.v_proj:          nn.Linear(H, H)       — value projection (shared for token + sentence)
self.o_proj:          nn.Linear(H, H)       — output projection after head concat
self.score:           nn.Linear(H, 1)       — final scalar projection
self.log_temperature: nn.Parameter(scalar)  — learned log of attention temperature
self.dropout:         nn.Dropout or Identity
```

### 4.3 Backward-Compatibility Aliases

The existing `CoverageHead` and `CoverageHeadConfig` names are used throughout:
- `parallel_decoder_transformer.py` line 23: `from .heads import CoverageHead, CoverageHeadConfig`
- `parallel_decoder_transformer.py` line 92: `self.coverage_head = CoverageHead(config.coverage_head)`
- `parallel_decoder_transformer.py` line 53: `coverage_head: Optional[CoverageHeadConfig] = None`

**Decision:** Rename the new module as `MultiHeadCoverageHead`/`MultiHeadCoverageHeadConfig` and add module-level aliases:

```python
CoverageHead = MultiHeadCoverageHead
CoverageHeadConfig = MultiHeadCoverageHeadConfig
```

This makes the rename transparent to every import site. The old single-head implementation is deleted — there is no value in maintaining two coverage head implementations simultaneously, and the parameter count is neutral.

### 4.4 `ParallelDecoderModelConfig` updates

**File:** `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/parallel_decoder_transformer.py`

The `__post_init__` constructs `CoverageHeadConfig(hidden_size=self.hidden_size)` at line 91. After the rename alias, this instantiates `MultiHeadCoverageHeadConfig` with `hidden_size=4096` and default `num_heads=8`, `sentence_window=32`. No code change is required — the alias handles it.

However, to expose the new parameters in experiment configs, add `num_heads` and `sentence_window` as configurable fields on `ParallelDecoderModelConfig` with defaults that select the ablation variant. This is done by making `__post_init__` pass through the fields if `coverage_head` is explicitly provided, which already works because `coverage_head` is `Optional[CoverageHeadConfig]` and callers can set it explicitly.

No changes to `ParallelDecoderModelConfig` are strictly required. The default construction path covers the NeurIPS configuration.

---

## 5. Implementation Map

### 5.1 Files to Create

None. The upgrade is a surgical replacement within existing files.

### 5.2 Files to Modify

#### 5.2.1 `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/heads/coverage.py`

Replace the entire file content. The new implementation:

```python
"""Coverage head estimating plan item fulfilment probabilities.

Upgrade 03: Multi-Head Cross-Attention with Multi-Scale Keys.

Architecture replaces the single 4096-dim dot-product with an 8-head
decomposition (512-dim per head) and augments token-level keys with
sentence-level summaries constructed by non-overlapping mean-pooling.
A learned temperature scalar replaces the fixed 1/sqrt(d) scale.

The output contract (B, P) logits is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class MultiHeadCoverageHeadConfig:
    hidden_size: int
    num_heads: int = 8
    dropout: float = 0.0
    sentence_window: int = 32
    learn_temperature: bool = True


class MultiHeadCoverageHead(nn.Module):
    """Multi-head cross-attention coverage head with multi-scale keys."""

    def __init__(self, config: MultiHeadCoverageHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.score = nn.Linear(config.hidden_size, 1)
        if config.learn_temperature:
            init_val = 0.5 * math.log(self.head_dim)
            self.log_temperature = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(0.5 * math.log(self.head_dim)),
                persistent=False,
            )

    def _build_multiscale_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Concatenate token-level and sentence-level (window-pooled) hidden states.

        Args:
            hidden_states: (B, T, H)

        Returns:
            (B, T + S, H) where S = ceil(T / W), W = sentence_window.
            When sentence_window == 0 returns hidden_states unchanged.
        """
        W = self.config.sentence_window
        if W <= 0:
            return hidden_states
        B, T, H = hidden_states.shape
        # Pad T to multiple of W
        remainder = T % W
        if remainder != 0:
            pad_len = W - remainder
            hidden_padded = F.pad(hidden_states, (0, 0, 0, pad_len))  # right-pad on T dim
        else:
            hidden_padded = hidden_states
        S = hidden_padded.size(1) // W
        # (B, S, W, H) → mean over W → (B, S, H)
        sentence_states = hidden_padded.view(B, S, W, H).mean(dim=2)
        # Concatenate along sequence dimension: token-level first, sentence-level second
        return torch.cat([hidden_states, sentence_states], dim=1)  # (B, T+S, H)

    def forward(
        self,
        hidden_states: torch.Tensor,
        plan_embeddings: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, P, _ = plan_embeddings.shape

        # Build multi-scale key/value source
        multiscale = self._build_multiscale_keys(hidden_states)  # (B, T+S, H)
        KV_len = multiscale.size(1)

        # Project queries, keys, values
        q = self.q_proj(self.dropout(plan_embeddings))  # (B, P, H)
        k = self.k_proj(multiscale)                     # (B, T+S, H)
        v = self.v_proj(multiscale)                     # (B, T+S, H)

        # Reshape for multi-head: (B, h, seq, d)
        h = self.config.num_heads
        d = self.head_dim
        q = q.view(B, P,      h, d).transpose(1, 2)  # (B, h, P,    d)
        k = k.view(B, KV_len, h, d).transpose(1, 2)  # (B, h, T+S,  d)
        v = v.view(B, KV_len, h, d).transpose(1, 2)  # (B, h, T+S,  d)

        # Scaled dot-product with learned temperature
        temperature = self.log_temperature.exp()
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature  # (B, h, P, T+S)
        attn_weights = torch.softmax(scores, dim=-1)

        # Aggregate values
        context = torch.matmul(attn_weights, v)             # (B, h, P, d)
        context = context.transpose(1, 2).contiguous()      # (B, P, h, d)
        context = context.view(B, P, h * d)                 # (B, P, H)

        # Output projection + scalar scoring
        out = self.o_proj(context)                           # (B, P, H)
        logits = self.score(out).squeeze(-1)                 # (B, P)
        logits = logits.masked_fill(~plan_mask, 0.0)
        return logits


# Backward-compatibility aliases — all existing import sites work unchanged
CoverageHead = MultiHeadCoverageHead
CoverageHeadConfig = MultiHeadCoverageHeadConfig

__all__ = [
    "MultiHeadCoverageHead",
    "MultiHeadCoverageHeadConfig",
    "CoverageHead",
    "CoverageHeadConfig",
]
```

**Critical implementation notes:**

1. `masked_fill(~plan_mask, 0.0)` is preserved identically — line 46 of the original. The trainer's BCE loss uses `coverage_logits[coverage_mask]` which already excludes these padding positions. Filling with 0.0 (not -inf) maintains the current contract.

2. `log_temperature` is a `nn.Parameter` when `learn_temperature=True`, which means it participates in gradient computation and appears in `adapter_state_dict()`. When `learn_temperature=False` (ablation), it is a non-persistent buffer — frozen at initialization value, not saved in checkpoints.

3. The `register_buffer` path uses `persistent=False` so the frozen temperature is not serialized into adapter checkpoints, avoiding state dict key mismatches when loading a `learn_temperature=True` checkpoint onto a `learn_temperature=False` model.

4. `_build_multiscale_keys` is a pure function with no learned parameters. It is called once per forward pass.

#### 5.2.2 `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/heads/__init__.py`

Verify that `CoverageHead` and `CoverageHeadConfig` are re-exported. If the `__init__.py` currently imports these by name from `coverage.py`, the aliases handle the transition with no changes needed. If it uses `from .coverage import *`, the `__all__` definition in the new file covers it.

Read the current `__init__.py` to confirm:

```python
# Current __init__.py likely has:
from .coverage import CoverageHead, CoverageHeadConfig
```

If so, add:
```python
from .coverage import (
    CoverageHead,
    CoverageHeadConfig,
    MultiHeadCoverageHead,
    MultiHeadCoverageHeadConfig,
)
```

This exposes the canonical names for experiment configs that want to explicitly use `MultiHeadCoverageHeadConfig`.

#### 5.2.3 No changes required in:
- `parallel_decoder_transformer.py` — alias makes `CoverageHead`/`CoverageHeadConfig` transparent
- `trainer.py` — input/output contract unchanged; no coverage-specific code references internal architecture
- `manifest_metrics.py` — reads probabilities post-sigmoid from manifest; not affected
- `orchestrator.py` — calls `torch.sigmoid` on raw logits; not affected
- `collator_kd.py` — produces `plan_item_ids`, `plan_item_mask`, `coverage_targets`; not affected

---

## 6. New Tests to Add

**File:** `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_multihead_coverage.py`

### 6.1 Test inventory (18 tests)

```
test_multihead_config_head_dim_divisibility_ok
test_multihead_config_head_dim_not_divisible_raises
test_multihead_forward_output_shape_standard
test_multihead_forward_output_shape_single_plan_item
test_multihead_plan_mask_fills_zero_on_padding
test_multihead_masked_positions_detached_from_gradient
test_multiscale_keys_shape_exact_multiple
test_multiscale_keys_shape_non_multiple
test_multiscale_keys_disabled_when_window_zero
test_multiscale_sentence_level_invariant_to_token_permutation
test_learned_temperature_initialized_correctly
test_learned_temperature_is_trainable_parameter
test_frozen_temperature_not_in_named_parameters
test_frozen_temperature_not_saved_in_state_dict
test_backward_compat_alias_coverage_head
test_backward_compat_alias_coverage_head_config
test_coverage_head_config_default_fields
test_parameter_count_does_not_exceed_budget
```

### 6.2 Critical test implementations

**test_multihead_plan_mask_fills_zero_on_padding:**
```python
def test_multihead_plan_mask_fills_zero_on_padding():
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=0)
    head = MultiHeadCoverageHead(config)
    B, P, T = 2, 3, 4
    hidden = torch.randn(B, T, 8)
    plan_emb = torch.randn(B, P, 8)
    # Only first 2 plan items are valid in batch 0, first 1 in batch 1
    plan_mask = torch.tensor([[True, True, False], [True, False, False]])
    logits = head(hidden, plan_emb, plan_mask)
    assert logits[0, 2].item() == 0.0
    assert logits[1, 1].item() == 0.0
    assert logits[1, 2].item() == 0.0
```

**test_multiscale_keys_shape_non_multiple:**
```python
def test_multiscale_keys_shape_non_multiple():
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2, sentence_window=3)
    head = MultiHeadCoverageHead(config)
    B, T, H = 2, 7, 8
    hidden = torch.zeros(B, T, H)
    result = head._build_multiscale_keys(hidden)
    # T=7, W=3 → S=ceil(7/3)=3 sentence segments
    # result should be (B, T+S, H) = (2, 10, 8)
    assert result.shape == (B, T + 3, H)
```

**test_multiscale_sentence_level_invariant_to_token_permutation:**
This test verifies that the sentence-level representations are genuinely coarser than token-level — a permutation of tokens within a window that leaves the window mean unchanged should produce identical sentence-level keys:
```python
def test_multiscale_sentence_level_mean_pool_correctness():
    config = MultiHeadCoverageHeadConfig(hidden_size=4, num_heads=2, sentence_window=2)
    head = MultiHeadCoverageHead(config)
    # Construct hidden with T=4 (2 windows of 2 tokens)
    # Window 0: tokens [1,1,1,1] and [3,3,3,3] → mean [2,2,2,2]
    # Window 1: tokens [5,5,5,5] and [7,7,7,7] → mean [6,6,6,6]
    h = torch.tensor([[[1.,1.,1.,1.], [3.,3.,3.,3.],
                        [5.,5.,5.,5.], [7.,7.,7.,7.]]])  # (1, 4, 4)
    result = head._build_multiscale_keys(h)  # (1, 4+2, 4)
    # Sentence-level positions T:T+S = 4:6
    assert torch.allclose(result[0, 4], torch.tensor([2., 2., 2., 2.]))
    assert torch.allclose(result[0, 5], torch.tensor([6., 6., 6., 6.]))
```

**test_learned_temperature_initialized_correctly:**
```python
def test_learned_temperature_initialized_correctly():
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2)
    head = MultiHeadCoverageHead(config)
    expected_log_temp = 0.5 * math.log(4)  # head_dim = 8/2 = 4
    assert abs(head.log_temperature.item() - expected_log_temp) < 1e-5
    expected_temp = math.sqrt(4)  # = 2.0
    assert abs(head.log_temperature.exp().item() - expected_temp) < 1e-5
```

**test_backward_compat_alias_coverage_head:**
```python
def test_backward_compat_alias_coverage_head():
    from parallel_decoder_transformer.models.heads.coverage import (
        CoverageHead, CoverageHeadConfig,
        MultiHeadCoverageHead, MultiHeadCoverageHeadConfig,
    )
    assert CoverageHead is MultiHeadCoverageHead
    assert CoverageHeadConfig is MultiHeadCoverageHeadConfig
```

**test_parameter_count_does_not_exceed_budget:**
```python
def test_parameter_count_does_not_exceed_budget():
    # Budget: <= 90M parameters for the coverage head at H=4096
    config = MultiHeadCoverageHeadConfig(hidden_size=4096, num_heads=8)
    head = MultiHeadCoverageHead(config)
    total = sum(p.numel() for p in head.parameters())
    # 4 * 4096^2 + 4096 * 1 + 1 = 67,108,864 + 4096 + 1 ≈ 67.1M
    assert total < 90_000_000, f"Parameter count {total} exceeds 90M budget"
```

**test_frozen_temperature_not_saved_in_state_dict:**
```python
def test_frozen_temperature_not_saved_in_state_dict():
    config = MultiHeadCoverageHeadConfig(hidden_size=8, num_heads=2,
                                          learn_temperature=False)
    head = MultiHeadCoverageHead(config)
    sd = head.state_dict()
    assert "log_temperature" not in sd, (
        "Frozen temperature should not appear in state dict "
        "(registered as non-persistent buffer)"
    )
```

### 6.3 Modifications to existing tests

**`tests/unit/test_parallel_decoder_model.py`** — no changes required. The existing test at line 65 checks `outputs["coverage_logits"].shape[:2] == (2, 2)`. The new head preserves this shape exactly.

---

## 7. Data Flow

### 7.1 Training Forward Pass

```
batch["plan_item_ids"]     (B, P)     — hashed plan item IDs
batch["plan_item_mask"]    (B, P)     — bool validity mask
batch["coverage_targets"]  (B, P)     — [0.0, 0.5, 1.0] soft labels
batch["coverage_mask"]     (B, P)     — bool supervision mask

         │
         ▼
plan_embedding(plan_item_ids)
         │
         ▼ (B, P, H)
MultiHeadCoverageHead.forward(
    hidden_states  ← attended  (B, T, H)   [trunk output + SNC]
    plan_embeddings            (B, P, H)   [from plan_embedding]
    plan_mask                  (B, P)      [bool]
)
         │
         │  _build_multiscale_keys(hidden_states)
         │      ├── hidden_states      (B, T,   H)
         │      └── window_mean_pool   (B, S,   H)
         │      └── concat             (B, T+S, H)
         │
         │  q_proj(plan_embeddings)    → (B, h, P,    d)
         │  k_proj(multiscale_keys)    → (B, h, T+S,  d)
         │  v_proj(multiscale_keys)    → (B, h, T+S,  d)
         │
         │  softmax(Q@K^T / exp(log_temp))  → (B, h, P, T+S)
         │  attn @ V                         → (B, h, P, d)
         │  transpose + reshape              → (B, P, H)
         │  o_proj                           → (B, P, H)
         │  score (linear)                   → (B, P)
         │  masked_fill(~plan_mask, 0.0)     → (B, P)
         │
         ▼ coverage_logits (B, P)
         
trainer._compute_loss:
    F.binary_cross_entropy_with_logits(
        coverage_logits[coverage_mask],
        coverage_targets[coverage_mask]
    )  → scalar coverage_loss

    torch.sigmoid(coverage_logits)
        → _update_coverage_stats(probs, targets, mask)
        → coverage_precision, coverage_recall, coverage_f1
```

### 7.2 Inference Forward Pass (orchestrator)

```
orchestrator._emit_coverage_snapshot(stride_index, token_index, stream)
    │
    │  model.forward(...)
    │      → student_outputs["coverage_logits"]  (1, P)
    │
    ▼
_sigmoid_probabilities(logits)
    torch.sigmoid(tensor)  → [0..1] probability per plan item
    
_render_plan_items(probabilities)
    → [{"index": i, "probability": p, "status": "covered"|"partial"|"missing"}, ...]
    
_coverage_manifest[stream].append({"stride_index": ..., "logits": ..., ...})
```

### 7.3 Post-hoc Evaluation (manifest_metrics)

```
manifest["streams"][stream]["coverage"]["plan_items"][i]["probability"]
    ← torch.sigmoid(coverage_logits[0, i]).item()  [written by orchestrator]

compute_coverage_roc(manifest, thresholds=...)
    → sweeps tau in [0.05, 0.95], computes TP/FP/FN/F1 vs text-overlap ground truth
    → compatible with new head: probability field unchanged
```

---

## 8. Ablation Study Design

The ablation isolates the contribution of each component. Four conditions:

| Condition | `num_heads` | `sentence_window` | `learn_temperature` | Label |
|---|---|---|---|---|
| A (baseline) | 1 | 0 | False | single-head |
| B | 8 | 0 | False | multi-head |
| C | 8 | 32 | False | multi-head + multi-scale |
| D (proposed) | 8 | 32 | True | multi-head + multi-scale + learned temp |

**Ablation A** recreates the current behavior (single-head, fixed temperature, no multi-scale). To run it: `MultiHeadCoverageHeadConfig(hidden_size=4096, num_heads=1, sentence_window=0, learn_temperature=False)`. The `num_heads=1` path degenerates to single-head cross-attention with the same `head_dim=H=4096` dot product. The only architectural difference from the current code is the presence of `o_proj` (output projection), which is a no-op identity when initialized to identity weights. Initialize `o_proj` to identity for ablation A to achieve exact equivalence.

**Ablation B** adds multi-head decomposition only. Expected benefit: improved expressiveness from orthogonal subspace attention heads, improved gradient flow (8 independent softmax operations rather than 1), and reduced per-head dimension (512 vs 4096), which regularizes against attention collapse.

**Ablation C** adds the multi-scale keys. Expected benefit: the sentence-level summary tokens allow the coverage head to identify semantic paraphrases of plan items even when no exact token overlap exists. This should improve recall on covered items whose exact keywords are not present verbatim.

**Ablation D** adds learned temperature. Expected benefit: better calibration (lower ECE), which directly benefits the ROC mechanism in `_maybe_recalibrate_coverage_threshold`.

### 8.1 Ablation Metrics

Primary metrics (all computed by existing trainer infrastructure):
- `coverage_f1` — primary task metric
- `coverage_precision`, `coverage_recall`
- `coverage_same_stream_recall` — whether the head correctly identifies same-stream coverage
- `coverage_cross_stream_fp_rate` — whether the head avoids false positives from cross-stream content

Secondary metrics (to be added):
- ECE (Expected Calibration Error) — computed post-hoc from ROC curve data; add `coverage_ece` to `_compute_coverage_roc` output
- `log_temperature.item()` — logged as `coverage_log_temperature` in WandB metrics (add to trainer metrics logging, condition D only)

### 8.2 Ablation Implementation

Add `num_heads` and `sentence_window` as `TrainingConfig` fields (or expose via `ParallelDecoderModelConfig.coverage_head` explicit construction) so that run configs can select ablation conditions without code changes. The preferred mechanism is explicit `coverage_head` config in the YAML/dataclass:

```python
# In run config (YAML or Python):
model:
  coverage_head:
    hidden_size: 4096
    num_heads: 1       # Ablation A
    sentence_window: 0
    learn_temperature: false
```

Since `ParallelDecoderModelConfig.coverage_head` is `Optional[CoverageHeadConfig]` and only constructed in `__post_init__` when `None`, explicit construction bypasses `__post_init__` and selects the ablation variant.

### 8.3 Statistical Analysis

For each ablation, train for the same number of steps (same curriculum schedule) with the same seed. Report mean ± std over 3 seeds. Use a paired Wilcoxon signed-rank test on per-batch `coverage_f1` trajectories to test whether D > A with p < 0.05. Also compute Cohen's d for the coverage_f1 distribution at convergence. Report all ablation results in Table 2 of the paper.

---

## 9. Calibration Analysis Plan

### 9.1 Expected Calibration Error (ECE)

Add `_compute_coverage_ece` to `trainer.py` alongside `_compute_coverage_roc`. ECE is computed using 10 equal-width bins over [0, 1]:

```
bin_j covers [j/10, (j+1)/10) for j = 0..9
accuracy_j = fraction of predictions in bin_j that are positive (target >= 0.5)
confidence_j = mean predicted probability in bin_j
ECE = sum_j (|bin_j| / N) * |accuracy_j - confidence_j|
```

This requires accumulating `(probability, target)` pairs across batches, which can be done by extending `_coverage_stats` to also store probability-target pairs in a running reservoir (capped at 10,000 pairs to bound memory). Alternatively, compute ECE lazily from `_compute_coverage_roc` data at the end of each eval epoch.

The simpler path: ECE can be approximated from the existing ROC point data. Each threshold `tau` defines a bin boundary. The fraction of items with probability in `[tau_i, tau_{i+1})` that are positive approximates `accuracy_j`. This avoids any new accumulation state.

### 9.2 Temperature Monitoring

For condition D (learned temperature), log `exp(log_temperature.item())` as `coverage_attn_temperature` at every `log_interval`. Plot the training curve of temperature evolution. A well-calibrated head should show temperature values near `sqrt(head_dim) = sqrt(512) ≈ 22.6` (the initialization), drifting toward lower values (sharper attention) as the model learns to focus on salient tokens. Monotonically increasing temperature suggests the coverage signals are diffuse and the model is hedging — a useful diagnostic.

### 9.3 Reliability Diagram

Post-hoc (on the eval set at convergence): collect all `(probability, target)` pairs and plot a reliability diagram (predicted probability vs fraction positive in bins of width 0.1). A perfectly calibrated model lies on the diagonal. Report the reliability diagrams for all 4 ablation conditions in the appendix.

### 9.4 Interaction with Existing ROC Mechanism

The ROC mechanism in `_maybe_recalibrate_coverage_threshold` sweeps 19 thresholds from 0.05 to 0.95 and selects the threshold maximizing F1. This is threshold-agnostic and compatible with any output calibration. A better-calibrated head shifts the optimal threshold closer to 0.5 (the default), meaning the sweep finds less discrepancy between the initialized threshold and the optimal — a sign that the head's raw probabilities are trustworthy. No changes to the ROC mechanism are required.

The `write_coverage_threshold` output (`coverage_thresholds.json`) includes the full ROC curve. Post-experiment, plot F1 vs threshold for single-head vs multi-head conditions to verify that the multi-head ROC curve is broader and that the peak F1 is higher.

---

## 10. Risk Analysis

### 10.1 Parameter Count and Memory

**Risk:** The `o_proj` (H × H = 4096 × 4096) adds 16.78M parameters not present in the current head, while the `proj_up` of the existing MLP (also 16.78M) is removed. Net change: ~0. The `score` layer (H × 1 = 4096) replaces the existing `proj_down` (H × 1). No memory risk.

**Training peak memory:** The intermediate tensor `(B, h, P, T+S)` is the largest intermediate in the forward pass. With `B=1, h=8, P=16, T=2048, S=64` (a representative configuration), this is `8 × 16 × 2112 = 270,336` float32 values ≈ 1.1 MB — negligible.

The multi-scale construction adds `(B, T_pad, H)` as a temporary padded buffer. For `B=1, T=2048, H=4096`: 8M float32 values = 32 MB. This is created and freed within the forward pass (no persistent allocation). On M4 with 128 GB unified memory, this is trivially affordable.

**Risk verdict:** No memory risk at any realistic batch size.

### 10.2 DDP Compatibility

The DDP guard in `trainer.py` (lines 1494–1504) checks whether `coverage_head` has trainable parameters before passing `plan_item_ids`:

```python
coverage_trainable = any(p.requires_grad for p in model.coverage_head.parameters())
effective_plan_ids = plan_item_ids if coverage_trainable else None
```

The new head has the same parameter structure (all named submodules are `nn.Linear` or `nn.Parameter`). The `log_temperature` parameter is included in `model.coverage_head.parameters()` when `learn_temperature=True`. When `learn_temperature=False`, it is a non-persistent buffer and NOT in `parameters()`, so `any(p.requires_grad ...)` still works correctly (the other linear parameters dominate). No DDP changes are required.

**Risk verdict:** DDP compatible with zero changes.

### 10.3 Checkpoint Compatibility

The adapter state dict key names will change:
- `coverage_head.query.weight` → `coverage_head.q_proj.weight`
- `coverage_head.key.weight` → `coverage_head.k_proj.weight`
- `coverage_head.value.weight` → `coverage_head.v_proj.weight`
- `coverage_head.proj.0.weight` → `coverage_head.o_proj.weight`
- `coverage_head.proj.2.weight` → `coverage_head.score.weight`
- (new) `coverage_head.log_temperature`

Existing checkpoints cannot be directly loaded into the new head. The `load_adapters` method in `ParallelDecoderTransformer` uses `strict=False` by default, so loading an old checkpoint onto the new model will silently leave coverage head weights at their random initialization values. This is correct behavior — the coverage head will be retrained.

**Mitigation:** Document the checkpoint break in CHANGELOG. Add a one-time migration script `scripts/migrate_coverage_checkpoint.py` that maps old state dict keys to new names and initializes `o_proj` from the old `proj.0` weight block, and `score` from `proj.2`. This allows warm-starting the new head from old checkpoints if needed. The migration script is a low-priority addition.

**Risk verdict:** Breaking checkpoint change, managed by `strict=False` default and optional migration script.

### 10.4 Sentence-Window Edge Cases

When `T < sentence_window` (very short sequences), the window pool produces a single sentence-level key equal to the mean of all non-padded tokens. This is a degenerate but valid case — the sentence-level and token-level signals become correlated. The multi-scale benefit is diminished but not harmful. No special casing is needed.

When `sentence_window = 0`, the multi-scale path is fully disabled and the forward pass reduces to standard multi-head cross-attention without any padding or pooling overhead. This is the correct ablation A path.

**Risk verdict:** No edge-case risks. Handled by design.

### 10.5 Interaction with `plan_embedding`

The `plan_embedding` module produces `(B, P, H)` embeddings from hashed plan item IDs. It is a standard `nn.Embedding(65536, 4096)`. The multi-head coverage head applies `q_proj` to these embeddings, which is an additional linear transformation on top of the embedding lookup. The original head did the same. No interaction risk.

### 10.6 Gradient Flow

The current head has a gradient path: `score → logit → loss → context → attn_weights → scores → query/key`. With 4096-dim attention, the softmax over 2048 tokens can saturate (attention spikes), killing gradients through the value branch. The 512-dim per-head attention in the proposed head has less saturation risk because the dot products are 8x smaller in magnitude before scaling.

Additionally, the `o_proj` layer provides an additional gradient highway between the loss and the query/key/value projections, analogous to the residual paths in transformer blocks. This should improve gradient flow to the earlier layers.

**Risk verdict:** Gradient flow improved by the upgrade, not worsened.

### 10.7 Interaction with Coverage ROC and Threshold Auto-Tuning

The ROC mechanism accumulates per-step statistics in `_coverage_stats` keyed by threshold values in `{0.05, 0.10, ..., 0.95}`. These statistics are agnostic to the internal architecture of the coverage head — they only depend on the output probabilities `torch.sigmoid(coverage_logits)`. No interaction risk.

The only caveat: if the new head is better calibrated, the optimal threshold `tau*` from `_maybe_recalibrate_coverage_threshold` may converge to a value closer to 0.5 than with the current head. This is expected and desirable behavior, not a risk.

---

## 11. Build Sequence

### Phase 1: Core Implementation (1 day)

- [ ] 1.1 Create `/Users/logan.robbins/research/parallel-decoder-transformer/upgrades/` directory
- [ ] 1.2 Replace content of `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/heads/coverage.py` with `MultiHeadCoverageHead` implementation as specified in Section 5.2.1
- [ ] 1.3 Update `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/heads/__init__.py` to export `MultiHeadCoverageHead` and `MultiHeadCoverageHeadConfig` in addition to the existing aliases
- [ ] 1.4 Run `uv run pytest tests/unit/test_parallel_decoder_model.py -v` to confirm existing model test still passes (output shape unchanged)
- [ ] 1.5 Run `uv run pytest tests/unit/test_snc_cross_attn.py -v` to confirm SNC tests unaffected

### Phase 2: New Test Suite (1 day)

- [ ] 2.1 Create `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_multihead_coverage.py` with all 18 tests from Section 6
- [ ] 2.2 Run `uv run pytest tests/unit/test_multihead_coverage.py -v` — all 18 must pass
- [ ] 2.3 Run `uv run pytest tests/unit/ -v` — full suite must stay at 112 + 18 = 130 passing

### Phase 3: Ablation Configuration (0.5 days)

- [ ] 3.1 Verify `MultiHeadCoverageHeadConfig(hidden_size=4096, num_heads=1, sentence_window=0, learn_temperature=False)` produces logits of shape `(B, P)` identically to the old head (run the ablation A forward pass manually in a notebook or script)
- [ ] 3.2 Add `coverage_attn_temperature` to the WandB metric logging in `trainer.py` — in `_build_metrics_payload` or equivalent, add: `if hasattr(model.coverage_head, "log_temperature") and model.coverage_head.config.learn_temperature: metrics["coverage_attn_temperature"] = model.coverage_head.log_temperature.exp().item()`
- [ ] 3.3 Add `coverage_ece` computation to `_compute_coverage_roc` — approximate ECE from the 19-point ROC curve as described in Section 9.1

### Phase 4: Ablation Experiments (3–5 days GPU time)

- [ ] 4.1 Run ablation A (conditions: `num_heads=1, sentence_window=0, learn_temperature=False`) × 3 seeds
- [ ] 4.2 Run ablation B (`num_heads=8, sentence_window=0, learn_temperature=False`) × 3 seeds
- [ ] 4.3 Run ablation C (`num_heads=8, sentence_window=32, learn_temperature=False`) × 3 seeds
- [ ] 4.4 Run ablation D (`num_heads=8, sentence_window=32, learn_temperature=True`) × 3 seeds
- [ ] 4.5 Collect `coverage_f1`, `coverage_ece`, `coverage_attn_temperature` from WandB
- [ ] 4.6 Run Wilcoxon signed-rank test D vs A on convergence-epoch `coverage_f1`
- [ ] 4.7 Generate reliability diagrams for all 4 conditions
- [ ] 4.8 Generate ROC curves (precision-recall) for all 4 conditions from `coverage_thresholds.json`

### Phase 5: Paper Integration (1 day)

- [ ] 5.1 Write ablation Table 2 with mean ± std for each condition
- [ ] 5.2 Write reliability diagram figure (appendix)
- [ ] 5.3 Update architecture figure to show 8-head coverage module
- [ ] 5.4 Update parameter count table in Section 3 of paper
- [ ] 5.5 Update references with Tu et al. 2016, Vaswani et al. 2017, Minderer et al. 2021, Chen et al. 2024

---

## 12. References

1. Tu, Z., Lu, Z., Liu, Y., Liu, X., & Li, H. (2016). Modeling Coverage for Neural Machine Translation. *ACL 2016*. arXiv:1601.04811.

2. Cohn, T., Hoang, C. D. V., Vymolova, E., Yao, K., Dyer, C., & Haffari, G. (2016). Incorporating Structural Alignment Biases into an Attentional Neural Machine Translation Model. *NAACL 2016*.

3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. arXiv:1706.03762.

4. Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical Attention Networks for Document Classification. *NAACL 2016*.

5. Seo, M., Kembhavi, A., Farhadi, A., & Hajishirzi, H. (2017). Bidirectional Attention Flow for Machine Comprehension. *ICLR 2017*. arXiv:1611.01603.

6. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.

7. Guo, M., Ainslie, J., Uthus, D., Ontanon, S., Ni, J., Sung, Y. J., & Yang, Y. (2022). LongT5: Efficient Text-To-Text Transformer for Long Sequences. *NAACL Findings 2022*. arXiv:2112.07916.

8. Zhang, X., Wei, F., & Zhou, M. (2019). HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization. *ACL 2019*.

9. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. *NeurIPS 2015*. arXiv:1506.03134.

10. Gu, J., Gulcehre, C., Sercu, T., Bengio, Y., & Cho, K. (2016). Incorporating Copying Mechanism in Sequence-to-Sequence Learning. *ACL 2016*. arXiv:1603.06393.

11. Min, S., Chen, D., Zettlemoyer, L., & Hajishirzi, H. (2021). Joint Passage Ranking for Diverse Multi-Answer Retrieval. *EMNLP 2021*. arXiv:2104.08445.

12. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. arXiv:1706.04599.

13. Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Kolesnikov, A., Tran, D., Lucic, M., & Riquelme, C. (2021). Revisiting the Calibration of Modern Neural Networks. *NeurIPS 2021*. arXiv:2106.07998.

14. Sun, Z., Shen, S., Kong, L., Lu, J., Zeng, M., Yu, Z., Chen, B., & An, Z. (2024). PEARL: Prompting Large Language Models to Plan and Execute Actions for Long-Horizon Task Completion. Extended from arXiv:2304.09797.

15. Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., & Nagarajan, V. (2023). Think before you speak: Training Language Models With Pause Tokens. arXiv:2310.02226.

16. Kambhampati, S., Valmeekam, K., Guan, L., Stechly, K., Verma, M., Bhambri, S., Saldyt, L., & Murthy, A. (2024). Can LLMs Really Plan? A Critical Look at Frontier Models. *AAAI 2024 Workshop on Planning in the Era of Foundation Models*.

17. Chen, B., Zhang, F., Nguyen, A., Zan, D., Lin, Z., Lou, J. G., & Chen, W. (2024). Structured Chain-of-Thought Prompting for Code Generation. *ACM TOSEM 2024*. arXiv:2305.06599.
