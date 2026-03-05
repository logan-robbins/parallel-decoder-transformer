# Learned Retention for the Dynamic Notes Bus

**Target venue:** NeurIPS 2026
**Status:** Planning
**Author:** Architecture review, 2026-03-04

---

## 1. Problem Statement

The `DynamicNotesBus._compact()` method (`src/parallel_decoder_transformer/inference/dnb_bus.py:98-100`) discards the oldest snapshot unconditionally:

```python
def _compact(self) -> None:
    while len(self._buffer) > self.config.max_snapshots:
        self._buffer.popleft()
```

With `max_snapshots=4` (the default in `DynamicNotesBusConfig`), any structural snapshot pushed at step 1 is permanently lost after four subsequent pushes. The bus is used in two distinct regimes:

1. **Inference** (`orchestrator.py:178-189`): one `DynamicNotesBus` per stream, snapshots read by `NotesWindowBuilder.build()` and consumed by `SharedNotesCrossAttention`.
2. **Training** (`trainer.py:1625-1769`): a dense batch tensor `student_notes_bus [B, max_snapshots, S, notes_dim]` plays the role of the bus; the FIFO shift is at lines 1669-1705.

Both regimes drop by position with no awareness of semantic relevance.

---

## 2. Literature Review

### 2.1 KV-Cache Eviction

**H2O: Heavy-Hitter Oracle (Zhang et al., NeurIPS 2023, arXiv:2306.14048)**
Defines an importance score for token position `i` as the cumulative sum of attention weights received across all decoding steps: `score_i = sum_t A_{t,i}`. Eviction removes the lowest-scoring position. Achieves near-lossless generation with 20% KV retention on LLaMA-class models. The core result: attention-weight accumulation is a robust proxy for downstream utility, and it is computable from signals already present in the SNC cross-attention layer.

**StreamingLLM (Xiao et al., ICLR 2024, arXiv:2309.17453)**
Demonstrates that the initial "attention sink" tokens (positions 0-3) receive anomalously high attention regardless of semantic content due to softmax normalization pressure, and that dropping them causes catastrophic perplexity spikes. StreamingLLM always retains the first four tokens plus a sliding window. For the DNB, this maps directly: early structural snapshots (plan preamble, introduction context) likely function as attention sinks in SNC cross-attention, meaning naive FIFO eviction removes precisely the tokens that SNC depends upon most.

**Scissorhands (Liu et al., NeurIPS 2023, arXiv:2305.17118)**
Establishes that heavy-hitter tokens are persistent: a position important at step `t` is very likely to remain important at step `t+k` (empirical correlation r > 0.95 across pivot windows). This validates lazy re-scoring — importance scores from the previous compaction window remain valid for the entire next cadence period, which aligns with the DNB's stride-based emission model.

**SnapKV (Li et al., 2024, arXiv:2404.14469)**
Pools attention weights from a short observation window to identify important KV positions during prefill. The observation-window pooling maps cleanly to the DNB's windowing design in `NotesWindowBuilder`: the set of snapshots visible in the current window provides the observation context for scoring.

### 2.2 Compressive Memory in Transformers

**Compressive Transformer (Rae et al., ICLR 2020, arXiv:1911.05507)**
Maintains two memory tiers: a recent FIFO window and a compressed memory of older activations. Compression applies a learned linear map `C: R^{m x d} -> R^{n x d}` trained via an auxiliary reconstruction loss. The compressed slot is differentiable — gradients flow through the compression operator back to the encoder. This is the canonical precedent for merging old snapshots rather than discarding them and for training the compression end-to-end.

**Memorizing Transformers (Wu et al., ICLR 2022, arXiv:2203.08913)**
Retrieval-augmented memory using k-NN lookup over stored (key, value) pairs. The retrieval key is the current query projection, so relevance is computed with respect to the current computation state. For the DNB this translates to: score old snapshots by their dot-product similarity with the current SNC query projection, then retain top-k. This is a stronger signal than pure attention-weight accumulation but requires keeping all old snapshots in memory for the query to be evaluated against.

**Infini-Attention (Munkhdalai et al., arXiv:2404.07143)**
Maintains a compressive memory matrix `M in R^{d_k x d_v}` updated by a delta rule: `M_t = M_{t-1} + sigma(K_t)^T V_t / (sigma(K_t)^T z_{t-1} + epsilon)`. Read is `A_mem = sigma(Q) M / (sigma(Q) z)`. This is a parameter-free, differentiable memory with linear complexity. The delta rule update is a natural candidate for compressing multiple evicted DNB snapshots into a single "residual slot," preserving an approximation of all evicted content at a fixed memory cost of `[d_k x d_v]`.

**Titan (Google DeepMind, 2024)**
Proposes surprise-gated episodic memory where the surprise score for token `t` is the gradient norm of a small memory MLP: `s_t = || nabla_M L_t ||_F`. High-surprise tokens are retained in full; low-surprise tokens are absorbed into the memory MLP. The gradient-norm surprise signal is computable from the NotesHead projector gradients and is worth exploring as an alternative scorer in Phase 5.

### 2.3 Priority Queues and Learned Eviction in Neural Systems

**Differentiable Neural Computer (Graves et al., Nature 2016)**
Content-based addressing via `w_i = softmax(beta * cosine(M_i, k))` plus a usage vector that tracks recency of writes and reads. Unused-and-old locations are preferentially freed. The usage-vector concept maps to maintaining a per-slot decay counter in the `SnapshotScoreBank` that decreases between reads and increases on write.

**Dynamic Memory Networks (Kumar et al., ICML 2016)**
Per-slot episodic gate `g_i = f_gate(m_i, q)` trained end-to-end to decide which memory slots are relevant to the current query `q`. This is the direct precedent for the differentiable eviction network in Phase 4.

---

## 3. Architecture Decision

The chosen approach is **Hybrid Scored Retention with Attention-Pooled Compression (HSRAC)**. It combines:

1. **Attention-weight accumulation scoring** (H2O-style): each snapshot slot accumulates a running EMA importance score from SNC cross-attention weights. Zero new parameters, zero latency impact.
2. **Recency pinning**: the most-recent `K_recent = ceil(K * 0.5)` visible snapshots (post-lag) are unconditionally retained. This directly implements the StreamingLLM "sink + window" insight.
3. **Compressive merge as fallback**: when the buffer is full, the two lowest-scoring non-pinned, non-lag-protected slots are merged via importance-weighted average rather than discarding one. The merge is differentiable.

This approach is chosen over pure retrieval (Memorizing Transformers) because the DNB preserves causal ordering via the lag constraint — retrieval would require scoring against a query vector that was not available at push time, breaking the lag guarantee. The compressive merge is preferred over pure eviction because it preserves an approximation of all historical content at no additional memory cost.

---

## 4. Mathematical Formulation

### 4.1 Importance Score Accumulation

Let `S = (s_1, ..., s_K)` be the buffer in temporal order, `s_1` oldest. Let `A_t in R^{H x T x K}` be the SNC cross-attention weight tensor at decode step `t` (computed at `snc_cross_attn.py:91`, shape `[batch=1, num_heads H, seq_len T, notes_len K]`).

Define the importance EMA for slot `i` after step `t`:

```
omega_i^t = alpha * omega_i^{t-1} + (1 - alpha) * mean_{h,j} A_t[h, j, i]
```

`alpha = 0.9` (configurable as `RetentionConfig.ema_decay`). Initial value: `omega_i^0 = score_floor = 1e-6`. The mean over `h` and `j` produces a scalar per slot, so the update is `O(HTK)` FLOP — negligible at H=32, T=512, K=4.

### 4.2 Eviction Selection

When `_compact()` is triggered, identify the candidate set:

```
candidates = {
    i : 0 <= i < K,
    i not in recency_window,       # recency_window = {K - K_recent, ..., K-1}
    i not in lag_window,            # lag_window = {K - lag, ..., K-1}
    not snapshots[i].metadata.get("pin", False)
}
```

where `K_recent = max(1, ceil(K * recency_pinned_fraction))`.

If `candidates` is empty: fall back to FIFO (evict `s_1`). This is a safety valve ensuring the buffer always makes progress when every slot is protected.

Otherwise:

```
i* = argmin_{i in candidates} omega_i
```

If `compression_enabled` and `|candidates| >= 2`:

```
j* = second argmin_{i in candidates, i != i*} omega_i
merged = compress(s_{i*}, s_{j*}, omega_{i*}, omega_{j*})
buffer = insert(buffer \ {s_{i*}, s_{j*}}, merged, at=min(i*, j*))
```

If not: `buffer = buffer \ {s_{i*}}`.

### 4.3 Compressive Merge

Given snapshots `s_a` and `s_b` with notes tensors `n_a, n_b in R^D` and importances `omega_a, omega_b`:

```
w_a = omega_a / (omega_a + omega_b)
w_b = omega_b / (omega_a + omega_b)    -- w_a + w_b = 1

n_merged = w_a * n_a + w_b * n_b          -- [D], differentiable
omega_merged = omega_a + omega_b           -- importance mass is conserved
version_merged = max(version_a, version_b) -- monotonicity preserved
stride_merged = stride_a + stride_b        -- tokens-covered additive
metadata_merged = {}                        -- merged slot has no pin/plan marker
```

The merge is fully differentiable. `n_a` and `n_b` are outputs of the `NotesHead` projector from earlier micro-steps, so gradients can flow from the current loss through `n_merged` back to the notes head weights (gated by `RetentionConfig.detach_compressed_grads`).

### 4.4 Differentiable Eviction Network (Phase 5 extension)

A 2-layer MLP `f_evict: R^{D+1} -> R` with GELU activation:

```
logit_i = f_evict(LayerNorm(n_i) || omega_i)   -- concat importance score
p_evict_i = sigmoid(logit_i)
```

During training, soft selection via Gumbel-softmax with temperature `tau`:

```
z_i = (logit_i + Gumbel(0, 1)) / tau
soft_mask_i = softmax(z_i)   -- [K], sums to 1
```

Retention auxiliary loss using future importance as label (stop-grad):

```
L_retain = -sum_i [
    sg(omega_i^{t+H}) * log(1 - p_evict_i)
  + sg(1 - omega_i^{t+H}) * log(p_evict_i)
]
```

where `sg(.)` is stop-gradient and `H` is the lookahead horizon (default: 4 steps, one full cadence cycle).

---

## 5. Component Design

### `src/parallel_decoder_transformer/inference/retention.py` (new file)

**Responsibilities:** All retention scoring and policy logic. No dependency on trainer, model, or collator.

**Key types:**

```python
@dataclass(slots=True)
class RetentionConfig:
    mode: Literal["fifo", "scored", "compressed", "hybrid"] = "fifo"
    ema_decay: float = 0.9
    recency_pinned_fraction: float = 0.5
    compression_enabled: bool = True
    detach_compressed_grads: bool = True
    eviction_network_enabled: bool = False
    eviction_network_hidden: int = 128
    score_floor: float = 1e-6
    min_score_spread: float = 0.05  # fallback-to-FIFO threshold


class SnapshotScoreBank:
    """Running EMA importance scores keyed by snapshot version."""
    # Internal state: Dict[int, float] -- version -> omega
    def update(self, attn_weights: torch.Tensor, slot_versions: List[int]) -> None:
        # attn_weights: [1, H, T, K] detached
        # slot_versions: [K] -- version of each visible slot
        ...
    def scores_for(self, snapshots: List[Snapshot]) -> torch.Tensor:
        # Returns [K] float32 tensor of omega values
        ...
    def reset_slot(self, version: int) -> None: ...


class RetentionPolicy:
    def __init__(self, config: RetentionConfig, max_snapshots: int) -> None: ...

    def select_eviction(
        self,
        buffer: List[Snapshot],
        scores: torch.Tensor,   # [K]
        lag: int,
    ) -> Tuple[int, Optional[int]]:
        # Returns (evict_idx, merge_with_idx)
        # merge_with_idx is None for pure-drop mode
        ...


def compress_snapshots(
    a: Snapshot,
    b: Snapshot,
    score_a: float,
    score_b: float,
    *,
    detach: bool = True,
) -> Snapshot: ...
```

### `src/parallel_decoder_transformer/inference/dnb_bus.py` (modified)

**New field on `DynamicNotesBusConfig`:**
```python
retention: Optional[RetentionConfig] = None
```

**New instance attributes on `DynamicNotesBus`:**
```python
self._score_bank: Optional[SnapshotScoreBank]   # None when retention is None
self._retention_policy: Optional[RetentionPolicy]
```

**New method:**
```python
def update_scores(
    self,
    attn_weights: torch.Tensor,  # [1, H, T, K] -- detached
    slot_versions: List[int],    # length K
) -> None:
    if self._score_bank is not None:
        self._score_bank.update(attn_weights, slot_versions)
```

**Modified `_compact()`:** as described in Section 5 of the Implementation Map above.

### `src/parallel_decoder_transformer/inference/snc_cross_attn.py` (modified)

**Modified `forward()` signature:**
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    notes: torch.Tensor,
    *,
    notes_mask: Optional[torch.Tensor] = None,
    force_gate: Optional[torch.Tensor | bool] = None,
    return_attn_weights: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

When `return_attn_weights=True`, returns `(output, attn_weights.detach())` where `attn_weights: [B, H, T, K]`.

### `src/parallel_decoder_transformer/models/snc_backend.py` (modified)

`PostTrunkSNC.apply()` calls `forward(return_attn_weights=True)` and caches the result in `_last_attn_weights`. `MidStackSNC.apply()` continues to return `hidden` unchanged and sets `_last_attn_weights = None`.

### `src/parallel_decoder_transformer/inference/orchestrator.py` (modified)

After each model forward pass within the per-stream decode step:

```python
snc_backend = self._resolve_snc_backend(stream)  # PostTrunkSNC
if snc_backend.last_attn_weights is not None:
    bus = self.bus_by_stream[stream]
    if bus.config.retention is not None:
        visible = bus.snapshot(lag=self.config.read_lag_delta, limit=self.config.max_snapshots_K)
        slot_versions = [s.version for s in visible]
        bus.update_scores(snc_backend.last_attn_weights, slot_versions)
```

### `src/parallel_decoder_transformer/inference/config.py` (modified)

```python
from .retention import RetentionConfig

@dataclass(slots=True)
class InferenceConfig:
    ...
    retention: Optional["RetentionConfig"] = None
```

Propagated through `build_inference_config()` as a pass-through optional.

### `src/parallel_decoder_transformer/training/trainer.py` (modified)

1. `LossWeights` gets `retain: float = 0.0`.
2. `_update_student_bus()` replaces the unconditional FIFO shift at lines 1669-1705 with coverage-scored eviction when `coverage_bus is not None` and score spread exceeds `min_coverage_confidence`.
3. New `_retention_loss()` method, gated by `self.config.loss_weights.retain > 0`, computing `L_retain` over the current batch's coverage bus.
4. `_compute_losses()` calls `_retention_loss()` with appropriate weight.
5. Metrics dict includes `"retention_loss"` key.

---

## 6. Data Flow Diagram

```
Inference (per decode step, stream s):
─────────────────────────────────────
 hidden_states [1, T, D]
      │
      ▼
 NotesWindowBuilder.build(state_s, bus_by_stream)
   └─ bus_s.snapshot(lag, limit)  ──► [K Snapshot objects]
      │
      ▼
 notes [1, K, notes_dim]
      │
      ▼
 PostTrunkSNC.apply(hidden, notes)
   └─ SharedNotesCrossAttention.forward(return_attn_weights=True)
         attn_weights [1, H, T, K]  ──detach──► _last_attn_weights
         output [1, T, D]
      │
      ▼
 Orchestrator reads snc_backend.last_attn_weights
   └─ bus_s.update_scores(attn_weights, slot_versions)
         SnapshotScoreBank.update()
         EMA: omega[v] = alpha*omega[v] + (1-alpha)*mean(A[:,v])
      │
 [on next bus_s.push()]
      ▼
 bus_s._compact()
   └─ RetentionPolicy.select_eviction(buffer, scores, lag)
         ► evict_idx, merge_idx
   └─ compress_snapshots(s_a, s_b, omega_a, omega_b)  (if merge_idx not None)
         n_merged = w_a*n_a + w_b*n_b   [D, differentiable]


Training (per micro-step, per batch item):
──────────────────────────────────────────
 student_notes_bus [B, K, S, notes_dim]
 student_bus_coverage [B, K, S]
      │
      ▼
 _update_student_bus()
   └─ coverage_scores = coverage_bus[b].mean(-1)  [K]
   └─ evict_idx = argmin(masked_scores)
   └─ shift bus/mask/coverage/streams/stride/version tensors
      │
      ▼
 L_retain (optional, loss_weights.retain > 0)
   └─ p_evict = sigmoid(f_evict(LayerNorm(notes) || omega))
   └─ loss = BCE(p_evict, sg(future_coverage))
```

---

## 7. Build Sequence Checklist

### Phase 1 — Foundation (no behavior change, all existing tests pass)

- [ ] Create `src/parallel_decoder_transformer/inference/retention.py` with `RetentionConfig`, `SnapshotScoreBank`, `RetentionPolicy`, `compress_snapshots`. `RetentionConfig.mode="fifo"` must be the default and must produce bit-identical behavior to current `_compact()` when wired in.
- [ ] Add `return_attn_weights: bool = False` parameter to `SharedNotesCrossAttention.forward()` in `snc_cross_attn.py`. When `False`, return signature is unchanged: single `torch.Tensor`. When `True`, return `Tuple[torch.Tensor, torch.Tensor]`. Run `uv run pytest tests/unit/test_snc_cross_attn.py -v` to verify no regressions.
- [ ] Add `_last_attn_weights: Optional[torch.Tensor]` caching to `PostTrunkSNC.apply()` in `snc_backend.py`. Set to `None` on `MidStackSNC`. Run `uv run pytest tests/unit/test_snc_cross_attn.py -v`.
- [ ] Add `retention: Optional[RetentionConfig] = None` to `DynamicNotesBusConfig`. When `None`, `_compact()` is the current two-liner unchanged. Run `uv run pytest tests/unit/test_dynamic_notes_bus.py -v`.

### Phase 2 — Scored eviction (inference path)

- [ ] Wire `SnapshotScoreBank` and `RetentionPolicy` into `DynamicNotesBus.__init__()` and `_compact()`.
- [ ] Implement `DynamicNotesBus.update_scores()`.
- [ ] Add `retention: Optional[RetentionConfig] = None` to `InferenceConfig` in `config.py`. Propagate through `build_inference_config()`.
- [ ] Call `bus.update_scores()` in `orchestrator.py` after each SNC application, guarded by `bus.config.retention is not None`.
- [ ] Write `tests/unit/test_retention_policy.py` (15 tests). Run `uv run pytest tests/unit/test_retention_policy.py -v`.
- [ ] Extend `tests/unit/test_dynamic_notes_bus.py` with 6 new tests. Run `uv run pytest tests/unit/test_dynamic_notes_bus.py -v`.
- [ ] Run full suite: `uv run pytest tests/unit/ -v` — all 112 + new tests must pass.

### Phase 3 — Compressive merge (inference path)

- [ ] Implement `compress_snapshots()` in `retention.py` including `detach` parameter.
- [ ] Activate the compression branch in `RetentionPolicy.select_eviction()` when `compression_enabled=True`.
- [ ] Write 6 compression-specific tests in `test_retention_policy.py` (gradient flow, version monotonicity, stride accumulation, buffer length after merge). Run `uv run pytest tests/unit/test_retention_policy.py -v`.
- [ ] Verify `test_notes_window_builder.py` passes — `NotesWindowBuilder` is transparent to snapshot internals. Run `uv run pytest tests/unit/test_notes_window_builder.py -v`.

### Phase 4 — Training integration

- [ ] Add `retain: float = 0.0` to `LossWeights` dataclass in `trainer.py`.
- [ ] Replace FIFO shift in `_update_student_bus()` with coverage-scored eviction. Preserve all tensor-slot side effects (coverage_bus, streams_bus, stride_bus, version_bus) by applying the same permutation to all.
- [ ] Implement `_retention_loss()` in `Trainer`. Pattern: identical structure to `_redundancy_loss()` (line 2561). Wire through `_compute_losses()`.
- [ ] Add `"retention_loss"` to the metrics dict at line 2238.
- [ ] Write `tests/unit/test_retention_training.py` (10 tests). Run `uv run pytest tests/unit/test_retention_training.py -v`.
- [ ] Run full suite: `uv run pytest tests/unit/ -v`.

### Phase 5 — Differentiable eviction network (optional)

- [ ] Implement `EvictionNetwork(nn.Module)` in `retention.py`.
- [ ] Register as optional submodule in `ParallelDecoderModelConfig` / `ParallelDecoderTransformer`.
- [ ] Implement Gumbel-softmax relaxation path in `RetentionPolicy.select_eviction()`.
- [ ] Implement `warmup_retain_steps` curriculum in `Trainer`: for the first N steps with `retain > 0`, only update `EvictionNetwork` parameters; beyond N steps, unlock `NotesHead` gradient path.
- [ ] Add ablation logging: log `eviction_network_entropy` (entropy of `soft_mask`) to metrics.

---

## 8. Ablation Study Design

Four conditions evaluated on the same model checkpoint (baseline FIFO-trained), fine-tuned for 10k steps, 3 seeds each.

| ID | Name | `RetentionConfig.mode` | Compression | Scoring source |
|---|---|---|---|---|
| A | FIFO (baseline) | `"fifo"` | off | None |
| B | H2O-scored | `"scored"` | off | SNC attn EMA |
| C | Coverage-scored | `"scored"` | off | Training coverage bus |
| D | HSRAC (proposed) | `"hybrid"` | on | SNC attn EMA + recency pin |

**Primary metrics per condition:**

1. Perplexity on held-out evaluation set (lower is better).
2. Agreement AUC-ROC at optimal tau (from `write_agreement_threshold()`).
3. Coverage AUC-ROC at optimal tau (from `write_coverage_threshold()`).
4. **Early-snapshot survival rate**: fraction of decode runs where `version=1` snapshot is still in the buffer after 8 pushes. Primary diagnostic for the pathology.
5. **Max-attention snapshot age**: for each SNC forward pass, compute `current_version - argmax_i(mean_{h,j} A[h,j,i])`. Distribution summary (p50, p95). Lower = recency-sufficient; higher = long-range retention exploited.
6. Peak GPU memory (MB) via `torch.cuda.memory_allocated()`.
7. SNC forward latency (ms/step), measured via `orchestrator._timings` (already collected).

**Reporting:** All metrics as mean ± std over 3 seeds. Table comparing A, B, C, D. Condition D is the proposed system; A is the baseline. The paper claims D improves over A on metrics 1-3 while metrics 6-7 are unchanged within measurement error.

---

## 9. Gradient Flow Analysis

### Inference path

No gradients. `attn_weights.detach()` before EMA update. `compress_snapshots(detach=True)` further ensures merged notes do not hold live references. Inference-time compaction is purely stateful.

### Training path — base scored eviction

The indexed shift `bus[index, evict_idx:-1] = bus[index, evict_idx+1:]` on line 1623+. This operation is on a tensor that was moved to device with `.to(self.device)` (creating a new tensor), not the original leaf. The autograd graph is not affected. Coverage scores are computed via `.mean(-1)` on `coverage_bus` which is also a device-moved copy. Gradient flow through `_update_student_bus()` is unchanged from the current FIFO implementation: gradients do not flow through the bus update (the bus is not in the backward graph; it is a running state tensor, not a leaf requiring grad).

### Training path — compressive merge with `detach_compressed_grads=False`

When compression is enabled in training and `detach_compressed_grads=False`, `n_merged = w_a * n_a + w_b * n_b` creates a node in the computation graph. `n_a` and `n_b` are slices of `student_notes_bus` which came from a previous micro-step's `NotesHead.forward()` output stored in the batch dict. The gradient path from current-step losses through `n_merged` back to the notes head is:

```
L_t --> snc_attn(notes_t, n_merged) --> n_merged = w_a * n_a + w_b * n_b
     --> n_a (from step t-k), n_b (from step t-j) --> NotesHead weights
```

This creates cross-micro-step gradient dependencies, which is desirable (the notes head learns to produce compressible-yet-informative outputs) but increases peak memory proportionally to the number of retained gradient-connected snapshots. Use `detach_compressed_grads=True` at stages 0-3 and `=False` at stage 4+ to manage memory.

### `L_retain` gradient path

`L_retain` flows gradients only through `p_evict_i = sigmoid(f_evict(LayerNorm(n_i) || omega_i))`. The `omega_i` is a Python float (not a tensor leaf), so no gradient flows through it. The gradient path is:

```
L_retain --> p_evict_i --> f_evict (EvictionNetwork weights)
                       --> LayerNorm(n_i) --> n_i (notes vector)
                                          --> NotesHead weights (if detach=False)
```

During `warmup_retain_steps`, apply `n_i.detach()` before passing to `f_evict` to isolate eviction network training from the notes head.

---

## 10. Risk Analysis

| Risk | Severity | Mitigation |
|---|---|---|
| Coverage-scored eviction is noisy early in training | Medium | Fall back to FIFO when `max(scores) - min(scores) < min_score_spread` (default 0.05) |
| Compressive merge introduces cross-step gradient dependencies and increases peak memory | Medium | `detach_compressed_grads=True` by default; unlock at stage 4+ |
| `L_retain` creates second-order dependency where notes head optimizes for eviction network | Medium | `warmup_retain_steps` isolates eviction network training for first 1000 steps |
| Lag invariance violated after merge | Low | `RetentionPolicy.select_eviction()` explicitly excludes lag-window slots; enforced by test `test_lag_slots_protected` |
| Version monotonicity violated after merge | Low | `version_merged = max(v_a, v_b)` always > both; enforced by test `test_compression_version_max` |
| Eviction network adds latency at inference | Negligible | 2-layer MLP over K<=8 vectors of dim 2048: <10 microseconds; within noise |
| FIFO regression (scored eviction is worse than FIFO) | Low | Mode defaults to `"fifo"`; scored mode is opt-in; ablation condition A provides regression baseline |
| Snapshot pinning (`metadata["pin"]`) abused to prevent all eviction | Low | If all active slots are pinned, policy falls back to FIFO with a warning log; cannot deadlock |
| deque O(n) deletion at large max_snapshots | Negligible | max_snapshots <= 8 in all current configs; O(8) is constant-time in practice |

---

## 11. Key File Paths

```
src/parallel_decoder_transformer/inference/dnb_bus.py         -- modify
src/parallel_decoder_transformer/inference/retention.py        -- create (new)
src/parallel_decoder_transformer/inference/snc_cross_attn.py  -- modify
src/parallel_decoder_transformer/inference/config.py           -- modify
src/parallel_decoder_transformer/inference/orchestrator.py     -- modify
src/parallel_decoder_transformer/models/snc_backend.py         -- modify
src/parallel_decoder_transformer/training/trainer.py           -- modify
tests/unit/test_retention_policy.py                            -- create (new)
tests/unit/test_retention_training.py                          -- create (new)
tests/unit/test_dynamic_notes_bus.py                           -- extend
```

---
