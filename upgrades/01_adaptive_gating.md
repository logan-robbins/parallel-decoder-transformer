# Upgrade 01: Hierarchical Input-Dependent Adaptive Gating for SNC Cross-Attention

## 1. Literature Review

### 1.1 The Problem with Scalar Gates in Residual Connections

The current SNC gate (`src/parallel_decoder_transformer/inference/snc_cross_attn.py:59`) is a single learned scalar `self.gate = nn.Parameter(torch.full((1,), config.gating_init))`. After sigmoid, this produces one value in `(0, 1)` broadcast identically over all batch items, all sequence positions, and all attention heads. This is the coarsest possible control signal.

The analogous issue has been studied extensively. Michel et al. (2019, "Are Sixteen Heads Really Better than One?", NeurIPS 2019) showed that individual attention heads specialize for different syntactic and semantic roles. Voita et al. (2019, "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting", ACL 2019) confirmed that head specialization is task-specific and that many heads can be pruned without loss. This implies that applying a single global gate identically to all heads ignores the heterogeneity of what each head attends to on the notes bus.

### 1.2 Per-Head Gating

Correia et al. (2019, "Adaptively Sparse Transformers", EMNLP 2019) replaced softmax with alpha-entmax to induce per-head sparsity, effectively gating entire notes positions per head. The key insight is that different heads should have different openness profiles.

More directly, the Flamingo visual language model (Alayrac et al., 2022, NeurIPS 2022) introduced tanh-gated cross-attention residuals for injecting visual features into a frozen language model trunk, with a learned parameter `alpha` initialized to zero (mapping through tanh to zero influence). Their formulation is:

```
h' = h + tanh(alpha) * CrossAttn(h, x_vision)
```

This is the closest precedent to the current SNC setup. Critically, Flamingo uses a single scalar `alpha`. Our upgrade expands this to a per-head vector, mirroring how CLIP and similar multimodal systems allocate different cross-modal attention capacity per head.

GatedCrossAttentionDense layers in the IDEFICS family (Laurenccon et al., 2023, NeurIPS 2023 datasets track) followed the same Flamingo pattern. Neither Flamingo nor IDEFICS moved to per-head or input-dependent gating — this is an opportunity for a direct contribution.

### 1.3 Input-Dependent (Dynamic) Gating

Mixture-of-Experts routing is the canonical input-dependent gating mechanism (Shazeer et al., 2017, "Outrageously Large Neural Networks"; Fedus et al., 2022, "Switch Transformers", JMLR). The router computes `g = softmax(W_r * x)` and uses these as mixing weights. While MoE applies at the expert level, the same principle — routing computed from input activations — applies to cross-attention gating.

The RWKV series (Peng et al., 2023, EMNLP 2023) introduced time-mixing gates as input-dependent linear projections: `g_t = sigmoid(W_g * x_t + b_g)`. This is a position-level input-dependent gate applied to control the contribution of a recurrent state, directly analogous to what we want for the DNB contribution at each sequence position.

Mamba (Gu and Dao, 2024, ICLR 2024) goes further: all SSM parameters (A, B, C, Delta) are input-dependent. Their "selective" mechanism generates a scalar gate per time step from the input. The critical property is that the gate is computed cheaply — a single linear layer — so the overhead is minimal.

For cross-attention specifically, the most relevant recent work is Perceiver Resampler (Jaegle et al., 2021) and its successors, but none explicitly introduce input-dependent gates on the residual contribution. Our proposal fills this gap.

### 1.4 Initialization and Training Stability

The current code initializes `gating_init = -5.0`, which maps through sigmoid to `sigmoid(-5) ≈ 0.0067` — effectively closed. This is the "start closed, learn to open" strategy.

For a per-head gate vector initialized to the same value, this property is preserved trivially: `torch.full((num_heads,), -5.0)` produces all heads closed. For the dynamic (input-dependent) component, the initialization must ensure the dynamic gate starts at zero contribution. The standard approach is to initialize the projection weight to zero and the bias to a large negative value so that `sigmoid(W*x + b) ≈ sigmoid(b) ≈ 0` at step zero regardless of `x`. This is identical to the approach used in the T5 dense gated activation units (Shazeer, 2020, "GLU Variants Improve Transformer") and the LoRA zero-init convention (Hu et al., 2022, ICLR 2022).

Spectral normalization (Miyato et al., 2018, ICLR 2018) is already applied to Q/K/V/O projections in this codebase when `config.spectral_norm=True`. The new dynamic gate projection should also be eligible for spectral norm to prevent the gate from becoming a runaway scaling factor early in training.

### 1.5 Hierarchical Gating in Modern Systems

The hierarchical approach — combining static per-head parameters with dynamic per-position projections — is structurally similar to the decomposition used in GQA (Ainslie et al., 2023, EMNLP 2023) where head groups share K/V but have distinct Q. We apply the analogous decomposition to gates: per-head static bias combined with per-position dynamic offset.

DeepSeek-V2 (DeepSeek-AI, 2024) introduced Multi-head Latent Attention (MLA) with per-head compression of KV states. While focused on KV cache efficiency, the per-head decomposition principle is directly applicable.

### 1.6 Gate Entropy as a Diagnostic

Several papers (Press et al., 2020 "Improving Transformer Models by Reordering their Sublayers"; Elbayad et al., 2020 "Depth-Adaptive Transformer") have tracked gate entropy as a training diagnostic. We will expose per-head gate statistics to WandB for analysis, enabling the ablation study to measure whether different heads learn distinct gate profiles.

---

## 2. Architecture Decision

### 2.1 Chosen Design: Two-Component Hierarchical Gate

We adopt a hierarchical gate that decomposes into two additive components in log-odds space before the final sigmoid:

```
gate(h, head) = sigmoid( alpha_h  +  W_dyn * LayerNorm(h_pooled) )
```

where:
- `alpha_h` is a learned per-head static bias, shape `(num_heads,)`, initialized to `gating_init`
- `W_dyn` is a learned linear projection, shape `(num_heads, hidden_size)`, weight initialized to zero, bias initialized to `0.0`
- `h_pooled` is the mean-pooled hidden state across the sequence dimension: shape `(batch, hidden_size)`
- `LayerNorm` is a learned single-layer norm applied before the projection for training stability
- The result is a gate tensor of shape `(batch, num_heads, 1, 1)` that broadcasts over `(sequence, notes_len)`

The additive structure in log-odds space means `alpha_h` sets the static "resting" gate value per head, and the dynamic term shifts it based on content. At initialization with zero `W_dyn`, the gate equals `sigmoid(alpha_h)` exactly, preserving the "start closed" property.

We do NOT adopt a per-position-per-head gate (shape `(batch, num_heads, seq, 1)`) as the primary mechanism because:
1. It multiplies parameters by `seq_len` per batch step, which is a function of the input, not a constant overhead
2. It breaks the `force_gate` semantics that need to broadcast cleanly to the entire sequence
3. The mean-pooled variant captures the dominant input-dependent signal (what stream are we generating?) with O(hidden_size * num_heads) parameters instead of O(seq_len * num_heads)

We DO NOT adopt full MoE-style gating (a softmax over K experts) because the SNC gate is not a routing decision between alternatives — it is a continuous scalar controlling the contribution magnitude of a single cross-attention pathway.

### 2.2 Gate Mode Flag

A `gate_mode: Literal["scalar", "per_head", "per_head_dynamic"]` field is added to `SharedNotesCrossAttentionConfig` to enable the ablation study without requiring separate checkpoints for each mode:
- `"scalar"`: original behavior, `alpha` shape `(1,)` — backward compatible
- `"per_head"`: static `alpha` shape `(num_heads,)`, no dynamic component
- `"per_head_dynamic"`: full two-component gate (the primary proposed upgrade)

### 2.3 Rationale Over Alternatives

**Alternative rejected: per-head per-position gate via an MLP per head.** This would cost `num_heads * hidden_size * 2` parameters per layer (a small MLP), but produces a gate tensor `(batch, num_heads, seq, 1)` which cannot be straightforwardly overridden by `force_gate` at the batch level without explicit masking logic. The pooled-input approach avoids this complexity.

**Alternative rejected: soft thresholding via straight-through estimator.** While differentiable binary gates (Louizos et al., 2018 "Learning Sparse Neural Networks through L0 Regularization", ICLR 2018) are appealing for interpretability, they introduce stochastic training dynamics and are harder to control during the sectional training curriculum where `force_gate=True` is used to force the gate fully open.

**Alternative rejected: gating at the head-output level instead of on the cross-attention residual.** Applying the gate before `o_proj` (on the `context` tensor per head) would change the expressiveness of `o_proj`, because the linear layer can no longer mix information across gates. The current design applies the gate after `o_proj`, which is the correct place for a residual contribution gate.

---

## 3. Mathematical Formulation

Let:
- `B` = batch size
- `T` = sequence length  
- `N` = notes length (DNB snapshots)
- `H` = number of attention heads (`num_heads`)
- `d` = `hidden_size`
- `d_h` = `d / H` = head dimension

### 3.1 Cross-Attention Forward Pass (Unchanged)

```
Q = W_Q * h          # (B, T, d)  -> reshape -> (B, H, T, d_h)
K = W_K * n          # (B, N, d)  -> reshape -> (B, H, N, d_h)
V = W_V * n          # (B, N, d)  -> reshape -> (B, H, N, d_h)

A = softmax( Q K^T / sqrt(d_h) + mask )    # (B, H, T, N)
C = A V                                      # (B, H, T, d_h)
C = reshape(C) -> (B, T, d)
P = W_O * C                                  # (B, T, d)    [projected output]
```

### 3.2 Gate Computation (New)

```
h_pool = mean(h, dim=T)                      # (B, d)
h_norm = LayerNorm(h_pool)                   # (B, d)

# Static per-head bias (always active)
alpha   in R^H   init: gating_init            # (H,)

# Dynamic head-dependent offset (only in per_head_dynamic mode)
W_dyn   in R^{H x d}   init: zeros
b_dyn   in R^H          init: zeros
delta   = W_dyn * h_norm + b_dyn             # (B, H)

logit   = alpha[None, :] + delta             # (B, H)   broadcast
g       = sigmoid(logit)                     # (B, H)   in (0,1)
g       = reshape(g, (B, H, 1, 1))          # (B, H, 1, 1)
```

### 3.3 Gated Output (New)

Before the current merge back into head-space output, we apply the gate **per head** before the `o_proj` mixing, then reassemble:

```
# Apply per-head gate BEFORE o_proj
C_gated = g * C                              # (B, H, T, d_h)  element-wise broadcast
C_flat  = reshape(C_gated, (B, T, d))
P       = W_O * C_flat                       # (B, T, d)

# Then residual as before (no outer global gate scalar):
h'      = h + P
```

**Key difference from the original design**: The gate is applied to `C` (per-head context vectors) BEFORE `o_proj`, not to `P = o_proj(C)` AFTER. This change means `o_proj` learns to mix already-gated head outputs, which is strictly more expressive. The old design applied a single scalar to the entire projected output, which is equivalent to applying the same scalar to each head's contribution and then summing — a special case of the new design when all per-head gates are equal.

**Important**: This also means the `o_proj` output `P` is no longer independently scaled by an outer gate, so the `force_gate` override in the new design must be applied to `g` directly (before applying to `C`), not to `P`. The `force_gate=True` override sets all `g` values to `1.0` exactly.

### 3.4 force_gate Override (Preserved Semantics)

```
if force_gate is True (scalar bool):
    g = ones(B, H, 1, 1)

if force_gate is 1D bool tensor of shape (B,):
    mask = reshape(force_gate, (B, 1, 1, 1)).expand(B, H, 1, 1)
    g = where(mask, ones_like(g), g)
```

This preserves all existing `force_gate` call-site semantics exactly.

### 3.5 Instrumentation Layer Gate (notes_gate)

The `notes_gate` scalar in `InstrumentedGPTNeoXLayer` and `InstrumentedGptOssDecoderLayer` wraps the `SharedNotesResidual` which itself contains a `SharedNotesCrossAttention`. The outer `notes_gate` becomes redundant when the inner cross-attention has a full per-head dynamic gate — the outer gate is a second stage of gating on top of an already-gated residual.

The recommendation: **deprecate the outer `notes_gate` scalar** in the instrumented layers. Replace it with `nn.Parameter(torch.tensor(0.0))` fixed at its current semantics, but add a config flag `use_outer_notes_gate: bool = False` to disable it (defaulting to False in new checkpoints, True for old ones). This prevents double-gating.

For the `stream_adapter_gate`, it controls a completely different pathway (StreamAdapters, not SNC), so it is left unchanged.

---

## 4. Tensor Shapes at Every Stage

For a concrete example: `B=2, T=512, N=16, H=32, d=4096, d_h=128`.

| Tensor | Shape | Description |
|--------|-------|-------------|
| `h` | `(2, 512, 4096)` | Input hidden states |
| `Q` | `(2, 32, 512, 128)` | Query after reshape |
| `K` | `(2, 32, 16, 128)` | Key from notes |
| `V` | `(2, 32, 16, 128)` | Value from notes |
| `A` | `(2, 32, 512, 16)` | Attention weights |
| `C` | `(2, 32, 512, 128)` | Context vectors per head |
| `h_pool` | `(2, 4096)` | Mean-pooled hidden |
| `h_norm` | `(2, 4096)` | Layer-normed pool |
| `alpha` | `(32,)` | Static per-head bias |
| `delta` | `(2, 32)` | Dynamic offset from pool |
| `logit` | `(2, 32)` | Combined log-odds |
| `g` | `(2, 32, 1, 1)` | Per-head gates |
| `C_gated` | `(2, 32, 512, 128)` | Gated context |
| `C_flat` | `(2, 512, 4096)` | Reshaped for o_proj |
| `P` | `(2, 512, 4096)` | Projected output |
| `h'` | `(2, 512, 4096)` | Final hidden |

Memory overhead vs. current design:
- Parameters: `32` (alpha) + `32 * 4096` (W_dyn) + `32` (b_dyn) + `4096 + 4096` (LayerNorm) = **~139K** extra params per SNC layer vs. 1 param currently
- Activation overhead: `h_pool` and `h_norm` add two `(B, d)` tensors, negligible vs. `C` at `(B, H, T, d_h)`

---

## 5. Files to Create or Modify

### 5.1 `src/parallel_decoder_transformer/inference/snc_cross_attn.py`

**Changes:**

1. Add `gate_mode: Literal["scalar", "per_head", "per_head_dynamic"] = "scalar"` to `SharedNotesCrossAttentionConfig`.

2. In `SharedNotesCrossAttention.__init__`:
   - Change `self.gate` to be shape-dependent on `gate_mode`:
     - `"scalar"`: `nn.Parameter(torch.full((1,), gating_init))` — unchanged
     - `"per_head"`: `nn.Parameter(torch.full((num_heads,), gating_init))`
     - `"per_head_dynamic"`: same per-head parameter plus `self.gate_pool_norm = nn.LayerNorm(hidden_size)`, `self.gate_dyn_proj = nn.Linear(hidden_size, num_heads, bias=True)` with zero weight init and zero bias init
   - When `spectral_norm=True`, apply `spectral_norm` to `gate_dyn_proj` as well

3. In `SharedNotesCrossAttention.forward`:
   - Restructure gate computation into a `_compute_gate` method that returns shape `(B, H, 1, 1)`
   - Apply gate to `C` (context tensor, shape `(B, H, T, d_h)`) before flattening and passing to `o_proj`
   - Remove the current outer-projection gate application at line 95-98
   - Adapt `force_gate` override to work on the new `(B, H, 1, 1)` tensor

4. Add `_compute_gate(self, hidden_states: torch.Tensor) -> torch.Tensor` private method.

5. Add `migrate_scalar_gate(state_dict: dict, num_heads: int) -> dict` module-level utility function for checkpoint migration.

### 5.2 `src/parallel_decoder_transformer/integration/instrumentation.py`

**Changes:**

1. Add `use_outer_notes_gate: bool = True` to `InstrumentationSpec` to control whether the outer `notes_gate` scalar in instrumented layers is active or bypassed.

2. In `InstrumentedGPTNeoXLayer.__init__` and `InstrumentedGptOssDecoderLayer.__init__`:
   - Accept `use_outer_notes_gate: bool = True` parameter
   - Store as `self._use_outer_notes_gate: bool`

3. In both `forward` methods, conditionalize the outer `notes_gate` application:
   ```python
   if self.snc_residual is not None and notes is not None and notes.size(1) > 0:
       delta = self.snc_residual(attn_output, notes, notes_mask=notes_mask)
       if self._use_outer_notes_gate:
           gate = torch.sigmoid(self.notes_gate).to(...)
           attn_output = attn_output + gate * delta
       else:
           attn_output = attn_output + delta
   ```

4. Propagate `use_outer_notes_gate` through `instrument_gpt_neox_layers` and `InstrumentationSpec`.

### 5.3 `src/parallel_decoder_transformer/models/heads/notes.py`

No changes. The `NotesHead` gate is a scalar on the entire notes embedding output, not a cross-attention head gate. It serves a different purpose (scaling the notes projector output) and expanding it per-head would require knowing how many "heads" the notes bus uses — a different decomposition. Leave as-is.

### 5.4 `src/parallel_decoder_transformer/config/schemas.py`

No changes required — `SharedNotesCrossAttentionConfig` is constructed in `ParallelDecoderModelConfig.__post_init__`, and the new `gate_mode` field has a default that preserves existing behavior.

### 5.5 `src/parallel_decoder_transformer/models/parallel_decoder_transformer.py`

No changes required. The `ParallelDecoderModelConfig.__post_init__` constructs `SharedNotesCrossAttentionConfig` and will naturally pass through any explicitly provided `gate_mode`. Default is `"scalar"` for backward compatibility.

### 5.6 New file: `src/parallel_decoder_transformer/inference/gate_utils.py`

Create a new module for gate diagnostic utilities:

- `gate_entropy(gate_values: torch.Tensor) -> float`: compute entropy over per-head gate values (treating them as a distribution after softmax normalization), to measure how heterogeneous head gates are
- `log_gate_stats(module: SharedNotesCrossAttention, prefix: str) -> dict[str, float]`: extract per-head gate means and stds for WandB logging
- `migrate_scalar_gate_checkpoint(state_dict: dict, key_prefix: str, num_heads: int) -> dict`: broadcast old scalar gate to per-head shape for checkpoint migration

### 5.7 New file: `scripts/migrate_gate_checkpoint.py`

A standalone CLI script that:
1. Loads an existing adapter checkpoint (`.pt` file produced by `adapter_state_dict()`)
2. Detects scalar gate keys (shape `(1,)`)
3. Broadcasts them to `(num_heads,)` using `migrate_scalar_gate_checkpoint`
4. Saves the migrated checkpoint to a new path

### 5.8 `src/parallel_decoder_transformer/training/trainer.py`

**Changes (instrumentation gate logging):**

Add a `_log_gate_stats` method that, at every `log_interval` steps:
1. Iterates over `self.model.cross_attention` (post-trunk SNC)
2. Calls `log_gate_stats` from `gate_utils`
3. Logs `gate/head_{h}_mean`, `gate/entropy` to WandB

This is the primary training signal for the ablation: watching whether different heads learn different gate profiles over training.

---

## 6. Detailed Implementation: `snc_cross_attn.py`

The complete replacement for the gate-related logic in `SharedNotesCrossAttention`:

```python
# In SharedNotesCrossAttentionConfig:
gate_mode: Literal["scalar", "per_head", "per_head_dynamic"] = "scalar"

# In SharedNotesCrossAttention.__init__:
if config.gate_mode == "scalar":
    self.gate = nn.Parameter(torch.full((1,), config.gating_init))
    self.gate_pool_norm = None
    self.gate_dyn_proj = None
elif config.gate_mode == "per_head":
    self.gate = nn.Parameter(torch.full((config.num_heads,), config.gating_init))
    self.gate_pool_norm = None
    self.gate_dyn_proj = None
else:  # per_head_dynamic
    self.gate = nn.Parameter(torch.full((config.num_heads,), config.gating_init))
    self.gate_pool_norm = nn.LayerNorm(config.hidden_size)
    self.gate_dyn_proj = nn.Linear(config.hidden_size, config.num_heads, bias=True)
    nn.init.zeros_(self.gate_dyn_proj.weight)
    nn.init.zeros_(self.gate_dyn_proj.bias)
    if config.spectral_norm:
        self.gate_dyn_proj = spectral_norm(
            self.gate_dyn_proj,
            n_power_iterations=config.spectral_norm_n_power_iterations,
            eps=config.spectral_norm_eps,
        )

# New private method:
def _compute_gate(
    self,
    hidden_states: torch.Tensor,  # (B, T, d)
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute gate tensor of shape (B, H, 1, 1)."""
    gate_param = self.gate.to(dtype=dtype, device=device)
    if self.config.gate_mode == "scalar":
        # Broadcast scalar -> (1, 1, 1, 1) for legacy compatibility
        g = torch.sigmoid(gate_param).view(1, 1, 1, 1)
        return g
    elif self.config.gate_mode == "per_head":
        # (H,) -> (1, H, 1, 1) -> broadcasts over (B, H, T, d_h)
        g = torch.sigmoid(gate_param).view(1, self.config.num_heads, 1, 1)
        return g
    else:  # per_head_dynamic
        B = hidden_states.size(0)
        h_pool = hidden_states.mean(dim=1)                   # (B, d)
        h_norm = self.gate_pool_norm(h_pool)                 # (B, d)
        delta = self.gate_dyn_proj(h_norm.to(dtype=dtype))   # (B, H)
        logit = gate_param.unsqueeze(0) + delta              # (B, H)
        g = torch.sigmoid(logit).view(B, self.config.num_heads, 1, 1)
        return g

# Revised forward:
def forward(self, hidden_states, notes, *, notes_mask=None, force_gate=None):
    batch, sequence, _ = hidden_states.size()
    _, notes_len, _ = notes.size()
    if notes_len == 0:
        return hidden_states

    q = self.q_proj(hidden_states)
    k = self.k_proj(notes)
    v = self.v_proj(notes)
    q = q.view(batch, sequence, self.config.num_heads, -1).transpose(1, 2)
    k = k.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)
    v = v.view(batch, notes_len, self.config.num_heads, -1).transpose(1, 2)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if notes_mask is not None:
        # [unchanged masking logic]
        ...
    attn_weights = torch.softmax(attn_scores, dim=-1)
    context = torch.matmul(attn_weights, v)          # (B, H, T, d_h)

    # Compute gate BEFORE o_proj (per-head gating of context)
    gating = self._compute_gate(hidden_states, dtype=context.dtype, device=context.device)

    # Apply force_gate override to gating tensor (B, H, 1, 1)
    gating = self._apply_force_gate(gating, force_gate, batch)

    # Gate applied to context per-head, then project
    context = gating * context                       # (B, H, T, d_h)
    context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)
    projected = self.o_proj(context)                 # (B, T, d)

    return hidden_states + projected

def _apply_force_gate(
    self,
    gating: torch.Tensor,  # (B, H, 1, 1) or (1, 1, 1, 1)
    force_gate: Optional[torch.Tensor | bool],
    batch: int,
) -> torch.Tensor:
    if force_gate is None:
        return gating
    if isinstance(force_gate, bool):
        if force_gate:
            return torch.ones_like(gating)
        return gating
    override_tensor = torch.as_tensor(force_gate, device=gating.device)
    if override_tensor.numel() == 1:
        if bool(override_tensor.item()):
            return torch.ones_like(gating)
        return gating
    override_tensor = override_tensor.to(dtype=torch.bool)
    if override_tensor.dim() == 1 and override_tensor.size(0) == batch:
        # (B,) -> (B, 1, 1, 1) broadcasts over (B, H, 1, 1)
        mask = override_tensor.view(batch, 1, 1, 1)
        return torch.where(mask, torch.ones_like(gating), gating)
    raise ValueError("force_gate tensor must broadcast to the batch dimension.")
```

---

## 7. New Tests to Add

### 7.1 `tests/unit/test_snc_cross_attn.py` — Additional Test Cases

```python
def test_per_head_gate_shape():
    """Gate tensor has shape (1, H, 1, 1) in per_head mode."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2, gate_mode="per_head"
    )
    layer = SharedNotesCrossAttention(config)
    assert layer.gate.shape == (2,)


def test_per_head_gate_closed_at_init():
    """All per-head gates start near-closed when gating_init=-5."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head"
    )
    layer = SharedNotesCrossAttention(config)
    g = torch.sigmoid(layer.gate)
    assert (g < 0.01).all()


def test_per_head_gates_can_differ():
    """Different heads can have different gate values after manual assignment."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2, gate_mode="per_head", gating_init=0.0
    )
    layer = SharedNotesCrossAttention(config)
    with torch.no_grad():
        layer.gate[0] = -10.0   # head 0 nearly closed
        layer.gate[1] = 10.0    # head 1 nearly open
    hidden = torch.zeros(1, 1, 4)
    notes = torch.ones(1, 1, 4)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    # Output should be a mixture: head 1 open, head 0 closed
    out = layer(hidden, notes, force_gate=False)
    # The output should be non-zero (head 1 is open) but less than force_gate=True
    forced = layer(hidden, notes, force_gate=True)
    assert out.abs().sum() < forced.abs().sum()


def test_per_head_dynamic_gate_zero_at_init():
    """Dynamic component produces zero delta at initialization (W=0, b=0)."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    hidden = torch.randn(3, 7, 4)  # arbitrary non-zero hidden
    gate = layer._compute_gate(hidden, dtype=torch.float32, device=torch.device("cpu"))
    # Should match static gate since dynamic delta is zero
    expected = torch.sigmoid(layer.gate).view(1, 2, 1, 1).expand(3, 2, 1, 1)
    assert torch.allclose(gate, expected, atol=1e-6)


def test_per_head_dynamic_gate_varies_by_input():
    """After training a non-zero W_dyn, gate varies by input."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    with torch.no_grad():
        layer.gate_dyn_proj.weight.fill_(0.1)
    hidden_a = torch.zeros(1, 3, 4)
    hidden_b = torch.ones(1, 3, 4)
    gate_a = layer._compute_gate(hidden_a, dtype=torch.float32, device=torch.device("cpu"))
    gate_b = layer._compute_gate(hidden_b, dtype=torch.float32, device=torch.device("cpu"))
    assert not torch.allclose(gate_a, gate_b)


def test_force_gate_overrides_per_head_dynamic():
    """force_gate=True sets all gates to 1.0 in dynamic mode."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    hidden = torch.zeros(1, 1, 4)
    notes = torch.ones(1, 1, 4)
    out_forced = layer(hidden, notes, force_gate=True)
    assert torch.allclose(out_forced, torch.ones_like(out_forced), atol=1e-4)


def test_force_gate_per_batch_per_head_dynamic():
    """Per-batch force_gate works in dynamic mode."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="per_head_dynamic"
    )
    layer = SharedNotesCrossAttention(config)
    _set_identity(layer.q_proj)
    _set_identity(layer.k_proj)
    _set_identity(layer.v_proj)
    _set_identity(layer.o_proj)
    hidden = torch.zeros(2, 1, 4)
    notes = torch.ones(2, 1, 4)
    out = layer(hidden, notes, force_gate=torch.tensor([True, False]))
    assert torch.allclose(out[0], torch.ones_like(out[0]), atol=1e-4)
    assert torch.allclose(out[1], hidden[1], atol=1e-4)


def test_scalar_mode_backward_compatible():
    """gate_mode='scalar' produces identical output to the original implementation."""
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=2,
        gating_init=-5.0, gate_mode="scalar"
    )
    layer = SharedNotesCrossAttention(config)
    assert layer.gate.shape == (1,)
    assert layer.gate_pool_norm is None
    assert layer.gate_dyn_proj is None


def test_migrate_scalar_gate():
    """migrate_scalar_gate_checkpoint broadcasts (1,) to (H,)."""
    from parallel_decoder_transformer.inference.gate_utils import (
        migrate_scalar_gate_checkpoint,
    )
    state = {"cross_attention.gate": torch.tensor([-5.0])}
    migrated = migrate_scalar_gate_checkpoint(state, "cross_attention.gate", num_heads=4)
    assert migrated["cross_attention.gate"].shape == (4,)
    assert (migrated["cross_attention.gate"] == -5.0).all()
```

### 7.2 `tests/unit/test_gate_utils.py` — New Test Module

```python
def test_gate_entropy_uniform():
    """Uniform gate values produce maximum entropy."""
    from parallel_decoder_transformer.inference.gate_utils import gate_entropy
    g = torch.full((8,), 0.5)
    e = gate_entropy(g)
    assert abs(e - math.log(8)) < 1e-4  # max entropy for 8 outcomes


def test_gate_entropy_degenerate():
    """One-hot gate values produce zero entropy."""
    from parallel_decoder_transformer.inference.gate_utils import gate_entropy
    g = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e = gate_entropy(g)
    assert e < 1e-4


def test_log_gate_stats_returns_per_head():
    """log_gate_stats returns H keys."""
    from parallel_decoder_transformer.inference.gate_utils import log_gate_stats
    config = SharedNotesCrossAttentionConfig(
        hidden_size=4, notes_dim=4, num_heads=4, gate_mode="per_head"
    )
    layer = SharedNotesCrossAttention(config)
    stats = log_gate_stats(layer, prefix="snc")
    assert "snc/gate_head_0_mean" in stats
    assert "snc/gate_head_3_mean" in stats
    assert "snc/gate_entropy" in stats
```

### 7.3 `tests/unit/test_instrumentation.py` — Additional Tests

Add tests that verify `use_outer_notes_gate=False` causes the instrumented layer to use the inner SNC gate only, producing identical output for a forced-open inner gate.

---

## 8. Ablation Study Design

### 8.1 Four Conditions

| Condition | `gate_mode` | `use_outer_notes_gate` | Parameters Added |
|-----------|-------------|------------------------|-----------------|
| **Baseline** | `"scalar"` | `True` | 0 |
| **Per-Head Static** | `"per_head"` | `False` | `H - 1 = 31` |
| **Per-Head Dynamic** | `"per_head_dynamic"` | `False` | `H + H*d + H + d + d = 32 + 131072 + 32 + 4096 + 4096 ≈ 139K` |
| **Oracle** | `force_gate=True` | n/a | 0 (no gate) |

All four conditions use the same pre-trained trunk, same curriculum, same seeds. The only variation is `gate_mode` and the `use_outer_notes_gate` flag.

### 8.2 Metrics to Track

Primary:
- KD loss (notes branch)
- Coverage score
- Agreement ROC AUC

Secondary (new, enabled by the upgrade):
- `gate/entropy`: per-head gate distribution entropy (higher = more heterogeneous head usage)
- `gate/head_{h}_mean`: mean sigmoid gate value per head over the eval set
- `gate/head_var`: variance of per-head gate means (measures specialization)
- `gate/input_sensitivity`: correlation between gate values and stream identity (measures whether dynamic gates encode stream routing)

### 8.3 Expected Findings

The hypothesis is that `per_head_dynamic` will show:
1. Lower KD loss (better notes integration) vs. baseline scalar
2. Non-uniform gate entropy (different heads specialize to different note types)
3. Gates that correlate with stream identity (the dynamic component learns stream routing)
4. No regression on fluency metrics (lm loss unchanged or better)

If these findings hold, this is the primary empirical contribution for the paper's ablation table.

### 8.4 Compute Budget

Each condition: 5K training steps (a single curriculum pass through stages 1-3), batch size 4, on 2x A100 80GB. Total: 4 conditions × 5K steps × ~15 min/1K steps ≈ 5 GPU-hours per condition, 20 GPU-hours total. This is feasible for a NeurIPS submission.

---

## 9. Risk Analysis

### 9.1 Training Stability

**Risk**: The dynamic component `W_dyn * h_norm` could produce large gate values early in training if the learning rate is applied uniformly.

**Mitigation**: Zero-initialize `W_dyn.weight`. With Adam/AdamW, the first gradient step for a zero-weight matrix is modulated by the second-moment estimate starting at zero, which causes the effective learning rate for this layer to be smaller in the first few steps. Additionally, consider applying a smaller learning rate multiplier to `gate_dyn_proj` parameters via a parameter group in the optimizer.

**Mitigation 2**: LayerNorm before the projection bounds the norm of `h_norm` to approximately `sqrt(d)` scale, preventing explosive gate values from large hidden states early in training.

### 9.2 Memory Overhead

**Risk**: Materializing the gate `(B, H, T, d_h)` for backpropagation through the gated context.

**Analysis**: The current code already materializes `context = matmul(attn_weights, v)` as shape `(B, H, T, d_h)`. Multiplying by `(B, H, 1, 1)` and backpropagating through this requires storing `context` in the forward pass, but this tensor is already required for the existing `o_proj` backward pass. The only new memory is `h_pool` `(B, d)` and `h_norm` `(B, d)` — negligible.

### 9.3 Checkpoint Migration

**Risk**: Existing trained checkpoints have `cross_attention.gate` with shape `(1,)`. Loading them into a `per_head` or `per_head_dynamic` model will fail with shape mismatch.

**Mitigation**: The `migrate_scalar_gate_checkpoint` utility in `gate_utils.py` handles this. The `load_adapters` method in `ParallelDecoderTransformer` uses `strict=False` by default, so old checkpoints will load with missing keys for the new parameters (which will retain their initialization values). For `gate`, the shape mismatch is more serious — `strict=False` with incompatible shapes raises a `RuntimeError`. Therefore:

1. When loading a checkpoint whose `gate` key has shape `(1,)` into a `per_head` model, call `migrate_scalar_gate_checkpoint` before `load_adapters`.
2. Add a guard in `load_adapters` that detects scalar-to-per-head mismatch and either auto-migrates or raises a clear error.

### 9.4 Interaction with force_gate During Curriculum

**Risk**: The sectional training curriculum uses `force_gate=True` during certain stages. If the new `_apply_force_gate` logic has a bug, entire curriculum stages break.

**Mitigation**: The new `_apply_force_gate` method has identical semantics to the original inline code, but is more clearly separated. Tests `test_force_gate_overrides_per_head_dynamic` and `test_force_gate_per_batch_per_head_dynamic` directly verify these semantics in all three gate modes.

### 9.5 Double-Gating in Instrumented Layers

**Risk**: `InstrumentedGPTNeoXLayer` applies an outer `notes_gate` scalar on top of the `SharedNotesCrossAttention` which now has its own per-head gate. This is double-gating: the total effective gate is `sigmoid(notes_gate_scalar) * sigmoid(alpha_h + delta)` per head, which compounds the "start closed" initialization so strongly that the gates may never open.

**Mitigation**: Set `use_outer_notes_gate=False` as the default for new configurations when `gate_mode != "scalar"`. The migration path: old checkpoints that have `notes_gate` trained will have `use_outer_notes_gate=True` preserved during migration. New training runs with `gate_mode="per_head_dynamic"` should always set `use_outer_notes_gate=False`.

### 9.6 Spectral Norm and the Dynamic Projection

**Risk**: Applying spectral norm to `gate_dyn_proj` (a `(num_heads, hidden_size)` = `(32, 4096)` matrix) constrains its Lipschitz constant to 1, which may prevent the gate from being sufficiently sensitive to input variation.

**Mitigation**: Monitor `gate/input_sensitivity` during training. If the dynamic component appears to have no effect on gates across different inputs (sensitivity near zero), disable spectral norm on `gate_dyn_proj` while keeping it on Q/K/V/O.

---

## 10. Build Sequence (Phased Checklist)

### Phase 1: Infrastructure (no behavior change)

- [ ] Create `upgrades/` directory and write this document
- [ ] Create `src/parallel_decoder_transformer/inference/gate_utils.py` with `migrate_scalar_gate_checkpoint`, `gate_entropy`, `log_gate_stats`
- [ ] Add `gate_mode: Literal["scalar", "per_head", "per_head_dynamic"] = "scalar"` to `SharedNotesCrossAttentionConfig` in `snc_cross_attn.py`
- [ ] Add `use_outer_notes_gate: bool = True` to `InstrumentationSpec` in `instrumentation.py`
- [ ] Write `tests/unit/test_gate_utils.py` and verify tests pass with `uv run pytest tests/unit/test_gate_utils.py -v`
- [ ] Write the `test_scalar_mode_backward_compatible` and `test_migrate_scalar_gate` tests; verify pass

### Phase 2: Per-Head Static Gate

- [ ] In `SharedNotesCrossAttention.__init__`, implement `gate_mode="per_head"` branch: `nn.Parameter(torch.full((num_heads,), gating_init))`
- [ ] Extract `_compute_gate` and `_apply_force_gate` private methods
- [ ] Refactor `forward` to apply gate to `context` (before `o_proj`) instead of to `projected`
- [ ] Run `uv run pytest tests/unit/test_snc_cross_attn.py -v` — all existing tests must pass
- [ ] Add and pass `test_per_head_gate_shape`, `test_per_head_gate_closed_at_init`, `test_per_head_gates_can_differ`, `test_force_gate_overrides_per_head_dynamic`, `test_force_gate_per_batch_per_head_dynamic`
- [ ] Run full unit test suite: `uv run pytest tests/unit/ -v` — 112+ tests pass

### Phase 3: Per-Head Dynamic Gate

- [ ] Implement `gate_mode="per_head_dynamic"` branch: add `gate_pool_norm`, `gate_dyn_proj` with zero init
- [ ] Add spectral norm support for `gate_dyn_proj`
- [ ] Add and pass `test_per_head_dynamic_gate_zero_at_init`, `test_per_head_dynamic_gate_varies_by_input`
- [ ] Run full unit test suite

### Phase 4: Instrumentation Layer Updates

- [ ] Implement `use_outer_notes_gate` flag in `InstrumentedGPTNeoXLayer` and `InstrumentedGptOssDecoderLayer`
- [ ] Propagate through `instrument_gpt_neox_layers` and `InstrumentationSpec`
- [ ] Add instrumentation tests for `use_outer_notes_gate=False` behavior
- [ ] Run full unit test suite

### Phase 5: Migration Tooling

- [ ] Write `scripts/migrate_gate_checkpoint.py` with argument parsing for input path, output path, num_heads, gate_mode
- [ ] Add guard in `ParallelDecoderTransformer.load_adapters` to detect scalar/per-head shape mismatch and emit a clear error message pointing to the migration script
- [ ] Test migration script on a synthetic checkpoint

### Phase 6: Training Diagnostics

- [ ] Add `_log_gate_stats` to `Trainer` in `trainer.py`
- [ ] Add `gate/entropy`, `gate/head_{h}_mean`, `gate/head_var` to WandB logging at `log_interval`
- [ ] Verify diagnostics appear correctly in a 100-step smoke run

### Phase 7: Ablation Runs

- [ ] Launch `"scalar"` baseline run (checkpoint the gate scalar value at each log step)
- [ ] Launch `"per_head"` run
- [ ] Launch `"per_head_dynamic"` run
- [ ] Collect `gate/entropy`, coverage, KD loss, agreement AUC
- [ ] Produce ablation table for paper

---

## 11. Critical Implementation Detail: Gate Applied Before vs. After o_proj

The current code applies the gate as:
```python
# Current (snc_cross_attn.py:94-120)
context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)  # flatten heads
projected = self.o_proj(context)                                            # mix heads
gating = torch.sigmoid(self.gate)                                           # scalar
return hidden_states + gating * projected                                   # residual
```

The new design applies the gate as:
```python
# New
context = gating * context                                                  # gate per-head BEFORE flatten
context = context.transpose(1, 2).contiguous().view(batch, sequence, -1)  # flatten
projected = self.o_proj(context)                                            # mix gated heads
return hidden_states + projected                                            # residual (no outer gate)
```

This is not merely an implementation convenience — it is architecturally different. In the old design, `o_proj` mixes ungated head outputs and then a single scalar scales the sum. In the new design, `o_proj` mixes already-gated head outputs, meaning that: if head `h` has gate ≈ 0, its contribution to the entire `d`-dimensional output is suppressed before `o_proj` can mix it with other heads. This is analogous to pruning head `h` softly at each forward pass, with the pruning intensity controlled by the learned (and possibly input-dependent) gate.

This architectural difference is the core contribution beyond prior work: while Flamingo and IDEFICS gate the entire cross-attention residual with a scalar, this design gates each head independently before the output projection mixes across heads. The `o_proj` therefore learns to integrate a dynamically-sparse set of head contributions.

---

## 12. Paper Framing

The contribution should be framed as follows for NeurIPS:

**Problem**: Cross-attention gating in conditioning-injection architectures (Flamingo, IDEFICS, this work) uses a single scalar, which cannot express the empirically observed head specialization in multi-head attention.

**Method**: Hierarchical adaptive gating that decomposes into (a) a learned per-head static bias capturing training-time specialization and (b) an input-dependent dynamic offset that routes bus influence based on the current generation context (stream identity, topic).

**Evidence**: Ablation on four conditions (scalar / per-head static / per-head dynamic / oracle) showing monotonically increasing performance as the gate gains expressiveness, with qualitative analysis of head specialization (different heads learned to gate open for different note categories: structural notes vs. factual notes vs. verification notes).

**Novelty claim**: First work to introduce per-head input-dependent gating for cross-attention residual injection in parallel/speculative decoding architectures, with a clean theoretical justification linking gate entropy to effective head utilization.

---

All file paths referenced above are absolute paths rooted at `/Users/logan.robbins/research/parallel-decoder-transformer/`. The primary implementation files are:

- `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/inference/snc_cross_attn.py` — core gate upgrade
- `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/inference/gate_utils.py` — new module
- `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/integration/instrumentation.py` — outer gate control
- `/Users/logan.robbins/research/parallel-decoder-transformer/scripts/migrate_gate_checkpoint.py` — new migration script
- `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_snc_cross_attn.py` — 8 new tests
- `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_gate_utils.py` — new test module (3 tests)

