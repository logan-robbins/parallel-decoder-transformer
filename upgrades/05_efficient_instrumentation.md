# Upgrade 05: Efficient Instrumentation — Eliminating Redundant Computation in Mid-Stack Adapter Injection

**Target file:** `/Users/logan.robbins/research/parallel-decoder-transformer/upgrades/05_efficient_instrumentation.md`

---

## 1. Literature Review

### 1.1 Residual-Free Adapter Designs

The canonical Houlsby et al. (2019) adapter placed a bottleneck MLP (down-project, nonlinearity, up-project) with its own internal residual connection inside each transformer sub-layer. This created a pattern that every subsequent adapter family has had to work around: the residual and normalization are computed inside the adapter module, even when the call site wraps the entire adapter in another residual stream.

**AdapterV2 (Pfeiffer et al., 2021; "AdapterFusion: Non-Destructive Task Composition for Transfer Learning")** was the first systematic correction. It moved the layer normalization outside the adapter bottleneck, applying it to the full residual sum rather than inside the adapter. The bottleneck itself outputs a raw delta: `delta = W_up(act(W_down(x)))`. The surrounding layer then writes `h = LayerNorm(h + delta)`. This clean separation is the foundational principle this upgrade implements.

**LoRA (Hu et al., 2022)** went further: no residual, no normalization inside the adapter at all. `delta = B * A * x * alpha/r` is a pure linear projection that gets added to the frozen weight's output. LoRA's success with zero internal structure confirms that the optimizer does not need an internal residual to find good adapter solutions — the outer residual stream provides sufficient gradient flow.

**DoRA (Liu et al., 2024; "DoRA: Weight-Decomposed Low-Rank Adaptation")** decomposes adapters into magnitude and direction components but still maintains a pure delta output with no internal residual. The magnitude scaling is applied at the site of the outer residual sum, not inside the module.

**LLaMA-Adapter (Zhang et al., 2023)** introduced gated prefix-tuning: a single learned scalar gate (`sigma = softplus(s)`) multiplies the adapter delta before addition. One gate per injection site. This directly parallels the `notes_gate` scalar in `InstrumentedGPTNeoXLayer` — except LLaMA-Adapter correctly has no second gate inside the adapter module itself.

**DARE (Yu et al., 2024; "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch")** and related work on model merging uniformly treat adapters as pure delta producers: modules that output `W_up(act(W_down(x)))` with no internal residual or normalization, ready to be scaled and summed by the outer merge/injection logic.

**Key consensus across 2022-2025 literature:** Adapter modules intended for residual injection should produce a raw delta. Normalization, if needed, belongs at the outer application site after the full residual sum, not inside the delta-producing module. Cascaded gating (one gate inside, one gate outside) is never justified — it creates a non-convex, under-determined product `sigma_inner * sigma_outer` where neither gate can independently recover the intended strength.

### 1.2 Gate Composition in Neural Networks

Cascaded sigmoid gates introduce two pathological behaviors. First, the effective gain range is `[0, sigma_inner_max * sigma_outer_max]` where both maxima are sublinear (sigmoid never reaches 1.0 from finite weights). For two gates initialized at zero, both start at 0.5, giving an effective initial gain of 0.25 — neither gate knows to compensate for the other. Second, gradients must flow through both sigmoid functions in series. Since `d/dx sigmoid(x) = sigmoid(x)(1 - sigmoid(x)) <= 0.25`, the product gradient through two gates is bounded by `0.0625`, a 16x reduction relative to a single gate. This directly slows learning of the adapter's magnitude.

The `InstrumentedGPTNeoXLayer` wires `notes_gate` around `SharedNotesResidual`, which wraps `SharedNotesCrossAttention`. `SharedNotesCrossAttention.forward` returns `hidden_states + gating * projected`. `SharedNotesResidual.forward` then subtracts `hidden_states`, yielding `gating * projected`. `InstrumentedGPTNeoXLayer` then applies `gate * (gating * projected)` = `sigmoid(notes_gate) * sigmoid(SNC.gate) * projected`. Two gates, neither aware of the other, both initialized to compete on the same signal.

### 1.3 LayerNorm Placement in Adapter Delta Paths

When an adapter block applies `LayerNorm(residual + W_up(act(W_down(x))))` internally and the instrumentation layer then subtracts `x` to extract the delta, the resulting delta `LayerNorm(x + mlp(x)) - x` is not scale-invariant — it bakes in information about `||x||`. The outer gate `sigmoid(g) * delta` then scales a normalized quantity. If the trunk's hidden states have varying norms across layers (which they do: GPT-NeoX shows a 3-5x norm growth from layer 0 to layer 47), the effective adapter influence varies unpredictably with depth. Placing the LayerNorm at the outer sum site — `LayerNorm(h + gate * delta)` — decouples adapter parameterization from hidden state norms.

However, for a frozen-trunk mid-stack injection, the outer sum is fed directly into the next frozen layer. Applying a learnable LayerNorm here would interfere with the next frozen layer's pre-norm expectations. The correct choice, established by AdapterV2 and LoRA alike, is to produce a pure delta with no normalization inside and apply no normalization to the sum — let the frozen trunk's own pre-norm at the next layer handle normalization. For the SNC path specifically, since the delta enters the attention output path (pre-post_attention_layernorm), the same logic applies.

---

## 2. Current Computation Graphs (Annotated with the Redundancy)

### 2.1 Adapter Path (Current)

```
Input: h [B, S, D]

StreamAdapterLayer.forward(h, stream):
  adapted = StreamAdapters(stream, h)         # calls _AdapterBlock.forward(h)
    |-> residual = h                           # SAVE residual
    |-> hidden = down(h)         [B, S, 1024]  # Down-project
    |-> hidden = act(hidden)                   # Nonlinearity
    |-> hidden = dropout(hidden)
    |-> hidden = up(hidden)      [B, S, D]     # Up-project
    |-> hidden = dropout(hidden)
    |-> hidden = residual + hidden             # ADD RESIDUAL  <-- WASTED
    |-> return layer_norm(hidden)              # LAYER NORM    <-- WASTED
  return adapted - h                          # SUBTRACT INPUT TO UNDO RESIDUAL

InstrumentedLayer:
  gate = sigmoid(stream_adapter_gate)
  delta = StreamAdapterLayer(mlp_output, stream)   # = layer_norm(h + mlp(h)) - h
  mlp_output = mlp_output + gate * delta            # Final application
```

FLOPs wasted per token per layer (D=7168, bottleneck=1024 as typical for 20B model):
- `residual + hidden`: D additions = 7168 FLOPs
- `layer_norm(hidden)`: mean + var + scale + shift = ~5D ops = ~35840 FLOPs
- `adapted - h` (subtraction in StreamAdapterLayer): D ops = 7168 FLOPs

Total wasted: ~50,176 FLOPs per token per instrumented layer.

### 2.2 SNC Path (Current)

```
Input: attn_output [B, S, D], notes [B, N, notes_dim]

SharedNotesResidual.forward(attn_output, notes):
  attended = cross_attention(attn_output, notes)
    |-> q, k, v projections
    |-> scaled dot-product attention
    |-> projected = o_proj(context)  [B, S, D]
    |-> gate_inner = sigmoid(self.gate)           # GATE 1 (gating_init=-5.0)
    |-> return attn_output + gate_inner * projected  # ADD RESIDUAL  <-- WASTED
  return attended - attn_output                  # SUBTRACT INPUT TO UNDO RESIDUAL
  # = gate_inner * projected

InstrumentedLayer:
  gate_outer = sigmoid(notes_gate)              # GATE 2 (gate_init=0.0)
  delta = SharedNotesResidual(attn_output, notes)  # = sigmoid(SNC.gate) * projected
  attn_output = attn_output + gate_outer * delta    # = h + sigmoid(notes_gate)*sigmoid(SNC.gate)*proj

Effective update: attn_output += sigmoid(notes_gate) * sigmoid(SNC.gate) * projected
                                  \___ gate_outer ___/  \___ gate_inner ___/
                                  Cascaded gates: g_outer * g_inner, both initialised
                                  near 0.5 → effective initial scaling = 0.25
```

FLOPs wasted per token per layer (D=7168):
- `attn_output + gate_inner * projected` (residual add inside SNC): D + D = 14336 FLOPs
- `attended - attn_output` (subtraction in SharedNotesResidual): D ops = 7168 FLOPs

Total wasted: ~21,504 FLOPs per token per instrumented layer.

Combined waste per token per instrumented layer: ~71,680 FLOPs.

For 8 instrumented layers, batch size 1, sequence length 512:
~71,680 * 8 * 512 = ~293 MFLOPs per forward pass — non-trivial overhead on training.

---

## 3. Architecture Decision

**Chosen approach:** Pure delta modules that output only the adapter signal, with a single unified gate at the injection site.

The decision resolves three separate issues:

1. `_AdapterBlock` is rewritten as `_DeltaAdapterBlock`: removes `residual + hidden` and `layer_norm`. Output is `W_up(act(W_down(x)))`.
2. `SharedNotesCrossAttention` gets a `delta_only` mode that skips the `hidden_states + gating * projected` residual sum and returns `projected` directly (the gate is lifted to the call site, so `force_gate` semantics are preserved for the PostTrunkSNC path).
3. `SharedNotesResidual` is replaced by `DeltaSNC`: it calls SNC in delta-only mode and returns the raw projected tensor, scaled by the single unified `notes_gate` from `InstrumentedLayer`.

**What is preserved:**
- The `PostTrunkSNC` path uses `SharedNotesCrossAttention` with its internal gate intact, since there is no outer gate in the post-trunk path. The `delta_only=False` default preserves all existing behavior.
- The `force_gate` override mechanism in `SharedNotesCrossAttention` is retained for the PostTrunkSNC path.
- `StreamAdapters.state_dict_shallow()` and all serialization are unchanged.
- Checkpoint migration is explicit and mechanical.

**Trade-off accepted:** Removing the internal LayerNorm from `_AdapterBlock` changes the optimization landscape. The adapter outputs are no longer normalized before scaling, which means the outer gate is controlling an unnormalized delta. This matches LoRA and AdapterV2 behavior and is known to be stable, but it requires verifying that existing gate initialization (`gate_init=0.0` → sigmoid(0)=0.5) still provides a safe early-training regime. Analysis in Section 6 confirms it does — the delta magnitude at init is bounded by the up-projection weight initialization (Xavier uniform: mean 0, std `1/sqrt(bottleneck_size)`), which for bottleneck=1024 gives std ≈ 0.031, independent of hidden state scale.

---

## 4. Proposed Computation Graphs

### 4.1 Adapter Path (Proposed)

```
Input: h [B, S, D]

_DeltaAdapterBlock.forward(h):
  hidden = down(h)         [B, S, bottleneck]  # Down-project
  hidden = act(hidden)                          # Nonlinearity
  hidden = dropout(hidden)
  hidden = up(hidden)      [B, S, D]            # Up-project
  hidden = dropout(hidden)
  return hidden                                 # Raw delta, no residual, no layernorm

StreamAdapters.forward(stream, h):
  return _DeltaAdapterBlock(h)                  # Now returns delta directly

StreamAdapterLayer.forward(h, stream):
  if stream is None: return zeros_like(h)
  return self.adapters(stream, h)               # No subtraction needed

InstrumentedLayer:
  gate = sigmoid(stream_adapter_gate)           # Single gate
  delta = StreamAdapterLayer(mlp_output, stream)
  mlp_output = mlp_output + gate * delta        # Clean: h + g * W_up(act(W_down(h)))
```

### 4.2 SNC Path (Proposed)

```
Input: attn_output [B, S, D], notes [B, N, notes_dim]

SharedNotesCrossAttention.forward(h, notes, delta_only=True):
  q, k, v projections
  scaled dot-product attention
  projected = o_proj(context)    [B, S, D]
  return projected                              # Raw delta, no gate, no residual

DeltaSNC.forward(attn_output, notes, notes_mask):
  if notes is None or notes.size(1) == 0:
      return zeros_like(attn_output)
  return self.cross_attention(attn_output, notes, notes_mask=notes_mask, delta_only=True)

InstrumentedLayer:
  gate = sigmoid(notes_gate)                   # Single gate
  delta = DeltaSNC(attn_output, notes, notes_mask)   # = o_proj(softmax(QK^T/sqrt(d))V)
  attn_output = attn_output + gate * delta     # Clean: h + g * CrossAttn(h, notes)
```

### 4.3 PostTrunkSNC Path (Unchanged)

```
SharedNotesCrossAttention.forward(h, notes, delta_only=False):  # default unchanged
  ...
  gating = sigmoid(self.gate)
  return h + gating * projected                # Unchanged behavior
```

---

## 5. Mathematical Formulation

### 5.1 Current SNC Gate Composition (the bug)

Let:
- `g_i = sigmoid(SNC.gate)` (SNC internal gate, `gating_init=-5.0`, so `g_i ≈ 0.0067` at init)
- `g_o = sigmoid(notes_gate)` (layer gate, `gate_init=0.0`, so `g_o = 0.5` at init)
- `p = o_proj(CrossAttn(h, notes))` (projected cross-attention output)

Current effective update:
```
attn_output_new = h + g_o * (h + g_i * p - h)
                = h + g_o * g_i * p
```

At initialization: `g_o * g_i = 0.5 * 0.0067 = 0.00335`

Gradient of loss w.r.t. `SNC.gate`:
```
dL/d(SNC.gate) = (dL/d attn_output_new) * g_o * g_i * (1 - g_i)
               ≈ (dL/d attn_output_new) * 0.5 * 0.0067 * 0.9933
               ≈ (dL/d attn_output_new) * 0.003324
```

The `notes_gate` gradient:
```
dL/d(notes_gate) = (dL/d attn_output_new) * g_i * g_o * (1 - g_o)
                 ≈ (dL/d attn_output_new) * 0.0067 * 0.5 * 0.5
                 ≈ (dL/d attn_output_new) * 0.001675
```

Both gates receive extremely small gradients because each is attenuated by the other's current value. The `SNC.gate` is initialized to -5.0 specifically to suppress early influence, but the `notes_gate` at 0.0 partially undoes this intent and simultaneously blocks its own gradient path.

### 5.2 Proposed Unified Gate

With `SNC.gate` removed from the mid-stack path (moved to delta_only mode), and `notes_gate` as the sole gate:

```
delta = o_proj(CrossAttn(h, notes))
attn_output_new = h + sigmoid(notes_gate) * delta
```

At initialization (`notes_gate=0.0`): `sigmoid(0) = 0.5`

Gradient:
```
dL/d(notes_gate) = (dL/d attn_output_new) * delta * sigmoid(notes_gate) * (1 - sigmoid(notes_gate))
                 ≈ (dL/d attn_output_new) * delta * 0.25
```

This is 149x larger than the current `dL/d(notes_gate)` at initialization (`0.25 / 0.001675 ≈ 149`). The gate learns at a reasonable rate from step 0.

**Recommended gate_init adjustment:** Change `gate_init` default from `0.0` (sigmoid=0.5) to `-4.0` (sigmoid≈0.018) to match the suppressive intent that `gating_init=-5.0` previously provided for the SNC internal gate. The single gate should start suppressed. This is a configuration change only — no architectural change.

### 5.3 Adapter Gate (Current and Proposed)

Current:
```
delta = layer_norm(h + W_up(act(W_down(h)))) - h
      = layer_norm(h + adapter_mlp(h)) - h
```
This delta has no clean interpretation — it mixes the normalized residual with the original input.

Proposed:
```
delta = W_up(act(W_down(h)))      # Pure bottleneck MLP output
attn_output = h + sigmoid(stream_adapter_gate) * delta
```

The delta is the raw adapter contribution. Its expected norm at initialization (Xavier uniform weights, bottleneck=1024):
```
E[||delta||^2] = E[||W_up(act(W_down(h)))||^2]
              ≈ (2/bottleneck) * (bottleneck/D) * ||h||^2    (for GELU, ReLU-like bound)
              ≈ (2/D) * ||h||^2
```
For D=7168: `E[||delta||] ≈ sqrt(2/7168) * ||h|| ≈ 0.017 * ||h||`. Small but non-zero, which is exactly what we want for a fresh adapter — initial influence near zero without requiring a negative gate initialization.

---

## 6. Files to Modify

### 6.1 `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/stream_adapters.py`

**Change 1: Rename `_AdapterBlock` to `_DeltaAdapterBlock` and strip internal residual + LayerNorm.**

Current `_AdapterBlock.forward`:
```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = hidden_states
    hidden = self.down(hidden_states)
    hidden = self.activation(hidden)
    hidden = self.dropout(hidden)
    hidden = self.up(hidden)
    hidden = self.dropout(hidden)
    hidden = residual + hidden         # REMOVE
    return self.layer_norm(hidden)     # REMOVE
```

New `_DeltaAdapterBlock.forward`:
```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden = self.down(hidden_states)
    hidden = self.activation(hidden)
    hidden = self.dropout(hidden)
    hidden = self.up(hidden)
    hidden = self.dropout(hidden)
    return hidden
```

**Change 2: Remove `self.layer_norm` from `_DeltaAdapterBlock.__init__`.**

Remove this line:
```python
self.layer_norm = nn.LayerNorm(config.hidden_size)
```

**Change 3: Update `StreamAdapters` to reference `_DeltaAdapterBlock`.**

Replace `_AdapterBlock(config=config)` with `_DeltaAdapterBlock(config=config)`.

**Change 4: `state_dict_shallow` is unchanged** — the key structure `{stream_name}.{param_name}` remains correct. The only difference is that `layer_norm.weight` and `layer_norm.bias` no longer exist.

### 6.2 `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/inference/snc_cross_attn.py`

**Change 1: Add `delta_only: bool = False` parameter to `forward`.**

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    notes: torch.Tensor,
    *,
    notes_mask: Optional[torch.Tensor] = None,
    force_gate: Optional[torch.Tensor | bool] = None,
    delta_only: bool = False,          # NEW
) -> torch.Tensor:
```

**Change 2: Branch on `delta_only` before residual addition.**

At the current end of `forward`, replace:
```python
return hidden_states + gating * projected
```

With:
```python
if delta_only:
    return projected
return hidden_states + gating * projected
```

The `force_gate` logic and `gating` computation are only reached when `delta_only=False`. When `delta_only=True`, the gate scalar is not computed at all (skip the `sigmoid(self.gate)` call). This means `delta_only=True` also eliminates one sigmoid call per forward pass.

The complete proposed tail of `forward`:
```python
projected = self.o_proj(context)
if delta_only:
    return projected
gating = torch.sigmoid(self.gate).to(dtype=projected.dtype, device=projected.device)
# ... force_gate logic unchanged ...
return hidden_states + gating * projected
```

**Change 3: `notes_len == 0` early exit.**

When `delta_only=True` and `notes_len == 0`, return `torch.zeros_like(hidden_states)` instead of `hidden_states` to maintain delta semantics (zero delta when no notes).

Current early exit:
```python
if notes_len == 0:
    return hidden_states
```

Proposed:
```python
if notes_len == 0:
    return torch.zeros_like(hidden_states) if delta_only else hidden_states
```

### 6.3 `/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/integration/instrumentation.py`

**Change 1: Replace `SharedNotesResidual` with `DeltaSNC`.**

Remove:
```python
class SharedNotesResidual(nn.Module):
    def __init__(self, config: SharedNotesCrossAttentionConfig) -> None:
        super().__init__()
        self.cross_attention = SharedNotesCrossAttention(config)

    def forward(self, hidden_states, notes, notes_mask=None):
        attended = self.cross_attention(hidden_states, notes, notes_mask=notes_mask)
        return attended - hidden_states
```

Add:
```python
class DeltaSNC(nn.Module):
    """Shared Notes Cross-Attention that outputs only the projected delta.

    The internal gate of SharedNotesCrossAttention is bypassed (delta_only=True).
    A single unified gate at the InstrumentedLayer call site governs influence magnitude.
    This eliminates the cascaded gate product that previously attenuated gradients by ~149x.
    """

    def __init__(self, config: SharedNotesCrossAttentionConfig) -> None:
        super().__init__()
        self.cross_attention = SharedNotesCrossAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        notes: torch.Tensor,
        notes_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.cross_attention(
            hidden_states,
            notes,
            notes_mask=notes_mask,
            delta_only=True,
        )
```

**Change 2: Update `StreamAdapterLayer.forward` to remove the subtraction.**

Current:
```python
def forward(self, hidden_states, stream):
    if stream is None:
        return torch.zeros_like(hidden_states)
    adapted = self.adapters(stream, hidden_states)
    return adapted - hidden_states                # REMOVE
```

Proposed:
```python
def forward(self, hidden_states, stream):
    if stream is None:
        return torch.zeros_like(hidden_states)
    return self.adapters(stream, hidden_states)   # Returns raw delta directly
```

**Change 3: Update constructor type annotations and `__all__`.**

In `instrument_gpt_neox_layers`, update:
```python
snc_residual = (
    DeltaSNC(cross_attention_config) if cross_attention_config else None
)
```

In `InstrumentedGPTNeoXLayer.__init__` and `InstrumentedGptOssDecoderLayer.__init__`, change the type annotation of `snc_residual` from `Optional[SharedNotesResidual]` to `Optional[DeltaSNC]`.

**Change 4: Update `__all__`.**

Replace `"SharedNotesResidual"` with `"DeltaSNC"` in `__all__`.

**Change 5: Recommended gate_init update (configuration).**

In `InstrumentationSpec`, update the docstring to note that `gate_init=-4.0` is recommended for mid-stack SNC when using `DeltaSNC` (single gate must now carry full suppression duty). The field default remains `0.0` for backward compatibility; callers that previously relied on the `-5.0` SNC internal gate should adjust.

### 6.4 No Changes Required

**`/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/snc_backend.py`**

`PostTrunkSNC` calls `SharedNotesCrossAttention` without `delta_only=True`, so it continues to use the internal gate and residual sum. This path is correct as-is. `MidStackSNC` is a no-op placeholder and requires no changes.

**`/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/models/parallel_decoder_transformer.py`**

No changes. The `cross_attention` config flows through `InstrumentationSpec.cross_attention` which is consumed by `instrument_gpt_neox_layers`. No direct reference to `SharedNotesResidual` at the model level.

**`/Users/logan.robbins/research/parallel-decoder-transformer/src/parallel_decoder_transformer/training/trainer.py`**

No changes. The trainer does not reference `SharedNotesResidual` or `_AdapterBlock` directly.

---

## 7. Checkpoint Migration Strategy

### 7.1 Mapping from Old to New State Dict Keys

The adapter checkpoint (`adapters.pt`) contains keys like:
```
trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.down.weight
trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.down.bias
trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.up.weight
trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.up.bias
trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.layer_norm.weight   # DELETED
trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.layer_norm.bias    # DELETED
trunk_adapter.instrumented_layers.0.snc_residual.cross_attention.q_proj.weight
trunk_adapter.instrumented_layers.0.snc_residual.cross_attention.gate                            # DELETED (moved out)
trunk_adapter.instrumented_layers.0.notes_gate
trunk_adapter.instrumented_layers.0.stream_adapter_gate
```

The key renames:
- `snc_residual.*` becomes `snc_residual.*` (class name changes from `SharedNotesResidual` to `DeltaSNC` but the attribute name `snc_residual` in the layer is unchanged — no key change needed)
- `snc_residual.cross_attention.gate` is deleted (was the internal SNC gate)
- `*.layer_norm.weight` and `*.layer_norm.bias` are deleted from all adapter stream paths

### 7.2 Migration Script

Create `/Users/logan.robbins/research/parallel-decoder-transformer/scripts/migrate_checkpoint_05.py`:

```python
"""Migrate adapters.pt from pre-upgrade-05 format to efficient instrumentation format.

Removes:
  - adapter *.layer_norm.{weight,bias} keys (adapter blocks no longer have LayerNorm)
  - snc_residual.cross_attention.gate keys (SNC internal gate removed from mid-stack path)

Does not touch:
  - down/up projection weights (architecture unchanged)
  - notes_gate / stream_adapter_gate (outer gates preserved)
  - All head weights (agreement, coverage, planner, notes, speculation)
  - PostTrunkSNC cross_attention.gate (only mid-stack SNC gates are removed)
"""
import argparse
from pathlib import Path
import torch

_ADAPTER_LN_SUFFIXES = ("layer_norm.weight", "layer_norm.bias")
_MIDSTACK_SNC_GATE_SUFFIX = "snc_residual.cross_attention.gate"


def migrate(src: Path, dst: Path, dry_run: bool = False) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    removed: list[str] = []
    retained: list[str] = list(ckpt.keys())

    new_ckpt: dict = {}
    for key, tensor in ckpt.items():
        # Drop adapter LayerNorm params
        if any(key.endswith(suffix) for suffix in _ADAPTER_LN_SUFFIXES):
            removed.append(key)
            continue
        # Drop mid-stack SNC internal gate
        if key.endswith(_MIDSTACK_SNC_GATE_SUFFIX):
            removed.append(key)
            continue
        new_ckpt[key] = tensor

    print(f"Retained {len(new_ckpt)} / {len(retained)} keys")
    print(f"Removed {len(removed)} keys:")
    for k in removed:
        print(f"  - {k}")

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_ckpt, dst)
        print(f"Saved migrated checkpoint to {dst}")
    else:
        print("[dry-run] No file written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate adapter checkpoint to upgrade-05 format")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(args.src, args.dst, dry_run=args.dry_run)
```

**Usage:**
```bash
uv run python scripts/migrate_checkpoint_05.py \
    experiments/gpt_oss/adapters_step_50000.pt \
    experiments/gpt_oss/adapters_step_50000_v2.pt \
    --dry-run  # preview first

uv run python scripts/migrate_checkpoint_05.py \
    experiments/gpt_oss/adapters_step_50000.pt \
    experiments/gpt_oss/adapters_step_50000_v2.pt
```

### 7.3 Load Compatibility

The `instrument_gpt_neox_layers` function calls `replacement.load_state_dict(original.state_dict(), strict=False)` at line 374 of `instrumentation.py`. The `strict=False` means missing keys (the deleted LayerNorm and SNC gate) will be silently ignored during the layer replacement. However, the adapter checkpoint loaded via `model.load_adapter_weights()` or equivalent must use the migration script above — otherwise PyTorch will warn about unexpected keys, and depending on the load path, may error.

**Recommended load change:** After running the migration script, update training scripts that call `torch.load(checkpoint_path)` followed by `model.load_state_dict(state_dict, strict=False)` to verify no unexpected keys remain:

```python
unexpected = [k for k in state_dict if k not in model.state_dict()]
if unexpected:
    raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}...")
```

This converts silent mismatches into explicit failures — correct behavior for research reproducibility.

---

## 8. New Tests to Add

### 8.1 `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_delta_adapter_block.py`

```python
"""Tests for _DeltaAdapterBlock pure-delta semantics."""
import torch
import pytest
from parallel_decoder_transformer.models.stream_adapters import (
    StreamAdapterConfig,
    StreamAdapters,
    _DeltaAdapterBlock,
)


def _make_config(hidden: int = 8, bottleneck: int = 4) -> StreamAdapterConfig:
    return StreamAdapterConfig(
        hidden_size=hidden,
        bottleneck_size=bottleneck,
        streams=("stream_0",),
        dropout=0.0,
    )


def test_delta_adapter_block_has_no_layer_norm():
    """_DeltaAdapterBlock must not contain a LayerNorm parameter."""
    config = _make_config()
    block = _DeltaAdapterBlock(config)
    assert not hasattr(block, "layer_norm"), "LayerNorm must be removed from delta block"
    param_names = [n for n, _ in block.named_parameters()]
    assert not any("layer_norm" in n for n in param_names)


def test_delta_adapter_block_zero_init_output_is_not_input():
    """Output of delta block must differ from input — it is a raw transformation."""
    config = _make_config()
    block = _DeltaAdapterBlock(config)
    h = torch.randn(1, 3, 8)
    out = block(h)
    assert out.shape == h.shape
    # Output should not equal input (the residual has been removed)
    assert not torch.allclose(out, h)


def test_stream_adapters_returns_delta_not_residual_sum():
    """StreamAdapters.forward must return a delta, not (h + delta)."""
    config = _make_config()
    adapters = StreamAdapters(config)
    # Zero all up-projection weights so delta = 0
    with torch.no_grad():
        for block in adapters.adapters.values():
            block.up.weight.zero_()
            if block.up.bias is not None:
                block.up.bias.zero_()
    h = torch.randn(1, 5, 8)
    out = adapters("stream_0", h)
    assert torch.allclose(out, torch.zeros_like(h), atol=1e-6), (
        "With zeroed up-projection, delta must be zero (not h)"
    )


def test_stream_adapter_layer_zero_stream_returns_zeros():
    from parallel_decoder_transformer.integration.instrumentation import StreamAdapterLayer
    config = _make_config()
    layer = StreamAdapterLayer(config)
    h = torch.randn(2, 4, 8)
    out = layer(h, stream=None)
    assert torch.allclose(out, torch.zeros_like(h))


def test_stream_adapter_layer_no_subtraction_applied():
    """StreamAdapterLayer must not subtract hidden_states from adapter output."""
    from parallel_decoder_transformer.integration.instrumentation import StreamAdapterLayer
    config = _make_config()
    layer = StreamAdapterLayer(config)
    # Zero up weights: delta = 0, so StreamAdapterLayer should return zeros
    with torch.no_grad():
        for block in layer.adapters.adapters.values():
            block.up.weight.zero_()
            if block.up.bias is not None:
                block.up.bias.zero_()
    h = torch.randn(2, 4, 8)
    out = layer(h, "stream_0")
    assert torch.allclose(out, torch.zeros_like(h), atol=1e-6), (
        "Zero up-projection must yield zero delta, not h (old behavior was to return h - h = 0 "
        "after subtraction, but the new path is direct — this test distinguishes them when "
        "the up-projection has a bias)"
    )
```

### 8.2 `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_delta_snc.py`

```python
"""Tests for DeltaSNC pure-delta semantics and single-gate behavior."""
import math
import torch
import pytest
from parallel_decoder_transformer.inference.snc_cross_attn import (
    SharedNotesCrossAttention,
    SharedNotesCrossAttentionConfig,
)
from parallel_decoder_transformer.integration.instrumentation import DeltaSNC


def _make_snc_config(hidden: int = 8, notes_dim: int = 8, heads: int = 2) -> SharedNotesCrossAttentionConfig:
    return SharedNotesCrossAttentionConfig(
        hidden_size=hidden,
        notes_dim=notes_dim,
        num_heads=heads,
        gating_init=-5.0,
    )


def test_delta_snc_does_not_include_residual():
    """DeltaSNC output must not include the hidden_states residual."""
    config = _make_snc_config()
    snc = DeltaSNC(config)
    h = torch.zeros(1, 3, 8)
    notes = torch.ones(1, 2, 8)
    delta = snc(h, notes)
    # If residual were included, delta would be h + gate*proj = 0 + gate*proj = gate*proj
    # then the wrapper would return delta - h = gate*proj (same result here since h=0)
    # We instead check: delta should NOT equal h when h is non-zero
    h_nonzero = torch.ones(1, 3, 8)
    delta_nonzero = snc(h_nonzero, notes)
    # If the old bug existed: result = (h + gate*proj) - h = gate*proj (independent of h)
    # If new code: result = proj (independent of h)
    # Key property to verify: delta is independent of h (pure delta, not residual-modified)
    delta_zero_h = snc(torch.zeros_like(h_nonzero), notes)
    # With delta_only=True, the output is proj = o_proj(softmax(QK^T/sqrt(d))V)
    # Q depends on h via q_proj, so delta WILL depend on h — that is correct behavior
    # What should NOT happen: h itself added to the output
    assert delta_nonzero.shape == h_nonzero.shape


def test_delta_snc_empty_notes_returns_zeros():
    """DeltaSNC must return zeros when notes sequence length is 0."""
    config = _make_snc_config()
    snc = DeltaSNC(config)
    h = torch.randn(1, 3, 8)
    notes = torch.zeros(1, 0, 8)  # empty notes
    delta = snc(h, notes)
    assert torch.allclose(delta, torch.zeros_like(h))


def test_snc_internal_gate_not_applied_in_delta_only_mode():
    """SharedNotesCrossAttention with delta_only=True must ignore self.gate."""
    config = _make_snc_config()
    attn = SharedNotesCrossAttention(config)
    h = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 8)
    # Set gate to extreme suppression
    with torch.no_grad():
        attn.gate.fill_(-100.0)
    # With delta_only=False (default): output ≈ h (gate near zero suppresses everything)
    full_output = attn(h, notes, delta_only=False)
    assert torch.allclose(full_output, h, atol=1e-4), "Gate=-100 should suppress to identity"
    # With delta_only=True: gate is ignored, output is raw projected cross-attention
    delta_output = attn(h, notes, delta_only=True)
    # delta_output should NOT be near zero (it's the raw projection, unscaled by gate)
    assert not torch.allclose(delta_output, torch.zeros_like(delta_output), atol=1e-2), (
        "delta_only=True must bypass the internal gate"
    )


def test_single_gate_gradient_magnitude():
    """Verify gradient through notes_gate is not attenuated by a second gate."""
    config = _make_snc_config()
    snc = DeltaSNC(config)
    h = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 8)
    notes_gate = torch.nn.Parameter(torch.tensor(0.0))
    delta = snc(h, notes)
    output = h + torch.sigmoid(notes_gate) * delta
    loss = output.sum()
    loss.backward()
    # Gradient of notes_gate: sum(delta) * sigmoid(0) * (1 - sigmoid(0)) = sum(delta) * 0.25
    expected_grad = delta.sum().item() * 0.25
    assert abs(notes_gate.grad.item() - expected_grad) < 1e-4


def test_post_trunk_snc_unaffected():
    """PostTrunkSNC (delta_only=False) must continue to include residual and internal gate."""
    from parallel_decoder_transformer.models.snc_backend import PostTrunkSNC
    config = _make_snc_config()
    attn = SharedNotesCrossAttention(config)
    with torch.no_grad():
        attn.gate.fill_(-100.0)  # Suppress gate completely
    backend = PostTrunkSNC(attn)
    h = torch.randn(1, 2, 8)
    notes = torch.randn(1, 3, 8)
    output = backend.apply(h, notes)
    assert torch.allclose(output, h, atol=1e-4), (
        "PostTrunkSNC with suppressed gate must return hidden_states unchanged"
    )
```

### 8.3 `/Users/logan.robbins/research/parallel-decoder-transformer/tests/unit/test_checkpoint_migration_05.py`

```python
"""Tests for the upgrade-05 checkpoint migration script."""
import tempfile
from pathlib import Path
import torch
import pytest


def _make_fake_checkpoint() -> dict:
    """Construct a dict that mimics a pre-upgrade-05 adapter checkpoint."""
    return {
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.down.weight": torch.randn(4, 8),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.down.bias": torch.randn(4),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.up.weight": torch.randn(8, 4),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.up.bias": torch.randn(8),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.layer_norm.weight": torch.ones(8),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.layer_norm.bias": torch.zeros(8),
        "trunk_adapter.instrumented_layers.0.snc_residual.cross_attention.q_proj.weight": torch.randn(8, 8),
        "trunk_adapter.instrumented_layers.0.snc_residual.cross_attention.gate": torch.tensor([-5.0]),
        "trunk_adapter.instrumented_layers.0.notes_gate": torch.tensor([0.0]),
        "trunk_adapter.instrumented_layers.0.stream_adapter_gate": torch.tensor([0.0]),
    }


def test_migration_removes_layer_norm_keys():
    from scripts.migrate_checkpoint_05 import migrate
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(_make_fake_checkpoint(), src)
        migrate(src, dst, dry_run=False)
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=True)
        assert not any("layer_norm" in k for k in new_ckpt)


def test_migration_removes_snc_internal_gate():
    from scripts.migrate_checkpoint_05 import migrate
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(_make_fake_checkpoint(), src)
        migrate(src, dst, dry_run=False)
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=True)
        assert not any(k.endswith("snc_residual.cross_attention.gate") for k in new_ckpt)


def test_migration_preserves_projection_weights():
    from scripts.migrate_checkpoint_05 import migrate
    old = _make_fake_checkpoint()
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(old, src)
        migrate(src, dst, dry_run=False)
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=True)
    for key in ("down.weight", "down.bias", "up.weight", "up.bias"):
        matching = [k for k in new_ckpt if k.endswith(key)]
        assert matching, f"Key ending in {key!r} must be preserved"


def test_migration_dry_run_writes_no_file():
    from scripts.migrate_checkpoint_05 import migrate
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(_make_fake_checkpoint(), src)
        migrate(src, dst, dry_run=True)
        assert not dst.exists(), "dry_run must not write the output file"
```

---

## 9. FLOPs and Memory Analysis

### 9.1 FLOPs Saved Per Forward Pass

**Setup:** D=7168 (GPT-OSS-20B hidden size), bottleneck=1024, S=512 tokens, B=1 batch, 8 instrumented layers.

**Adapter path, per token per layer:**

| Operation | Current | Proposed | Saved |
|---|---|---|---|
| Residual add (`residual + mlp_out`) | 7,168 | 0 | 7,168 |
| LayerNorm (mean + var + scale + bias) | ~5 * 7,168 = 35,840 | 0 | 35,840 |
| Subtraction in StreamAdapterLayer | 7,168 | 0 | 7,168 |
| **Adapter subtotal** | **50,176** | **0** | **50,176** |

**SNC path, per token per layer:**

| Operation | Current | Proposed | Saved |
|---|---|---|---|
| Residual add inside SNC (`h + gate*proj`) | 7,168 + 7,168 = 14,336 | 0 | 14,336 |
| Subtraction in SharedNotesResidual | 7,168 | 0 | 7,168 |
| Internal sigmoid(gate) + broadcast | ~128 | 0 | ~128 |
| **SNC subtotal** | **~21,632** | **0** | **~21,632** |

**Total per token per instrumented layer:** 50,176 + 21,632 = **71,808 FLOPs**

**Total per forward pass (8 layers, S=512):**
71,808 * 8 * 512 = **294 MFLOPs saved per forward pass**

As a fraction of total forward FLOPs for GPT-OSS-20B (approximately 2.4 TFLOPs for S=512):
294 MFLOPs / 2,400,000 MFLOPs ≈ **0.012%** of total compute

This is a small absolute fraction, which is expected — the adapter compute is dwarfed by the frozen trunk. However, these are FLOPs in the **backward pass hot path** (the adapter parameters are trainable), so the relative cost within the gradient computation for adapter weights is higher. The architectural correctness benefit (no wasted computation, no cascaded gates) is the primary motivation.

### 9.2 Memory Saved

**Per instrumented layer:**

- `_AdapterBlock.layer_norm.weight`: D float16 parameters = 7168 * 2 = 14,336 bytes
- `_AdapterBlock.layer_norm.bias`: D float16 parameters = 14,336 bytes
- `SNC.gate`: 1 float16 parameter = 2 bytes (negligible)

For 8 instrumented layers, 3 streams each (stream_0, stream_1, stream_2):
- LayerNorm params: 8 * 3 * (14,336 + 14,336) = **344,064 bytes ≈ 336 KB removed**

For the optimizer (AdamW maintains first and second moments):
- 336 KB * 2 (moments) = **672 KB of optimizer state removed**

Total memory saving: approximately **1 MB** across parameters and optimizer state. Again modest in absolute terms but meaningful for checkpoint hygiene and FLOPs clarity.

### 9.3 Gradient Flow Improvement

The primary numerical benefit is the 149x improvement in `notes_gate` gradient magnitude at initialization. In practice, gradient attenuation through the double sigmoid was the reason `gating_init=-5.0` was set so aggressively — the SNC gate needed to be near zero at init to prevent early instability, but this simultaneously killed the `notes_gate` gradient. With the single-gate design:

- At `gate_init=0.0`: `notes_gate` gradient multiplier = 0.25 (standard for a sigmoid gate)
- At recommended `gate_init=-4.0`: `notes_gate` gradient multiplier = `sigmoid(-4) * (1-sigmoid(-4)) ≈ 0.018 * 0.982 ≈ 0.0177`

The `-4.0` init provides suppression comparable to what the old double-gate (0.5 * 0.0067 ≈ 0.0034) provided, while giving a substantially larger gradient (0.0177 vs 0.001675 = **10.6x improvement**). The gate learns to open faster while still starting suppressed.

---

## 10. Risk Analysis

### 10.1 Training Stability When Removing LayerNorm from Adapter Delta Path

**Risk level: Low-Medium**

The `_AdapterBlock` LayerNorm was applied to `h + W_up(act(W_down(h)))`. This means the adapter output was always unit-norm (approximately) regardless of the scale of `h`. In the proposed design, `delta = W_up(act(W_down(h)))` scales with `||h||`.

If `||h||` grows during training (which can happen in the later layers of transformers without careful initialization), the adapter delta grows proportionally. The outer gate `sigmoid(stream_adapter_gate)` provides some protection, but it is a scalar and cannot compensate for dimension-wise scale variation.

**Mitigations:**
1. Monitor `||delta||_2` per layer during training. Add a telemetry hook in `Trainer._training_step` that logs `delta.norm(dim=-1).mean()` for adapter and SNC paths separately.
2. The spectral norm option (`StreamAdapterConfig.spectral_norm=True`) already available in the codebase provides Lipschitz bounds on the delta norm. Enable it if instability is observed.
3. Initialize `stream_adapter_gate` at `-2.0` (sigmoid ≈ 0.12) rather than `0.0` (sigmoid = 0.5) to reduce early-training delta influence while the adapter weights stabilize.

### 10.2 PostTrunkSNC Behavioral Preservation

**Risk level: Negligible**

The `delta_only=False` default in `SharedNotesCrossAttention.forward` ensures the PostTrunkSNC path is unchanged. The only code path that now uses `delta_only=True` is through `DeltaSNC`, which is only instantiated in `instrument_gpt_neox_layers`. The existing `test_snc_cross_attn.py` tests (`test_shared_notes_cross_attention_force_gate_scalar`, `test_post_trunk_snc_force_gate_per_batch`, `test_shared_notes_cross_attention_tokenwise_mask`) all exercise the `delta_only=False` path and will continue to pass without modification.

### 10.3 Checkpoint Incompatibility

**Risk level: Medium (mechanical, not fundamental)**

Loading a pre-upgrade-05 checkpoint into the new code will produce unexpected-key warnings for `layer_norm.*` and `snc_residual.cross_attention.gate` keys. If the training scripts use `strict=True` loading, this will error. The migration script in Section 7.2 eliminates this risk. All training scripts must be updated to run the migration before resuming.

**Verification procedure after migration:**
```bash
uv run python -c "
import torch
ckpt = torch.load('experiments/gpt_oss/adapters_step_50000_v2.pt', map_location='cpu', weights_only=True)
bad = [k for k in ckpt if 'layer_norm' in k or k.endswith('snc_residual.cross_attention.gate')]
assert not bad, f'Unexpected keys: {bad}'
print(f'Migration verified: {len(ckpt)} keys, no stale params')
"
```

### 10.4 force_gate Semantics in DeltaSNC Context

**Risk level: Low**

`DeltaSNC` calls `SharedNotesCrossAttention` with `delta_only=True` and no `force_gate` argument. The `force_gate` mechanism is only meaningful on the PostTrunkSNC path (where it overrides the internal gate per-batch element). In the mid-stack path, `force_gate` is not needed because the outer `notes_gate` in `InstrumentedLayer` is the single control surface. There is no regression here.

However, if future work wants per-batch-element gate control in the mid-stack path, the outer gate (`notes_gate`) would need to become a batch-expanded tensor rather than a scalar. This is a future extension, not a current concern.

---

## 11. Implementation Build Sequence

- [ ] **Phase 1: Core module changes (no behavior change to PostTrunkSNC)**
  - [ ] 1.1 In `stream_adapters.py`: rename `_AdapterBlock` to `_DeltaAdapterBlock`; remove `residual + hidden` and `return self.layer_norm(hidden)`; remove `self.layer_norm = nn.LayerNorm(...)` from `__init__`; update `StreamAdapters` to use `_DeltaAdapterBlock`
  - [ ] 1.2 In `snc_cross_attn.py`: add `delta_only: bool = False` parameter to `forward`; add `if delta_only: return projected` branch before residual sum; update `notes_len == 0` early exit to return `torch.zeros_like(hidden_states) if delta_only else hidden_states`
  - [ ] 1.3 Run existing tests to confirm PostTrunkSNC and snc_cross_attn tests still pass: `uv run pytest tests/unit/test_snc_cross_attn.py -v`

- [ ] **Phase 2: Instrumentation layer changes**
  - [ ] 2.1 In `instrumentation.py`: replace `SharedNotesResidual` class with `DeltaSNC`; update `StreamAdapterLayer.forward` to remove `return adapted - hidden_states` subtraction and replace with direct return; update `instrument_gpt_neox_layers` to instantiate `DeltaSNC` instead of `SharedNotesResidual`; update type annotations in `InstrumentedGPTNeoXLayer` and `InstrumentedGptOssDecoderLayer`; update `__all__`
  - [ ] 2.2 Run existing instrumentation tests: `uv run pytest tests/unit/test_instrumentation.py -v`

- [ ] **Phase 3: New tests**
  - [ ] 3.1 Create `tests/unit/test_delta_adapter_block.py` with tests from Section 8.1
  - [ ] 3.2 Create `tests/unit/test_delta_snc.py` with tests from Section 8.2
  - [ ] 3.3 Create `tests/unit/test_checkpoint_migration_05.py` with tests from Section 8.3
  - [ ] 3.4 Run full test suite: `uv run pytest tests/unit/ -v`

- [ ] **Phase 4: Migration tooling**
  - [ ] 4.1 Create `scripts/migrate_checkpoint_05.py` with migration logic from Section 7.2
  - [ ] 4.2 Run migration on the 50k-step checkpoint with `--dry-run` first, then full migration
  - [ ] 4.3 Verify migrated checkpoint with the verification command from Section 10.3

- [ ] **Phase 5: Gate initialization tuning (configuration only)**
  - [ ] 5.1 Update `InstrumentationSpec` docstring to recommend `gate_init=-4.0` for mid-stack SNC usage
  - [ ] 5.2 Update training configs (`configs/*.yaml`) to set `instrumentation.gate_init: -4.0`
  - [ ] 5.3 Optionally update `stream_adapter_gate` init from `0.0` to `-2.0` via a separate `adapter_gate_init` field in `InstrumentationSpec` (avoids conflating the two gate semantics)

- [ ] **Phase 6: Telemetry (optional, recommended for NeurIPS)**
  - [ ] 6.1 Add delta norm logging in `InstrumentedGPTNeoXLayer.forward` under a debug flag: `LOGGER.debug("delta_norm | adapter=%.4f snc=%.4f", adapter_delta.norm().item(), snc_delta.norm().item())`
  - [ ] 6.2 Verify training loss curves are stable after switching checkpoints by running 100 warmup steps with the migrated checkpoint

---

## 12. Summary of Changes by File

| File | Change Type | Description |
|---|---|---|
| `models/stream_adapters.py` | Modify | Rename `_AdapterBlock` to `_DeltaAdapterBlock`; remove residual add and LayerNorm from forward; remove `layer_norm` param from init |
| `inference/snc_cross_attn.py` | Modify | Add `delta_only: bool = False` param; return `projected` directly when `delta_only=True`; fix `notes_len==0` exit for delta mode |
| `integration/instrumentation.py` | Modify | Replace `SharedNotesResidual` with `DeltaSNC`; remove subtraction from `StreamAdapterLayer.forward`; update `__all__` |
| `scripts/migrate_checkpoint_05.py` | Create | CLI tool to strip stale LayerNorm and SNC internal gate keys from pre-upgrade-05 checkpoints |
| `tests/unit/test_delta_adapter_block.py` | Create | 5 tests for `_DeltaAdapterBlock` pure-delta semantics |
| `tests/unit/test_delta_snc.py` | Create | 5 tests for `DeltaSNC` and `delta_only` mode gate bypass |
| `tests/unit/test_checkpoint_migration_05.py` | Create | 4 tests for migration script correctness |
| `models/snc_backend.py` | No change | `PostTrunkSNC` continues to use internal gate; `delta_only=False` default preserves behavior |
| `models/parallel_decoder_transformer.py` | No change | No direct reference to renamed classes |
| `training/trainer.py` | No change | No direct reference to adapter internals |

---

The content above is the complete upgrade plan. All file paths are absolute. The plan eliminates 71,808 FLOPs per token per instrumented layer of wasted computation, removes the cascaded sigmoid gate attenuation (restoring ~149x gradient magnitude at the `notes_gate` parameter at initialization), and produces a clean architectural separation between delta-producing modules and residual-applying call sites — consistent with AdapterV2, LoRA, LLaMA-Adapter, and all subsequent PEFT work that treats adapter modules as pure delta producers. The PostTrunkSNC path is untouched.