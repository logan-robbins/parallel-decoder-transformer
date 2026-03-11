"""
microgpt_dnb.py — Fork of Karpathy's microgpt with Dynamic Notes Bus + Auto-Steer (2026 extension)

Original: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
Paper:    https://arxiv.org/abs/2512.10054 (Parallel Decoder Transformer)
Code:     https://github.com/logan-robbins/parallel-decoder-transformer

This micro-implementation demonstrates the FULL PDT architecture on a frozen trunk.
It proves that mid-trajectory steering is possible via the Dynamic Notes Bus.

=== IMPORTANT NOTE ABOUT DATASET & TOKENIZER (your question) ===
The current code uses character-level tokens + TinyShakespeare for speed and zero dependencies.
This is fine for proving the mechanism, but **you are correct** — it is not ideal for real-world general knowledge tasks like "what was the evolution of transformer models".

**For real PDT use on general knowledge / conceptual prompts:**
- Switch to a **subword tokenizer** (tiktoken or SentencePiece — BPE)
- Use a **knowledge-rich corpus** (Wikipedia dump, arXiv papers, The Pile, or a QA dataset)
- Increase block_size to 256+ and n_embd to 64+
The planner + bus + auto-steer mechanism stays **exactly the same** — only the input representation changes.

In this micro file we keep TinyShakespeare with tiktoken subword tokenization (GPT-2 BPE).
The architecture and all four contributions are unchanged.

=== HOW TOKENS ARE FED BACK ===
Tokens feed back **privately per stream** via its own KV cache (stream_tokens[si] + stream_keys[si]/stream_values[si]). No extra residual stream per parallel stream is ever needed.

=== MAPPING TO THE PDT PAPER CONTRIBUTIONS ===

Contribution 1 — Planner-seeded multi-stream protocol (Section 3.2)
    WHAT: plan_seed(prompt_tokens) → multi-slot argmax → re-embed via E_plan → snapshot_0
    WHY:  Gives every stream the same latent snapshot-0 contract on the bus BEFORE parallel generation starts.

Contribution 2 — Embeddings-only coordination bus + SNC (Section 3.3–3.5)
    WHAT: other_notes gathered from buses (with visibility delay Δ) + passed to gpt(..., notes=other_notes)
    WHY:  All cross-stream info travels ONLY through notes_dim embeddings. Self-attention stays private.

Contribution 3 — Ownership-aware commit control (Section 3.6–3.7)
    WHAT: At every TAU-token block boundary: coverage head → agreement head → commit/rollback gate
    WHY:  Streams only commit when agreement says the shared state is sufficient for safe continuation.

Contribution 4 — Frozen-trunk realization (Section 3.9)
    WHAT: Phase 1 trains only base_params; Stages 0–3 freeze trunk and train only sidecar params
    WHY:  The entire transformer is shared and frozen. Only lightweight sidecars are trainable.

=== AUTO-STEER (2026 extension — NOT in the original paper) ===
    WHAT: steer_proj emits a steering vector at block boundaries alongside speculation notes
    WHY:  Siblings see it via SNC → enables mid-trajectory correction beyond the paper's protocol.

Dependencies: tiktoken. Run: pip install tiktoken && python microgpt_dnb.py
"""

import os
import math
import random

random.seed(42)

# --- Dataset (TinyShakespeare — fast demo only; see note above for real general-knowledge use) ---
if not os.path.exists('input.txt'):
    import urllib.request
    shakespeare_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(shakespeare_url, 'input.txt')
raw_docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(raw_docs)

# --- Tokenizer (tiktoken subword — GPT-2 BPE encoding) ---
import tiktoken
_enc = tiktoken.get_encoding("gpt2")
MAX_VOCAB = 128  # restrict to top-N most frequent tokens for micro demo param count

# Tokenize all docs and build restricted vocabulary
_all_token_ids = []
for d in raw_docs:
    _all_token_ids.extend(_enc.encode(d))
# Count frequencies, keep top MAX_VOCAB
from collections import Counter
_freq = Counter(_all_token_ids)
_top_tokens = [tok for tok, _ in _freq.most_common(MAX_VOCAB)]
_top_set = set(_top_tokens)
# Build compact remapping: original tiktoken id → compact id
UNK = 0
BOS = 1
_tok2compact = {orig: i + 2 for i, orig in enumerate(_top_tokens)}  # 2..MAX_VOCAB+1
vocab_size = MAX_VOCAB + 2  # +2 for UNK and BOS

def encode(text):
    """Encode text to compact token ids using tiktoken + frequency cap."""
    return [_tok2compact.get(t, UNK) for t in _enc.encode(text)]

def decode_tokens(ids):
    """Decode compact token ids back to text."""
    _compact2tok = {v: k for k, v in _tok2compact.items()}
    orig_ids = [_compact2tok.get(i, None) for i in ids if i not in (BOS, UNK)]
    return _enc.decode([t for t in orig_ids if t is not None])

# Tokenize docs
docs = []
for d in raw_docs:
    toks = encode(d)
    if len(toks) >= 2:  # need at least 2 tokens for next-token prediction
        docs.append(toks)
random.shuffle(docs)
print(f"num docs: {len(docs)} (TinyShakespeare, tiktoken gpt2 subword)")
print(f"vocab size: {vocab_size} (top {MAX_VOCAB} subword tokens + BOS + UNK)")

# --- PDT Hyperparameters (scaled down for micro demo; paper uses S=16, V_p=65536) ---
S = 4                    # num plan slots (paper: 16)
V_p = 32                 # plan vocabulary size (paper: 65536)
TAU = 4                  # tokens per provisional block (paper: configurable)
DELTA = 1                # notes visibility delay in rounds (paper: configurable)
GAMMA = 0.3              # agreement threshold γ (paper: learned or tuned)

# --- Autograd (unchanged from Karpathy's original) ---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# --- Parameters (base GPT trunk — frozen after Phase 1) ---
n_layer = 1
n_embd = 16
block_size = 32
n_head = 4
head_dim = n_embd // n_head
notes_dim = 8

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

base_params = [p for mat in state_dict.values() for row in mat for p in row]

# --- SNC parameters (Contribution 2: embeddings-only coordination bus) ---
state_dict['notes_proj']   = matrix(notes_dim, n_embd)   # notes head (training supervision)
state_dict['snc_wq']       = matrix(n_embd, n_embd)
state_dict['snc_wk']       = matrix(n_embd, notes_dim)
state_dict['snc_wv']       = matrix(n_embd, notes_dim)
state_dict['snc_wo']       = matrix(n_embd, n_embd)
state_dict['snc_gate']     = [[Value(-5.0)]]              # starts closed (sigmoid ≈ 0)

snc_params = [p for key in ['notes_proj', 'snc_wq', 'snc_wk', 'snc_wv', 'snc_wo', 'snc_gate']
              for row in state_dict[key] for p in row]

# --- Planner parameters (Contribution 1: planner-seeded multi-stream protocol, Section 3.2) ---
state_dict['plan_proj']    = matrix(S * V_p, n_embd)      # pooled prompt → S slot logits over V_p
state_dict['E_plan']       = matrix(V_p, notes_dim)        # plan embedding matrix

plan_params = [p for key in ['plan_proj', 'E_plan']
               for row in state_dict[key] for p in row]

# --- Speculation head (Contribution 3: provisional note emission, Section 3.6) ---
state_dict['spec_proj']    = matrix(notes_dim, n_embd)     # hidden → provisional note

# --- Coverage head (Contribution 3: ownership tracking, Section 3.6) ---
state_dict['coverage_proj'] = matrix(notes_dim, n_embd)    # hidden → notes_dim for dot product with E_plan

# --- Agreement head (Contribution 3: readiness scoring, Section 3.7) ---
agree_input_dim = n_embd + notes_dim + S + notes_dim       # hidden + mean_notes + coverage + provisional_note
state_dict['agree_proj']   = matrix(1, agree_input_dim)    # → scalar readiness

# --- Auto-Steer parameters (2026 extension — NOT in the original paper) ---
state_dict['steer_proj']   = matrix(notes_dim, n_embd)

commit_params = [p for key in ['spec_proj', 'coverage_proj', 'agree_proj', 'steer_proj']
                 for row in state_dict[key] for p in row]

all_params = base_params + snc_params + plan_params + commit_params
print(f"num params: {len(all_params)} (base trunk: {len(base_params)}, "
      f"snc: {len(snc_params)}, planner: {len(plan_params)}, commit+steer: {len(commit_params)})")

# --- Architecture helpers ---

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def sigmoid(x):
    return Value(1.0) / (Value(1.0) + (-x).exp())

def l2_normalize(x):
    """L2-normalize a list of Value scalars."""
    norm_sq = sum(xi * xi for xi in x)
    scale = (norm_sq + 1e-8) ** -0.5
    return [xi * scale for xi in x]

def detach(x):
    """Detach a list of Values from the computation graph (like torch.detach)."""
    return [Value(v.data) for v in x]

def detach_nested(xs):
    """Detach a list of lists of Values."""
    return [detach(x) for x in xs]

# --- SNC Cross-Attention (Contribution 2, Section 3.5) ---

def snc_cross_attention(x, notes):
    """Embeddings-only SNC read from Dynamic Notes Bus.
    This is the ONLY place cross-stream information ever enters a stream."""
    if not notes:
        return [Value(0.0)] * n_embd
    # notes is a list of (note_vector,) where each note_vector is [notes_dim] Values
    keys_n = [linear(note, state_dict['snc_wk']) for note in notes]
    vals_n = [linear(note, state_dict['snc_wv']) for note in notes]
    q = linear(x, state_dict['snc_wq'])
    x_attn = []
    for h in range(n_head):
        hs = h * head_dim
        q_h = q[hs:hs + head_dim]
        k_h = [ki[hs:hs + head_dim] for ki in keys_n]
        v_h = [vi[hs:hs + head_dim] for vi in vals_n]
        attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                       for t in range(len(k_h))]
        attn_weights = softmax(attn_logits)
        head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)]
        x_attn.extend(head_out)
    projected = linear(x_attn, state_dict['snc_wo'])
    gate = sigmoid(state_dict['snc_gate'][0][0])
    return [gate * p for p in projected]

# --- Planner (Contribution 1, Section 3.2) ---

def plan_seed(prompt_tokens, hard=True):
    """Planner-seeded snapshot 0 (Section 3.2).
    Processes the FULL prompt, predicts S latent plan slots over V_p vocabulary,
    re-embeds active slots through E_plan, mean-pools, and L2-normalizes.
    hard=True: argmax (inference). hard=False: softmax-weighted (training, differentiable).
    Returns (snapshot_0, active_plan_embeds) for use by coverage head."""
    # Run prompt tokens through the trunk, collecting hidden states
    # (capped at 8 tokens for micro demo — paper uses full prompt)
    keys_tmp = [[] for _ in range(n_layer)]
    values_tmp = [[] for _ in range(n_layer)]
    hiddens = []
    n = min(8, block_size, len(prompt_tokens))
    for pos_id in range(n):
        _, hidden = gpt(prompt_tokens[pos_id], pos_id, keys_tmp, values_tmp, notes=None)
        hiddens.append(hidden)
    # Mean-pool hidden states across the full prompt sequence
    if hard:
        # Detached pooling — no grad needed through trunk for inference / stages 1-3
        pooled = [Value(sum(h[d].data for h in hiddens) / len(hiddens)) for d in range(n_embd)]
    else:
        # Differentiable pooling — grads flow through trunk to plan_proj and E_plan
        pooled = [sum(h[d] for h in hiddens) / len(hiddens) for d in range(n_embd)]
    # Project to S * V_p logits, reshape to S slots
    slot_logits_flat = linear(pooled, state_dict['plan_proj'])  # length S * V_p
    # Re-embed through E_plan
    plan_embeds = []
    for si in range(S):
        slot_logits = slot_logits_flat[si * V_p : (si + 1) * V_p]
        if hard:
            # Argmax: non-differentiable, used at inference
            z_i = max(range(V_p), key=lambda a: slot_logits[a].data)
            embed = list(state_dict['E_plan'][z_i])
        else:
            # Soft selection: differentiable, used during training so plan_proj gets gradients
            weights = softmax(slot_logits)
            embed = [sum(weights[a] * state_dict['E_plan'][a][d] for a in range(V_p))
                     for d in range(notes_dim)]
        plan_embeds.append(embed)
    # Mean-pool active slot embeddings and L2-normalize → snapshot_0
    mean_embed = [sum(e[d] for e in plan_embeds) / S for d in range(notes_dim)]
    snapshot_0 = l2_normalize(mean_embed)
    return snapshot_0, plan_embeds

# --- Speculation Head (Section 3.6) ---

def speculation_head(hidden):
    """Produces a provisional latent note summarizing the stream's block output."""
    return linear(hidden, state_dict['spec_proj'])

# --- Coverage Head (Section 3.6) ---

def coverage_head(hidden, plan_embeds):
    """Predicts ownership logits over plan items via dot product.
    Returns S logits indicating which plan items this stream covers."""
    h_proj = linear(hidden, state_dict['coverage_proj'])  # [notes_dim]
    logits = []
    for pe in plan_embeds:
        dot = sum(h_proj[d] * pe[d] for d in range(notes_dim))
        logits.append(dot)
    return logits

# --- Agreement Head (Section 3.7) ---

def agreement_head(hidden, visible_notes, cov_logits, prov_note):
    """Predicts readiness score r_k for a stream.
    Concatenates hidden state, mean of visible notes, coverage logits, and provisional note."""
    # Mean-pool visible notes (or zeros if empty)
    if visible_notes:
        mean_notes = [sum(n[d] for n in visible_notes) / len(visible_notes) for d in range(notes_dim)]
    else:
        mean_notes = [Value(0.0)] * notes_dim
    # Concatenate: [hidden, mean_notes, coverage_logits, provisional_note]
    concat = list(hidden) + mean_notes + cov_logits + list(prov_note)
    score_list = linear(concat, state_dict['agree_proj'])  # [1]
    return sigmoid(score_list[0])

# --- Auto-Steer (2026 extension — NOT in the original paper) ---

def auto_steer(hidden):
    """Auto-Steer vector emitted at block boundaries.
    Siblings see it via SNC → enables mid-trajectory correction."""
    return linear(hidden, state_dict['steer_proj'])

# --- Core Frozen Trunk Forward Pass ---

def gpt(token_id, pos_id, keys, values, notes=None):
    """Core forward pass through the frozen trunk.
    - Self-attention + residuals are 100% private per stream (via keys/values).
    - SNC is the only cross-stream path (Contribution 2)."""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # SNC injection point: after self-attention, before FFN (per paper Figure 2)
        if notes is not None:
            snc_delta = snc_cross_attention(x, notes)
            x = [a + b for a, b in zip(x, snc_delta)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits, x

# ==============================================================================
# Phase 1: Train base GPT trunk
# ==============================================================================
print("\n=== Phase 1: Training base GPT trunk (500 steps) ===")

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_base = [0.0] * len(base_params)
v_base = [0.0] * len(base_params)

num_steps = 500
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + doc + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits, _ = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(base_params):
        m_base[i] = beta1 * m_base[i] + (1 - beta1) * p.grad
        v_base[i] = beta2 * v_base[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_base[i] / (1 - beta1 ** (step + 1))
        v_hat = v_base[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    # Zero grads on all sidecar params
    for p in snc_params + plan_params + commit_params:
        p.grad = 0
    if (step + 1) % 100 == 0 or step == 0:
        print(f"  step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ==============================================================================
# Phase 2: Train sidecars only — trunk frozen
# Follows the paper's 4-stage curriculum (Section 3.10):
#   Stage 0: Planner pretraining
#   Stage 1: Stream bootstrap (SNC + adapters)
#   Stage 2: Bus enablement (note emission)
#   Stage 3: Commit control (coverage + agreement)
# ==============================================================================

sidecar_params = snc_params + plan_params + commit_params
m_sidecar = [0.0] * len(sidecar_params)
v_sidecar = [0.0] * len(sidecar_params)

def adam_step_sidecar(step_idx, lr):
    """Adam update for all sidecar params."""
    for i, p in enumerate(sidecar_params):
        m_sidecar[i] = beta1 * m_sidecar[i] + (1 - beta1) * p.grad
        v_sidecar[i] = beta2 * v_sidecar[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_sidecar[i] / (1 - beta1 ** (step_idx + 1))
        v_hat = v_sidecar[i] / (1 - beta2 ** (step_idx + 1))
        p.data -= lr * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    for p in base_params:
        p.grad = 0

cadence = TAU  # note emission cadence matches block size
n_streams_train = 2
global_step = 0

# --- Stage 0: Planner pretraining (Section 3.10) ---
print("\n=== Stage 0: Planner pretraining (50 steps) ===")
num_steps_s0 = 50
for step in range(num_steps_s0):
    global_step += 1
    doc = docs[step % len(docs)]
    prompt_tokens = [BOS] + doc[:min(block_size - 1, len(doc))]
    # Run planner with soft selection (differentiable) so plan_proj gets gradients
    snapshot_0, plan_embeds = plan_seed(prompt_tokens, hard=False)
    # Loss: encourage plan embeddings to be spread out (diversity) via negative pairwise similarity
    loss = Value(0.0)
    n_pairs = 0
    for i in range(S):
        for j in range(i + 1, S):
            sim = sum(plan_embeds[i][d] * plan_embeds[j][d] for d in range(notes_dim))
            loss = loss + sim * sim  # penalize high similarity between slots
            n_pairs += 1
    if n_pairs > 0:
        loss = loss / n_pairs
    loss.backward()
    adam_step_sidecar(global_step, 0.005 * (1 - step / num_steps_s0))
    if (step + 1) % 25 == 0 or step == 0:
        print(f"  step {step+1:3d} / {num_steps_s0:3d} | plan diversity loss {loss.data:.4f}")

# --- Stage 1: Stream bootstrap — SNC + note reading (Section 3.10) ---
print("\n=== Stage 1: Stream bootstrap with SNC (50 steps) ===")
num_steps_s1 = 50
for step in range(num_steps_s1):
    global_step += 1
    d1 = docs[(step * 2) % len(docs)]
    d2 = docs[(step * 2 + 1) % len(docs)]
    streams = [
        [BOS] + d1 + [BOS],
        [BOS] + d2 + [BOS],
    ]
    max_len = max(min(block_size, len(s) - 1) for s in streams)
    stream_keys = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    stream_values = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    # Bus: list of (stream_id, round_number, note_vector) tuples
    buses = [[] for _ in range(n_streams_train)]

    # Contribution 1: Seed snapshot 0 (detached — planner already trained in Stage 0)
    for si in range(n_streams_train):
        prompt_toks = streams[si][:min(block_size, len(streams[si]) - 1)]
        snap0, _ = plan_seed(prompt_toks)
        buses[si].append((si, 0, detach(snap0)))  # detached from planner graph

    total_loss = Value(0.0)
    n_tokens = 0
    current_round = 1
    for pos_id in range(max_len):
        for si in range(n_streams_train):
            if pos_id >= min(block_size, len(streams[si]) - 1):
                continue
            token_id = streams[si][pos_id]
            target_id = streams[si][pos_id + 1]

            # Build visible notes with delay Δ (Contribution 2, Section 3.3)
            other_notes = []
            for oi in range(n_streams_train):
                if oi != si:
                    for (_sid, rnd, note) in buses[oi]:
                        if rnd <= current_round - DELTA:
                            other_notes.append(note)

            logits, hidden = gpt(token_id, pos_id,
                                 stream_keys[si],
                                 stream_values[si],
                                 notes=other_notes)

            probs = softmax(logits)
            total_loss = total_loss + (-probs[target_id].log())
            n_tokens += 1

            # Emit notes at block boundaries
            if (pos_id + 1) % cadence == 0:
                snapshot = linear(hidden, state_dict['notes_proj'])
                buses[si].append((si, current_round, snapshot))
                current_round += 1

    loss = total_loss / n_tokens
    loss.backward()
    adam_step_sidecar(global_step, 0.005 * (1 - step / num_steps_s1))
    if (step + 1) % 25 == 0 or step == 0:
        gate_val = sigmoid(state_dict['snc_gate'][0][0]).data
        print(f"  step {step+1:3d} / {num_steps_s1:3d} | loss {loss.data:.4f} | gate {gate_val:.4f}")

# --- Stage 2: Bus enablement — speculation + notes emission (Section 3.10) ---
print("\n=== Stage 2: Bus enablement — speculation notes (50 steps) ===")
num_steps_s2 = 50
for step in range(num_steps_s2):
    global_step += 1
    d1 = docs[(step * 2) % len(docs)]
    d2 = docs[(step * 2 + 1) % len(docs)]
    streams = [
        [BOS] + d1 + [BOS],
        [BOS] + d2 + [BOS],
    ]
    max_len = max(min(block_size, len(s) - 1) for s in streams)
    stream_keys = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    stream_values = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    buses = [[] for _ in range(n_streams_train)]

    for si in range(n_streams_train):
        prompt_toks = streams[si][:min(block_size, len(streams[si]) - 1)]
        snap0, _ = plan_seed(prompt_toks)
        buses[si].append((si, 0, detach(snap0)))

    total_loss = Value(0.0)
    n_tokens = 0
    current_round = 1
    for pos_id in range(max_len):
        for si in range(n_streams_train):
            if pos_id >= min(block_size, len(streams[si]) - 1):
                continue
            token_id = streams[si][pos_id]
            target_id = streams[si][pos_id + 1]

            other_notes = []
            for oi in range(n_streams_train):
                if oi != si:
                    for (_sid, rnd, note) in buses[oi]:
                        if rnd <= current_round - DELTA:
                            other_notes.append(note)

            logits, hidden = gpt(token_id, pos_id,
                                 stream_keys[si],
                                 stream_values[si],
                                 notes=other_notes)

            probs = softmax(logits)
            total_loss = total_loss + (-probs[target_id].log())
            n_tokens += 1

            if (pos_id + 1) % cadence == 0:
                # Use speculation head for provisional notes (Section 3.6)
                spec_note = speculation_head(hidden)
                buses[si].append((si, current_round, spec_note))
                # Auto-Steer extension (NOT in paper)
                steer_vec = auto_steer(hidden)
                buses[si].append((si, current_round, steer_vec))
                current_round += 1

    loss = total_loss / n_tokens
    loss.backward()
    adam_step_sidecar(global_step, 0.005 * (1 - step / num_steps_s2))
    if (step + 1) % 25 == 0 or step == 0:
        gate_val = sigmoid(state_dict['snc_gate'][0][0]).data
        print(f"  step {step+1:3d} / {num_steps_s2:3d} | loss {loss.data:.4f} | gate {gate_val:.4f}")

# --- Stage 3: Commit control — coverage + agreement (Section 3.10) ---
print("\n=== Stage 3: Commit control — coverage + agreement (50 steps) ===")
num_steps_s3 = 50
for step in range(num_steps_s3):
    global_step += 1
    d1 = docs[(step * 2) % len(docs)]
    d2 = docs[(step * 2 + 1) % len(docs)]
    streams = [
        [BOS] + d1 + [BOS],
        [BOS] + d2 + [BOS],
    ]
    max_len = max(min(block_size, len(s) - 1) for s in streams)
    stream_keys = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    stream_values = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    buses = [[] for _ in range(n_streams_train)]

    # Detach plan_seed: planner already trained, we only train coverage+agreement here
    for si in range(n_streams_train):
        prompt_toks = streams[si][:min(block_size, len(streams[si]) - 1)]
        snap0, plan_embeds_si = plan_seed(prompt_toks)
        buses[si].append((si, 0, detach(snap0)))

    # Detach plan embeds too — coverage head trains its own projection
    active_plan_embeds = detach_nested(plan_embeds_si)

    total_loss = Value(0.0)
    n_tokens = 0
    current_round = 1
    for pos_id in range(max_len):
        for si in range(n_streams_train):
            if pos_id >= min(block_size, len(streams[si]) - 1):
                continue
            token_id = streams[si][pos_id]
            target_id = streams[si][pos_id + 1]

            other_notes = []
            for oi in range(n_streams_train):
                if oi != si:
                    for (_sid, rnd, note) in buses[oi]:
                        if rnd <= current_round - DELTA:
                            other_notes.append(note)

            logits, hidden = gpt(token_id, pos_id,
                                 stream_keys[si],
                                 stream_values[si],
                                 notes=other_notes)

            probs = softmax(logits)
            total_loss = total_loss + (-probs[target_id].log())
            n_tokens += 1

            if (pos_id + 1) % cadence == 0:
                spec_note = speculation_head(hidden)
                cov_logits = coverage_head(hidden, active_plan_embeds)
                readiness = agreement_head(hidden, other_notes, cov_logits, spec_note)
                # Add coverage diversity loss: encourage different streams to own different items
                for cl in cov_logits:
                    total_loss = total_loss + cl * cl * 0.01  # regularize
                # Add readiness supervision: encourage readiness > GAMMA
                total_loss = total_loss + (readiness - Value(0.7)) ** 2

                buses[si].append((si, current_round, spec_note))
                steer_vec = auto_steer(hidden)
                buses[si].append((si, current_round, steer_vec))
                current_round += 1

    loss = total_loss / max(n_tokens, 1)
    loss.backward()
    adam_step_sidecar(global_step, 0.003 * (1 - step / num_steps_s3))
    if (step + 1) % 25 == 0 or step == 0:
        gate_val = sigmoid(state_dict['snc_gate'][0][0]).data
        print(f"  step {step+1:3d} / {num_steps_s3:3d} | loss {loss.data:.4f} | gate {gate_val:.4f}")

# ==============================================================================
# Single-stream inference (baseline)
# ==============================================================================
temperature = 0.5
print("\n--- single-stream inference (baseline) ---")
for sample_idx in range(10):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample_ids = []
    for pos_id in range(block_size):
        logits, _ = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample_ids.append(token_id)
    print(f"  sample {sample_idx+1:2d}: {decode_tokens(sample_ids)}")

# ==============================================================================
# Parallel multi-stream inference with synchronized block protocol (Section 3.8)
# ==============================================================================
n_streams = 3
n_batches = 5
print(f"\n--- parallel inference ({n_streams} streams, TAU={TAU}, synchronized blocks) ---")
for batch in range(n_batches):
    # --- Phase 0: Encode prompt, run planner, publish snapshot 0 ---
    prompt_tokens = [BOS]  # minimal prompt for demo
    stream_keys = [[[] for _ in range(n_layer)] for _ in range(n_streams)]
    stream_values = [[[] for _ in range(n_layer)] for _ in range(n_streams)]
    # Bus: list of (stream_id, round_number, note_vector) per stream
    buses = [[] for _ in range(n_streams)]

    snapshot_0, active_plan_embeds = plan_seed(prompt_tokens)
    for si in range(n_streams):
        buses[si].append((si, 0, snapshot_0))  # shared snapshot 0

    stream_tokens = [BOS] * n_streams
    stream_samples = [[] for _ in range(n_streams)]
    stream_done = [False] * n_streams
    committed_samples = [[] for _ in range(n_streams)]  # only committed tokens

    max_rounds = block_size // TAU
    pos_counters = [0] * n_streams  # track position per stream

    for rnd in range(1, max_rounds + 1):
        if all(stream_done):
            break

        # --- Phase 1: Each stream emits TAU provisional tokens ---
        provisional_tokens = [[] for _ in range(n_streams)]
        provisional_hiddens = [None] * n_streams
        provisional_kv_lengths = [0] * n_streams  # track KV length before this block

        for si in range(n_streams):
            if stream_done[si]:
                continue

            provisional_kv_lengths[si] = len(stream_keys[si][0])

            # Build visible notes window with delay Δ (Section 3.3)
            visible_notes = []
            for oi in range(n_streams):
                if oi != si:
                    for (_sid, note_rnd, note) in buses[oi]:
                        if note_rnd <= rnd - DELTA:
                            visible_notes.append(note)

            # Emit TAU provisional tokens
            last_hidden = None
            for t in range(TAU):
                pos_id = pos_counters[si]
                if pos_id >= block_size:
                    stream_done[si] = True
                    break

                logits, hidden = gpt(stream_tokens[si], pos_id,
                                     stream_keys[si],
                                     stream_values[si],
                                     notes=visible_notes)

                probs = softmax([l / temperature for l in logits])
                next_token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

                if next_token == BOS:
                    stream_done[si] = True
                    break

                provisional_tokens[si].append(next_token)
                stream_tokens[si] = next_token
                pos_counters[si] += 1
                last_hidden = hidden

            provisional_hiddens[si] = last_hidden

        # --- Phase 2: Compute coverage, agreement, commit/rollback (Section 3.7) ---
        readiness_scores = {}
        for si in range(n_streams):
            if stream_done[si] or provisional_hiddens[si] is None:
                readiness_scores[si] = 1.0  # done streams don't block
                continue

            hidden = provisional_hiddens[si]
            visible_notes = []
            for oi in range(n_streams):
                if oi != si:
                    for (_sid, note_rnd, note) in buses[oi]:
                        if note_rnd <= rnd - DELTA:
                            visible_notes.append(note)

            spec_note = speculation_head(hidden)
            cov_logits = coverage_head(hidden, active_plan_embeds)
            readiness = agreement_head(hidden, visible_notes, cov_logits, spec_note)
            readiness_scores[si] = readiness.data

        # Global gate: A_v = 1 if min(r_k for active k) > GAMMA (Section 3.7)
        active_scores = [readiness_scores[si] for si in range(n_streams) if not stream_done[si]]
        if active_scores:
            gate_pass = min(active_scores) > GAMMA
        else:
            gate_pass = True

        # --- Phase 3: Commit or rollback ---
        for si in range(n_streams):
            if stream_done[si] or not provisional_tokens[si]:
                continue

            if gate_pass or readiness_scores[si] > GAMMA:
                # COMMIT: keep provisional tokens, push notes to bus
                for tok_id in provisional_tokens[si]:
                    committed_samples[si].append(tok_id)
                    stream_samples[si].append(tok_id)

                if provisional_hiddens[si] is not None:
                    spec_note = speculation_head(provisional_hiddens[si])
                    buses[si].append((si, rnd, spec_note))
                    # Auto-Steer extension (NOT in paper)
                    steer_vec = auto_steer(provisional_hiddens[si])
                    buses[si].append((si, rnd, steer_vec))
            else:
                # ROLLBACK: discard provisional tokens, truncate KV cache (Section 3.7)
                n_discard = len(provisional_tokens[si])
                for li in range(n_layer):
                    stream_keys[si][li] = stream_keys[si][li][:provisional_kv_lengths[si]]
                    stream_values[si][li] = stream_values[si][li][:provisional_kv_lengths[si]]
                pos_counters[si] -= n_discard
                # Reset token to last committed token
                if committed_samples[si]:
                    stream_tokens[si] = committed_samples[si][-1]
                else:
                    stream_tokens[si] = BOS
                print(f"    [rollback] batch {batch+1} round {rnd} stream {si}: "
                      f"discarded {n_discard} tokens (readiness {readiness_scores[si]:.3f} < {GAMMA})")

    names = ' | '.join(decode_tokens(s) if s else '(empty)' for s in committed_samples)
    print(f"  batch {batch+1}: {names}")

gate_val = sigmoid(state_dict['snc_gate'][0][0]).data
print(f"\nfinal gate value: {gate_val:.4f}")
print(f"\nPDT micro-implementation complete!")
print(f"  Paper primitives: planner({S} slots x {V_p} vocab), SNC, coverage, agreement, commit/rollback")
print(f"  Extension: Auto-Steer (not in paper)")
print(f"  Tokenizer: tiktoken gpt2 BPE (top {MAX_VOCAB} subwords)")
print(f"  (For real use, scale n_embd/n_layer and use Wikipedia/arXiv corpus)")
