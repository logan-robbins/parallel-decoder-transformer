"""
microgpt_dnb.py — Fork of Karpathy's microgpt with Dynamic Notes Bus

Original: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
Paper:    https://arxiv.org/abs/2512.10054
Code:     https://github.com/logan-robbins/parallel-decoder-transformer

Karpathy's microgpt is the complete GPT algorithm in ~200 lines of pure Python.
This fork adds ~100 lines to show where the Dynamic Notes Bus (DNB), Shared
Notes Cross-Attention (SNC), and Planner Head fit inside the transformer to
enable parallel decoding.

Phase 1 trains the base GPT exactly as in the original.
Phase 2 freezes the trunk and trains only the SNC + planner parameters so
  multiple independent streams can coordinate via compressed embedding snapshots.
  The planner seeds each stream's bus with a plan snapshot at t=0 so SNC has
  context to cross-attend to from the very first token.
Inference shows both single-stream (baseline) and parallel multi-stream generation.

No dependencies. Run: python microgpt_dnb.py
"""

import os
import math
import random

random.seed(42)

# --- Dataset ---
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# --- Tokenizer ---
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# --- Autograd ---
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

# --- Parameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

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

# --- SNC + Planner parameters (Dynamic Notes Bus) ---
notes_dim = 8  # compressed cross-stream channel
state_dict['notes_proj']   = matrix(notes_dim, n_embd)  # hidden(16) -> notes(8) for evolving snapshots
state_dict['planner_proj'] = matrix(notes_dim, n_embd)  # hidden(16) -> plan seed(8) at t=0
state_dict['snc_wq']   = matrix(n_embd, n_embd)         # query from hidden
state_dict['snc_wk']   = matrix(n_embd, notes_dim)      # key from notes
state_dict['snc_wv']   = matrix(n_embd, notes_dim)      # value from notes
state_dict['snc_wo']   = matrix(n_embd, n_embd)         # output projection
state_dict['snc_gate'] = [[Value(-5.0)]]                 # sigmoid(-5) ≈ 0.007, starts closed

snc_params = [p for key in ['notes_proj', 'planner_proj', 'snc_wq', 'snc_wk', 'snc_wv', 'snc_wo', 'snc_gate']
              for row in state_dict[key] for p in row]

all_params = base_params + snc_params
print(f"num params: {len(all_params)} (base: {len(base_params)}, snc: {len(snc_params)})")

# --- Architecture ---

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
    """Sigmoid using Value ops: 1 / (1 + exp(-x))"""
    return Value(1.0) / (Value(1.0) + (-x).exp())

def snc_cross_attention(x, notes):
    """Shared Notes Cross-Attention over bus snapshots.
    x: list[Value] of length n_embd (current hidden state)
    notes: list[list[Value]] each of length notes_dim (snapshots from other streams)
    Returns a gated delta (not residual) of length n_embd.
    """
    if not notes:
        return [Value(0.0)] * n_embd
    # Precompute K, V for all notes
    keys_n = [linear(note, state_dict['snc_wk']) for note in notes]
    vals_n = [linear(note, state_dict['snc_wv']) for note in notes]
    # Query from current hidden
    q = linear(x, state_dict['snc_wq'])
    # Multi-head attention over notes
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
    # Output projection + learned gate
    projected = linear(x_attn, state_dict['snc_wo'])
    gate = sigmoid(state_dict['snc_gate'][0][0])
    return [gate * p for p in projected]

def plan_seed(token_id):
    """Planner head: run a token through the frozen trunk to produce a plan seed.
    Returns a notes_dim vector that seeds a stream's bus as snapshot 0 at t=0,
    giving SNC something to cross-attend to before any cadence snapshots exist.
    """
    keys_tmp = [[] for _ in range(n_layer)]
    values_tmp = [[] for _ in range(n_layer)]
    _, hidden = gpt(token_id, 0, keys_tmp, values_tmp)  # trunk only, no SNC (notes=None)
    return linear(hidden, state_dict['planner_proj'])

def gpt(token_id, pos_id, keys, values, notes=None):
    """GPT forward pass. Returns (logits, hidden_state).
    notes: optional list of snapshots from other streams for SNC.
    """
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        # 1) Multi-head Attention
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
        # SNC cross-attention (inserted between self-attention and MLP)
        if notes is not None:
            snc_delta = snc_cross_attention(x, notes)
            x = [a + b for a, b in zip(x, snc_delta)]
        # 2) MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits, x  # return hidden state for snapshot projection

# ==============================================================================
# Phase 1: Train base GPT (identical to original microgpt)
# ==============================================================================
print("\n=== Phase 1: Training base GPT (1000 steps) ===")

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_base = [0.0] * len(base_params)
v_base = [0.0] * len(base_params)

num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
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
    # Zero SNC grads too (they participate in forward but aren't updated yet)
    for p in snc_params:
        p.grad = 0
    if (step + 1) % 100 == 0 or step == 0:
        print(f"  step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ==============================================================================
# Phase 2: Train SNC (trunk frozen, only snc_params updated)
# ==============================================================================
print("\n=== Phase 2: Training SNC / Dynamic Notes Bus (300 steps) ===")

cadence = 3  # emit a snapshot every 3 tokens
n_streams_train = 2  # train with 2 parallel streams

m_snc = [0.0] * len(snc_params)
v_snc = [0.0] * len(snc_params)

num_steps_p2 = 300
for step in range(num_steps_p2):
    # Sample 2 documents as parallel streams
    d1 = docs[(step * 2) % len(docs)]
    d2 = docs[(step * 2 + 1) % len(docs)]
    streams = [
        [BOS] + [uchars.index(ch) for ch in d1] + [BOS],
        [BOS] + [uchars.index(ch) for ch in d2] + [BOS],
    ]
    max_len = max(min(block_size, len(s) - 1) for s in streams)
    # Per-stream KV caches and notes buses
    stream_keys = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    stream_values = [[[] for _ in range(n_layer)] for _ in range(n_streams_train)]
    buses = [[] for _ in range(n_streams_train)]  # list of snapshots per stream
    # Planner: seed each stream's bus with snapshot 0 from BOS hidden state
    for si in range(n_streams_train):
        buses[si].append(plan_seed(streams[si][0]))
    total_loss = Value(0.0)
    n_tokens = 0
    # Round-robin: for each position, advance all streams
    for pos_id in range(max_len):
        for si in range(n_streams_train):
            if pos_id >= min(block_size, len(streams[si]) - 1):
                continue
            token_id = streams[si][pos_id]
            target_id = streams[si][pos_id + 1]
            # Gather notes from OTHER streams' buses
            other_notes = []
            for oi in range(n_streams_train):
                if oi != si:
                    other_notes.extend(buses[oi])
            logits, hidden = gpt(token_id, pos_id, stream_keys[si], stream_values[si],
                                 notes=other_notes)
            probs = softmax(logits)
            total_loss = total_loss + (-probs[target_id].log())
            n_tokens += 1
            # Emit snapshot at cadence intervals
            if (pos_id + 1) % cadence == 0:
                snapshot = linear(hidden, state_dict['notes_proj'])
                buses[si].append(snapshot)
    loss = total_loss / n_tokens
    loss.backward()
    # Update only SNC params (trunk is frozen)
    lr_t = 0.005 * (1 - step / num_steps_p2)
    for i, p in enumerate(snc_params):
        m_snc[i] = beta1 * m_snc[i] + (1 - beta1) * p.grad
        v_snc[i] = beta2 * v_snc[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_snc[i] / (1 - beta1 ** (step + 1))
        v_hat = v_snc[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    # Zero base param grads (they accumulate but we don't use them)
    for p in base_params:
        p.grad = 0
    if (step + 1) % 50 == 0 or step == 0:
        gate_val = sigmoid(state_dict['snc_gate'][0][0]).data
        print(f"  step {step+1:3d} / {num_steps_p2:3d} | loss {loss.data:.4f} | gate {gate_val:.4f}")

# ==============================================================================
# Single-stream inference (baseline, identical to original)
# ==============================================================================
temperature = 0.5
print("\n--- single-stream inference (baseline) ---")
for sample_idx in range(10):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits, _ = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"  sample {sample_idx+1:2d}: {''.join(sample)}")

# ==============================================================================
# Parallel multi-stream inference with Dynamic Notes Bus
# ==============================================================================
n_streams = 3
n_batches = 5
print(f"\n--- parallel inference ({n_streams} streams, DNB) ---")
for batch in range(n_batches):
    stream_keys = [[[] for _ in range(n_layer)] for _ in range(n_streams)]
    stream_values = [[[] for _ in range(n_layer)] for _ in range(n_streams)]
    buses = [[] for _ in range(n_streams)]
    # Planner: seed each stream's bus with snapshot 0
    for si in range(n_streams):
        buses[si].append(plan_seed(BOS))
    stream_tokens = [BOS] * n_streams
    stream_samples = [[] for _ in range(n_streams)]
    stream_done = [False] * n_streams
    # Round-robin generation
    for pos_id in range(block_size):
        for si in range(n_streams):
            if stream_done[si]:
                continue
            # Gather notes from other streams' buses
            other_notes = []
            for oi in range(n_streams):
                if oi != si:
                    other_notes.extend(buses[oi])
            logits, hidden = gpt(stream_tokens[si], pos_id, stream_keys[si], stream_values[si],
                                 notes=other_notes)
            probs = softmax([l / temperature for l in logits])
            next_token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if next_token == BOS:
                stream_done[si] = True
                continue
            stream_samples[si].append(uchars[next_token])
            stream_tokens[si] = next_token
            # Emit snapshot at cadence
            if (pos_id + 1) % cadence == 0:
                snapshot = linear(hidden, state_dict['notes_proj'])
                buses[si].append(snapshot)
        if all(stream_done):
            break
    names = ' | '.join(''.join(s) if s else '(empty)' for s in stream_samples)
    print(f"  batch {batch+1}: {names}")

gate_val = sigmoid(state_dict['snc_gate'][0][0]).data
print(f"\nfinal gate value: {gate_val:.4f}")
