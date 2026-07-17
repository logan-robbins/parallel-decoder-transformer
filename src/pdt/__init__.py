"""Parallel Decoder Transformer (PDT) coordination research package.

The canonical system uses a revision-pinned, frozen
``Qwen3-4B-Instruct-2507`` trunk augmented with trainable coordination
modules:

- ``pdt.trunk``: frozen Qwen3 CausalLM wrapper and instrumented decoder layers
- ``pdt.sidecar``: a continuous unordered planner, persistent plan projection,
  semantic heads, SNC, shared plan adapters, and dynamic writer
- ``pdt.runtime``: addressed Dynamic Notes Bus and synchronized decoding
- ``pdt.training``: packed three-lane recurrent rollout and curriculum
- ``pdt.diagnostics`` / ``pdt.evaluation``: token-weighted causal metrics
- ``pdt.datasets``: source-grounded Batch schemas and pinned retokenization
- ``pdt.baselines``: parameter-matched falsification controls
- ``pdt.cli``: train, inference, and ablation entry points

The falsifiable mechanism claim is that locally causal streams can transmit
otherwise unavailable dependency state through a narrow, delayed latent bus.
The package reports paired dependency/nondependency effects; it does not
claim trained empirical evidence by itself.
"""

__version__ = "0.2.0"
