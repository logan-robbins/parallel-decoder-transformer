"""Parallel Decoder Transformer (PDT) coordination research package.

The canonical system uses a revision-pinned, frozen
``Qwen3-4B-Instruct-2507`` trunk augmented with trainable coordination
modules:

- ``pdt.trunk``: frozen Qwen3 CausalLM wrapper and instrumented decoder layers
- ``pdt.sidecar``: SNC, stream adapters, a VQ planner, plan-note projection,
  speculation writer, and stream classifier
- ``pdt.runtime``: addressed Dynamic Notes Bus and synchronized decoding
- ``pdt.training``: cached temporal rollout, functional KD, and curriculum
- ``pdt.diagnostics`` / ``pdt.evaluation``: token-weighted causal metrics
- ``pdt.datasets``: temporal dependency benchmark retokenization
- ``pdt.baselines``: parameter-matched falsification controls
- ``pdt.cli``: train, inference, and ablation entry points

The falsifiable mechanism claim is that locally causal streams can transmit
otherwise unavailable dependency state through a narrow, delayed latent bus.
The package reports paired dependency/nondependency effects; it does not
claim trained empirical evidence by itself.
"""

__version__ = "0.2.0"
