"""Parallel Decoder Transformer (PDT) -- Qwen3-4B Base rebuild.

Clean-slate rebuild of the PDT coordination stack targeting a frozen
Qwen3-4B Base trunk augmented with a trainable sidecar tree:

- ``pdt.trunk``: frozen Qwen3ForCausalLM wrapper + instrumented decoder layer
- ``pdt.sidecar``: all trainable \u03c6 modules (SNC, per-stream adapters, heads,
  plan embedding, plan-notes projection)
- ``pdt.runtime``: Dynamic Notes Bus, window builder, multi-stream
  orchestrator, per-stream state, counterfactual hooks
- ``pdt.training``: staged curriculum (Stage 0 \u2192 Stage 3), loss assembly, trainer
- ``pdt.diagnostics``: codebook-utilization diagnostics for the planner
- ``pdt.datasets``: dependency benchmark generation and retokenization
- ``pdt.cli``: entry points for train / infer / ablate

The thesis this package defends is *concept-space co-referencing*: K streams
trained to read a shared latent workspace and write back to it so that each
stream's trajectory is shaped by awareness of where siblings are operating
in concept-space. The K=3 text-generation setting is the existence-proof
domain; see docs/arxiv_submission/PAPER.md.
"""

__version__ = "0.2.0"
