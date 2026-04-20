"""Top-level PDT model: frozen Qwen3 trunk + instrumented layers + sidecar tree.

This is the only place where \u03b8_pre and \u03c6 are assembled together. The
sidecar tree lives in ``self.sidecar`` as a clean subtree; stream
adapters + SNC + speculation tap live inside the instrumented decoder
layers (wired there by ``instrument_trunk``) but all their state is also
reachable via ``self.instrumented_layers`` for the curriculum resolver.

Param group discipline:
    - ``trunk_parameters()``         yields \u03b8_pre (always requires_grad=False)
    - ``sidecar_parameters()``       yields the ``self.sidecar`` subtree
    - ``per_layer_phi_parameters()`` yields the SNC + adapter + outer gate
                                     parameters that live inside the
                                     instrumented decoder layers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List

import torch
from torch import nn

from pdt.config.schemas import (
    InstrumentationConfig,
    PDTConfig,
    SidecarConfig,
    TrunkConfig,
)
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.heads.agreement import AgreementHead
from pdt.sidecar.heads.coverage import CoverageHead
from pdt.sidecar.heads.notes import NotesHead
from pdt.sidecar.heads.plan_notes_proj import PlanNotesProjection
from pdt.sidecar.heads.planner import PlannerHead
from pdt.sidecar.heads.speculation import SpeculationHead
from pdt.sidecar.heads.stream_classifier import StreamClassifierHead
from pdt.sidecar.plan_embedding import PlanEmbedding
from pdt.sidecar.snc import SharedNotesCrossAttention
from pdt.trunk.instrumentation import (
    InstrumentedQwen3DecoderLayer,
    instrument_trunk,
)
from pdt.trunk.qwen3_adapter import Qwen3TrunkAdapter


LOGGER = logging.getLogger("pdt.model")

__all__ = ["PDTModel", "Sidecar"]


class Sidecar(nn.Module):
    """All \u03c6 heads in a single named subtree.

    The per-layer SNC and per-layer stream adapters live inside the
    instrumented decoder layers (they have to, to be reachable by the
    trunk's forward pass); everything else lives here so the trainer's
    name resolver can reach it by ``model.sidecar.<name>``.
    """

    def __init__(self, config: SidecarConfig) -> None:
        super().__init__()
        self.config = config
        self.planner_head = PlannerHead(config.planner_head)
        self.plan_embedding = PlanEmbedding(
            plan_vocab_size=config.plan_vocab_size,
            hidden_size=config.hidden_size,
        )
        self.plan_notes_proj = PlanNotesProjection(config.plan_notes_proj)
        self.notes_head = NotesHead(config.notes_head)
        self.speculation_head = SpeculationHead(config.speculation_head)
        self.coverage_head = CoverageHead(config.coverage_head)
        self.agreement_head = AgreementHead(config.agreement_head)
        self.stream_classifier = StreamClassifierHead(config.stream_classifier)


class PDTModel(nn.Module):
    """Frozen Qwen3 trunk + sidecar + instrumented layers.

    Construction:
        1. Load and freeze the Qwen3 trunk (via ``Qwen3TrunkAdapter``).
        2. Build the ``Sidecar`` subtree.
        3. Instrument selected trunk layers: wrap each with
           ``InstrumentedQwen3DecoderLayer`` carrying a fresh SNC module
           and a fresh ``StreamAdapterLayer``.
    """

    def __init__(self, config: PDTConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk_adapter = Qwen3TrunkAdapter(config.trunk)
        self.sidecar = Sidecar(config.sidecar)

        # Build per-layer \u03c6 modules and land them in the trunk.
        def make_snc() -> SharedNotesCrossAttention:
            return SharedNotesCrossAttention(
                config.sidecar.snc,
                gating_init=config.instrumentation.snc_gate_init,
            )

        def make_adapter() -> StreamAdapterLayer:
            return StreamAdapterLayer(config.sidecar.adapters)

        self.instrumented_layers: List[InstrumentedQwen3DecoderLayer] = (
            instrument_trunk(
                self.trunk_adapter,
                config.instrumentation,
                config.sidecar,
                make_snc=make_snc,
                make_adapter=make_adapter,
            )
        )
        LOGGER.info(
            "PDTModel ready: trunk=%s, instrumented=%d/%d layers, "
            "sidecar_params=%s, per_layer_phi_params=%s",
            config.trunk.base_model,
            len(self.instrumented_layers),
            self.trunk_adapter.num_layers(),
            _fmt_params(self.sidecar_parameters()),
            _fmt_params(self.per_layer_phi_parameters()),
        )

    # ------------------------------------------------------------------ #
    # Param group discipline
    # ------------------------------------------------------------------ #

    def trunk_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter(self.trunk_adapter.frozen_parameters())

    def sidecar_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.sidecar.parameters()

    def per_layer_phi_parameters(self) -> Iterator[torch.nn.Parameter]:
        for layer in self.instrumented_layers:
            # SNC + adapter + outer gates. The super().__init__ params
            # (self_attn, mlp, layernorms) are part of \u03b8_pre and are frozen.
            if layer.snc is not None:
                yield from layer.snc.parameters()
            if layer.stream_adapter is not None:
                yield from layer.stream_adapter.parameters()
            if layer.notes_gate is not None:
                yield layer.notes_gate
            if layer.adapter_gate is not None:
                yield layer.adapter_gate

    def all_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """All \u03c6 parameters. What the optimizer should see."""
        yield from self.sidecar_parameters()
        yield from self.per_layer_phi_parameters()

    # ------------------------------------------------------------------ #
    # Trunk forward pass-through
    # ------------------------------------------------------------------ #

    def forward(self, *args, **kwargs):
        return self.trunk_adapter.model(*args, **kwargs)


def _fmt_params(it) -> str:
    total = sum(p.numel() for p in it)
    if total >= 1_000_000:
        return f"{total / 1e6:.1f}M"
    if total >= 1_000:
        return f"{total / 1e3:.1f}K"
    return str(total)
