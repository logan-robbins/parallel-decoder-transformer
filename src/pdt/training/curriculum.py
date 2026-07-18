"""Staged-curriculum controller.

Maps the numeric stage index (0-3) to a set of frozen / unfrozen modules
and per-stage loss-weight overrides. Handles two responsibilities:

1. **Resolution**: look up the named identifiers from ``StagePolicy`` and
   return the concrete ``nn.Module`` instances they point to. This is the
   part that silently failed in the previous codebase -- it could not
   reach per-layer SNC modules or plan memory. We reach them
   explicitly here.

2. **Transition**: called on every train step; compares the current and
   previous stage indices and flips ``requires_grad`` accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast

from torch import nn

from pdt.config.schemas import (
    CURRICULUM_IDENTIFIERS,
    CurriculumConfig,
    PDTConfig,
    StagePolicy,
)
from pdt.model import PDTModel
from pdt.trunk.physical_decoder import PhysicalDecoderLayerBank


LOGGER = logging.getLogger("pdt.training.curriculum")


__all__ = ["CurriculumController"]


_SIDECAR_MODULE_NAMES = (
    "planner_head",
    "plan_memory_proj",
    "semantic_heads",
    "speculation_head",
)


@dataclass(slots=True)
class _ModuleHandle:
    """Points to a discoverable module or parameter subset."""

    name: str
    module: nn.Module | None = None
    parameters: Tuple[nn.Parameter, ...] = ()


class CurriculumController:
    def __init__(self, model: PDTModel, config: PDTConfig) -> None:
        self.model = model
        self.config = config
        self.curriculum: CurriculumConfig = config.training.curriculum
        self.stage_schedule = tuple(self.curriculum.stage_schedule)
        if len(self.stage_schedule) != 4:
            raise ValueError("stage_schedule must have 4 entries.")
        self.current_stage: int = -1  # Force a transition on first tick.
        self._validate_policy_resolution()

    # ------------------------------------------------------------------ #
    # Name resolution
    # ------------------------------------------------------------------ #

    def resolve_handles(self, identifier: str) -> List[_ModuleHandle]:
        """Return the ``(name, module, params)`` triples for an identifier.

        Supported identifiers:
            - ``"trunk"``: frozen shared embeddings/lower layers/final head.
            - ``"decoder_branches"``: all independent upper Qwen parameters.
            - sidecar module names: ``planner_head``, ``plan_memory_proj``,
              ``semantic_heads``, ``speculation_head``.
            - ``"snc"``: every per-layer SNC module.
            - ``"plan_attention"``: every persistent Plan-KV read.
            - ``"snc_gate"``: the per-layer outer notes_gate scalar.
            - ``"plan_gate"``: the per-layer outer plan_gate scalar.
        """
        key = identifier.strip().lower()
        handles: List[_ModuleHandle] = []

        if key == "trunk":
            parameters = tuple(self.model.trunk_parameters())
            if not parameters:
                raise ValueError("Curriculum identifier 'trunk' resolved no shared parameters.")
            handles.append(
                _ModuleHandle(
                    name="trunk",
                    parameters=parameters,
                )
            )
            return handles

        if key == "decoder_branches":
            parameters = tuple(self.model.decoder_branch_parameters())
            if not parameters:
                raise ValueError(
                    "Curriculum identifier 'decoder_branches' resolved no parameters."
                )
            return [
                _ModuleHandle(
                    name="decoder_branches",
                    module=self.model.physical_decoder,
                    parameters=parameters,
                )
            ]

        if key in _SIDECAR_MODULE_NAMES:
            module = getattr(self.model.sidecar, key, None)
            if module is None:
                raise ValueError(
                    f"Curriculum identifier {identifier!r} has no model.sidecar.{key} module."
                )
            handles.append(
                _ModuleHandle(
                    name=key,
                    module=module,
                    parameters=tuple(module.parameters()),
                )
            )
            return handles

        if key == "snc":
            for raw_layer in self.model.instrumented_layers:
                layer = cast(PhysicalDecoderLayerBank, raw_layer)
                handles.append(
                    _ModuleHandle(
                        name=f"snc@layer_{layer.pdt_layer_idx}",
                        module=layer.snc,
                        parameters=tuple(layer.snc.parameters()),
                    )
                )
            return self._require_handles(identifier, handles)

        if key == "plan_attention":
            for raw_layer in self.model.instrumented_layers:
                layer = cast(PhysicalDecoderLayerBank, raw_layer)
                handles.append(
                    _ModuleHandle(
                        name=f"plan_attention@layer_{layer.pdt_layer_idx}",
                        module=layer.plan_attention,
                        parameters=tuple(layer.plan_attention.parameters()),
                    )
                )
            return self._require_handles(identifier, handles)

        if key == "snc_gate":
            for raw_layer in self.model.instrumented_layers:
                layer = cast(PhysicalDecoderLayerBank, raw_layer)
                if layer.notes_gate is not None:
                    handles.append(
                        _ModuleHandle(
                            name=f"snc_gate@layer_{layer.pdt_layer_idx}",
                            parameters=(layer.notes_gate,),
                        )
                    )
            return self._require_handles(identifier, handles)

        if key == "plan_gate":
            for raw_layer in self.model.instrumented_layers:
                layer = cast(PhysicalDecoderLayerBank, raw_layer)
                handles.append(
                    _ModuleHandle(
                        name=f"plan_gate@layer_{layer.pdt_layer_idx}",
                        parameters=(layer.plan_gate,),
                    )
                )
            return self._require_handles(identifier, handles)

        raise ValueError(
            f"Unknown curriculum identifier {identifier!r}; "
            f"valid identifiers are {list(CURRICULUM_IDENTIFIERS)}."
        )

    @staticmethod
    def _require_handles(
        identifier: str,
        handles: List[_ModuleHandle],
    ) -> List[_ModuleHandle]:
        if not handles:
            raise ValueError(
                f"Curriculum identifier {identifier!r} resolved no parameters; "
                "the configured instrumentation does not provide this control."
            )
        return handles

    def _validate_policy_resolution(self) -> None:
        expected = set(CURRICULUM_IDENTIFIERS)
        if set(self.curriculum.stages) != set(range(4)):
            raise ValueError("Curriculum stages must be exactly 0, 1, 2, 3.")
        for stage_idx, policy in self.curriculum.stages.items():
            declared = tuple(policy.freeze) + tuple(policy.unfreeze)
            if len(set(declared)) != len(declared):
                raise ValueError(
                    f"Curriculum stage {stage_idx} contains duplicate freeze/unfreeze identifiers."
                )
            actual = set(declared)
            if actual != expected:
                raise ValueError(
                    f"Curriculum stage {stage_idx} is not exhaustive: "
                    f"missing={sorted(expected - actual)}, "
                    f"unknown={sorted(actual - expected)}."
                )
        for identifier in CURRICULUM_IDENTIFIERS:
            self.resolve_handles(identifier)

    # ------------------------------------------------------------------ #
    # Stage transitions
    # ------------------------------------------------------------------ #

    def determine_stage(self, global_step: int) -> int:
        stage = 0
        for idx, threshold in enumerate(self.stage_schedule):
            if global_step >= threshold:
                stage = idx
        return min(stage, 3)

    def on_step(self, global_step: int) -> int:
        """Called once per training step. Returns the current stage index.

        If the stage has changed since the last call, freezes + unfreezes
        modules per ``StagePolicy`` and returns the new index. Otherwise
        returns the cached index.
        """
        new_stage = self.determine_stage(global_step)
        if new_stage == self.current_stage:
            return new_stage

        prev = self.current_stage
        self.current_stage = new_stage
        policy = self.curriculum.stages[new_stage]
        LOGGER.info(
            "Curriculum transition: stage %d -> %d (name=%r, step=%d)",
            prev,
            new_stage,
            policy.name,
            global_step,
        )
        self._apply_policy(policy)
        return new_stage

    def _apply_policy(self, policy: StagePolicy) -> None:
        changed: List[str] = []
        for identifier in policy.freeze:
            for handle in self.resolve_handles(identifier):
                for p in handle.parameters:
                    if p.requires_grad:
                        p.requires_grad_(False)
                changed.append(f"freeze:{handle.name}")
        for identifier in policy.unfreeze:
            for handle in self.resolve_handles(identifier):
                for p in handle.parameters:
                    if not p.requires_grad:
                        p.requires_grad_(True)
                changed.append(f"unfreeze:{handle.name}")
        LOGGER.info("Applied stage policy: %s", ", ".join(changed) if changed else "(noop)")

    def active_loss_weights(self, stage: int):
        """Return the stage-specific loss weights if any, else global."""
        policy = self.curriculum.stages.get(stage)
        if policy is None or policy.loss_weights is None:
            return self.config.training.loss_weights
        return policy.loss_weights

    def active_modules_snapshot(self) -> Dict[str, bool]:
        """Return ``{identifier -> any_trainable}`` for observability."""
        result: Dict[str, bool] = {}
        for ident in CURRICULUM_IDENTIFIERS:
            handles = self.resolve_handles(ident)
            any_train = False
            for h in handles:
                for p in h.parameters:
                    if p.requires_grad:
                        any_train = True
                        break
                if any_train:
                    break
            result[ident] = any_train
        return result
