"""Oracle executor, planner distillation, and joint-stage freeze contracts."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from pdt.config.schemas import PDTConfig
from pdt.training.curriculum import CurriculumController


class _Sidecar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.planner_head = nn.Linear(4, 4)
        self.plan_memory_proj = nn.Linear(4, 4)
        self.semantic_heads = nn.Linear(4, 4)
        self.speculation_head = nn.Linear(4, 4)


class _Layer:
    def __init__(self, index: int) -> None:
        self.pdt_layer_idx = index
        self.snc = nn.Linear(4, 4)
        self.plan_adapter = nn.Linear(4, 4)
        self.notes_gate = nn.Parameter(torch.tensor(-4.0))
        self.adapter_gate = nn.Parameter(torch.tensor(-4.0))


def _model(config: PDTConfig):
    layers = [_Layer(index) for index in config.instrumentation.target_layers]
    return SimpleNamespace(
        config=config,
        sidecar=_Sidecar(),
        instrumented_layers=layers,
        trunk_adapter=SimpleNamespace(model=nn.Linear(4, 4)),
    )


def test_curriculum_exposes_only_the_scientifically_intended_modules() -> None:
    config = PDTConfig()
    controller = CurriculumController(_model(config), config)

    assert controller.on_step(0) == 0
    stage_zero = controller.active_modules_snapshot()
    assert stage_zero["trunk"] is False
    assert stage_zero["planner_head"] is False
    assert stage_zero["plan_memory_proj"] is True
    assert stage_zero["semantic_heads"] is True
    assert stage_zero["speculation_head"] is True
    assert stage_zero["snc"] is True
    assert stage_zero["plan_adapters"] is True

    assert controller.on_step(config.training.curriculum.stage_schedule[1]) == 1
    stage_one = controller.active_modules_snapshot()
    assert stage_one["planner_head"] is True
    assert all(
        not active
        for name, active in stage_one.items()
        if name != "planner_head"
    )

    assert controller.on_step(config.training.curriculum.stage_schedule[2]) == 2
    stage_two = controller.active_modules_snapshot()
    assert stage_two["trunk"] is False
    assert all(
        active
        for name, active in stage_two.items()
        if name != "trunk"
    )


def test_stage_loss_overrides_separate_executor_and_planner_objectives() -> None:
    config = PDTConfig()
    controller = CurriculumController(_model(config), config)
    executor = controller.active_loss_weights(0)
    planner = controller.active_loss_weights(1)
    joint = controller.active_loss_weights(2)

    assert executor.lm_ce > 0 and executor.plan_semantic == 0
    assert planner.plan_semantic > 0 and planner.lm_ce == 0
    assert planner.fact_route > 0 and planner.fact_write == 0
    assert joint.lm_ce > 0 and joint.plan_semantic > 0
