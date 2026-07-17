"""Non-empirical execution contract for packed self-only free generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from pdt.config import load_config
from pdt.runtime.counterfactuals import CounterfactualConfig
from pdt.runtime.orchestrator import MultiStreamOrchestrator
from pdt.trunk.instrumentation import LayerRuntimeContext


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TokenizerContractDouble:
    chat_template = "contract"
    pad_token_id = 0
    eos_token_id = 7

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class CacheContractDouble:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class ContextRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[LayerRuntimeContext | None] = []

    def set_runtime_context(self, context: LayerRuntimeContext | None) -> None:
        self.contexts.append(context)


class PlannerContractDouble(nn.Module):
    def __init__(self, *, hidden_size: int, planner_width: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.planner_width = planner_width

    def forward(
        self,
        prompt_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        if prompt_hidden.shape[:2] != attention_mask.shape:
            raise ValueError("Planner contract received misaligned prompt tensors.")
        device = prompt_hidden.device
        return SimpleNamespace(
            nodes=torch.ones(1, 3, 2, self.planner_width, device=device),
            node_validity_logits=torch.ones(1, 3, 2, device=device),
            presentation_order_logits=torch.zeros(1, 3, device=device),
        )


class PlanMemoryContractDouble(nn.Module):
    def __init__(self, notes_dim: int) -> None:
        super().__init__()
        self.notes_dim = notes_dim

    def forward(
        self,
        nodes: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if nodes.shape[:-1] != mask.shape:
            raise ValueError("Plan-memory contract received a misaligned plan mask.")
        return torch.ones(*nodes.shape[:-1], self.notes_dim, device=nodes.device)


class SpeculationContractDouble(nn.Module):
    """Unreachable in self-only generation; any access is a contract failure."""

    @property
    def width(self) -> int:
        raise AssertionError("Self-only generation must not construct a notes bus.")


class TrunkContractDouble(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.hidden_size = hidden_size

    def frozen_parameters(self) -> list[nn.Parameter]:
        return [self.anchor]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, 8, device=input_ids.device)
        logits[..., 3] = 1.0
        hidden_states = (
            torch.ones(
                batch,
                width,
                self.hidden_size,
                dtype=self.anchor.dtype,
                device=input_ids.device,
            ),
        ) if output_hidden_states else None
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=(
                CacheContractDouble(attention_mask.size(1))
                if use_cache
                else None
            ),
        )


class ModelContractDouble(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.trunk_adapter = TrunkContractDouble(config.sidecar.hidden_size)
        self.sidecar = nn.Module()
        self.sidecar.anchor = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.sidecar.planner_head = PlannerContractDouble(
            hidden_size=config.sidecar.hidden_size,
            planner_width=config.sidecar.planner_head.planner_width,
        )
        self.sidecar.plan_memory_proj = PlanMemoryContractDouble(
            config.sidecar.notes_dim
        )
        self.sidecar.speculation_head = SpeculationContractDouble()
        self.context_recorder = ContextRecorder()
        self.instrumented_layers = [self.context_recorder]


def _self_only_config() -> Any:
    config = load_config(PROJECT_ROOT / "configs" / "pdt_qwen3_4b.yaml")
    config.instrumentation.coordination_source = "self_only"
    # A two-token block makes the same canonical boundary mechanics observable
    # in a tiny contract test. Production config validation still requires 32.
    config.runtime.block_size = 2
    config.runtime.notes_bus.history_blocks = 2
    return config


def test_self_only_free_generation_reads_only_prior_owned_blocks() -> None:
    config = _self_only_config()
    model = ModelContractDouble(config)
    orchestrator = MultiStreamOrchestrator(
        cast(Any, model),
        cast(Any, TokenizerContractDouble()),
        config,
    )

    result = orchestrator.generate("Explain one historical event.", max_new_tokens=4)

    assert result.tokens_by_stream == {
        "stream_0": [3, 3, 3, 3],
        "stream_1": [3, 3, 3, 3],
        "stream_2": [3, 3, 3, 3],
    }
    assert result.dynamic_codes_by_stream == {
        "stream_0": [],
        "stream_1": [],
        "stream_2": [],
    }
    contexts = [
        context
        for context in model.context_recorder.contexts
        if context is not None
    ]
    assert contexts
    assert all(context.notes is None for context in contexts)
    assert all(context.self_only_memory is not None for context in contexts)
    second_block = next(
        context
        for context in contexts
        if context.self_only_memory is not None
        and context.self_only_memory.hidden_states.size(1) == 1
    )
    memory = second_block.self_only_memory
    assert memory is not None
    assert memory.owner_streams == config.runtime.streams
    assert memory.hidden_states.dtype is torch.bfloat16
    assert memory.mask.tolist() == [[True], [True], [True]]
    assert memory.positions.tolist() == [[3], [3], [3]]
    assert second_block.self_only_query_positions is not None
    assert second_block.self_only_query_positions.tolist() == [[4], [4], [4]]


@pytest.mark.parametrize("mode", ["bus_mutation", "norm_scramble", "source_swap"])
def test_self_only_runtime_rejects_bus_interventions(mode: str) -> None:
    config = _self_only_config()
    with pytest.raises(ValueError, match="requires a bus checkpoint"):
        MultiStreamOrchestrator(
            cast(Any, ModelContractDouble(config)),
            cast(Any, TokenizerContractDouble()),
            config,
            counterfactual=CounterfactualConfig(mode=mode),  # type: ignore[arg-type]
        )
