"""Frozen Qwen3 trunk adapter.

Loads the revision-pinned ``Qwen/Qwen3-4B-Instruct-2507`` trunk via
``AutoModelForCausalLM``, freezes every parameter, and exposes a list of
the trunk's decoder layers for instrumentation. Canonical PDT prompts use
the trunk's existing chat vocabulary and never mutate frozen token rows.

**Critical fix relative to the previous codebase:** layer access returns the
actual ``nn.ModuleList`` (not a shallow Python list). Subclass replacement
via ``trunk.model.layers[idx] = replacement`` writes into the forward
graph. A post-instrumentation ``is``-identity assertion verifies the
installation did in fact land on the real module.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from pdt.config.schemas import TrunkConfig


LOGGER = logging.getLogger("pdt.trunk")

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

__all__ = ["Qwen3TrunkAdapter"]


class Qwen3TrunkAdapter:
    """Wrapper around a frozen Qwen3ForCausalLM.

    This is deliberately *not* an ``nn.Module`` -- it holds the underlying
    HuggingFace model by composition and exposes precisely the surface PDT
    needs (layer access, tokenizer access, frozen param iterator). Treating
    the trunk as a non-module prevents accidental inclusion of its
    parameters in the sidecar module tree and makes the \u03b8_pre / \u03c6 split
    inspectable by module path alone.
    """

    def __init__(self, config: TrunkConfig) -> None:
        self.config = config
        self.dtype = _resolve_dtype(config.torch_dtype)
        self.model: PreTrainedModel = self._load_model()
        self.tokenizer: PreTrainedTokenizerBase = self._load_tokenizer()
        self._freeze()
        self._instrumented_layer_indices: Tuple[int, ...] = tuple()

    def _load_model(self) -> PreTrainedModel:
        source = self.config.local_path or self.config.base_model
        LOGGER.info(
            "Loading Qwen3 trunk from %s (dtype=%s, attn=%s)",
            source,
            self.dtype,
            self.config.attn_implementation,
        )
        kwargs = {
            "dtype": self.dtype,
            "attn_implementation": self.config.attn_implementation,
            "revision": self.config.revision,
        }
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        # Incremental differentiable rollout requires the real KV cache.
        # Hugging Face disables use_cache when gradient checkpointing is active
        # in train mode, so the frozen trunk is intentionally kept in eval mode
        # and gradient checkpointing is not part of the canonical configuration.
        model.config.use_cache = True
        model.eval()
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        source = self.config.local_path or self.config.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            use_fast=True,
            revision=self.config.revision,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Layer access
    # ------------------------------------------------------------------ #

    @property
    def layers(self) -> nn.ModuleList:
        """Return the actual ``nn.ModuleList`` of decoder layers.

        This is the canonical place to read and write trunk layers. **Do not
        wrap the result in ``list(...)``** -- that detaches the reference and
        mutations on the returned object will not propagate to the forward
        graph. The audit found this exact bug in the previous codebase.
        """
        inner = self.model.model  # Qwen3Model
        layers = inner.layers
        if not isinstance(layers, nn.ModuleList):
            raise RuntimeError(
                f"Expected model.model.layers to be nn.ModuleList, got {type(layers).__name__}. "
                f"This adapter assumes the standard Qwen3 layout."
            )
        return layers

    def num_layers(self) -> int:
        return len(self.layers)

    def replace_layer(self, index: int, replacement: nn.Module) -> None:
        """Swap the decoder layer at ``index`` with ``replacement`` in place.

        Verifies the identity post-swap: ``self.layers[index] is replacement``.
        """
        if not 0 <= index < self.num_layers():
            raise IndexError(f"Layer index {index} out of range [0, {self.num_layers()})")
        src_layer = self.layers[index]
        device = next(src_layer.parameters()).device
        dtype = next(src_layer.parameters()).dtype
        replacement.to(device=device, dtype=dtype)
        self.layers[index] = replacement
        if self.layers[index] is not replacement:
            raise RuntimeError(
                f"Layer replacement at index {index} failed identity check; "
                f"the swap did not land on model.model.layers."
            )

    def instrumented_layer_indices(self) -> Tuple[int, ...]:
        return self._instrumented_layer_indices

    def record_instrumented_indices(self, indices: Tuple[int, ...]) -> None:
        self._instrumented_layer_indices = tuple(indices)

    # ------------------------------------------------------------------ #
    # Parameter helpers
    # ------------------------------------------------------------------ #

    def frozen_parameters(self) -> List[torch.nn.Parameter]:
        """Frozen base-trunk parameters, excluding instrumented PDT phi."""

        phi_ids = self._instrumented_phi_parameter_ids()
        parameters = [
            parameter for parameter in self.model.parameters() if id(parameter) not in phi_ids
        ]
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("A frozen base-trunk parameter unexpectedly requires gradients.")
        return parameters

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Currently trainable parameters inside the instrumented trunk tree."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def _instrumented_phi_parameter_ids(self) -> set[int]:
        parameter_ids: set[int] = set()
        for index in self._instrumented_layer_indices:
            layer = self.layers[index]
            for component_name in ("snc", "stream_adapter"):
                component = getattr(layer, component_name, None)
                if component is not None:
                    parameter_ids.update(id(parameter) for parameter in component.parameters())
            for gate_name in ("notes_gate", "adapter_gate"):
                gate = getattr(layer, gate_name, None)
                if gate is not None:
                    parameter_ids.add(id(gate))
        return parameter_ids

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
    ):
        """Thin wrapper around the HF model's forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


def _resolve_dtype(alias: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[alias]
    except KeyError as exc:
        raise ValueError(f"Unsupported trunk dtype {alias!r}. Known: {sorted(_DTYPE_MAP)}") from exc
