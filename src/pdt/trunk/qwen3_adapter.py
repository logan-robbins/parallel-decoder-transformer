"""Frozen Qwen3 trunk adapter.

Loads ``Qwen/Qwen3-4B-Base`` (or any Qwen3 variant) via
``AutoModelForCausalLM``, freezes every parameter, optionally resizes
the embedding matrix to accommodate PDT special tokens, and exposes a
list of the trunk's decoder layers for instrumentation.

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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

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
        self._resize_for_special_tokens()
        self._freeze()
        self._instrumented_layer_indices: Tuple[int, ...] = tuple()

    def _load_model(self) -> PreTrainedModel:
        source = self.config.local_path or self.config.base_model
        LOGGER.info("Loading Qwen3 trunk from %s (dtype=%s, attn=%s)",
                    source, self.dtype, self.config.attn_implementation)
        kwargs = {
            "torch_dtype": self.dtype,
            "attn_implementation": self.config.attn_implementation,
        }
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.eval()
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        source = self.config.local_path or self.config.base_model
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _resize_for_special_tokens(self) -> None:
        if not self.config.extra_special_tokens:
            return
        existing = set(self.tokenizer.get_vocab().keys())
        new_tokens = [tok for tok in self.config.extra_special_tokens if tok not in existing]
        if not new_tokens:
            return
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if added > 0:
            # Qwen3's embedding matrix has padded rows (vocab > len(tokenizer)
            # out of the box). Resize only if the token additions overflow
            # the current embedding capacity; otherwise the tokens already
            # have a home.
            old_embed = self.model.get_input_embeddings().weight
            old_vocab = old_embed.size(0)
            new_vocab = max(old_vocab, len(self.tokenizer))
            if new_vocab == old_vocab:
                LOGGER.info(
                    "Added %d special tokens; vocab stays at %d (Qwen3 embedding has %d padding rows reused)",
                    added, old_vocab, old_vocab - len(self.tokenizer),
                )
                return
            self.model.resize_token_embeddings(new_vocab)
            # Initialize the new rows with small Gaussian noise so they are
            # not exactly zero (which would land on the softmax tail forever).
            with torch.no_grad():
                embed = self.model.get_input_embeddings().weight
                embed[old_vocab:].normal_(mean=0.0, std=0.02)
                out_embed = self.model.get_output_embeddings()
                if out_embed is not None and out_embed.weight.size(0) != new_vocab:
                    # Tied embeddings are resized above; if untied, copy too.
                    out_w = out_embed.weight
                    if out_w.size(0) < new_vocab:
                        out_w.data = torch.cat(
                            [out_w.data, torch.randn(new_vocab - out_w.size(0), out_w.size(1),
                                                     dtype=out_w.dtype, device=out_w.device) * 0.02],
                            dim=0,
                        )
            LOGGER.info("Added %d special tokens; vocab: %d -> %d", added, old_vocab, new_vocab)

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
            raise IndexError(
                f"Layer index {index} out of range [0, {self.num_layers()})"
            )
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
        """Every parameter of the trunk. All are frozen by construction."""
        return list(self.model.parameters())

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Parameters still marked ``requires_grad=True`` inside the trunk.

        For a pure frozen-trunk setup this should always be empty -- useful
        as a sanity check after instrumentation has landed \u03c6 modules inside
        the replaced layers.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
    ):
        """Thin wrapper around the HF model's forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


def _resolve_dtype(alias: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[alias]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported trunk dtype {alias!r}. Known: {sorted(_DTYPE_MAP)}"
        ) from exc
