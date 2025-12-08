"""Wrappers that expose GPT-OSS trunks to the Parallel Decoder Transformer integration stack."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, TYPE_CHECKING


from ...utils import resolve_device
from .accelerated_model import _accelerate_model_in_place

try:  # Optional heavy dependency
    from transformers import AutoModelForCausalLM, PreTrainedModel
except ImportError:  # pragma: no cover - handled dynamically at runtime
    AutoModelForCausalLM = None  # type: ignore[assignment]
    PreTrainedModel = object  # type: ignore[assignment]

try:
    import torch
except ImportError as exc:  # pragma: no cover - ensures clearer error upstream
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn as torch_nn


@dataclass(slots=True)
class TrunkAdapterConfig:
    """Configuration for loading and preparing the GPT-OSS trunk."""

    base_model: str = "openai/gpt-oss-20b"
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"
    trust_remote_code: bool = True
    revision: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    freeze_lower_layers: int = 0
    unfreeze_modules: tuple[str, ...] = field(default_factory=lambda: ("lm_head",))
    parameter_dtype_overrides: Dict[str, str] = field(default_factory=dict)
    peft_checkpoint: Optional[str] = None
    num_key_value_heads: Optional[int] = None
    attn_implementation: str = (
        "flash_attention_2"  # Options: "flash_attention_2", "eager", "sdpa", "flex"
    )
    gradient_checkpointing: bool = (
        False  # Enable gradient checkpointing to trade compute for memory
    )


class GptOssTrunkAdapter:
    """Adapter that anchors a GPT-OSS model inside the Parallel Decoder Transformer stack.

    The adapter intentionally loads no weights at construction time. Call
    :meth:`load_model` or :meth:`attach_model` once an instantiated trunk is
    available. This keeps destructive refactors lightweight while wiring the
    new architecture.
    """

    def __init__(
        self, config: TrunkAdapterConfig, *, model: Optional[PreTrainedModel] = None
    ) -> None:
        self.config = config
        self._logger = logging.getLogger("parallel decoder transformer.gpt_oss.trunk")
        self._model: Optional[PreTrainedModel] = None
        if model is not None:
            self.attach_model(model)
        else:
            self._logger.debug("trunk_adapter_initialised | base=%s", config.base_model)

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("GPT-OSS trunk not loaded. Call load_model() or attach_model().")
        return self._model

    def load_model(self) -> PreTrainedModel:
        """Materialise the GPT-OSS trunk using Hugging Face weights."""

        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is required to load GPT-OSS. Install it before calling load_model()."
            )
        if self._model is not None:
            return self._model
        base_model_ref = self.config.base_model
        base_model_path: Optional[Path] = None
        try:
            candidate = Path(base_model_ref)
            if candidate.exists():
                base_model_path = candidate
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            base_model_path = None

        device_preference = resolve_device()
        dtype_alias = self.config.torch_dtype or "float32"
        if (
            device_preference == "mps"
            and isinstance(dtype_alias, str)
            and dtype_alias.lower() in {"bfloat16", "bf16"}
        ):
            dtype_alias = "float16"
        torch_dtype = _resolve_torch_dtype(dtype_alias)
        device_map = self.config.device_map
        if device_preference != "cuda":
            device_map = None
        self._logger.info(
            "load_trunk_start | model=%s | dtype=%s | device_map=%s | device=%s | local_path=%s",
            self.config.base_model,
            torch_dtype,
            device_map,
            device_preference,
            base_model_path if base_model_path is not None else "<remote>",
        )
        self._model = AutoModelForCausalLM.from_pretrained(  # type: ignore[call-arg]
            self.config.base_model,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            local_files_only=bool(base_model_path is not None),
            low_cpu_mem_usage=True,
            # If 'flex' is requested, we load as 'eager' first, then patch it.
            attn_implementation=(
                "eager"
                if self.config.attn_implementation == "flex"
                else self.config.attn_implementation
            ),
        )

        if self.config.attn_implementation == "flex":
            self._logger.info(
                "load_trunk_applying_flex_attention | patch=AcceleratedGptOssAttention"
            )
            _accelerate_model_in_place(self._model)

        if torch is not None and device_preference in {"cuda", "mps"} and device_map is None:
            try:
                target = torch.device(device_preference)
                self._model.to(target)
            except Exception as exc:  # pragma: no cover - defensive guard for unsupported devices
                self._logger.warning(
                    "load_trunk_device_move_failed | device=%s | error=%s", device_preference, exc
                )
        self._post_attach()
        return self._model

    def attach_model(self, model: PreTrainedModel) -> None:
        """Attach an already-instantiated GPT-OSS trunk."""

        self._model = model
        self._post_attach()

    def _post_attach(self) -> None:
        """Apply freezing policies and dtype overrides after attaching a model."""

        if self._model is None:
            return

        # Enable gradient checkpointing if requested (trades compute for memory)
        if self.config.gradient_checkpointing:
            if hasattr(self._model, "gradient_checkpointing_enable"):
                self._model.gradient_checkpointing_enable()
                self._logger.info("trunk_gradient_checkpointing_enabled")
            else:
                self._logger.warning(
                    "trunk_gradient_checkpointing_not_supported | model_type=%s",
                    type(self._model).__name__,
                )

        self._apply_freeze_policy()
        self._apply_gqa_override()
        self._apply_dtype_overrides()
        if self.config.peft_checkpoint:
            self._load_peft_checkpoint(self.config.peft_checkpoint)

    def _apply_freeze_policy(self) -> None:
        model = self._model
        if model is None:
            return
        frozen, unfrozen = 0, 0
        for param in model.parameters():
            param.requires_grad = False
            frozen += 1
        target_layers = _resolve_transformer_layers(model)
        if target_layers and self.config.freeze_lower_layers > 0:
            trainable_layers = target_layers[-self.config.freeze_lower_layers :]
            for layer in trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen += 1
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in self.config.unfreeze_modules):
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += 1
        self._logger.info("freeze_policy_applied | frozen=%d | unfrozen=%d", frozen, unfrozen)

    def _apply_dtype_overrides(self) -> None:
        model = self._model
        if model is None or not self.config.parameter_dtype_overrides:
            return
        overrides = {
            prefix: _resolve_torch_dtype(dtype)
            for prefix, dtype in self.config.parameter_dtype_overrides.items()
        }
        for name, param in model.named_parameters():
            for prefix, dtype in overrides.items():
                if name.startswith(prefix):
                    param.data = param.data.to(dtype)
                    break

    def _apply_gqa_override(self) -> None:
        model = self._model
        heads = self.config.num_key_value_heads
        if model is None or heads is None:
            return
        config = getattr(model, "config", None)
        if config is None or not hasattr(config, "num_key_value_heads"):
            self._logger.warning("gqa_override_skipped | attribute_missing=num_key_value_heads")
            return
        config.num_key_value_heads = int(heads)
        self._logger.info("gqa_override_applied | num_key_value_heads=%d", heads)

    def iter_trainable_parameters(self) -> Iterable["torch_nn.Parameter"]:
        model = self._model
        if model is None:
            return tuple()
        return tuple(param for param in model.parameters() if param.requires_grad)

    def _load_peft_checkpoint(self, checkpoint: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot load PEFT checkpoint before the trunk model is attached.")
        if PeftModel is None:
            raise RuntimeError(
                "peft is required to load adapter checkpoints. Install it via `pip install peft`."
            )
        path = checkpoint
        self._logger.info("peft_load_start | path=%s", path)
        self._model = PeftModel.from_pretrained(self._model, path, is_trainable=True)
        self._logger.info("peft_load_complete | path=%s", path)


def _resolve_transformer_layers(model: PreTrainedModel) -> Optional[Iterable["torch_nn.Module"]]:
    """Best-effort resolution of transformer block stack."""

    potential_paths = ("model.layers", "model.decoder.layers", "transformer.h")
    for path in potential_paths:
        layers = _dig(model, path)
        if layers is not None:
            return list(layers)  # type: ignore[return-value]
    return None


def _dig(obj: object, path: str) -> Optional[object]:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _resolve_torch_dtype(alias: str) -> torch.dtype:
    if torch is None:  # pragma: no cover - exercised only in environments without torch
        raise RuntimeError(
            "PyTorch is required to resolve dtype aliases but is not installed."
        ) from _TORCH_IMPORT_ERROR
    alias_lower = alias.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    if alias_lower in mapping:
        return mapping[alias_lower]
    stripped = alias_lower.replace("float", "")
    if stripped in mapping:
        return mapping[stripped]
    raise ValueError(f"Unsupported torch dtype alias: {alias!r}")


__all__ = ["TrunkAdapterConfig", "GptOssTrunkAdapter"]
