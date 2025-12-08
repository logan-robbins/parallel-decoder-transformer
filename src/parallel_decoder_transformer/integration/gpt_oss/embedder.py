"""Embedding utilities that expose GPT-OSS token embeddings for preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from ...utils import resolve_device

try:  # pragma: no cover - heavy dependency loaded at runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The optional 'transformers' dependency is required for GPT-OSS embeddings. "
        "Install it via `pip install parallel-decoder-transformer[data]`."
    ) from exc

try:  # pragma: no cover - optional dependency used for local checkpoints
    from safetensors.torch import safe_open
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'safetensors' package is required to read GPT-OSS checkpoints. "
        "Install it via `pip install safetensors`."
    ) from exc


@dataclass(slots=True)
class GptOssEmbedderConfig:
    """Configuration used to instantiate :class:`GptOssEmbedder`."""

    model_reference: str | Path
    tokenizer_reference: str | Path | None = None
    device: Optional[str] = None
    torch_dtype: str = "float16"
    max_length: int = 2048
    target_dimension: Optional[int] = None


class GptOssEmbedder:
    """Loads the GPT-OSS trunk and exposes pooled input embeddings."""

    def __init__(
        self,
        *,
        model_reference: str | Path,
        tokenizer_reference: str | Path | None = None,
        device: Optional[str] = None,
        torch_dtype: str = "float16",
        max_length: int = 2048,
        target_dimension: Optional[int] = None,
    ) -> None:
        if tokenizer_reference is None:
            tokenizer_reference = model_reference

        self.model_reference = str(model_reference)
        self.tokenizer_reference = str(tokenizer_reference)
        self.max_length = max_length
        self.target_dimension = target_dimension

        resolved_device = self._resolve_device(device)
        self.device = torch.device(resolved_device)
        dtype = self._resolve_dtype(torch_dtype, resolved_device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_reference,
            trust_remote_code=True,
        )
        self.model = None
        self.embed_layer = None

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_reference,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except (ValueError, OSError):  # fall back to direct embedding weight load
            embedding_weight = self._load_embedding_weight(Path(self.model_reference))
            self.embed_layer = torch.nn.Embedding.from_pretrained(embedding_weight, freeze=True)
            self.embed_layer = self.embed_layer.to(self.device)
        else:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            self.embed_layer = self.model.get_input_embeddings()
            self.embed_layer = self.embed_layer.to(self.device)

    def embed(self, texts: Sequence[str], *, target_dim: Optional[int] = None) -> list[list[float]]:
        if not texts:
            return []

        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        with torch.no_grad():
            embeddings = self.embed_layer(input_ids)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
            summed = (embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        else:
            pooled = embeddings.mean(dim=1)

        pooled = pooled.to(torch.float32).cpu()
        pooled = self._align_dimension(pooled, target_dim or self.target_dimension)
        return pooled.tolist()

    def _align_dimension(self, tensor: torch.Tensor, target_dim: Optional[int]) -> torch.Tensor:
        if target_dim is None or tensor.size(-1) == target_dim:
            return tensor
        if tensor.size(-1) > target_dim:
            return tensor[:, :target_dim]
        pad = torch.zeros(
            tensor.size(0),
            target_dim - tensor.size(-1),
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)

    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve the runtime device with CUDA preference.

        Uses the shared utility which prefers CUDA, then MPS, then CPU, and
        respects the PDT_DEVICE environment variable when set.
        """
        if device:
            return device
        return resolve_device()

    def _resolve_dtype(self, alias: str, device: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = mapping.get(alias.lower())
        if dtype is None:
            dtype = torch.float16 if device != "cpu" else torch.float32
        if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:  # pragma: no cover
            return torch.float32
        return dtype

    def _load_embedding_weight(self, model_path: Path) -> torch.Tensor:
        tensor_candidates = (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "tok_embeddings.weight",
            "embedding.weight",
        )
        index_candidates = [
            model_path / "model.safetensors.index.json",
            model_path / "pytorch_model.bin.index.json",
        ]
        weight_files = [
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin",
        ]

        for weight_file in weight_files:
            if not weight_file.exists():
                continue
            with safe_open(weight_file, framework="pt") as handle:
                for name in tensor_candidates:
                    if name in handle.keys():
                        return handle.get_tensor(name)

        for index_path in index_candidates:
            if not index_path.exists():
                continue
            mapping = json.loads(index_path.read_text())
            weight_map = mapping.get("weight_map", {})
            for name in tensor_candidates:
                shard_name = weight_map.get(name)
                if shard_name is None:
                    continue
                shard_path = model_path / shard_name
                if not shard_path.exists():
                    continue
                with safe_open(shard_path, framework="pt") as shard:
                    return shard.get_tensor(name)

        raise FileNotFoundError(
            f"Failed to locate an embedding weight under {model_path}. Ensure the GPT-OSS checkpoint "
            "includes the token embedding tensor."
        )


__all__ = ["GptOssEmbedder", "GptOssEmbedderConfig"]
