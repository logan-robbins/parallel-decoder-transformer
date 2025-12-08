from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..datasets.config import GenerationConfig, LocalLLMConfig


class LocalLLMClient:
    """LLM client for running local models using the Hugging Face transformers stack."""

    def __init__(self, config: LocalLLMConfig) -> None:
        torch_dtype = getattr(torch, config.dtype, torch.bfloat16)
        device_map = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer or config.model,
            trust_remote_code=config.trust_remote_code,
        )
        if hasattr(self._tokenizer, "pad_token") and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, prompts: Sequence[str], config: GenerationConfig) -> list[str]:
        outputs: list[str] = []
        for prompt in prompts:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._tokenizer.model_max_length,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generation = self._model.generate(
                    **inputs,
                    do_sample=config.temperature > 0,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k if config.top_k is not None else 0,
                    max_new_tokens=config.max_new_tokens,
                    repetition_penalty=config.repetition_penalty,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            generated_tokens = generation[0][input_length:]
            text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            outputs.append(text.strip())
        return outputs
