"""Token-level baseline runners (Medusa/Lookahead/EAGLE) for manifest parity."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..inference import DecodeConfig


@dataclass(frozen=True)
class TokenBaselineConfig:
    """Configuration preset describing a token-level baseline."""

    name: str
    chunk_size: int
    branch_factor: int


_BASELINE_PRESETS: Dict[str, TokenBaselineConfig] = {
    "medusa": TokenBaselineConfig(name="medusa", chunk_size=4, branch_factor=3),
    "lookahead": TokenBaselineConfig(name="lookahead", chunk_size=2, branch_factor=2),
    "eagle": TokenBaselineConfig(name="eagle", chunk_size=3, branch_factor=2),
}


def build_token_baseline_config(name: str) -> TokenBaselineConfig:
    preset = _BASELINE_PRESETS.get(name.lower())
    if preset is None:
        raise ValueError(f"Unknown token baseline '{name}'.")
    return preset


BaselineEventCallback = Callable[[Dict[str, Any], int], None]


def run_token_baseline(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    decode_config: DecodeConfig,
    baseline_config: TokenBaselineConfig,
    *,
    max_new_tokens: Optional[int] = None,
    event_callback: Optional[BaselineEventCallback] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run a token-level baseline and return a PDT-compatible manifest."""

    if baseline_config.chunk_size <= 0:
        raise ValueError("Token baseline chunk_size must be positive.")
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    generated_tokens: List[int] = []
    generated_texts: List[str] = []
    per_token_records: List[Dict[str, Any]] = []
    gate_trace: List[Dict[str, Any]] = []
    cadence_events: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    chunk_tokens: List[int] = []
    chunk_index = 0
    chunk_start = time.perf_counter()
    max_steps = int(max_new_tokens or decode_config.max_new_tokens or 1)
    max_steps = max(1, max_steps)
    start_time = time.perf_counter()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for step in range(max_steps):
        step_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :].detach()
        token_id, top2_margin = _sample_token(logits, decode_config, generated_tokens)
        duration = time.perf_counter() - step_start
        per_token_records.append(
            {
                "step": step + 1,
                "stream": baseline_config.name,
                "stride_index": chunk_index,
                "token_index": step,
                "duration_s": duration,
                "top2_margin": top2_margin,
            }
        )
        chunk_tokens.append(token_id)
        gate_trace.append({"step": step + 1, "stream": baseline_config.name, "value": 0.0})
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        generated_tokens.append(token_id)
        generated_texts.append(token_text)
        stride_completed = len(chunk_tokens) >= baseline_config.chunk_size
        event = {
            "stream": baseline_config.name,
            "token_id": token_id,
            "token_text": token_text,
            "stride_index": chunk_index,
            "stride_completed": stride_completed,
            "stream_completed": False,
            "agreement": 1.0,
            "notes_emitted": False,
            "rollback_performed": False,
            "coverage_logits": None,
            "counterfactuals": None,
            "top2_margin": top2_margin,
            "cadence_mode": "token_chunk",
            "cadence_probability": 1.0,
            "cadence_multiplier": 1.0,
            "cadence_forced": False,
        }
        events.append(event)
        if event_callback is not None:
            event_callback(event, step + 1)
        next_token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if attention_mask is not None:
            ones = torch.ones(
                (attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([attention_mask, ones], dim=-1)
        eos_hit = eos_token_id is not None and token_id == int(eos_token_id)
        forced_end = step + 1 == max_steps
        if stride_completed or eos_hit or forced_end:
            cadence_events.append(
                {
                    "chunk_index": chunk_index,
                    "draft_tokens": baseline_config.chunk_size,
                    "accepted_tokens": list(chunk_tokens),
                    "latency_s": time.perf_counter() - chunk_start,
                    "branch_factor": baseline_config.branch_factor,
                }
            )
            chunk_tokens.clear()
            chunk_index += 1
            chunk_start = time.perf_counter()
        if eos_hit:
            event["stride_completed"] = True
            event["stream_completed"] = True
            break
        if forced_end and not stride_completed and events:
            events[-1]["stride_completed"] = True
    total_duration = time.perf_counter() - start_time
    stream_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    manifest = {
        "baseline": baseline_config.name,
        "timings": {"total": total_duration, "per_token": per_token_records},
        "integration": {"mode": "baseline", "instrumented_layers": []},
        "config": {
            "stride_B": baseline_config.chunk_size,
            "commit_L": baseline_config.chunk_size,
            "read_lag_delta": 0,
            "max_snapshots_K": 1,
            "topology": "sequential",
            "gate_g": 0.0,
            "tau": 1.0,
            "M_by_stream": {baseline_config.name: baseline_config.chunk_size},
            "alpha": 0.0,
            "gate_annealing": {
                "enabled": False,
                "decay": 1.0,
                "min_value": 0.0,
                "recovery": 0.0,
                "stability_margin": 0.0,
                "cooldown": 0,
            },
            "cadence_policy": {
                "mode": "deterministic",
                "min_probability": 1.0,
                "max_interval": baseline_config.chunk_size,
                "multiplier_min": 1.0,
                "multiplier_max": 1.0,
                "agreement_low": 0.0,
                "agreement_high": 1.0,
                "age_boost": 0.0,
            },
            "rng_seed": decode_config.seed,
            "decode": decode_config.as_sampling_kwargs(),
            "baseline_params": {
                "chunk_size": baseline_config.chunk_size,
                "branch_factor": baseline_config.branch_factor,
            },
        },
        "streams": {
            baseline_config.name: {
                "text": stream_text,
                "token_texts": generated_texts,
                "token_ids": generated_tokens,
                "latest_version": len(generated_tokens),
                "rollback_buffer": [],
                "gate": 0.0,
                "coverage": None,
            }
        },
        "rollbacks": [],
        "steps": len(events),
        "cadence_events": cadence_events,
        "gate_trace": gate_trace,
    }
    return manifest, events


def _sample_token(
    logits: torch.Tensor,
    decode_config: DecodeConfig,
    history: Sequence[int],
) -> Tuple[int, float]:
    scores = logits.clone()
    if decode_config.temperature > 0 and decode_config.temperature != 1.0:
        scores = scores / float(decode_config.temperature)
    if decode_config.repetition_penalty != 1.0 and history:
        scores = _apply_repetition_penalty(scores, history, decode_config.repetition_penalty)
    scores = _apply_top_k_top_p(scores, decode_config.top_k, decode_config.top_p)
    if decode_config.do_sample:
        probs = F.softmax(scores, dim=-1)
        token = int(torch.multinomial(probs, num_samples=1).item())
    else:
        token = int(torch.argmax(scores, dim=-1).item())
    margin = _top2_margin(scores)
    return token, margin


def _apply_repetition_penalty(
    scores: torch.Tensor,
    history: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    adjusted = scores
    for token in history:
        if 0 <= token < adjusted.size(-1):
            value = adjusted[..., token]
            adjusted[..., token] = torch.where(value < 0, value * penalty, value / penalty)
    return adjusted


def _apply_top_k_top_p(scores: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    filtered = scores
    vocab = scores.size(-1)
    if top_k > 0 and top_k < vocab:
        kth_values = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        mask = filtered < kth_values
        filtered = filtered.masked_fill(mask, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        cutoff_indices = torch.where(cutoff, float("-inf"), 0.0)
        filtered = filtered.scatter(-1, sorted_indices, sorted_logits + cutoff_indices)
    return filtered


def _top2_margin(scores: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.0
    row = scores.view(-1)
    values, _ = torch.topk(row, k=min(2, row.numel()))
    if values.numel() < 2:
        return 0.0
    return float((values[0] - values[1]).item())
