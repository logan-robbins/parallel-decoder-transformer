# ruff: noqa: E402
"""Inference CLI for the GPT-OSS backed Parallel Decoder Transformer."""

from __future__ import annotations

# Ensure local src/ is on sys.path when running from the repo without installation
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_SRC_PATH = _os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in _sys.path and _os.path.isdir(_SRC_PATH):
    _sys.path.insert(0, _SRC_PATH)

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from parallel_decoder_transformer.config import ModelConfig, TrainingConfig
from parallel_decoder_transformer.baselines import build_token_baseline_config, run_token_baseline
from parallel_decoder_transformer.inference import (
    CounterfactualConfig,
    DecodeConfig,
    InferenceConfig,
    MultiStreamOrchestrator,
    StepOutcome,
    build_inference_config,
)
from parallel_decoder_transformer.inference.replay import ReplayArtifactWriter
from parallel_decoder_transformer.models import ParallelDecoderTransformer
from parallel_decoder_transformer.integration.gpt_oss import TrunkAdapterConfig
from parallel_decoder_transformer.models.stream_adapters import StreamAdapterConfig
from parallel_decoder_transformer.models.heads import (
    PlannerHeadConfig,
    NotesHeadConfig,
    SpeculationHeadConfig,
    AgreementHeadConfig,
)
from parallel_decoder_transformer.inference.dnb_bus import DynamicNotesBusConfig
from parallel_decoder_transformer.inference.snc_cross_attn import SharedNotesCrossAttentionConfig
from parallel_decoder_transformer.training.trainer import (
    CurriculumConfig,
    GradNormConfig,
    LossWeights,
    MetricsConfig,
    NegativeSamplingConfig,
    NotesNoiseConfig,
    StagePolicyConfig,
    TeacherBranchConfig,
)
from parallel_decoder_transformer.data.collator_kd import TwoBranchKDCollatorConfig
from parallel_decoder_transformer.data.teacher_runner import DatasetTeacherConfig
from parallel_decoder_transformer.data.tokenizer import TokenizerConfig, resolve_tokenizer
from parallel_decoder_transformer.utils import configure_logging, seed_everything, get_git_metadata
from parallel_decoder_transformer.utils.plan_catalog import resolve_plan_hash_params

try:  # pragma: no cover - optional dependency for hidden size probing
    from transformers import AutoConfig
except ImportError:  # pragma: no cover - runtime path without transformers config
    AutoConfig = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Parallel Decoder Transformer inference with GPT-OSS backbone."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_oss_transfer.yaml"),
        help="Path to YAML configuration file used for training.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to decode. Mutually exclusive with --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file containing the prompt. Overrides --prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/infer/manifest.json"),
        help="Where to write the inference manifest JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional adapter checkpoint file or directory containing adapters.pt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed overriding the inference config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override for maximum tokens generated per stream.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override sampling top-k.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override sampling top-p.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    # Device is auto-detected via the trunk adapter; override via PDT_DEVICE env if needed.
    parser.add_argument(
        "--hf-verbosity",
        choices=["critical", "error", "warning", "info", "debug"],
        default=None,
        help="Optional override for Hugging Face `TRANSFORMERS_VERBOSITY`.",
    )
    parser.add_argument(
        "--cadence-mode",
        choices=["deterministic", "stochastic", "adaptive"],
        default=None,
        help="Override the cadence policy mode used during inference.",
    )
    parser.add_argument(
        "--cadence-max-interval",
        type=int,
        default=None,
        help="Force an emission after this many tokens without notes when using stochastic/adaptive cadence.",
    )
    parser.add_argument(
        "--cadence-min-prob",
        type=float,
        default=None,
        help="Minimum emission probability clamp for stochastic/adaptive cadence.",
    )
    parser.add_argument(
        "--cadence-m-min",
        type=float,
        default=None,
        help="Lower multiplier bound for adaptive cadence modulation.",
    )
    parser.add_argument(
        "--cadence-m-max",
        type=float,
        default=None,
        help="Upper multiplier bound for adaptive cadence modulation.",
    )
    parser.add_argument(
        "--cadence-agreement-low",
        type=float,
        default=None,
        help="Agreement threshold below which adaptive cadence applies maximum multiplier.",
    )
    parser.add_argument(
        "--cadence-agreement-high",
        type=float,
        default=None,
        help="Agreement threshold above which adaptive cadence applies minimum multiplier.",
    )
    parser.add_argument(
        "--cadence-age-boost",
        type=float,
        default=None,
        help="Additional multiplier growth per cadence interval when notes age exceeds the base cadence.",
    )
    parser.add_argument(
        "--cadence-M",
        action="append",
        default=None,
        metavar="ROLE=M",
        help=(
            "Override deterministic cadence per stream without editing configs (repeatable, use ROLE=M or all=M)."
        ),
    )
    parser.add_argument(
        "--cf-swap",
        action="append",
        default=[],
        metavar="ROLE_A:ROLE_B",
        help="Swap notes windows between two streams before SNC (repeatable).",
    )
    parser.add_argument(
        "--cf-shuffle",
        action="append",
        default=[],
        metavar="ROLE",
        help="Shuffle notes window order for the specified stream (repeatable, use 'all' for every stream).",
    )
    parser.add_argument(
        "--cf-freeze",
        action="append",
        default=[],
        metavar="ROLE",
        help="Freeze the first observed notes window for the specified stream (repeatable, use 'all' for every stream).",
    )
    parser.add_argument(
        "--cf-stale",
        action="append",
        default=[],
        metavar="ROLE:DELTA",
        help="Increase the effective read lag for ROLE by DELTA snapshots (repeatable).",
    )
    parser.add_argument(
        "--cf-stale-default",
        type=int,
        default=0,
        help="Global stale lag delta applied to all streams unless overridden by --cf-stale.",
    )
    parser.add_argument(
        "--cf-ablate",
        action="append",
        default=[],
        metavar="ROLE",
        help="Zero the notes window for ROLE before SNC (repeatable, use 'all' for every stream).",
    )
    parser.add_argument(
        "--cf-tag",
        type=str,
        default=None,
        help="Optional manifest tag describing the counterfactual intervention.",
    )
    parser.add_argument(
        "--memory-report",
        action="store_true",
        help="Record per-step device/host memory usage in the inference manifest.",
    )
    parser.add_argument(
        "--gate-g",
        type=float,
        default=None,
        help="Override SNC gate scalar in [0,1] (set 0 for no cross-lane influence).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional override for logit blend alpha in [0,1] (1 uses attended logits only).",
    )
    parser.add_argument(
        "--read-lag-delta",
        type=int,
        default=None,
        help="Override Dynamic Notes Bus read lag Δ (default taken from training curriculum).",
    )
    parser.add_argument(
        "--seed-text",
        action="append",
        default=None,
        metavar="ROLE=TEXT",
        help="Inject seed text for a stream prior to decoding (repeat per stream, format stream=text).",
    )
    parser.add_argument(
        "--seed-text-file",
        type=Path,
        default=None,
        help="JSON file mapping stream -> seed text string to preload the notes bus.",
    )
    parser.add_argument(
        "--seed-notes-file",
        type=Path,
        default=None,
        help="JSON file mapping stream -> list[float] seed vector (length notes_dim) for initial bus snapshots.",
    )
    parser.add_argument(
        "--stream-prefix-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping { stream: prefix } to prepend per-stream before the shared prompt. "
            "Useful for demos to steer each lane to a distinct part."
        ),
    )
    parser.add_argument(
        "--plan-text-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping { stream: [plan item strings...] } used to seed the coverage/catalog metadata. "
            "When provided, coverage logits are aligned to these entries instead of the model's planner output."
        ),
    )
    parser.add_argument(
        "--plan-contract",
        type=Path,
        default=None,
        help=(
            "Optional path to a plan JSON (matching the dataset schema) used to derive initial Dynamic Notes Bus "
            "snapshots before decoding."
        ),
    )
    parser.add_argument(
        "--stream",
        action="append",
        default=None,
        help=(
            "Override stream names (repeat for each lane, e.g. --stream stream_1 --stream stream_2 --stream stream_3). "
            "Names are normalised to lowercase."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream per-token outputs to stdout.",
    )
    parser.add_argument(
        "--stream-jsonl",
        action="store_true",
        help="Emit a JSON line per token step with rich telemetry.",
    )
    parser.add_argument(
        "--replay-artifact-dir",
        type=Path,
        default=None,
        help="Optional directory where a logit replay artifact should be written.",
    )
    parser.add_argument(
        "--replay-chunk-size",
        type=int,
        default=4096,
        help="Maximum rows per tensor chunk when writing replay artifacts.",
    )
    parser.add_argument(
        "--log-margins",
        action="store_true",
        help="Record per-token top-2 logit margins in the manifest.",
    )
    parser.add_argument(
        "--sync-profile",
        action="store_true",
        help="Synchronize the device each stride to capture sync timings.",
    )
    parser.add_argument(
        "--notes-text-file",
        type=Path,
        default=None,
        help="Optional JSON mapping { stream: [note strings...] } to attach reference notes for evaluation.",
    )
    parser.add_argument(
        "--baseline",
        choices=["sequential", "medusa", "lookahead", "eagle"],
        default=None,
        help=(
            "Optional runtime baseline: 'sequential' disables SNC, 'medusa'/'lookahead'/'eagle' "
            "run token-level draft baselines with manifest parity."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        text = args.prompt_file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file {args.prompt_file} is empty.")
        return text
    if args.prompt is None or not args.prompt.strip():
        raise ValueError("Provide either --prompt or --prompt-file.")
    return args.prompt


def _coerce_model_config(payload: Dict[str, Any]) -> ModelConfig:
    data = dict(payload)
    if isinstance(data.get("trunk"), dict):
        data["trunk"] = TrunkAdapterConfig(**data["trunk"])
    if isinstance(data.get("stream_adapters"), dict):
        data["stream_adapters"] = StreamAdapterConfig(**data["stream_adapters"])
    if isinstance(data.get("notes_bus"), dict):
        data["notes_bus"] = DynamicNotesBusConfig(**data["notes_bus"])
    if isinstance(data.get("cross_attention"), dict):
        data["cross_attention"] = SharedNotesCrossAttentionConfig(**data["cross_attention"])
    if isinstance(data.get("planner_head"), dict):
        data["planner_head"] = PlannerHeadConfig(**data["planner_head"])
    if isinstance(data.get("notes_head"), dict):
        data["notes_head"] = NotesHeadConfig(**data["notes_head"])
    if isinstance(data.get("speculation_head"), dict):
        data["speculation_head"] = SpeculationHeadConfig(**data["speculation_head"])
    if isinstance(data.get("agreement_head"), dict):
        data["agreement_head"] = AgreementHeadConfig(**data["agreement_head"])
    if isinstance(data.get("collator"), dict):
        data["collator"] = TwoBranchKDCollatorConfig(**data["collator"])
    return ModelConfig(**data)


def _coerce_training_config(payload: Dict[str, Any]) -> TrainingConfig:
    data = dict(payload)
    if isinstance(data.get("curriculum"), dict):
        data["curriculum"] = CurriculumConfig(**data["curriculum"])
    if isinstance(data.get("teacher"), dict):
        data["teacher"] = TeacherBranchConfig(**data["teacher"])
    if isinstance(data.get("dataset_teacher"), dict):
        data["dataset_teacher"] = DatasetTeacherConfig(**data["dataset_teacher"])
    if isinstance(data.get("loss_weights"), dict):
        data["loss_weights"] = LossWeights(**data["loss_weights"])
    if isinstance(data.get("notes_noise"), dict):
        data["notes_noise"] = NotesNoiseConfig(**data["notes_noise"])
    if isinstance(data.get("metrics"), dict):
        data["metrics"] = MetricsConfig(**data["metrics"])
    if isinstance(data.get("negative_sampling"), dict):
        data["negative_sampling"] = NegativeSamplingConfig(**data["negative_sampling"])
    if isinstance(data.get("gradnorm"), dict):
        data["gradnorm"] = GradNormConfig(**data["gradnorm"])
    if isinstance(data.get("stage_policies"), dict):
        policies: Dict[int, StagePolicyConfig] = {}
        for key, policy_payload in data["stage_policies"].items():
            try:
                index = int(key)
            except (TypeError, ValueError) as err:
                raise ValueError(f"Stage policy keys must be integers, received {key!r}.") from err
            payload_copy = dict(policy_payload or {})
            if isinstance(payload_copy.get("notes_noise"), dict):
                payload_copy["notes_noise"] = NotesNoiseConfig(**payload_copy["notes_noise"])
            policies[index] = StagePolicyConfig(**payload_copy)
        data["stage_policies"] = policies
    return TrainingConfig(**data)


def _resolve_trunk_hidden_size(trunk_cfg: TrunkAdapterConfig) -> Optional[int]:
    if AutoConfig is None:
        return None
    base_model = trunk_cfg.base_model
    kwargs: Dict[str, Any] = {
        "trust_remote_code": trunk_cfg.trust_remote_code,
    }
    if trunk_cfg.revision is not None:
        kwargs["revision"] = trunk_cfg.revision
    try:
        candidate = Path(base_model)
    except (TypeError, ValueError):  # pragma: no cover - defensive path
        candidate = None
    else:
        if candidate.exists():
            kwargs["local_files_only"] = True
    try:
        config = AutoConfig.from_pretrained(base_model, **kwargs)
    except Exception:  # pragma: no cover - best effort probe
        return None
    hidden_size = getattr(config, "hidden_size", None)
    return int(hidden_size) if hidden_size is not None else None


def _align_model_hidden_size(model_cfg: ModelConfig, hidden_size: Optional[int], logger) -> None:
    if hidden_size is None or hidden_size == model_cfg.hidden_size:
        return
    logger.info(
        "model_hidden_size_adjust | configured=%d | trunk=%d",
        model_cfg.hidden_size,
        hidden_size,
    )
    model_cfg.hidden_size = hidden_size
    if model_cfg.stream_adapters is not None:
        model_cfg.stream_adapters = replace(model_cfg.stream_adapters, hidden_size=hidden_size)
    if model_cfg.cross_attention is not None:
        model_cfg.cross_attention = replace(model_cfg.cross_attention, hidden_size=hidden_size)
    if model_cfg.planner_head is not None:
        model_cfg.planner_head = replace(model_cfg.planner_head, hidden_size=hidden_size)
    if model_cfg.notes_head is not None:
        model_cfg.notes_head = replace(model_cfg.notes_head, hidden_size=hidden_size)
    if model_cfg.speculation_head is not None:
        model_cfg.speculation_head = replace(model_cfg.speculation_head, hidden_size=hidden_size)
    if model_cfg.agreement_head is not None:
        model_cfg.agreement_head = replace(model_cfg.agreement_head, hidden_size=hidden_size)
    if model_cfg.coverage_head is not None:
        model_cfg.coverage_head = replace(model_cfg.coverage_head, hidden_size=hidden_size)
    if model_cfg.stream_classifier_head is not None:
        model_cfg.stream_classifier_head = replace(
            model_cfg.stream_classifier_head, hidden_size=hidden_size
        )


def _parse_seed_texts(
    arg_entries: Optional[Sequence[str]], file_path: Optional[Path]
) -> Dict[str, str]:
    seed_map: Dict[str, str] = {}
    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(f"seed text file not found: {file_path}")
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("--seed-text-file must contain a JSON object mapping stream -> text")
        for key, value in raw.items():
            if not isinstance(value, str):
                raise ValueError("Seed text values must be strings.")
            seed_map[str(key).lower()] = value
    if arg_entries:
        for entry in arg_entries:
            if "=" not in entry:
                raise ValueError("Seed text entries must use ROLE=TEXT format.")
            stream, text = entry.split("=", 1)
            stream = stream.strip().lower()
            if not stream:
                raise ValueError("Seed text stream cannot be empty.")
            seed_map[stream] = text.lstrip()
    return seed_map


def _parse_seed_notes(file_path: Optional[Path]) -> Dict[str, List[float]]:
    if file_path is None:
        return {}
    if not file_path.exists():
        raise FileNotFoundError(f"seed notes file not found: {file_path}")
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "--seed-notes-file must contain a JSON object mapping stream -> list of floats."
        )
    seed_notes: Dict[str, List[float]] = {}
    for key, value in raw.items():
        if not isinstance(value, list) or not value:
            raise ValueError("Seed note values must be non-empty lists of floats.")
        try:
            vector = [float(item) for item in value]
        except (TypeError, ValueError) as err:
            raise ValueError("Seed note vectors must contain numeric values.") from err
        seed_notes[str(key).lower()] = vector
    return seed_notes


def _load_plan_text_map(
    file_path: Optional[Path], streams: Sequence[str]
) -> Optional[Dict[str, List[str]]]:
    if file_path is None:
        return None
    if not file_path.exists():
        raise FileNotFoundError(f"plan text file not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--plan-text-file must contain a JSON object mapping stream -> list[str].")
    stream_set = {stream.lower() for stream in streams}
    plan_map: Dict[str, List[str]] = {}
    for key, value in payload.items():
        stream = str(key).lower()
        if stream not in stream_set:
            raise ValueError(
                f"Stream '{stream}' specified in --plan-text-file not present in inference streams {streams}."
            )
        if not isinstance(value, Sequence):
            raise ValueError("Plan text mapping values must be lists of strings.")
        entries = [
            str(item).strip() for item in value if isinstance(item, str) and str(item).strip()
        ]
        if entries:
            plan_map[stream] = entries
    return plan_map or None


def _load_plan_contract(file_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if file_path is None:
        return None
    if not file_path.exists():
        raise FileNotFoundError(f"plan contract file not found: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--plan-contract must contain a JSON object.")
    return payload


def _parse_notes_texts(file_path: Optional[Path]) -> Dict[str, List[str]]:
    if file_path is None:
        return {}
    if not file_path.exists():
        raise FileNotFoundError(f"notes text file not found: {file_path}")
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "--notes-text-file must contain a JSON object mapping stream -> list of strings."
        )
    notes_map: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise ValueError("Notes text values must be lists of strings.")
        entries: List[str] = []
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError("Notes text entries must be strings.")
            trimmed = entry.strip()
            if trimmed:
                entries.append(trimmed)
        notes_map[str(key).lower()] = entries
    return notes_map


def _parse_cadence_overrides(
    entries: Optional[Sequence[str]], streams: Sequence[str]
) -> Dict[str, int]:
    if not entries:
        return {}
    valid_streams = {stream.lower() for stream in streams}
    overrides: Dict[str, int] = {}
    for raw in entries:
        if not raw:
            continue
        if "=" in raw:
            stream_part, value_part = raw.split("=", 1)
        elif ":" in raw:
            stream_part, value_part = raw.split(":", 1)
        else:
            raise ValueError("--cadence-M entries must use ROLE=M format.")
        stream_norm = stream_part.strip().lower()
        try:
            cadence_val = int(value_part.strip())
        except ValueError as err:
            raise ValueError(
                f"Invalid cadence value '{value_part}' for stream '{stream_norm}'."
            ) from err
        if cadence_val <= 0:
            raise ValueError("Cadence overrides must be positive integers.")
        targets: Sequence[str]
        if stream_norm == "all":
            targets = sorted(valid_streams)
        else:
            if stream_norm not in valid_streams:
                raise ValueError(
                    f"Unknown stream '{stream_norm}' in --cadence-M; valid streams: {sorted(valid_streams)}"
                )
            targets = (stream_norm,)
        for target in targets:
            overrides[target] = cadence_val
    return overrides


def _counterfactual_requested(args: argparse.Namespace) -> bool:
    return bool(
        (args.cf_swap and len(args.cf_swap) > 0)
        or (args.cf_shuffle and len(args.cf_shuffle) > 0)
        or (args.cf_freeze and len(args.cf_freeze) > 0)
        or (args.cf_stale and len(args.cf_stale) > 0)
        or (args.cf_stale_default and args.cf_stale_default > 0)
        or (args.cf_ablate and len(args.cf_ablate) > 0)
        or (args.cf_tag and args.cf_tag.strip())
    )


def _normalise_stream_list(
    values: Optional[Sequence[str]], valid_streams: Sequence[str]
) -> List[str]:
    if not values:
        return []
    valid_set = {stream.lower() for stream in valid_streams}
    if not valid_set:
        raise ValueError("No streams available to apply counterfactual overrides.")
    resolved: List[str] = []
    for entry in values:
        if entry is None:
            continue
        stream_norm = str(entry).strip().lower()
        if not stream_norm:
            continue
        if stream_norm == "all":
            return sorted(valid_set)
        if stream_norm not in valid_set:
            raise ValueError(
                f"Unknown stream '{stream_norm}' provided in counterfactual arguments; valid streams: {sorted(valid_set)}"
            )
        if stream_norm not in resolved:
            resolved.append(stream_norm)
    return resolved


def _build_counterfactual_config(
    args: argparse.Namespace,
    streams: Sequence[str],
) -> Optional[CounterfactualConfig]:
    if not _counterfactual_requested(args):
        return None
    stream_list = tuple(stream.lower() for stream in streams)
    stream_set = set(stream_list)
    swap_pairs: List[Tuple[str, str]] = []
    for spec in args.cf_swap or []:
        if not spec or ":" not in spec:
            raise ValueError("--cf-swap entries must use ROLE_A:ROLE_B format.")
        left, right = spec.split(":", 1)
        lhs = left.strip().lower()
        rhs = right.strip().lower()
        if lhs not in stream_set or rhs not in stream_set:
            raise ValueError(f"Streams '{lhs}:{rhs}' in --cf-swap must exist within {stream_list}.")
        if lhs == rhs:
            continue
        swap_pairs.append((lhs, rhs))
    shuffle_streams = tuple(_normalise_stream_list(args.cf_shuffle, stream_list))
    freeze_streams = tuple(_normalise_stream_list(args.cf_freeze, stream_list))
    stale_map: Dict[str, int] = {}
    for spec in args.cf_stale or []:
        if spec is None:
            continue
        if ":" in spec:
            stream_part, delta_part = spec.split(":", 1)
        elif "=" in spec:
            stream_part, delta_part = spec.split("=", 1)
        else:
            raise ValueError("--cf-stale entries must use ROLE:DELTA format.")
        stream_norm = stream_part.strip().lower()
        if stream_norm == "all":
            raise ValueError(
                "Use --cf-stale-default for global stale lag adjustments instead of 'all'."
            )
        if stream_norm not in stream_set:
            raise ValueError(
                f"Stream '{stream_norm}' in --cf-stale must exist within {stream_list}."
            )
        try:
            delta_val = int(delta_part.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid stale lag delta '{delta_part}' for stream '{stream_norm}'."
            ) from exc
        if delta_val < 0:
            raise ValueError("--cf-stale deltas must be non-negative integers.")
        stale_map[stream_norm] = delta_val
    default_stale = max(0, int(args.cf_stale_default or 0))
    ablate_streams = tuple(_normalise_stream_list(args.cf_ablate, stream_list))
    tag = args.cf_tag.strip() if args.cf_tag and args.cf_tag.strip() else None
    return CounterfactualConfig(
        swap_pairs=tuple(swap_pairs),
        shuffle_streams=shuffle_streams,
        freeze_streams=freeze_streams,
        ablate_streams=ablate_streams,
        stale_overrides=stale_map,
        default_stale_extra=default_stale,
        tag=tag,
    )


def _summarize_counterfactuals(config: Optional[CounterfactualConfig]) -> List[str]:
    if config is None or not getattr(config, "enabled", False):
        return []
    tags: List[str] = []
    if config.ablate_streams:
        tags.append("ablate:" + ",".join(sorted(config.ablate_streams)))
    if config.freeze_streams:
        tags.append("freeze:" + ",".join(sorted(config.freeze_streams)))
    if config.shuffle_streams:
        tags.append("shuffle:" + ",".join(sorted(config.shuffle_streams)))
    for lhs, rhs in config.swap_pairs:
        tags.append(f"swap:{lhs}->{rhs}")
    for stream, delta in sorted(config.stale_overrides.items()):
        tags.append(f"stale:{stream}+{delta}")
    if config.default_stale_extra > 0:
        tags.append(f"stale_default:+{config.default_stale_extra}")
    if config.tag:
        tags.append(f"tag:{config.tag}")
    return tags


def override_decode_config(config: DecodeConfig, args: argparse.Namespace) -> None:
    if args.max_new_tokens is not None:
        config.max_new_tokens = int(args.max_new_tokens)
    if args.temperature is not None:
        config.temperature = float(args.temperature)
    if args.top_k is not None:
        config.top_k = int(args.top_k)
    if args.top_p is not None:
        config.top_p = float(args.top_p)
    if args.no_sample:
        config.do_sample = False


def format_event(event: StepOutcome) -> str:
    token_repr = event.token_text
    if "\n" in token_repr:
        token_repr = token_repr.replace("\n", "\\n")
    margin_part = f" margin={event.top2_margin:.3f}" if event.top2_margin is not None else ""
    cf_part = ""
    if event.counterfactuals:
        cf_part = f" cf={','.join(event.counterfactuals)}"
    return (
        f"[stream={event.stream} stride={event.stride_index}] token={event.token_id} "
        f"text='{token_repr}' agree={event.agreement:.3f}{margin_part}{cf_part} "
        f"notes={'Y' if event.notes_emitted else 'N'} rollback={'Y' if event.rollback_performed else 'N'}"
    )


def _apply_sequential_baseline_model_overrides(
    model_cfg: ModelConfig, training_cfg: TrainingConfig
) -> None:
    """Clamp model/training configuration to a single sequential stream."""

    seq_stream = "seq"
    collator_cfg = model_cfg.collator
    model_cfg.collator = replace(
        collator_cfg,
        stream_to_id={seq_stream: 0},
    )
    if model_cfg.stream_adapters is not None:
        model_cfg.stream_adapters = replace(model_cfg.stream_adapters, streams=(seq_stream,))
    if getattr(training_cfg, "dataset_teacher", None) is not None:
        training_cfg.dataset_teacher = replace(training_cfg.dataset_teacher, streams=(seq_stream,))
    if getattr(model_cfg, "stream_classifier_head", None) is not None:
        model_cfg.stream_classifier_head = replace(model_cfg.stream_classifier_head, num_streams=1)


def _apply_sequential_baseline_runtime_overrides(inference_cfg: InferenceConfig) -> None:
    """Disable cross-lane features when running the sequential baseline."""

    # Disable SNC influence and treat cadence as never emitting.
    inference_cfg.gate_g = 0.0
    inference_cfg.logit_blend_alpha = 0.0
    disabled_cadence = max(1_000_000, inference_cfg.decode.max_new_tokens * 10)
    inference_cfg.emission_cadence_M_by_stream = {
        stream: disabled_cadence for stream in inference_cfg.streams
    }
    inference_cfg.gate_annealing = replace(inference_cfg.gate_annealing, enabled=False)
    inference_cfg.cadence_policy = replace(
        inference_cfg.cadence_policy,
        mode="deterministic",
        max_interval=0,
    )


def main() -> None:
    args = parse_args()
    if args.hf_verbosity is not None:
        _os.environ["TRANSFORMERS_VERBOSITY"] = args.hf_verbosity
    logger = configure_logging(
        name="parallel decoder transformer.cli.infer",
        extra_loggers=[
            "parallel decoder transformer.gpt_oss.trunk",
            "parallel decoder transformer.inference",
            "parallel decoder transformer.training",
            "transformers",
        ],
    )
    # Device selection is automatic; PDT_DEVICE env can override if set.
    prompt = resolve_prompt(args)
    raw_cfg = load_config(args.config)
    model_cfg = _coerce_model_config(raw_cfg.get("model", {}))
    training_cfg = _coerce_training_config(raw_cfg.get("training", {}))

    baseline_mode = args.baseline
    token_baseline_modes = {"medusa", "lookahead", "eagle"}
    is_sequential_baseline = baseline_mode == "sequential"
    is_token_baseline = baseline_mode in token_baseline_modes
    if (is_sequential_baseline or is_token_baseline) and args.stream:
        raise ValueError(
            "--baseline %s cannot be combined with --stream overrides." % baseline_mode
        )
    if is_sequential_baseline:
        _apply_sequential_baseline_model_overrides(model_cfg, training_cfg)
    elif args.stream:
        stream_list = [
            stream.strip().lower() for stream in args.stream if stream and stream.strip()
        ]
        if not stream_list:
            raise ValueError(
                "At least one --stream value must be provided when overriding streams."
            )
        collator_cfg = model_cfg.collator
        model_cfg.collator = replace(
            collator_cfg,
            stream_to_id={name: index for index, name in enumerate(stream_list)},
        )
        adapter_cfg = model_cfg.stream_adapters
        model_cfg.stream_adapters = replace(adapter_cfg, streams=tuple(stream_list))
        if getattr(training_cfg, "dataset_teacher", None) is not None:
            training_cfg.dataset_teacher = replace(
                training_cfg.dataset_teacher, streams=tuple(stream_list)
            )

    _align_model_hidden_size(model_cfg, _resolve_trunk_hidden_size(model_cfg.trunk), logger)

    counterfactual_cfg = _build_counterfactual_config(
        args, tuple(model_cfg.collator.stream_to_id.keys())
    )

    # No device override via CLI; trunk adapter resolves device and dtype (incl. MPS-safe downgrade).

    base_model_ref = model_cfg.trunk.base_model
    tokenizer_override: Optional[Path] = None
    # Strict tokenizer requirement: require sibling tokenizer dir
    try:
        base_model_path = Path(base_model_ref)
    except (TypeError, ValueError):  # pragma: no cover - defensive path
        base_model_path = None  # type: ignore[assignment]
    if base_model_path is None or not base_model_path.exists():
        raise FileNotFoundError(
            "Model base path must be a local directory, and a sibling 'tokenizer/' directory is required. "
            f"Got base_model={base_model_ref!r}. Place weights under e.g. 'gpt-oss-20b/original' and tokenizer at 'gpt-oss-20b/tokenizer'."
        )
    candidate = base_model_path.parent / "tokenizer"
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Tokenizer directory is required and was not found at: {candidate}. "
            "Create a sibling 'tokenizer/' next to your 'original/' directory."
        )
    tokenizer_override = candidate

    tokenizer_cfg = TokenizerConfig(
        pretrained_name=str(base_model_ref),
        custom_path=tokenizer_override,
    )
    tokenizer, tokenizer_manifest = resolve_tokenizer(tokenizer_cfg)

    seed_override = args.seed if args.seed is not None else getattr(training_cfg, "seed", None)
    seed_everything(seed_override)

    alpha_override: Optional[float] = None
    if args.alpha is not None:
        if not (0.0 <= args.alpha <= 1.0):
            raise ValueError("--alpha must lie within [0,1].")
        alpha_override = float(args.alpha)

    read_lag_override: Optional[int] = None
    if args.read_lag_delta is not None:
        if args.read_lag_delta < 0:
            raise ValueError("--read-lag-delta must be non-negative.")
        read_lag_override = int(args.read_lag_delta)

    seed_text_map = _parse_seed_texts(args.seed_text, args.seed_text_file)
    seed_notes_map = _parse_seed_notes(args.seed_notes_file)
    if is_sequential_baseline or is_token_baseline:
        if seed_text_map:
            logger.info("%s_baseline | seed text ignored", baseline_mode)
            seed_text_map = {}
        if seed_notes_map:
            logger.info("%s_baseline | seed notes ignored", baseline_mode)
            seed_notes_map = {}
    notes_reference_map = _parse_notes_texts(args.notes_text_file)

    logger.info(
        "infer_start | config=%s | prompt_chars=%d | base_model=%s",
        str(args.config),
        len(prompt),
        model_cfg.trunk.base_model,
    )

    model = ParallelDecoderTransformer(model_cfg)
    model.trunk_adapter.load_model()
    # Align lightweight heads/adapters to the trunk device/dtype
    model.to_trunk_device_and_dtype()
    plan_params = resolve_plan_hash_params(model.config)
    logger.info(
        "trunk_loaded | base_model=%s | device=%s",
        model_cfg.trunk.base_model,
        (
            model.trunk_adapter.model.device
            if hasattr(model.trunk_adapter.model, "device")
            else "unknown"
        ),
    )
    if args.checkpoint is not None:
        import torch

        state = None
        if args.checkpoint.is_dir():
            for candidate in (
                args.checkpoint / "adapters.pt",
                args.checkpoint / "adapter_state.pt",
                args.checkpoint / "state_dict.pt",
            ):
                if candidate.exists():
                    state = torch.load(candidate, map_location="cpu")
                    break
        else:
            state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict):
            model.load_adapters(state, strict=False)
            # Ensure any newly loaded tensors are on the trunk device/dtype
            model.to_trunk_device_and_dtype()

    inference_cfg = build_inference_config(
        training_cfg,
        stream_to_id=dict(model_cfg.collator.stream_to_id),
        rng_seed=seed_override,
        logit_blend_alpha=alpha_override,
        read_lag_delta=read_lag_override,
        counterfactuals=counterfactual_cfg,
        memory_report=args.memory_report,
    )
    if inference_cfg.decode.max_new_tokens < 512 and args.max_new_tokens is None:
        inference_cfg.decode.max_new_tokens = 512
    if seed_override is not None:
        inference_cfg.rng_seed = seed_override
        inference_cfg.decode.seed = seed_override
    override_decode_config(inference_cfg.decode, args)
    if notes_reference_map:
        valid_streams = set(inference_cfg.streams)
        for key in list(notes_reference_map.keys()):
            if key not in valid_streams:
                raise ValueError(
                    f"Stream '{key}' in --notes-text-file not present in inference streams {inference_cfg.streams}."
                )

    cadence_policy = inference_cfg.cadence_policy
    if args.cadence_mode is not None:
        cadence_policy.mode = args.cadence_mode
    if args.cadence_max_interval is not None:
        cadence_policy.max_interval = int(args.cadence_max_interval)
    if args.cadence_min_prob is not None:
        cadence_policy.min_probability = float(args.cadence_min_prob)
    if args.cadence_m_min is not None:
        cadence_policy.multiplier_min = float(args.cadence_m_min)
    if args.cadence_m_max is not None:
        cadence_policy.multiplier_max = float(args.cadence_m_max)
    if args.cadence_agreement_low is not None:
        cadence_policy.agreement_low = float(args.cadence_agreement_low)
    if args.cadence_agreement_high is not None:
        cadence_policy.agreement_high = float(args.cadence_agreement_high)
    if args.cadence_age_boost is not None:
        cadence_policy.age_boost = float(args.cadence_age_boost)
    inference_cfg._validate_cadence_policy()

    cadence_override_map = _parse_cadence_overrides(args.cadence_M, inference_cfg.streams)
    if cadence_override_map:
        for stream, value in cadence_override_map.items():
            inference_cfg.emission_cadence_M_by_stream[stream] = value
            logger.info("cadence_override | stream=%s | M=%d", stream, value)
    else:
        cadence_override_map = {}

    if args.gate_g is not None:
        try:
            gate_val = float(args.gate_g)
        except Exception as err:  # pragma: no cover - CLI guard
            raise ValueError("--gate-g must be a float in [0,1].") from err
        if not (0.0 <= gate_val <= 1.0):
            raise ValueError("--gate-g must lie within [0,1].")
        inference_cfg.gate_g = gate_val

    if is_sequential_baseline:
        _apply_sequential_baseline_runtime_overrides(inference_cfg)

    if is_token_baseline:
        baseline_cfg = build_token_baseline_config(baseline_mode or "medusa")
        max_tokens = args.max_new_tokens or inference_cfg.decode.max_new_tokens

        def handle_baseline_event(event: Dict[str, Any], step_index: int) -> None:
            token_repr = event["token_text"].replace("\n", "\\n")
            if args.verbose:
                logger.info(
                    "[baseline=%s step=%d stride=%d] token=%d text='%s'",
                    baseline_cfg.name,
                    step_index,
                    event["stride_index"],
                    event["token_id"],
                    token_repr,
                )
            if args.stream_jsonl:
                payload = dict(event)
                payload["step"] = step_index
                print(json.dumps(payload, ensure_ascii=False), flush=True)

        manifest, baseline_events = run_token_baseline(
            model.trunk_adapter.model,
            tokenizer,
            prompt,
            inference_cfg.decode,
            baseline_cfg,
            max_new_tokens=max_tokens,
            event_callback=handle_baseline_event,
        )
        git_meta = get_git_metadata()
        manifest["git_sha"] = git_meta.sha
        manifest["git_dirty"] = git_meta.dirty
        manifest["prompt"] = prompt
        manifest["tokenizer"] = tokenizer_manifest.to_dict()
        manifest["events"] = baseline_events
        manifest.update(plan_params.as_dict())
        if cadence_override_map:
            manifest["cadence_overrides"] = dict(cadence_override_map)
        cf_tags = _summarize_counterfactuals(counterfactual_cfg)
        if cf_tags:
            manifest["counterfactual_tags"] = cf_tags
        manifest_path = args.output
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        logger.info(
            "infer_complete | manifest=%s | baseline=%s", str(manifest_path), baseline_cfg.name
        )
        return

    replay_writer: Optional[ReplayArtifactWriter] = None
    if args.replay_artifact_dir is not None:
        chunk_size = max(1, int(args.replay_chunk_size))
        model_cfg_runtime = getattr(model, "config", None)
        if model_cfg_runtime is None:
            raise RuntimeError("Model configuration missing; cannot emit replay artifact.")
        notes_dim = getattr(model_cfg_runtime, "notes_dim", None)
        hidden_size = getattr(model_cfg_runtime, "hidden_size", None)
        plan_vocab_size = getattr(model_cfg_runtime, "plan_vocab_size", None)
        lm_vocab_size = getattr(model_cfg_runtime, "vocab_size", None)
        if any(value is None for value in (notes_dim, hidden_size, plan_vocab_size, lm_vocab_size)):
            raise RuntimeError(
                "Model configuration lacks dimensions required to write replay artifacts."
            )
        replay_writer = ReplayArtifactWriter(
            args.replay_artifact_dir,
            prompt=prompt,
            tokenizer_config=tokenizer_cfg,
            tokenizer_manifest=tokenizer_manifest,
            inference_config=inference_cfg,
            notes_dim=int(notes_dim),
            hidden_size=int(hidden_size),
            plan_vocab_size=int(plan_vocab_size),
            lm_vocab_size=int(lm_vocab_size),
            chunk_size=chunk_size,
            git_metadata=get_git_metadata(),
            plan_hash_buckets=plan_params.hash_buckets,
            plan_hash_salt=plan_params.salt,
        )

    orchestrator = MultiStreamOrchestrator(
        model,
        tokenizer,
        inference_cfg,
        log_margins=args.log_margins,
        sync_profile=args.sync_profile,
        replay_writer=replay_writer,
    )
    if is_sequential_baseline:
        orchestrator.agreement_gate.threshold = -1.0  # Skip rollback triggers in sequential mode.
        orchestrator._gate_values = {stream: 0.0 for stream in inference_cfg.streams}
    stream_prefix_map = None
    if args.stream_prefix_file is not None:
        if not args.stream_prefix_file.exists():
            raise FileNotFoundError(f"stream prefix file not found: {args.stream_prefix_file}")
        try:
            with args.stream_prefix_file.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError(
                    "--stream-prefix-file must contain a JSON object mapping stream -> prefix string."
                )
            stream_prefix_map = {}
            for key, value in raw.items():
                if not isinstance(value, str):
                    raise ValueError("stream prefix values must be strings.")
                stream_key = str(key).lower()
                if stream_key not in inference_cfg.streams:
                    raise ValueError(
                        f"stream '{stream_key}' in stream-prefix mapping not present in inference streams {inference_cfg.streams}."
                    )
                stream_prefix_map[stream_key] = value
        except Exception:
            raise

    plan_text_map = _load_plan_text_map(args.plan_text_file, inference_cfg.streams)
    plan_contract = _load_plan_contract(args.plan_contract)

    orchestrator.start(
        prompt,
        prefix_by_stream=stream_prefix_map,
        seed_text_by_stream=seed_text_map or None,
        seed_notes_by_stream=seed_notes_map or None,
        plan_text_by_stream=plan_text_map,
        plan_contract=plan_contract,
    )

    events: list[StepOutcome] = []
    while True:
        outcome = orchestrator.step()
        if outcome is None:
            break
        events.append(outcome)
        if args.verbose:
            logger.info(format_event(outcome))
        if args.stream_jsonl:
            step_payload = {
                "step": len(events),
                "stream": outcome.stream,
                "token_id": outcome.token_id,
                "token_text": outcome.token_text,
                "stride_index": outcome.stride_index,
                "stride_completed": outcome.stride_completed,
                "stream_completed": outcome.stream_completed,
                "agreement": outcome.agreement,
                "notes_emitted": outcome.notes_emitted,
                "rollback_performed": outcome.rollback_performed,
                "cadence_mode": outcome.cadence_mode,
                "cadence_probability": outcome.cadence_probability,
                "cadence_multiplier": outcome.cadence_multiplier,
                "cadence_forced": outcome.cadence_forced,
                "coverage_logits": outcome.coverage_logits,
                "top2_margin": outcome.top2_margin,
                "counterfactuals": outcome.counterfactuals,
            }
            print(json.dumps(step_payload, ensure_ascii=False), flush=True)

    manifest = orchestrator.finalize()
    git_meta = get_git_metadata()
    manifest["git_sha"] = git_meta.sha
    manifest["git_dirty"] = git_meta.dirty
    manifest["prompt"] = prompt
    manifest["tokenizer"] = tokenizer_manifest.to_dict()
    manifest["events"] = [
        {
            "stream": event.stream,
            "token_id": event.token_id,
            "token_text": event.token_text,
            "stride_index": event.stride_index,
            "stride_completed": event.stride_completed,
            "stream_completed": event.stream_completed,
            "agreement": event.agreement,
            "notes_emitted": event.notes_emitted,
            "rollback_performed": event.rollback_performed,
            "cadence_mode": event.cadence_mode,
            "cadence_probability": event.cadence_probability,
            "cadence_multiplier": event.cadence_multiplier,
            "cadence_forced": event.cadence_forced,
            "top2_margin": event.top2_margin,
            "coverage_logits": event.coverage_logits,
            "counterfactuals": event.counterfactuals,
        }
        for event in events
    ]
    if replay_writer is not None:
        artifact_dir = replay_writer.finalize()
        manifest["replay_artifact"] = str(artifact_dir)
    if cadence_override_map:
        manifest["cadence_overrides"] = dict(cadence_override_map)
    cf_tags = _summarize_counterfactuals(counterfactual_cfg)
    if cf_tags:
        manifest["counterfactual_tags"] = cf_tags
    if notes_reference_map:
        reference_payload = {
            stream: list(notes_reference_map.get(stream, [])) for stream in inference_cfg.streams
        }
        manifest["reference_notes"] = reference_payload
        for stream, stream_payload in manifest.get("streams", {}).items():
            stream_key = stream.lower()
            stream_payload["reference_notes"] = list(reference_payload.get(stream_key, []))
    if is_sequential_baseline:
        manifest["baseline"] = "sequential"
    manifest_path = args.output
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "infer_complete | manifest=%s | streams=%s",
        str(manifest_path),
        ", ".join(inference_cfg.streams),
    )


if __name__ == "__main__":
    main()
