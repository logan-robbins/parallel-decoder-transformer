"""CPU-only logit replay harness feeding the orchestration stack."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from types import SimpleNamespace

import torch
from torch import nn

from .config import (
    CadencePolicyConfig,
    DecodeConfig,
    GateAnnealingConfig,
    InferenceConfig,
)
from ..data.tokenizer import TokenizerConfig, TokenizerManifest
from ..utils.git import GitMetadata, get_git_metadata

REPLAY_ARTIFACT_VERSION = "1.0"
DEFAULT_CHUNK_SIZE = 4096


def _as_path(root: Path, file_name: Optional[str]) -> Optional[Path]:
    if not file_name:
        return None
    candidate = root / str(file_name)
    if not candidate.exists():
        raise FileNotFoundError(f"Replay artifact file not found: {candidate}")
    return candidate


def _load_tensor_dict(file_path: Optional[Path]) -> Dict[str, torch.Tensor]:
    if file_path is None:
        return {}
    payload = torch.load(file_path, map_location="cpu")
    if isinstance(payload, Mapping):
        return {str(key).lower(): torch.as_tensor(value) for key, value in payload.items()}
    raise ValueError(f"Replay tensor file {file_path} must contain a mapping.")


def _load_chunks(root: Path, files: Sequence[str], *, dim: int = 0) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for name in files:
        path = _as_path(root, name)
        if path is None:
            continue
        tensor = torch.load(path, map_location="cpu")
        tensors.append(torch.as_tensor(tensor))
    if not tensors:
        return torch.zeros(0)
    return torch.cat(tensors, dim=dim)


def _normalise_chunk_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw if str(item)]
    return []


def _serialise_tokenizer_config(config: TokenizerConfig) -> Dict[str, Any]:
    payload = asdict(config)
    custom_path = payload.get("custom_path")
    if custom_path:
        payload["custom_path"] = str(custom_path)
    return payload


class ReplayArtifactWriter:
    """Records inference-time tensors for later CPU-only replay."""

    def __init__(
        self,
        root: Path,
        *,
        prompt: str,
        tokenizer_config: TokenizerConfig,
        tokenizer_manifest: TokenizerManifest,
        inference_config: InferenceConfig,
        notes_dim: int,
        hidden_size: int,
        plan_vocab_size: int,
        lm_vocab_size: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        git_metadata: Optional[GitMetadata] = None,
        logit_dtype: torch.dtype = torch.bfloat16,
        plan_hash_buckets: Optional[int] = None,
        plan_hash_salt: str = "",
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.streams = tuple(stream.lower() for stream in inference_config.streams)
        self.chunk_size = max(1, int(chunk_size))
        self._notes_dtype = torch.bfloat16
        self._logit_dtype = logit_dtype
        git_meta = git_metadata or get_git_metadata()

        hash_buckets = int(plan_hash_buckets or plan_vocab_size)
        hash_salt = str(plan_hash_salt or "")
        self.manifest: Dict[str, Any] = {
            "version": REPLAY_ARTIFACT_VERSION,
            "prompt": prompt,
            "streams": list(self.streams),
            "notes_dim": int(notes_dim),
            "hidden_size": int(hidden_size),
            "plan_vocab_size": int(plan_vocab_size),
            "plan_hash_buckets": hash_buckets,
            "plan_hash_salt": hash_salt,
            "lm_vocab_size": int(lm_vocab_size),
            "git_sha": git_meta.sha,
            "git_dirty": git_meta.dirty,
            "chunk_size": self.chunk_size,
            "tokenizer_config": _serialise_tokenizer_config(tokenizer_config),
            "tokenizer_manifest": tokenizer_manifest.to_dict(),
            "config": self._config_snapshot(inference_config),
            "token_ids": {stream: [] for stream in self.streams},
        }

        self._agreement: Dict[str, List[float]] = {stream: [] for stream in self.streams}
        self._coverage: Dict[str, List[List[float]]] = {stream: [] for stream in self.streams}
        self._notes: Dict[str, List[torch.Tensor]] = {stream: [] for stream in self.streams}
        self._bootstrap: Dict[str, torch.Tensor] = {}
        self._attended_probe: Dict[str, List[float]] = {stream: [] for stream in self.streams}
        self._plan_token_ids: Optional[torch.Tensor] = None
        self._plan_mask: Optional[torch.Tensor] = None
        self._plan_logits: Optional[torch.Tensor] = None
        self._plan_logits_shape: Optional[List[int]] = None
        self._plan_source: str = "model"
        self._plan_catalog: Optional[List[Dict[str, Any]]] = None

    def _config_snapshot(self, config: InferenceConfig) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "stride_B": config.stride_B,
            "commit_L": config.commit_L,
            "read_lag_delta": config.read_lag_delta,
            "max_snapshots_K": config.max_snapshots_K,
            "topology": config.topology,
            "gate_g": config.gate_g,
            "tau": config.agreement_threshold_tau,
            "alpha": config.logit_blend_alpha,
            "coverage_threshold": config.coverage_threshold,
            "coverage_partial_band": config.coverage_partial_band,
            "rng_seed": config.rng_seed,
            "M_by_stream": dict(config.emission_cadence_M_by_stream),
            "decode": asdict(config.decode),
            "gate_annealing": asdict(config.gate_annealing),
            "cadence_policy": asdict(config.cadence_policy),
            "counterfactuals": config.counterfactuals.as_dict(),
        }
        if getattr(config, "memory_report", False):
            payload["memory_report"] = True
        safeguards = getattr(config, "safeguards", None)
        if safeguards is not None:
            payload["safeguards"] = asdict(safeguards)
        return payload

    def record_plan(
        self,
        *,
        plan_token_ids: torch.Tensor,
        plan_mask: torch.Tensor,
        plan_logits: Optional[torch.Tensor],
        source: str,
        catalog: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self._plan_token_ids = plan_token_ids.detach().to("cpu", torch.long).clone()
        self._plan_mask = plan_mask.detach().to("cpu", torch.long).clone()
        if plan_logits is not None:
            self._plan_logits_shape = list(plan_logits.shape)
            squeezed = plan_logits.detach().to("cpu", self._logit_dtype)
            if squeezed.dim() == 3 and squeezed.size(0) == 1:
                squeezed = squeezed.squeeze(0)
            self._plan_logits = squeezed.contiguous()
        self._plan_source = source
        if catalog:
            self._plan_catalog = [dict(entry) for entry in catalog]

    def record_bootstrap(self, stream: str, vector: torch.Tensor) -> None:
        stream_norm = self._normalise_stream(stream)
        flattened = vector.detach().to("cpu", self._notes_dtype).reshape(-1)
        self._bootstrap[stream_norm] = flattened

    def record_step(
        self,
        stream: str,
        *,
        token_id: int,
        agreement: float,
        coverage_logits: Optional[Sequence[float]],
        note_emitted: bool,
        note_vector: Optional[torch.Tensor],
        delta_norm: Optional[float] = None,
    ) -> None:
        stream_norm = self._normalise_stream(stream)
        try:
            self.manifest["token_ids"][stream_norm].append(int(token_id))
        except KeyError as exc:
            raise ValueError(
                f"Unknown stream {stream!r} provided to ReplayArtifactWriter."
            ) from exc
        self._agreement[stream_norm].append(float(max(0.0, min(1.0, agreement))))
        if coverage_logits is not None:
            self._coverage[stream_norm].append([float(value) for value in coverage_logits])
        if note_emitted and note_vector is not None:
            flattened = note_vector.detach().to("cpu", self._notes_dtype).reshape(-1)
            self._notes[stream_norm].append(flattened)
        if delta_norm is not None:
            self._attended_probe[stream_norm].append(float(delta_norm))

    def _normalise_stream(self, stream: str) -> str:
        lowered = stream.strip().lower()
        if lowered not in self.manifest["token_ids"]:
            raise ValueError(f"Stream {stream!r} is not part of the inference config.")
        return lowered

    def finalize(self) -> Path:
        files_section: Dict[str, Any] = {}
        if self._plan_logits is not None:
            files_section["planner_logits"] = self._write_tensor_chunks(
                self._plan_logits, base_name="planner_logits", dim=0
            )
        agreement_files = self._write_stream_series(
            self._agreement, "agreement", dtype=torch.float32
        )
        if agreement_files:
            files_section["agreement"] = agreement_files
        coverage_files = self._write_stream_matrix(self._coverage, "coverage", dtype=torch.float32)
        if coverage_files:
            files_section["coverage"] = coverage_files
        notes_files = self._write_stream_tensor_map(self._notes, "notes")
        if notes_files:
            files_section["notes"] = notes_files
        bootstrap_files = self._write_bootstrap()
        if bootstrap_files:
            files_section["bootstrap"] = bootstrap_files
        probe_files = self._write_stream_series(
            self._attended_probe, "attended_probe", dtype=torch.float32
        )
        if probe_files:
            files_section["attended_probe"] = probe_files

        if self._plan_token_ids is not None:
            plan_section: Dict[str, Any] = {
                "source": self._plan_source,
                "token_ids": self._plan_token_ids.tolist(),
                "mask": self._plan_mask.tolist() if self._plan_mask is not None else None,
            }
            if self._plan_logits_shape is not None:
                plan_section["logits_shape"] = list(self._plan_logits_shape)
            if "planner_logits" in files_section:
                plan_section["logits_chunks"] = list(files_section["planner_logits"])
                plan_section["logits_dtype"] = str(self._logit_dtype).split(".")[-1]
            if self._plan_catalog:
                catalog_path = self.root / "plan_catalog.json"
                catalog_path.write_text(
                    json.dumps(self._plan_catalog, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                plan_section["catalog_file"] = catalog_path.name
            self.manifest["plan"] = plan_section

        if files_section:
            self.manifest["files"] = files_section
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self.root

    def _write_tensor_chunks(
        self, tensor: torch.Tensor, *, base_name: str, dim: int = 0
    ) -> List[str]:
        if tensor.numel() == 0:
            return []
        files: List[str] = []
        length = tensor.size(dim) if tensor.dim() > 0 else 1
        start = 0
        chunk_index = 0
        while start < length:
            end = min(length, start + self.chunk_size)
            width = end - start
            chunk = tensor.narrow(dim, start, width).clone()
            suffix = "" if chunk_index == 0 else f"-{chunk_index:04d}"
            file_name = f"{base_name}{suffix}.pt"
            torch.save(chunk, self.root / file_name)
            files.append(file_name)
            chunk_index += 1
            start = end
        return files

    def _write_stream_series(
        self,
        mapping: Mapping[str, List[float]],
        base_name: str,
        *,
        dtype: torch.dtype,
    ) -> Dict[str, List[str]]:
        files: Dict[str, List[str]] = {}
        for stream, values in mapping.items():
            if not values:
                continue
            tensor = torch.tensor(values, dtype=dtype)
            files[stream] = self._write_tensor_chunks(
                tensor, base_name=f"{base_name}-{stream}", dim=0
            )
        return files

    def _write_stream_matrix(
        self,
        mapping: Mapping[str, List[List[float]]],
        base_name: str,
        *,
        dtype: torch.dtype,
    ) -> Dict[str, List[str]]:
        files: Dict[str, List[str]] = {}
        for stream, rows in mapping.items():
            if not rows:
                continue
            tensor = torch.tensor(rows, dtype=dtype)
            files[stream] = self._write_tensor_chunks(
                tensor, base_name=f"{base_name}-{stream}", dim=0
            )
        return files

    def _write_stream_tensor_map(
        self,
        mapping: Mapping[str, List[torch.Tensor]],
        base_name: str,
    ) -> Dict[str, List[str]]:
        files: Dict[str, List[str]] = {}
        for stream, vectors in mapping.items():
            if not vectors:
                continue
            tensor = torch.stack([vec.to(dtype=self._notes_dtype) for vec in vectors], dim=0)
            files[stream] = self._write_tensor_chunks(
                tensor, base_name=f"{base_name}-{stream}", dim=0
            )
        return files

    def _write_bootstrap(self) -> Dict[str, str]:
        files: Dict[str, str] = {}
        for stream, vector in self._bootstrap.items():
            file_name = f"bootstrap-{stream}.pt"
            torch.save(vector.to(dtype=self._notes_dtype), self.root / file_name)
            files[stream] = file_name
        return files


def _load_stream_tensor_map(
    root: Path,
    payload: Any,
    *,
    dim: int = 0,
    required: bool = False,
) -> Dict[str, torch.Tensor]:
    if not payload:
        if required:
            raise ValueError("Replay artifact missing required tensor payload.")
        return {}
    if isinstance(payload, str):
        return _load_tensor_dict(_as_path(root, payload))
    if isinstance(payload, Mapping):
        tensors: Dict[str, torch.Tensor] = {}
        for stream, files in payload.items():
            chunks = _normalise_chunk_list(files)
            if not chunks:
                continue
            tensors[str(stream).lower()] = _load_chunks(root, chunks, dim=dim)
        if tensors or not required:
            return tensors
    raise ValueError("Replay artifact tensor mapping is malformed.")


def _load_tokenizer_cfg(manifest: Mapping[str, Any]) -> TokenizerConfig:
    payload = manifest.get("tokenizer_config") or manifest.get("tokenizer") or {}
    payload = dict(payload)
    custom_path = payload.get("custom_path")
    if custom_path:
        payload["custom_path"] = Path(custom_path)
    return TokenizerConfig(**payload)


def _load_token_ids_map(data: Mapping[str, Any], streams: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    token_ids_payload = data.get("token_ids")
    if isinstance(token_ids_payload, Mapping):
        for stream, sequence in token_ids_payload.items():
            mapping[str(stream).lower()] = [int(token) for token in sequence]
    if mapping:
        return mapping
    streams_payload = data.get("streams", {})
    if isinstance(streams_payload, Mapping):
        for stream in streams:
            payload = streams_payload.get(stream) or streams_payload.get(stream.lower())
            if isinstance(payload, Mapping) and payload.get("token_ids"):
                mapping[str(stream).lower()] = [int(token) for token in payload["token_ids"]]
    if mapping:
        return mapping
    raise ValueError("Replay manifest missing token_ids mapping.")


@dataclass(slots=True)
class LogitReplayArtifact:
    """Loads manifests/tensors describing a recorded inference run."""

    root: Path
    manifest: Mapping[str, Any]
    streams: Tuple[str, ...]
    prompt: str
    notes_dim: int
    hidden_size: int
    plan_vocab_size: int
    plan_hash_buckets: int
    plan_hash_salt: str
    lm_vocab_size: int
    token_ids: Dict[str, List[int]]
    planner_logits: torch.Tensor
    planner_token_ids: torch.Tensor
    planner_mask: torch.Tensor
    agreement: Dict[str, torch.Tensor]
    note_snapshots: Dict[str, torch.Tensor]
    bootstrap_notes: Dict[str, torch.Tensor]
    coverage_logits: Dict[str, torch.Tensor]
    tokenizer_cfg: TokenizerConfig

    @classmethod
    def load(cls, root: Path) -> "LogitReplayArtifact":
        root = root.expanduser().resolve()
        manifest_path = root / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Replay manifest must include a non-empty prompt.")
        streams_payload = data.get("streams")
        if not streams_payload:
            raise ValueError("Replay manifest must include the stream list.")
        streams = tuple(str(stream).lower() for stream in streams_payload)
        token_ids_map = _load_token_ids_map(data, streams)
        notes_dim = int(data.get("notes_dim", 0))
        hidden_size = int(data.get("hidden_size", 0))
        plan_vocab_size = int(data.get("plan_vocab_size", 65536))
        plan_hash_buckets = int(data.get("plan_hash_buckets", plan_vocab_size))
        plan_hash_salt = str(data.get("plan_hash_salt", ""))
        lm_vocab_size = int(data.get("lm_vocab_size", plan_vocab_size))
        if notes_dim <= 0 or hidden_size <= 0:
            raise ValueError(
                "Replay manifest must specify positive notes_dim and hidden_size values."
            )
        files_section = data.get("files", {})
        planner_payload = data.get("plan") or data.get("planner") or {}
        plan_ids_value = planner_payload.get("token_ids")
        if plan_ids_value is None:
            raise ValueError("Replay manifest missing plan token IDs.")
        planner_token_ids = torch.as_tensor(plan_ids_value, dtype=torch.long)
        if planner_token_ids.dim() == 1:
            planner_token_ids = planner_token_ids.unsqueeze(0)
        mask_value = planner_payload.get("mask")
        if mask_value is None:
            mask_value = torch.ones_like(planner_token_ids)
        planner_mask = torch.as_tensor(mask_value, dtype=torch.long)
        logits_chunks = planner_payload.get("logits_chunks")
        if logits_chunks:
            chunk_list = _normalise_chunk_list(logits_chunks)
            planner_logits = _load_chunks(root, chunk_list, dim=0)
            logits_shape = planner_payload.get("logits_shape")
            if logits_shape:
                shape = tuple(int(dim) for dim in logits_shape)
                planner_logits = planner_logits.view(*shape)
            elif planner_logits.dim() == 2:
                planner_logits = planner_logits.unsqueeze(0)
        else:
            planner_logits_path = _as_path(root, planner_payload.get("logits_file"))
            if planner_logits_path is None:
                raise ValueError("Replay manifest must provide planner logits.")
            planner_logits = torch.load(planner_logits_path, map_location="cpu")
        agreement = _load_stream_tensor_map(
            root, files_section.get("agreement"), dim=0, required=True
        )
        note_snapshots = _load_stream_tensor_map(
            root, files_section.get("notes"), dim=0, required=False
        )
        bootstrap_notes = _load_stream_tensor_map(
            root, files_section.get("bootstrap"), dim=0, required=False
        )
        coverage_logits = _load_stream_tensor_map(
            root, files_section.get("coverage"), dim=0, required=False
        )
        tokenizer_cfg = _load_tokenizer_cfg(data)
        return cls(
            root=root,
            manifest=data,
            streams=streams,
            prompt=prompt,
            notes_dim=notes_dim,
            hidden_size=hidden_size,
            plan_vocab_size=plan_vocab_size,
            plan_hash_buckets=plan_hash_buckets,
            plan_hash_salt=plan_hash_salt,
            lm_vocab_size=lm_vocab_size,
            token_ids=token_ids_map,
            planner_logits=planner_logits,
            planner_token_ids=planner_token_ids,
            planner_mask=planner_mask,
            agreement=agreement,
            note_snapshots=note_snapshots,
            bootstrap_notes=bootstrap_notes,
            coverage_logits=coverage_logits,
            tokenizer_cfg=tokenizer_cfg,
        )

    def build_inference_config(self) -> InferenceConfig:
        cfg_payload = self.manifest.get("config")
        if not isinstance(cfg_payload, Mapping):
            raise ValueError(
                "Replay manifest must contain a 'config' section mirroring inference config output."
            )
        decode_cfg = DecodeConfig(**cfg_payload.get("decode", {}))
        gate_cfg = GateAnnealingConfig(**cfg_payload.get("gate_annealing", {}))
        cadence_cfg = CadencePolicyConfig(**cfg_payload.get("cadence_policy", {}))
        emission_cadence = cfg_payload.get("M_by_stream") or {}
        rng_seed = cfg_payload.get("rng_seed")
        topology_value = str(cfg_payload.get("topology", "all_to_all")).strip().lower()
        if topology_value != "all_to_all":
            topology_value = "all_to_all"
        return InferenceConfig(
            streams=self.streams,
            stride_B=int(cfg_payload.get("stride_B", 1)),
            commit_L=int(cfg_payload.get("commit_L", 1)),
            read_lag_delta=int(cfg_payload.get("read_lag_delta", 0)),
            max_snapshots_K=int(cfg_payload.get("max_snapshots_K", len(self.streams))),
            topology=topology_value,  # type: ignore[arg-type]
            gate_g=float(cfg_payload.get("gate_g", 1.0)),
            agreement_threshold_tau=float(
                cfg_payload.get("tau", cfg_payload.get("agreement_threshold_tau", 0.15))
            ),
            emission_cadence_M_by_stream={
                stream: int(emission_cadence.get(stream, cfg_payload.get("stride_B", 1)))
                for stream in self.streams
            },
            logit_blend_alpha=float(cfg_payload.get("alpha", 1.0)),
            coverage_threshold=float(cfg_payload.get("coverage_threshold", 0.5)),
            decode=decode_cfg,
            gate_annealing=gate_cfg,
            cadence_policy=cadence_cfg,
            rng_seed=rng_seed,
        )

    def planner_payload(self) -> Mapping[str, Any]:
        return {
            "plan_token_ids": self.planner_token_ids.clone(),
            "plan_mask": self.planner_mask.clone(),
            "plan_logits": self.planner_logits.clone(),
            "source": "replay",
        }


class ReplayTimeline:
    """Tracks per-stream cursors for replaying recorded tensors."""

    def __init__(self, artifact: LogitReplayArtifact) -> None:
        self.streams = artifact.streams
        self.notes_dim = artifact.notes_dim
        self.tokens = {stream: list(artifact.token_ids.get(stream, [])) for stream in self.streams}
        self.agreement = {
            stream: artifact.agreement.get(stream, torch.ones(1)) for stream in self.streams
        }
        self.coverage = {stream: artifact.coverage_logits.get(stream) for stream in self.streams}
        self.note_snapshots = {
            stream: artifact.note_snapshots.get(stream, torch.zeros(0, artifact.notes_dim))
            for stream in self.streams
        }
        self.bootstrap = {
            stream: artifact.bootstrap_notes.get(stream, torch.zeros(artifact.notes_dim))
            for stream in self.streams
        }
        self._step_ptr = {stream: 0 for stream in self.streams}
        self._emission_ptr = {stream: 0 for stream in self.streams}
        self._active_stream: Optional[str] = None

    def set_active_stream(self, stream: str) -> None:
        self._active_stream = stream

    def active_stream(self) -> str:
        if self._active_stream is None:
            raise RuntimeError("Replay timeline active stream is undefined.")
        return self._active_stream

    def token_id(self, stream: str) -> int:
        pointer = self._step_ptr[stream]
        sequence = self.tokens.get(stream) or []
        if pointer >= len(sequence):
            return sequence[-1] if sequence else 0
        return int(sequence[pointer])

    def advance_step(self, stream: str) -> None:
        self._step_ptr[stream] = self._step_ptr.get(stream, 0) + 1

    def agreement_score(self, stream: str) -> float:
        pointer = self._step_ptr[stream]
        tensor = self.agreement.get(stream)
        if tensor is None or pointer >= tensor.numel():
            return 1.0
        return float(torch.clamp(tensor.flatten()[pointer], 0.0, 1.0).item())

    def coverage_logits_for(self, stream: str) -> Optional[torch.Tensor]:
        pointer = self._step_ptr[stream]
        tensor = self.coverage.get(stream)
        if tensor is None or tensor.numel() == 0:
            return None
        width = tensor.size(-1)
        if pointer >= tensor.size(0):
            pointer = tensor.size(0) - 1
        return tensor[pointer].view(1, width)

    def next_note_vector(self, stream: str) -> torch.Tensor:
        pointer = self._emission_ptr[stream]
        tensor = self.note_snapshots.get(stream)
        if tensor is None or tensor.numel() == 0:
            # Return zeros if no data available
            return torch.zeros(1, self.notes_dim)
        if pointer >= tensor.size(0):
            # Return last vector if we've exhausted the data
            vector = tensor[-1]
        else:
            vector = tensor[pointer]
        self._emission_ptr[stream] = pointer + 1
        return vector.view(1, -1)

    def bootstrap_vector(self, stream: str) -> torch.Tensor:
        tensor = self.bootstrap.get(stream)
        if tensor is None or tensor.numel() == 0:
            raise ValueError(f"Bootstrap vector not available for stream '{stream}'")
        return tensor.view(1, -1)


class ReplayTrunkModel(nn.Module):
    """Minimal trunk stub returning zeroed hidden states."""

    def __init__(self, timeline: ReplayTimeline, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.timeline = timeline
        self.hidden_size = hidden_size
        self.register_parameter("_stub", nn.Parameter(torch.zeros(1)))
        self.lm_head = ReplayLMHead(timeline, vocab_size)

    def forward(self, *_, **__) -> "_ReplayOutput":  # type: ignore[override]
        hidden = torch.zeros(1, 1, self.hidden_size)
        # Return stub KV cache with at least one layer (required for health check)
        key_stub = torch.zeros(1, 1, 1, self.hidden_size)
        value_stub = torch.zeros(1, 1, 1, self.hidden_size)
        past_kv = ((key_stub, value_stub),)  # One layer worth of KV pairs
        return _ReplayOutput(
            hidden_states=[hidden], past_key_values=past_kv, logits=torch.zeros(1, 1, 1)
        )


@dataclass(slots=True)
class _ReplayOutput:
    hidden_states: List[torch.Tensor]
    past_key_values: Tuple[Any, ...]
    logits: torch.Tensor


class ReplayLMHead(nn.Module):
    def __init__(self, timeline: ReplayTimeline, vocab_size: int) -> None:
        super().__init__()
        self.timeline = timeline
        self.vocab_size = vocab_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        stream = self.timeline.active_stream()
        token_id = self.timeline.token_id(stream)
        logits = torch.full(
            (hidden_states.size(0), hidden_states.size(1), self.vocab_size),
            fill_value=-1e9,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        logits[..., int(token_id)] = 10.0
        return logits


class ReplaySpeculationHead(nn.Module):
    def __init__(self, timeline: ReplayTimeline, notes_dim: int) -> None:
        super().__init__()
        self.timeline = timeline
        self.notes_dim = notes_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        stream = self.timeline.active_stream()
        vector = self.timeline.bootstrap_vector(stream)
        return vector.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(1)


class ReplayNotesHead(nn.Module):
    def __init__(self, timeline: ReplayTimeline) -> None:
        super().__init__()
        self.timeline = timeline

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        stream = self.timeline.active_stream()
        vector = self.timeline.next_note_vector(stream)
        expanded = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch, seq_len, _ = hidden_states.shape
        return expanded.unsqueeze(1).expand(batch, seq_len, expanded.size(-1))


class ReplayAgreementHead(nn.Module):
    def __init__(self, timeline: ReplayTimeline) -> None:
        super().__init__()
        self.timeline = timeline

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        score = self.timeline.agreement_score(self.timeline.active_stream())
        tensor = torch.tensor(score, device=hidden_states.device, dtype=hidden_states.dtype)
        return tensor.view(1, 1, 1)


class ReplayCoverageHead(nn.Module):
    def __init__(self, timeline: ReplayTimeline) -> None:
        super().__init__()
        self.timeline = timeline

    def forward(self, hidden_states: torch.Tensor, plan_embeddings: torch.Tensor, plan_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.timeline.coverage_logits_for(self.timeline.active_stream())
        if logits is None:
            width = plan_mask.numel() if plan_mask is not None else plan_embeddings.size(1)
            return torch.zeros((1, width), device=hidden_states.device, dtype=hidden_states.dtype)
        return logits.to(device=hidden_states.device, dtype=hidden_states.dtype)


class ReplayStreamAdapters(nn.Module):
    def forward(self, stream: str, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return hidden_states


class ReplaySNCBackend:
    def apply(self, adapted: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return adapted


class ReplayPlanEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, plan_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.embedding(plan_ids)


class _ReplayContextManager:
    """No-op context manager for replay trunk adapter."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class _ReplayTrunkAdapter:
    """Minimal trunk adapter stub for replay compatibility."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.instrumentation_enabled = True
        self._notes_provider = None
        self.selected_layers = (0,)  # Stub layer index

    def set_notes_provider(self, provider) -> None:
        self._notes_provider = provider

    def activate_context(self, *, stream=None, notes=None, notes_mask=None):
        """Return a no-op context manager for replay compatibility."""
        return _ReplayContextManager()


class ReplayModel(nn.Module):
    """Minimal model implementation that replays recorded logits/notes."""

    def __init__(self, artifact: LogitReplayArtifact) -> None:
        super().__init__()
        self.timeline = ReplayTimeline(artifact)
        trunk_model = ReplayTrunkModel(self.timeline, artifact.hidden_size, artifact.lm_vocab_size)
        self.trunk_adapter = _ReplayTrunkAdapter(trunk_model)
        self.stream_adapters = ReplayStreamAdapters()
        self.snc_backend = ReplaySNCBackend()
        self.speculation_head = ReplaySpeculationHead(self.timeline, artifact.notes_dim)
        self.notes_head = ReplayNotesHead(self.timeline)
        self.agreement_head = ReplayAgreementHead(self.timeline)
        self.coverage_head = ReplayCoverageHead(self.timeline)
        self.plan_embedding = ReplayPlanEmbedding(artifact.plan_vocab_size, artifact.hidden_size)
        collator_cfg = SimpleNamespace(
            plan_hash_buckets=int(artifact.plan_hash_buckets),
            plan_hash_salt=str(artifact.plan_hash_salt),
        )
        self.config = SimpleNamespace(
            plan_vocab_size=int(artifact.plan_vocab_size),
            plan_hash_salt=str(artifact.plan_hash_salt),
            notes_dim=int(artifact.notes_dim),
            collator=collator_cfg,
        )

    def to_trunk_device_and_dtype(self) -> None:
        return None

    # Hooks used by the orchestrator
    def on_bootstrap_stream(self, stream: str, **_: Any) -> None:
        self.timeline.set_active_stream(stream)

    def on_step_begin(self, stream: str, **_: Any) -> None:
        self.timeline.set_active_stream(stream)

    def on_step_end(self, stream: str, **_: Any) -> None:
        self.timeline.advance_step(stream)


__all__ = [
    "ReplayArtifactWriter",
    "LogitReplayArtifact",
    "ReplayModel",
]
