"""Topology-aware utilities for building Dynamic Notes Bus windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple, Literal

import torch

from .config import InferenceConfig
from .dnb_bus import DynamicNotesBus
from .state import StreamState


@dataclass(slots=True)
class NotesWindow:
    """Structured window of notes tensors and metadata for SNC reads."""

    notes: torch.Tensor  # [1, S, notes_dim]
    mask: torch.Tensor  # [1, S]
    producers: Tuple[str, ...]
    versions: torch.Tensor  # [S]
    strides: torch.Tensor  # [S]


class TopologyMask:
    """Resolves which producers feed a consumer stream for the all-to-all topology."""

    def __init__(
        self,
        streams: Sequence[str],
        topology: Literal["all_to_all"] = "all_to_all",
        *,
        allow_self: bool = False,
    ) -> None:
        if not streams:
            raise ValueError("TopologyMask requires at least one stream.")
        if topology != "all_to_all":
            raise ValueError("Only the 'all_to_all' topology is supported.")
        self.streams = tuple(stream.lower() for stream in streams)
        self.topology = topology
        self.allow_self = allow_self
        self._index = {stream: idx for idx, stream in enumerate(self.streams)}

    def producers_for(self, consumer: str) -> Tuple[str, ...]:
        consumer = consumer.lower()
        if consumer not in self._index:
            raise ValueError(f"Unknown consumer stream: {consumer!r}")
        # All-to-all always exposes every stream (including self) to each consumer.
        return self.streams


class NotesWindowBuilder:
    """Builds lagged note windows from producer buses."""

    def __init__(
        self,
        *,
        notes_dim: int,
        topology_mask: TopologyMask,
        read_lag: int,
        max_snapshots: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        self_lag_offset: int = 1,
        self_only_tokens: int = 0,
    ) -> None:
        if notes_dim <= 0:
            raise ValueError("notes_dim must be positive.")
        if read_lag < 0:
            raise ValueError("read_lag must be non-negative.")
        if max_snapshots <= 0:
            raise ValueError("max_snapshots must be positive.")
        if self_lag_offset < 0:
            raise ValueError("self_lag_offset must be non-negative.")
        self.notes_dim = notes_dim
        self.topology_mask = topology_mask
        self.read_lag = read_lag
        self.max_snapshots = max_snapshots
        self.device = device
        self.dtype = dtype
        self.self_lag_offset = self_lag_offset
        self.self_only_tokens = max(0, int(self_only_tokens))

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        notes_dim: int,
        *,
        topology_mask: Optional[TopologyMask] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "NotesWindowBuilder":
        mask = topology_mask or TopologyMask(
            config.streams,
            config.topology,
        )
        return cls(
            notes_dim=notes_dim,
            topology_mask=mask,
            read_lag=config.read_lag_delta,
            max_snapshots=config.max_snapshots_K,
            device=device,
            dtype=dtype,
            self_lag_offset=1,
            self_only_tokens=config.sectional_self_tokens,
        )

    def build(
        self,
        consumer_state: StreamState,
        bus_by_stream: Mapping[str, DynamicNotesBus],
    ) -> NotesWindow:
        consumer_stream = consumer_state.stream
        restrict_self = (
            self.self_only_tokens > 0 and consumer_state.generated_count < self.self_only_tokens
        )
        if restrict_self:
            producers = (consumer_stream,)
        else:
            producers = self.topology_mask.producers_for(consumer_stream)
            if consumer_stream not in producers:
                producers = tuple(producers) + (consumer_stream,)
        note_vectors: List[torch.Tensor] = []
        producer_ids: List[str] = []
        versions: List[int] = []
        strides: List[int] = []
        for producer in producers:
            bus = bus_by_stream.get(producer)
            if bus is None or len(bus) == 0:
                continue
            lag = self.read_lag + (self.self_lag_offset if producer == consumer_stream else 0)
            snapshots = bus.snapshot(lag=lag, limit=self.max_snapshots)
            last_seen = consumer_state.last_seen_version.get(producer, 0)
            for snapshot in snapshots:
                if snapshot.version <= last_seen:
                    continue
                vector = self._normalize_notes(snapshot.notes)
                note_vectors.append(vector)
                producer_ids.append(producer)
                versions.append(snapshot.version)
                strides.append(snapshot.stride)
        if len(note_vectors) > self.max_snapshots:
            note_vectors = note_vectors[-self.max_snapshots :]
            producer_ids = producer_ids[-self.max_snapshots :]
            versions = versions[-self.max_snapshots :]
            strides = strides[-self.max_snapshots :]
        if not note_vectors:
            empty_notes = torch.zeros(
                (1, 0, self.notes_dim),
                dtype=self.dtype or torch.float32,
                device=self.device or consumer_state.device,
            )
            empty_mask = torch.zeros((1, 0), dtype=torch.bool, device=empty_notes.device)
            return NotesWindow(
                notes=empty_notes,
                mask=empty_mask,
                producers=tuple(),
                versions=torch.zeros(0, dtype=torch.long, device=empty_notes.device),
                strides=torch.zeros(0, dtype=torch.long, device=empty_notes.device),
            )
        target_device = self.device or consumer_state.device
        target_dtype = self.dtype or note_vectors[0].dtype
        converted = [vec.to(device=target_device, dtype=target_dtype) for vec in note_vectors]
        stacked = torch.stack(converted, dim=0).unsqueeze(0)
        mask = torch.ones((1, stacked.size(1)), dtype=torch.bool, device=target_device)
        return NotesWindow(
            notes=stacked,
            mask=mask,
            producers=tuple(producer_ids),
            versions=torch.tensor(versions, dtype=torch.long, device=stacked.device),
            strides=torch.tensor(strides, dtype=torch.long, device=stacked.device),
        )

    def _normalize_notes(self, notes: torch.Tensor) -> torch.Tensor:
        if notes.dim() == 0:
            raise ValueError("Snapshot notes tensor must have at least one dimension.")
        if notes.dim() == 1:
            vector = notes
        elif notes.dim() >= 2:
            vector = notes.reshape(-1, notes.size(-1))[0]
        else:  # pragma: no cover - exhaustive guard
            raise ValueError("Unsupported notes tensor rank.")
        if vector.numel() < self.notes_dim:
            padded = torch.zeros(self.notes_dim, dtype=vector.dtype, device=vector.device)
            padded[: vector.numel()] = vector.view(-1)
            vector = padded
        elif vector.numel() > self.notes_dim:
            vector = vector.view(-1)[: self.notes_dim]
        return vector


__all__ = ["NotesWindow", "NotesWindowBuilder", "TopologyMask"]
