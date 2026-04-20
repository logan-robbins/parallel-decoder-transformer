"""Lag-aware notes-window builder for multi-stream consumers.

Given a mapping ``stream -> DynamicNotesBus`` and a consumer stream's
current state, assembles a flat ``(1, S, notes_dim)`` tensor + mask of the
most recent eligible sibling notes. Enforces monotonic version tracking
per producer to prevent stale replays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, Sequence, Tuple

import torch

from pdt.runtime.dnb_bus import DynamicNotesBus
from pdt.runtime.state import StreamState


__all__ = ["NotesWindow", "NotesWindowBuilder", "TopologyMask"]


@dataclass(slots=True)
class NotesWindow:
    notes: torch.Tensor  # (1, S, notes_dim)
    mask: torch.Tensor  # (1, S) bool
    producers: Tuple[str, ...]
    versions: torch.Tensor  # (S,)
    strides: torch.Tensor  # (S,)


class TopologyMask:
    """Which producers feed a consumer. ``all_to_all`` is the only topology."""

    def __init__(
        self,
        streams: Sequence[str],
        topology: Literal["all_to_all"] = "all_to_all",
    ) -> None:
        if topology != "all_to_all":
            raise ValueError("Only 'all_to_all' topology is supported.")
        self.streams = tuple(s.lower() for s in streams)
        self.topology = topology

    def producers_for(self, consumer: str) -> Tuple[str, ...]:
        consumer = consumer.lower()
        if consumer not in self.streams:
            raise ValueError(f"Unknown consumer stream: {consumer!r}")
        return self.streams


class NotesWindowBuilder:
    """Assembles the visible workspace window for a consumer stream."""

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
        self.notes_dim = notes_dim
        self.topology_mask = topology_mask
        self.read_lag = read_lag
        self.max_snapshots = max_snapshots
        self.device = device
        self.dtype = dtype
        self.self_lag_offset = self_lag_offset
        self.self_only_tokens = max(0, int(self_only_tokens))

    def build(
        self,
        consumer: StreamState,
        bus_by_stream: Mapping[str, DynamicNotesBus],
    ) -> NotesWindow:
        consumer_name = consumer.stream.lower()
        restrict_self = (
            self.self_only_tokens > 0 and consumer.generated_count < self.self_only_tokens
        )
        if restrict_self:
            producers: Tuple[str, ...] = (consumer_name,)
        else:
            producers = self.topology_mask.producers_for(consumer_name)
            if consumer_name not in producers:
                producers = producers + (consumer_name,)

        note_vectors: List[torch.Tensor] = []
        producer_ids: List[str] = []
        versions: List[int] = []
        strides: List[int] = []

        active = sum(
            1
            for p in producers
            if (bus := bus_by_stream.get(p)) is not None and len(bus) > 0
        )
        per_producer_limit = max(1, self.max_snapshots // max(1, active))

        for producer in producers:
            bus = bus_by_stream.get(producer)
            if bus is None or len(bus) == 0:
                continue
            lag = self.read_lag + (self.self_lag_offset if producer == consumer_name else 0)
            snapshots = bus.snapshot(lag=lag, limit=per_producer_limit)
            last_seen = consumer.last_seen_version.get(producer, 0)
            for snap in snapshots:
                if snap.version <= last_seen:
                    continue
                note_vectors.append(self._normalize(snap.notes))
                producer_ids.append(producer)
                versions.append(snap.version)
                strides.append(snap.stride)

        if len(note_vectors) > self.max_snapshots:
            note_vectors = note_vectors[-self.max_snapshots :]
            producer_ids = producer_ids[-self.max_snapshots :]
            versions = versions[-self.max_snapshots :]
            strides = strides[-self.max_snapshots :]

        if not note_vectors:
            device = self.device or consumer.device
            empty_notes = torch.zeros(
                (1, 0, self.notes_dim),
                dtype=self.dtype or torch.float32,
                device=device,
            )
            empty_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
            return NotesWindow(
                notes=empty_notes,
                mask=empty_mask,
                producers=tuple(),
                versions=torch.zeros(0, dtype=torch.long, device=device),
                strides=torch.zeros(0, dtype=torch.long, device=device),
            )

        target_device = self.device or consumer.device
        target_dtype = self.dtype or note_vectors[0].dtype
        converted = [v.to(device=target_device, dtype=target_dtype) for v in note_vectors]
        stacked = torch.stack(converted, dim=0).unsqueeze(0)  # (1, S, notes_dim)
        mask = torch.ones((1, stacked.size(1)), dtype=torch.bool, device=target_device)
        return NotesWindow(
            notes=stacked,
            mask=mask,
            producers=tuple(producer_ids),
            versions=torch.tensor(versions, dtype=torch.long, device=target_device),
            strides=torch.tensor(strides, dtype=torch.long, device=target_device),
        )

    def _normalize(self, notes: torch.Tensor) -> torch.Tensor:
        if notes.dim() == 0:
            raise ValueError("Snapshot notes must have at least one dim.")
        if notes.dim() == 1:
            vec = notes
        else:
            vec = notes.reshape(-1, notes.size(-1))[0]
        if vec.numel() < self.notes_dim:
            padded = torch.zeros(self.notes_dim, dtype=vec.dtype, device=vec.device)
            padded[: vec.numel()] = vec.view(-1)
            return padded
        if vec.numel() > self.notes_dim:
            return vec.view(-1)[: self.notes_dim]
        return vec
