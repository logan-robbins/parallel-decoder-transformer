"""Shared CRDT/LWW read for training and runtime Dynamic Notes Bus state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from pdt.runtime.dnb_bus import DynamicNotesBus, Snapshot
from pdt.runtime.state import StreamState


__all__ = ["NotesWindow", "NotesWindowBuilder", "read_notes_lww"]


@dataclass(slots=True)
class NotesWindow:
    """Fixed ``2K`` slots: K prompt anchors followed by K dynamic LWW notes."""

    notes: torch.Tensor  # (B, 2K, notes_dim)
    mask: torch.Tensor  # (B, 2K) bool
    producers: tuple[str, ...]
    producer_indices: torch.Tensor  # (2K,)
    versions: torch.Tensor  # (2K,); -1 means absent
    published_blocks: torch.Tensor  # (2K,); -1 for anchors/absent
    lags: torch.Tensor  # (2K,); 0 for anchors/absent
    anchor_mask: torch.Tensor  # (2K,) bool


def read_notes_lww(
    delivered_updates: Iterable[Snapshot],
    *,
    producers: tuple[str, ...],
    consumer_block: int,
    notes_dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> NotesWindow:
    """Merge a delivered update set by per-producer version maximum.

    The merge is associative, commutative, and idempotent. Equal-version,
    unequal-payload updates are rejected because they violate the LWW key's
    single-writer invariant.
    """
    if consumer_block < 0:
        raise ValueError("consumer_block must be non-negative.")
    normalized = tuple(producer.lower() for producer in producers)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("producers must be non-empty and unique.")
    if notes_dim <= 0:
        raise ValueError("notes_dim must be positive.")

    updates = tuple(delivered_updates)
    latest: dict[tuple[str, str], Snapshot] = {}
    seen_versions: dict[tuple[str, str, int], Snapshot] = {}
    for update in updates:
        if update.producer not in normalized:
            raise ValueError(f"Delivered update has unknown producer {update.producer!r}.")
        key = (update.kind, update.producer)
        version_key = (update.kind, update.producer, update.version)
        previous_version = seen_versions.get(version_key)
        if previous_version is not None and not _same_update(update, previous_version):
            raise ValueError(f"Conflicting updates for {key} at version {update.version}.")
        seen_versions[version_key] = update
        if update.kind == "dynamic" and update.published_block > consumer_block:
            raise ValueError(
                f"Dynamic update {key} version {update.version} was published "
                f"in future block {update.published_block} for consumer block "
                f"{consumer_block}."
            )
        current = latest.get(key)
        if current is None or update.version > current.version:
            latest[key] = update

    sample = next(iter(latest.values()), None)
    target_device = device or (sample.notes.device if sample is not None else torch.device("cpu"))
    target_dtype = dtype or (sample.notes.dtype if sample is not None else torch.float32)
    batch_size = _batch_size(sample.notes) if sample is not None else 1

    ordered_keys = tuple(("anchor", producer) for producer in normalized) + tuple(
        ("dynamic", producer) for producer in normalized
    )
    vectors: list[torch.Tensor] = []
    masks: list[bool] = []
    versions: list[int] = []
    published_blocks: list[int] = []
    lags: list[int] = []
    for kind, producer in ordered_keys:
        selected_update = latest.get((kind, producer))
        if selected_update is None:
            vectors.append(
                torch.zeros(
                    (batch_size, notes_dim),
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            masks.append(False)
            versions.append(-1)
            published_blocks.append(-1)
            lags.append(0)
            continue
        vector = _normalize_note(selected_update.notes, notes_dim=notes_dim)
        if vector.size(0) != batch_size:
            raise ValueError("All delivered note tensors must share the same batch size.")
        vectors.append(vector.to(device=target_device, dtype=target_dtype))
        masks.append(True)
        versions.append(selected_update.version)
        published_blocks.append(selected_update.published_block)
        lags.append(
            0
            if selected_update.kind == "anchor"
            else consumer_block - selected_update.published_block
        )

    notes = torch.stack(vectors, dim=1)
    slot_mask = torch.tensor(masks, dtype=torch.bool, device=target_device)
    producer_ids = tuple(producer for _, producer in ordered_keys)
    producer_index = {producer: index for index, producer in enumerate(normalized)}
    return NotesWindow(
        notes=notes,
        mask=slot_mask.unsqueeze(0).expand(batch_size, -1),
        producers=producer_ids,
        producer_indices=torch.tensor(
            [producer_index[producer] for producer in producer_ids],
            dtype=torch.long,
            device=target_device,
        ),
        versions=torch.tensor(versions, dtype=torch.long, device=target_device),
        published_blocks=torch.tensor(published_blocks, dtype=torch.long, device=target_device),
        lags=torch.tensor(lags, dtype=torch.long, device=target_device),
        anchor_mask=torch.tensor(
            [kind == "anchor" for kind, _ in ordered_keys],
            dtype=torch.bool,
            device=target_device,
        ),
    )


class NotesWindowBuilder:
    """Runtime adapter over the same pure LWW merge used by training."""

    def __init__(
        self,
        *,
        producers: tuple[str, ...],
        notes_dim: int,
        block_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.producers = tuple(producer.lower() for producer in producers)
        if not self.producers or len(set(self.producers)) != len(self.producers):
            raise ValueError("producers must be non-empty and unique.")
        if notes_dim <= 0:
            raise ValueError("notes_dim must be positive.")
        self.notes_dim = notes_dim
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

    def build(self, consumer: StreamState, bus: DynamicNotesBus) -> NotesWindow:
        if consumer.stream not in self.producers:
            raise ValueError(f"Unknown consumer stream {consumer.stream!r}.")
        consumer_block = consumer.generated_count // self.block_size
        return self.build_for_block(consumer, bus, consumer_block=consumer_block)

    def build_for_block(
        self,
        consumer: StreamState,
        bus: DynamicNotesBus,
        *,
        consumer_block: int,
    ) -> NotesWindow:
        """Build at an explicit block while a just-sampled token is being consumed."""

        if consumer.stream not in self.producers:
            raise ValueError(f"Unknown consumer stream {consumer.stream!r}.")
        if consumer_block < 0:
            raise ValueError("consumer_block must be non-negative.")
        delivered = bus.delivered_updates(consumer_block=consumer_block)
        return read_notes_lww(
            delivered,
            producers=self.producers,
            consumer_block=consumer_block,
            notes_dim=self.notes_dim,
            device=self.device,
            dtype=self.dtype,
        )


def _batch_size(notes: torch.Tensor) -> int:
    if notes.dim() == 1:
        return 1
    if notes.dim() == 2:
        return notes.size(0)
    raise ValueError(f"Notes must be rank 1 or 2, got shape {tuple(notes.shape)}.")


def _normalize_note(notes: torch.Tensor, *, notes_dim: int) -> torch.Tensor:
    vector = notes.unsqueeze(0) if notes.dim() == 1 else notes
    if vector.dim() != 2:
        raise ValueError(f"Notes must be rank 1 or 2, got shape {tuple(notes.shape)}.")
    if vector.size(-1) != notes_dim:
        raise ValueError(f"Note width mismatch: expected {notes_dim}, got {vector.size(-1)}.")
    return vector


def _same_update(left: Snapshot, right: Snapshot) -> bool:
    return (
        left.producer == right.producer
        and left.version == right.version
        and left.published_block == right.published_block
        and left.stride == right.stride
        and left.kind == right.kind
        and torch.equal(left.notes, right.notes)
    )
