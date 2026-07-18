"""Shared versioned-history read for training and runtime bus state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from pdt.runtime.dnb_bus import DynamicNotesBus, Snapshot
from pdt.runtime.state import StreamState


__all__ = ["NotesWindow", "NotesWindowBuilder", "read_notes_history"]


@dataclass(slots=True)
class NotesWindow:
    """Fixed addressed slots containing age-major dynamic writes only."""

    notes: torch.Tensor  # (B, S, notes_dim)
    mask: torch.Tensor  # (B, S) bool
    producers: tuple[str, ...]
    producer_indices: torch.Tensor  # (S,)
    versions: torch.Tensor  # (S,); -1 means absent
    published_blocks: torch.Tensor  # (S,); -1 for absent
    lags: torch.Tensor  # (S,); explicit dynamic age even when absent


def read_notes_history(
    delivered_updates: Iterable[Snapshot],
    *,
    producers: tuple[str, ...],
    consumer_block: int,
    notes_dim: int,
    history_blocks: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> NotesWindow:
    """Build a deterministic versioned dynamic-history window.

    Dynamic slots are ordered by age, then producer. Ages run from one
    through ``history_blocks``; a write becomes readable at the next block.
    Equal-version,
    unequal-payload updates are rejected as single-writer corruption.
    """
    if consumer_block < 0:
        raise ValueError("consumer_block must be non-negative.")
    normalized = tuple(producer.lower() for producer in producers)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("producers must be non-empty and unique.")
    if notes_dim <= 0:
        raise ValueError("notes_dim must be positive.")
    if type(history_blocks) is not int or history_blocks <= 0:
        raise ValueError("history_blocks must be a positive integer.")

    updates = tuple(delivered_updates)
    dynamics: dict[tuple[int, str], Snapshot] = {}
    seen_versions: dict[tuple[str, str, int], Snapshot] = {}
    for update in updates:
        if update.producer not in normalized:
            raise ValueError(f"Delivered update has unknown producer {update.producer!r}.")
        version_key = (update.kind, update.producer, update.version)
        previous_version = seen_versions.get(version_key)
        if previous_version is not None and not _same_update(update, previous_version):
            raise ValueError(
                f"Conflicting updates for {(update.kind, update.producer)} "
                f"at version {update.version}."
            )
        seen_versions[version_key] = update
        if update.published_block > consumer_block:
            raise ValueError(
                f"Dynamic update {(update.kind, update.producer)} version {update.version} "
                "was published "
                f"in future block {update.published_block} for consumer block "
                f"{consumer_block}."
            )
        age = consumer_block - update.published_block
        if age < 1:
            raise ValueError(
                "Delivered dynamic notes must be delayed by at least one block."
            )
        if age > history_blocks:
            continue
        slot = (age, update.producer)
        current = dynamics.get(slot)
        if current is not None and current.version != update.version:
            raise ValueError(
                f"Producer {update.producer!r} published multiple versions for block "
                f"{update.published_block}."
            )
        dynamics[slot] = update

    sample = next(iter(updates), None)
    target_device = device or (sample.notes.device if sample is not None else torch.device("cpu"))
    target_dtype = dtype or (sample.notes.dtype if sample is not None else torch.float32)
    batch_size = _batch_size(sample.notes) if sample is not None else 1

    ordered_keys = tuple(
        (age, producer)
        for age in range(1, history_blocks + 1)
        for producer in normalized
    )
    vectors: list[torch.Tensor] = []
    masks: list[bool] = []
    versions: list[int] = []
    published_blocks: list[int] = []
    lags: list[int] = []
    for age, producer in ordered_keys:
        selected_update = dynamics.get((age, producer))
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
            lags.append(age)
            continue
        vector = _normalize_note(selected_update.notes, notes_dim=notes_dim)
        if vector.size(0) != batch_size:
            raise ValueError("All delivered note tensors must share the same batch size.")
        vectors.append(vector.to(device=target_device, dtype=target_dtype))
        masks.append(True)
        versions.append(selected_update.version)
        published_blocks.append(selected_update.published_block)
        lags.append(age)

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
    )


class NotesWindowBuilder:
    """Runtime adapter over the same versioned-history read used by training."""

    def __init__(
        self,
        *,
        producers: tuple[str, ...],
        notes_dim: int,
        block_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        history_blocks: int = 16,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.producers = tuple(producer.lower() for producer in producers)
        if not self.producers or len(set(self.producers)) != len(self.producers):
            raise ValueError("producers must be non-empty and unique.")
        if notes_dim <= 0:
            raise ValueError("notes_dim must be positive.")
        if type(history_blocks) is not int or history_blocks <= 0:
            raise ValueError("history_blocks must be a positive integer.")
        self.notes_dim = notes_dim
        self.block_size = block_size
        self.history_blocks = history_blocks
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
        return read_notes_history(
            delivered,
            producers=self.producers,
            consumer_block=consumer_block,
            notes_dim=self.notes_dim,
            history_blocks=self.history_blocks,
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
        and left.code_indices == right.code_indices
    )
