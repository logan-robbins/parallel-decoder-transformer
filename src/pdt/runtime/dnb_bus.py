"""Dynamic Notes Bus as an inflationary set of addressed generated-fragment updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Protocol, Sequence

import torch

from pdt.config.schemas import NotesBusConfig


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


__all__ = ["DynamicNoteCodec", "DynamicNotesBus", "Snapshot"]


class DynamicNoteCodec(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def num_codebooks(self) -> int: ...

    @property
    def codes_per_codebook(self) -> int: ...

    def decode(self, indices: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One immutable update in the addressed version history.

    ``version`` is monotone per producer and begins at one. Static plans are
    not bus writes: they live in persistent read-only plan memory.
    """

    producer: str
    version: int
    published_block: int
    stride: int
    kind: Literal["dynamic"]
    notes: torch.Tensor
    code_indices: Optional[tuple[int, ...]]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        producer = self.producer.lower()
        if not producer:
            raise ValueError("Snapshot producer must be non-empty.")
        object.__setattr__(self, "producer", producer)
        if self.stride < 0:
            raise ValueError("Snapshot stride must be non-negative.")
        if self.kind != "dynamic":
            raise ValueError(f"Snapshot kind must be 'dynamic', got {self.kind!r}.")
        if self.version <= 0 or self.published_block < 0:
            raise ValueError("Dynamic snapshots require version>0 and published_block>=0.")
        if self.code_indices is None or not self.code_indices:
            raise ValueError("Dynamic snapshots require a non-empty finite code tuple.")


class DynamicNotesBus:
    """Canonical multi-producer update set for runtime note delivery.

    Writes are inflationary and dynamic versions must increase. Publishers
    provide only finite code indices; this object
    performs the sole decode into the local SNC tensor. Reads are delegated to
    the shared pure versioned-window builder in :mod:`pdt.runtime.window`,
    which training mirrors exactly.
    """

    def __init__(
        self,
        config: NotesBusConfig,
        *,
        producers: tuple[str, ...],
        device: torch.device,
        codec: DynamicNoteCodec,
    ) -> None:
        normalized = tuple(producer.lower() for producer in producers)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("DynamicNotesBus producers must be non-empty and unique.")
        if config.lag < 0:
            raise ValueError("Notes-bus lag must be non-negative.")
        if config.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported notes-bus dtype {config.dtype!r}; "
                f"expected one of {tuple(sorted(_DTYPE_MAP))}."
            )
        if codec.width != config.snapshot_dim:
            raise ValueError(
                f"Dynamic-note codec width {codec.width} != snapshot_dim {config.snapshot_dim}."
            )
        if codec.num_codebooks != config.num_codebooks:
            raise ValueError("Dynamic-note codec num_codebooks does not match bus config.")
        if codec.codes_per_codebook != config.codes_per_codebook:
            raise ValueError("Dynamic-note codec code count does not match bus config.")
        self.config = config
        self.producers = normalized
        self.device = device
        self.dtype = _DTYPE_MAP[config.dtype]
        self.codec = codec
        self._updates: list[Snapshot] = []
        self._latest_version: dict[str, int] = {producer: 0 for producer in normalized}

    def publish(
        self,
        producer: str,
        *,
        published_block: int,
        stride: int,
        code_indices: Sequence[int],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Snapshot:
        producer = self._validate_producer(producer)
        if published_block < 0:
            raise ValueError("published_block must be non-negative.")
        version = self._latest_version[producer] + 1
        codes = tuple(code_indices)
        if len(codes) != self.config.num_codebooks:
            raise ValueError(
                "Dynamic note code tuple length must equal notes_bus.num_codebooks; "
                f"expected {self.config.num_codebooks}, got {len(codes)}."
            )
        if any(
            type(index) is not int or not 0 <= index < self.config.codes_per_codebook
            for index in codes
        ):
            raise ValueError(
                "Every dynamic note code must be an integer in "
                f"[0, {self.config.codes_per_codebook})."
            )
        code_tensor = torch.tensor([codes], dtype=torch.long, device=self.device)
        decoded = self.codec.decode(code_tensor)
        if decoded.shape != (1, self.config.snapshot_dim):
            raise RuntimeError(
                "Dynamic-note codec returned the wrong decoded shape: "
                f"expected {(1, self.config.snapshot_dim)}, got {tuple(decoded.shape)}."
            )
        snapshot = Snapshot(
            producer=producer,
            version=version,
            published_block=published_block,
            stride=stride,
            kind="dynamic",
            notes=self._prepare_notes(decoded[0]),
            code_indices=codes,
            metadata=dict(metadata or {}),
        )
        self._updates.append(snapshot)
        self._latest_version[producer] = version
        return snapshot

    def delivered_updates(self, *, consumer_block: int) -> tuple[Snapshot, ...]:
        if consumer_block < 0:
            raise ValueError("consumer_block must be non-negative.")
        lag = self.config.lag
        return tuple(
            update
            for update in self._updates
            if update.published_block + lag <= consumer_block
        )

    def all_updates(self) -> tuple[Snapshot, ...]:
        return tuple(self._updates)

    def _validate_producer(self, producer: str) -> str:
        normalized = producer.lower()
        if normalized not in self.producers:
            raise ValueError(f"Unknown producer {producer!r}; expected one of {self.producers}.")
        return normalized

    def _prepare_notes(self, notes: torch.Tensor) -> torch.Tensor:
        if notes.dim() not in (1, 2):
            raise ValueError(f"Snapshot notes must be rank 1 or 2, got shape {tuple(notes.shape)}.")
        if notes.size(-1) != self.config.snapshot_dim:
            raise ValueError(
                f"Snapshot dim mismatch: expected {self.config.snapshot_dim}, got {notes.size(-1)}."
            )
        return notes.to(device=self.device, dtype=self.dtype).clone()

    def __len__(self) -> int:
        return len(self._updates)
