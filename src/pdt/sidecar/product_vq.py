"""Finite-alphabet product vector quantization for dynamic PDT notes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


__all__ = ["ProductVQOutput", "ProductVectorQuantizer"]


@dataclass(frozen=True, slots=True)
class ProductVQOutput:
    """A decoded finite message and the exact transmitted code indices."""

    pre_quantized: torch.Tensor
    quantized: torch.Tensor
    indices: torch.Tensor
    assignment_logits: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    capacity_bits: int


class ProductVectorQuantizer(nn.Module):
    """Quantize ``D`` features as ``M`` independently coded sub-vectors.

    With the canonical ``M=4`` and ``C=256``, each dynamic snapshot has at
    most ``C**M`` values and therefore carries exactly 32 capacity bits. The
    reconstructed float tensor is a local decoder representation; the
    ``(M,)`` integer index tuple is the message on the logical bus.
    """

    def __init__(self, *, width: int, num_codebooks: int, codes_per_codebook: int) -> None:
        super().__init__()
        for name, value in (
            ("width", width),
            ("num_codebooks", num_codebooks),
            ("codes_per_codebook", codes_per_codebook),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if width % num_codebooks != 0:
            raise ValueError(f"width={width} must be divisible by num_codebooks={num_codebooks}.")
        if codes_per_codebook < 2 or codes_per_codebook & (codes_per_codebook - 1):
            raise ValueError("codes_per_codebook must be a power of two greater than one.")

        self.width = width
        self.num_codebooks = num_codebooks
        self.codes_per_codebook = codes_per_codebook
        self.subvector_width = width // num_codebooks
        self.capacity_bits = num_codebooks * int(math.log2(codes_per_codebook))
        self.codebook = nn.Parameter(
            torch.empty(num_codebooks, codes_per_codebook, self.subvector_width)
        )
        nn.init.normal_(self.codebook, mean=0.0, std=0.02)

    def forward(self, vectors: torch.Tensor) -> ProductVQOutput:
        if vectors.size(-1) != self.width:
            raise ValueError(
                f"Product VQ expected final width {self.width}, got {vectors.size(-1)}."
            )
        if not vectors.is_floating_point():
            raise TypeError("Product VQ input must use a floating-point dtype.")

        original_shape = vectors.shape
        input_dtype = vectors.dtype
        flat = vectors.to(dtype=self.codebook.dtype).reshape(
            -1, self.num_codebooks, self.subvector_width
        )
        squared_inputs = flat.square().sum(dim=-1, keepdim=True)
        squared_codes = self.codebook.square().sum(dim=-1).unsqueeze(0)
        cross = torch.einsum("nmd,mcd->nmc", flat, self.codebook)
        distances = squared_inputs - 2.0 * cross + squared_codes
        indices = distances.argmin(dim=-1)

        selected = self.decode(indices).reshape(
            -1,
            self.num_codebooks,
            self.subvector_width,
        )
        quantized_flat = flat + (selected - flat).detach()
        quantized = quantized_flat.reshape(original_shape).to(dtype=input_dtype)
        embedded = selected.reshape(original_shape)
        continuous = flat.reshape(original_shape)
        return ProductVQOutput(
            pre_quantized=vectors,
            quantized=quantized,
            indices=indices.reshape(*original_shape[:-1], self.num_codebooks),
            assignment_logits=(-distances).reshape(
                *original_shape[:-1],
                self.num_codebooks,
                self.codes_per_codebook,
            ),
            commitment_loss=F.mse_loss(continuous, embedded.detach()),
            codebook_loss=F.mse_loss(embedded, continuous.detach()),
            capacity_bits=self.capacity_bits,
        )

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode the transmitted index tuples with no caller-supplied payload."""

        if indices.size(-1) != self.num_codebooks:
            raise ValueError(
                "Product VQ indices must end in num_codebooks; "
                f"expected {self.num_codebooks}, got {indices.size(-1)}."
            )
        if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise TypeError("Product VQ indices must use a non-bool integer dtype.")
        if bool(((indices < 0) | (indices >= self.codes_per_codebook)).any()):
            raise ValueError(f"Product VQ indices must lie in [0, {self.codes_per_codebook}).")
        flat = indices.to(device=self.codebook.device, dtype=torch.long).reshape(
            -1, self.num_codebooks
        )
        selected = torch.stack(
            [self.codebook[group, flat[:, group]] for group in range(self.num_codebooks)],
            dim=1,
        )
        return selected.reshape(*indices.shape[:-1], self.width)
