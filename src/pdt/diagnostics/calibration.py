"""Checkpoint-free calibration for the PDT coordination instrument.

This is the canonical implementation of the Tier-0 artifact specified by
``07_15.md`` Appendix C. It uses the real sidecar modules with a tiny synthetic
graph, so it needs neither model weights nor a dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from pdt.config.schemas import (
    PlanNotesProjectionConfig,
    PlannerHeadConfig,
    SNCConfig,
    SpeculationHeadConfig,
    StreamAdapterConfig,
)
from pdt.sidecar.adapters import StreamAdapterLayer
from pdt.sidecar.heads.plan_notes_proj import PlanNotesProjection
from pdt.sidecar.heads.planner import PlannerHead
from pdt.sidecar.heads.speculation import SpeculationHead
from pdt.sidecar.snc import SharedNotesCrossAttention


_DTYPE = torch.float64
_BATCH = 32
_HIDDEN = 32
_NOTES = 8
_SEQUENCE = 4
_VOCAB = 17
_SEED = 20260715
_SWEEP = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    gradient_norms: dict[str, float]
    slopes: dict[str, float]
    initial_scale: float
    two_factor_scale: float
    suppression_ratio: float
    plateau_ratio: float
    producer_permutation_delta_ce: float
    delivery_reorder_delta_ce: float


class _TinyCoordinationGraph(nn.Module):
    """Small graph retaining every multiplicative path named in Appendix C."""

    def __init__(self) -> None:
        super().__init__()
        self.planner = PlannerHead(
            PlannerHeadConfig(
                hidden_size=_HIDDEN,
                vocab_size=16,
                num_slots=4,
                dropout=0.0,
            )
        )
        self.plan_notes = PlanNotesProjection(
            PlanNotesProjectionConfig(hidden_size=_HIDDEN, notes_dim=_NOTES)
        )
        self.speculation = SpeculationHead(
            SpeculationHeadConfig(
                hidden_size=_HIDDEN,
                notes_dim=_NOTES,
                dropout=0.0,
            )
        )
        self.snc = SharedNotesCrossAttention(
            SNCConfig(
                hidden_size=_HIDDEN,
                notes_dim=_NOTES,
                num_heads=4,
                dropout=0.0,
            ),
            gating_init=-4.0,
        )
        self.adapter = StreamAdapterLayer(
            StreamAdapterConfig(
                hidden_size=_HIDDEN,
                bottleneck_size=8,
                streams=("stream_0", "stream_1"),
                dropout=0.0,
            )
        )
        self.notes_gate = nn.Parameter(torch.tensor(-4.0))
        self.adapter_gate = nn.Parameter(torch.tensor(-4.0))
        self.to(dtype=_DTYPE)

    def forward(
        self,
        prompt_hidden: torch.Tensor,
        writer_hidden: torch.Tensor,
        receiver_hidden: torch.Tensor,
    ) -> torch.Tensor:
        planner = self.planner(prompt_hidden)
        ownership = torch.zeros(
            prompt_hidden.size(0),
            2,
            4,
            dtype=torch.bool,
            device=prompt_hidden.device,
        )
        ownership[:, 0, (0, 2)] = True
        ownership[:, 1, (1, 3)] = True
        anchors = self.plan_notes(planner.quantized, ownership)
        written = self.speculation(writer_hidden).mean(dim=1)
        notes = torch.stack((anchors[:, 0], written), dim=1)
        note_mask = torch.ones(notes.shape[:2], dtype=torch.bool, device=notes.device)
        snc_delta = self.snc(receiver_hidden, notes, notes_mask=note_mask)
        adapter_delta = self.adapter(receiver_hidden, "stream_0")
        return (
            receiver_hidden
            + torch.sigmoid(self.notes_gate) * snc_delta
            + torch.sigmoid(self.adapter_gate) * adapter_delta
        )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_SEED)
    tensors = tuple(
        torch.randn(
            _BATCH,
            _SEQUENCE,
            _HIDDEN,
            generator=generator,
            dtype=_DTYPE,
        )
        for _ in range(3)
    )
    probe = torch.randn(
        _BATCH,
        _SEQUENCE,
        _HIDDEN,
        generator=generator,
        dtype=_DTYPE,
    )
    return tensors[0], tensors[1], tensors[2], probe


def _linear_probe_loss(output: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    return (output * probe).sum() / output.size(0)


def _group_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=_DTYPE)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().to(_DTYPE).square().sum()
    return float(squared.sqrt().item())


def step_zero_gradient_norms() -> dict[str, float]:
    """Return Appendix C Table 1 norms from the exact zero-init graph."""
    torch.manual_seed(_SEED)
    model = _TinyCoordinationGraph()
    prompt, writer, receiver, probe = _inputs()
    _linear_probe_loss(model(prompt, writer, receiver), probe).backward()

    adapter = model.adapter.adapters.adapters["stream_0"]
    groups: dict[str, tuple[nn.Parameter, ...]] = {
        "notes_gate": (model.notes_gate,),
        "snc.gate": (cast(nn.Parameter, model.snc.gate),),
        "snc.o_proj.weight": (cast(nn.Parameter, model.snc.o_proj.weight),),
        "snc.o_proj.bias": (cast(nn.Parameter, model.snc.o_proj.bias),),
        "snc.q_proj": tuple(model.snc.q_proj.parameters()),
        "snc.k_proj": tuple(model.snc.k_proj.parameters()),
        "snc.v_proj": tuple(model.snc.v_proj.parameters()),
        "speculation_head": tuple(model.speculation.parameters()),
        "planner.slot_projector": tuple(model.planner.slot_projector.parameters()),
        "planner.codebook": tuple(model.planner.codebook.parameters()),
        "plan_notes_proj": tuple(model.plan_notes.parameters()),
        "adapter.up_proj": tuple(
            parameter for name, parameter in adapter.named_parameters() if name.startswith("up.")
        ),
        "adapter.down_proj": tuple(
            parameter for name, parameter in adapter.named_parameters() if name.startswith("down.")
        ),
        "adapter_gate": (model.adapter_gate,),
    }
    norms = {name: _group_grad_norm(params) for name, params in groups.items()}

    expected_nonzero = {
        "snc.o_proj.weight",
        "snc.o_proj.bias",
        "adapter.up_proj",
    }
    for name, value in norms.items():
        if name in expected_nonzero and value <= 0.0:
            raise AssertionError(f"Expected a nonzero step-zero gradient for {name}.")
        if name not in expected_nonzero and value != 0.0:
            raise AssertionError(
                f"Expected a bitwise-zero step-zero gradient for {name}, got {value}."
            )
    return norms


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _speculation_grad_norm(*, outer_scale: float, inner_scale: float, output_scale: float) -> float:
    torch.manual_seed(_SEED)
    model = _TinyCoordinationGraph()
    prompt, writer, receiver, probe = _inputs()
    with torch.no_grad():
        model.notes_gate.fill_(_logit(outer_scale))
        model.snc.gate.fill_(_logit(inner_scale))
        model.snc.o_proj.weight.normal_(mean=0.0, std=0.02)
        model.snc.o_proj.weight.mul_(output_scale)
        model.snc.o_proj.bias.zero_()
    _linear_probe_loss(model(prompt, writer, receiver), probe).backward()
    return _group_grad_norm(model.speculation.parameters())


def _fit_log_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_tensor = torch.log(torch.tensor(tuple(x), dtype=_DTYPE))
    y_tensor = torch.log(torch.tensor(tuple(y), dtype=_DTYPE))
    design = torch.stack((x_tensor, torch.ones_like(x_tensor)), dim=1)
    solution = torch.linalg.lstsq(design, y_tensor.unsqueeze(1)).solution
    return float(solution[0, 0].item())


def escape_law_curves() -> tuple[dict[str, list[float]], dict[str, float]]:
    """Measure all three multiplicative slopes in the corrected escape law."""
    curves = {
        "outer_gate": [
            _speculation_grad_norm(outer_scale=value, inner_scale=0.2, output_scale=0.2)
            for value in _SWEEP
        ],
        "inner_gate": [
            _speculation_grad_norm(outer_scale=0.2, inner_scale=value, output_scale=0.2)
            for value in _SWEEP
        ],
        "o_proj_norm": [
            _speculation_grad_norm(outer_scale=0.2, inner_scale=0.2, output_scale=value)
            for value in _SWEEP
        ],
    }
    slopes = {name: _fit_log_slope(_SWEEP, values) for name, values in curves.items()}
    for name, slope in slopes.items():
        if not 0.85 <= slope <= 1.15:
            raise AssertionError(f"Escape-law slope {name}={slope:.6f} is outside 1.0 +/- 0.15.")
    return curves, slopes


def _ce_from_notes(
    snc: SharedNotesCrossAttention,
    hidden: torch.Tensor,
    notes: torch.Tensor,
    classifier: nn.Linear,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = torch.ones(notes.shape[:2], dtype=torch.bool, device=notes.device)
    delta = snc(hidden, notes, notes_mask=mask, force_gate=True)
    logits = classifier((hidden + delta).mean(dim=1))
    return F.cross_entropy(logits, labels)


def paired_null_deltas() -> tuple[float, float]:
    """Return the §16 structural null and order-dependent positive control.

    Producer labels are deliberately absent from the current unaddressed SNC
    operator. The delivery control applies the old mutable-window operation
    itself: take the last four delivered notes. This is a scientific control,
    not a retained runtime path.
    """
    torch.manual_seed(_SEED)
    snc = SharedNotesCrossAttention(
        SNCConfig(hidden_size=_HIDDEN, notes_dim=_NOTES, num_heads=4, dropout=0.0),
        gating_init=-4.0,
    ).to(dtype=_DTYPE)
    classifier = nn.Linear(_HIDDEN, _VOCAB).to(dtype=_DTYPE)
    with torch.no_grad():
        snc.o_proj.weight.normal_(mean=0.0, std=0.1)
        snc.o_proj.bias.zero_()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_SEED + 1)
    hidden = torch.randn(1, _SEQUENCE, _HIDDEN, generator=generator, dtype=_DTYPE)
    delivered = torch.randn(1, 6, _NOTES, generator=generator, dtype=_DTYPE)
    labels = torch.tensor([3], dtype=torch.long)

    # Producer attribution is metadata the unaddressed operator never receives.
    producer_order = (0, 1, 2, 0, 1, 2)
    permuted_attribution = (2, 0, 1, 2, 0, 1)
    if sorted(producer_order) != sorted(permuted_attribution):
        raise AssertionError("Producer permutation must preserve attribution counts.")
    ce_original = _ce_from_notes(snc, hidden, delivered[:, :4], classifier, labels)
    ce_permuted = _ce_from_notes(snc, hidden, delivered[:, :4], classifier, labels)
    permutation_delta = float((ce_permuted - ce_original).abs().item())

    schedule_a = delivered[:, -4:]
    reordered_indices = torch.tensor([4, 5, 0, 1, 2, 3])
    schedule_b = delivered.index_select(1, reordered_indices)[:, -4:]
    ce_schedule_a = _ce_from_notes(snc, hidden, schedule_a, classifier, labels)
    ce_schedule_b = _ce_from_notes(snc, hidden, schedule_b, classifier, labels)
    reorder_delta = float((ce_schedule_b - ce_schedule_a).abs().item())

    if permutation_delta != 0.0:
        raise AssertionError(
            f"Producer-permutation delta CE must be exactly zero, got {permutation_delta}."
        )
    if reorder_delta <= 0.0:
        raise AssertionError("Delivery-reorder delta CE must be positive.")
    return permutation_delta, reorder_delta


def run_calibration() -> tuple[CalibrationResult, dict[str, list[float]]]:
    gradient_norms = step_zero_gradient_norms()
    curves, slopes = escape_law_curves()
    permutation_delta, reorder_delta = paired_null_deltas()
    two_factor_scale = torch.sigmoid(torch.tensor(-4.0, dtype=_DTYPE)).item()
    initial_scale = two_factor_scale**2
    suppression_ratio = two_factor_scale / initial_scale
    plateau_ratio = suppression_ratio**2
    result = CalibrationResult(
        gradient_norms=gradient_norms,
        slopes=slopes,
        initial_scale=initial_scale,
        two_factor_scale=two_factor_scale,
        suppression_ratio=suppression_ratio,
        plateau_ratio=plateau_ratio,
        producer_permutation_delta_ce=permutation_delta,
        delivery_reorder_delta_ce=reorder_delta,
    )
    return result, curves


def _write_table(path: Path, rows: Iterable[tuple[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def _write_figure(path: Path, curves: dict[str, list[float]], slopes: dict[str, float]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    labels = {
        "outer_gate": "sigmoid(g_outer)",
        "inner_gate": "sigmoid(g_inner)",
        "o_proj_norm": "W_o scale",
    }
    for axis, (name, values) in zip(axes, curves.items(), strict=True):
        axis.loglog(_SWEEP, values, marker="o")
        axis.set_xlabel(labels[name])
        axis.set_ylabel("speculation grad norm")
        axis.set_title(f"slope={slopes[name]:.4f}")
        axis.grid(True, which="both", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_calibration(output_dir: Path) -> CalibrationResult:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Calibration output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result, curves = run_calibration()
    _write_table(output_dir / "table1_gradient_norms.csv", result.gradient_norms.items())
    _write_figure(output_dir / "figure1_escape_law.png", curves, result.slopes)
    _write_table(
        output_dir / "table2_paired_nulls.csv",
        (
            ("producer_permutation_delta_ce", result.producer_permutation_delta_ce),
            ("delivery_reorder_delta_ce", result.delivery_reorder_delta_ce),
        ),
    )
    summary = {
        "slopes": result.slopes,
        "initial_scale": result.initial_scale,
        "two_factor_scale": result.two_factor_scale,
        "suppression_ratio": result.suppression_ratio,
        "plateau_ratio": result.plateau_ratio,
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the two tables, one figure, and numeric summary.",
    )
    args = parser.parse_args()
    result = write_calibration(args.output_dir)
    print(
        json.dumps(
            {
                "gradient_norms": result.gradient_norms,
                "slopes": result.slopes,
                "initial_scale": result.initial_scale,
                "suppression_ratio": result.suppression_ratio,
                "plateau_ratio": result.plateau_ratio,
                "producer_permutation_delta_ce": result.producer_permutation_delta_ce,
                "delivery_reorder_delta_ce": result.delivery_reorder_delta_ce,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
