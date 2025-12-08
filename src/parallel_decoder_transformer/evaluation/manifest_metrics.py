"""Telemetry post-processing helpers for inference manifests."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_ROLLBACK_BINS: Tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class PlanEntry:
    """Canonicalised planner entry derived from a manifest or override map."""

    stream: str
    text: str
    index: Optional[int]
    plan_item_id: Optional[int]
    position: int

    def key(self) -> Tuple[str, int]:
        index = self.index if self.index is not None else self.position
        return self.stream, index


@dataclass(frozen=True)
class AgreementStats:
    """Sufficient statistics for recomputing Pearson correlation."""

    count: int
    sum_agreement: float
    sum_agreement_sq: float
    sum_flag: float
    sum_flag_sq: float
    sum_cross: float


@dataclass
class RollbackSummary:
    """Distributional summary over rollback events."""

    stride_count: int
    token_count: int
    total_events: int
    total_tokens_removed: int
    per_stream_events: Dict[str, int]
    per_stream_tokens: Dict[str, int]
    histogram: Dict[str, int]
    length_p50: Optional[float]
    length_p95: Optional[float]
    per_stride_rate: Optional[float]
    per_stream_rate: Dict[str, Optional[float]]
    agreement_correlation: Optional[float]
    length_samples: List[int] = field(default_factory=list)
    agreement_stats: Optional[AgreementStats] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stride_count": self.stride_count,
            "token_count": self.token_count,
            "total_events": self.total_events,
            "total_tokens_removed": self.total_tokens_removed,
            "per_stream_events": dict(self.per_stream_events),
            "per_stream_tokens": dict(self.per_stream_tokens),
            "histogram": dict(self.histogram),
            "length_p50": self.length_p50,
            "length_p95": self.length_p95,
            "per_stride_rate": self.per_stride_rate,
            "per_stream_rate": dict(self.per_stream_rate),
            "agreement_correlation": self.agreement_correlation,
        }


@dataclass
class GateStreamSummary:
    """Gate statistics for one stream."""

    stream: str
    count: int
    mean: Optional[float]
    stddev: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    entropy: Optional[float]
    high_fraction: Optional[float]
    low_fraction: Optional[float]
    dwell_high_mean: Optional[float]
    dwell_low_mean: Optional[float]
    dwell_mid_mean: Optional[float]
    oscillation_count: int
    oscillation_rate: Optional[float]
    thresholds: Tuple[float, float]
    series: List[float] = field(default_factory=list)

    def to_payload(self, *, max_points: int = 512) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "count": self.count,
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.minimum,
            "max": self.maximum,
            "entropy": self.entropy,
            "high_fraction": self.high_fraction,
            "low_fraction": self.low_fraction,
            "dwell_high_mean": self.dwell_high_mean,
            "dwell_low_mean": self.dwell_low_mean,
            "dwell_mid_mean": self.dwell_mid_mean,
            "oscillation_count": self.oscillation_count,
            "oscillation_rate": self.oscillation_rate,
            "thresholds": {
                "low": self.thresholds[0],
                "high": self.thresholds[1],
            },
            "series": _downsample_series(self.series, max_points=max_points),
        }


@dataclass
class GateSummary:
    """Wrapper for per-stream gate summaries."""

    per_stream: Dict[str, GateStreamSummary]

    def to_payload(self, *, max_points: int = 512) -> Dict[str, Any]:
        return {
            "per_stream": {
                stream: summary.to_payload(max_points=max_points)
                for stream, summary in self.per_stream.items()
            }
        }


@dataclass
class CoverageStreamSummary:
    """Coverage precision/recall metrics for a single stream."""

    stream: str
    tp: float
    fp: float
    fn: float
    support: int
    evaluated: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    source: str
    missing_predictions: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "evaluated": self.evaluated,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "source": self.source,
            "missing_predictions": self.missing_predictions,
        }


@dataclass
class CoverageSummary:
    """Aggregate coverage metrics."""

    total_tp: float
    total_fp: float
    total_fn: float
    support: int
    evaluated: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    per_stream: Dict[str, CoverageStreamSummary]
    source: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "support": self.support,
            "evaluated": self.evaluated,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "source": self.source,
            "per_stream": {
                stream: summary.to_payload() for stream, summary in self.per_stream.items()
            },
        }


@dataclass
class ManifestMetrics:
    """Container for per-manifest summaries."""

    path: str
    rollback: RollbackSummary
    gate: GateSummary
    coverage: CoverageSummary

    def to_payload(self, *, gate_series_max_points: int = 512) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rollback": self.rollback.to_payload(),
            "gate": self.gate.to_payload(max_points=gate_series_max_points),
            "coverage": self.coverage.to_payload(),
        }


@dataclass
class AggregateMetrics:
    """Cross-manifest aggregation results."""

    count: int
    rollback: RollbackSummary
    gate: GateSummary
    coverage: CoverageSummary

    def to_payload(self, *, gate_series_max_points: int = 512) -> Dict[str, Any]:
        return {
            "manifest_count": self.count,
            "rollback": self.rollback.to_payload(),
            "gate": self.gate.to_payload(max_points=gate_series_max_points),
            "coverage": self.coverage.to_payload(),
        }


def summarize_manifest(
    manifest: Mapping[str, Any],
    *,
    path: str,
    plan_text_override: Optional[Mapping[str, Sequence[str]]] = None,
    coverage_partial_threshold: float = 0.4,
    gate_low_threshold: float = 0.25,
    gate_high_threshold: float = 0.75,
    gate_entropy_bins: int = 20,
) -> ManifestMetrics:
    """Build summaries for a single manifest."""

    rollback = compute_rollback_summary(manifest)
    gate = compute_gate_summary(
        manifest,
        gate_low_threshold=gate_low_threshold,
        gate_high_threshold=gate_high_threshold,
        entropy_bins=gate_entropy_bins,
    )
    coverage = compute_coverage_summary(
        manifest,
        plan_text_override=plan_text_override,
        coverage_partial_threshold=coverage_partial_threshold,
    )
    return ManifestMetrics(path=path, rollback=rollback, gate=gate, coverage=coverage)


def aggregate_metrics(metrics: Sequence[ManifestMetrics]) -> AggregateMetrics:
    """Aggregate summaries across manifests."""

    if not metrics:
        empty = RollbackSummary(
            stride_count=0,
            token_count=0,
            total_events=0,
            total_tokens_removed=0,
            per_stream_events={},
            per_stream_tokens={},
            histogram={},
            length_p50=None,
            length_p95=None,
            per_stride_rate=None,
            per_stream_rate={},
            agreement_correlation=None,
        )
        return AggregateMetrics(
            count=0,
            rollback=empty,
            gate=GateSummary(per_stream={}),
            coverage=CoverageSummary(
                total_tp=0.0,
                total_fp=0.0,
                total_fn=0.0,
                support=0,
                evaluated=0,
                precision=None,
                recall=None,
                f1=None,
                per_stream={},
                source="none",
            ),
        )
    roll_lengths: List[int] = []
    roll_hist = Counter()
    roll_per_stream_events: Counter[str] = Counter()
    roll_per_stream_tokens: Counter[str] = Counter()
    total_stride_count = 0
    total_token_count = 0
    total_events = 0
    total_tokens_removed = 0
    agreement_stats = AgreementStats(
        count=0,
        sum_agreement=0.0,
        sum_agreement_sq=0.0,
        sum_flag=0.0,
        sum_flag_sq=0.0,
        sum_cross=0.0,
    )
    coverage_counts = Counter()
    coverage_tp = 0.0
    coverage_fp = 0.0
    coverage_fn = 0.0
    coverage_support = 0
    coverage_evaluated = 0
    coverage_source = "logits"
    coverage_per_stream: Dict[str, CoverageStreamSummary] = {}
    gate_series: Dict[str, List[float]] = defaultdict(list)
    for item in metrics:
        roll = item.rollback
        roll_lengths.extend(roll.length_samples)
        roll_hist.update(roll.histogram)
        total_stride_count += roll.stride_count
        total_token_count += roll.token_count
        total_events += roll.total_events
        total_tokens_removed += roll.total_tokens_removed
        roll_per_stream_events.update(roll.per_stream_events)
        roll_per_stream_tokens.update(roll.per_stream_tokens)
        if roll.agreement_stats:
            stats = roll.agreement_stats
            agreement_stats = AgreementStats(
                count=agreement_stats.count + stats.count,
                sum_agreement=agreement_stats.sum_agreement + stats.sum_agreement,
                sum_agreement_sq=agreement_stats.sum_agreement_sq + stats.sum_agreement_sq,
                sum_flag=agreement_stats.sum_flag + stats.sum_flag,
                sum_flag_sq=agreement_stats.sum_flag_sq + stats.sum_flag_sq,
                sum_cross=agreement_stats.sum_cross + stats.sum_cross,
            )
        cov = item.coverage
        coverage_tp += cov.total_tp
        coverage_fp += cov.total_fp
        coverage_fn += cov.total_fn
        coverage_support += cov.support
        coverage_evaluated += cov.evaluated
        coverage_counts[cov.source] += 1
        for stream, summary in cov.per_stream.items():
            existing = coverage_per_stream.get(stream)
            if existing is None:
                coverage_per_stream[stream] = summary
            else:
                merged = _merge_stream_coverage(existing, summary)
                coverage_per_stream[stream] = merged
        for stream, summary in item.gate.per_stream.items():
            gate_series[stream].extend(summary.series)
    roll_hist_dict = dict(roll_hist)
    length_p50 = _percentile(roll_lengths, 50.0) if roll_lengths else None
    length_p95 = _percentile(roll_lengths, 95.0) if roll_lengths else None
    per_stride_rate = (
        float(total_events) / float(total_stride_count) if total_stride_count > 0 else None
    )
    per_stream_rate = {
        stream: (count / total_stride_count if total_stride_count > 0 else None)
        for stream, count in roll_per_stream_events.items()
    }
    correlation = _pearson_from_stats(agreement_stats) if agreement_stats.count > 1 else None
    aggregated_roll = RollbackSummary(
        stride_count=total_stride_count,
        token_count=total_token_count,
        total_events=total_events,
        total_tokens_removed=total_tokens_removed,
        per_stream_events=dict(roll_per_stream_events),
        per_stream_tokens=dict(roll_per_stream_tokens),
        histogram=roll_hist_dict,
        length_p50=length_p50,
        length_p95=length_p95,
        per_stride_rate=per_stride_rate,
        per_stream_rate=per_stream_rate,
        agreement_correlation=correlation,
        length_samples=roll_lengths,
        agreement_stats=agreement_stats,
    )
    coverage_precision, coverage_recall, coverage_f1 = _compute_prf(
        coverage_tp, coverage_fp, coverage_fn
    )
    coverage_source = (
        "logits"
        if coverage_counts.get("logits", 0) == len(metrics)
        else ("text" if coverage_counts.get("logits", 0) == 0 else "mixed")
    )
    aggregated_cov = CoverageSummary(
        total_tp=coverage_tp,
        total_fp=coverage_fp,
        total_fn=coverage_fn,
        support=coverage_support,
        evaluated=coverage_evaluated,
        precision=coverage_precision,
        recall=coverage_recall,
        f1=coverage_f1,
        per_stream={stream: summary for stream, summary in coverage_per_stream.items()},
        source=coverage_source,
    )
    aggregated_gate = GateSummary(
        per_stream={
            stream: _build_gate_summary_for_series(stream, series)
            for stream, series in gate_series.items()
        }
    )
    return AggregateMetrics(
        count=len(metrics),
        rollback=aggregated_roll,
        gate=aggregated_gate,
        coverage=aggregated_cov,
    )


def compute_rollback_summary(manifest: Mapping[str, Any]) -> RollbackSummary:
    """Collect rollback distribution statistics."""

    timings = manifest.get("timings", {})
    stride_durations = _resolve_stride_durations(timings)
    stride_count = len(stride_durations)
    token_count = _count_tokens(manifest)
    rollbacks = manifest.get("rollbacks", [])
    per_stream_events: Counter[str] = Counter()
    per_stream_tokens: Counter[str] = Counter()
    lengths: List[int] = []
    total_tokens_removed = 0
    for entry in rollbacks:
        if not isinstance(entry, Mapping):
            continue
        stream = _normalize_stream(entry.get("stream", ""))
        tokens_removed = entry.get("tokens_removed")
        if isinstance(tokens_removed, Sequence):
            length = len(tokens_removed)
        else:
            length = 0
        per_stream_events[stream] += 1
        per_stream_tokens[stream] += length
        total_tokens_removed += length
        lengths.append(length)
    histogram = _build_histogram(lengths, bins=_DEFAULT_ROLLBACK_BINS)
    per_stride_rate = (sum(per_stream_events.values()) / stride_count) if stride_count > 0 else None
    per_stream_rate = {
        stream: (count / stride_count if stride_count > 0 else None)
        for stream, count in per_stream_events.items()
    }
    agreements, flags, stats = _collect_agreement_series(manifest)
    correlation = _pearson_from_samples(agreements, flags)
    return RollbackSummary(
        stride_count=stride_count,
        token_count=token_count,
        total_events=sum(per_stream_events.values()),
        total_tokens_removed=total_tokens_removed,
        per_stream_events=dict(per_stream_events),
        per_stream_tokens=dict(per_stream_tokens),
        histogram=histogram,
        length_p50=_percentile(lengths, 50.0) if lengths else None,
        length_p95=_percentile(lengths, 95.0) if lengths else None,
        per_stride_rate=per_stride_rate,
        per_stream_rate=per_stream_rate,
        agreement_correlation=correlation,
        length_samples=lengths,
        agreement_stats=stats,
    )


def compute_gate_summary(
    manifest: Mapping[str, Any],
    *,
    gate_low_threshold: float = 0.25,
    gate_high_threshold: float = 0.75,
    entropy_bins: int = 20,
) -> GateSummary:
    """Compute gate statistics per stream."""

    gate_trace = manifest.get("gate_trace", [])
    per_stream_values: Dict[str, List[float]] = defaultdict(list)
    for entry in gate_trace:
        if not isinstance(entry, Mapping):
            continue
        stream = _normalize_stream(entry.get("stream", ""))
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            continue
        per_stream_values[stream].append(value)
    summaries = {
        stream: _build_gate_summary_for_series(
            stream,
            values,
            gate_low_threshold=gate_low_threshold,
            gate_high_threshold=gate_high_threshold,
            entropy_bins=entropy_bins,
        )
        for stream, values in per_stream_values.items()
    }
    return GateSummary(per_stream=summaries)


def compute_coverage_summary(
    manifest: Mapping[str, Any],
    *,
    plan_text_override: Optional[Mapping[str, Sequence[str]]] = None,
    coverage_partial_threshold: float = 0.4,
) -> CoverageSummary:
    """Compute coverage precision/recall metrics per stream."""

    plan_entries = _collect_plan_entries(manifest, override=plan_text_override)
    if not plan_entries:
        return CoverageSummary(
            total_tp=0.0,
            total_fp=0.0,
            total_fn=0.0,
            support=0,
            evaluated=0,
            precision=None,
            recall=None,
            f1=None,
            per_stream={},
            source="none",
        )
    notes_text = _collect_stream_notes(manifest)
    coverage_threshold = _resolve_coverage_threshold(manifest)
    observed = _compute_observed_coverage(
        plan_entries, notes_text, overlap_threshold=coverage_partial_threshold
    )
    predictions = _collect_coverage_predictions(manifest)
    per_stream: Dict[str, CoverageStreamSummary] = {}
    total_tp = total_fp = total_fn = 0.0
    total_support = 0
    total_evaluated = 0
    saw_logits = False
    for stream, entries in plan_entries.items():
        summary = _score_stream_coverage(
            stream,
            entries,
            observed,
            predictions,
            coverage_threshold=coverage_threshold,
        )
        per_stream[stream] = summary
        total_tp += summary.tp
        total_fp += summary.fp
        total_fn += summary.fn
        total_support += summary.support
        total_evaluated += summary.evaluated
        if summary.source == "logits":
            saw_logits = True
    precision, recall, f1 = _compute_prf(total_tp, total_fp, total_fn)
    source = "logits" if saw_logits else "text"
    return CoverageSummary(
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        support=total_support,
        evaluated=total_evaluated,
        precision=precision,
        recall=recall,
        f1=f1,
        per_stream=per_stream,
        source=source,
    )


def _merge_stream_coverage(
    current: CoverageStreamSummary,
    incoming: CoverageStreamSummary,
) -> CoverageStreamSummary:
    tp = current.tp + incoming.tp
    fp = current.fp + incoming.fp
    fn = current.fn + incoming.fn
    support = current.support + incoming.support
    evaluated = current.evaluated + incoming.evaluated
    precision, recall, f1 = _compute_prf(tp, fp, fn)
    source = "logits" if current.source == "logits" and incoming.source == "logits" else "mixed"
    return CoverageStreamSummary(
        stream=current.stream,
        tp=tp,
        fp=fp,
        fn=fn,
        support=support,
        evaluated=evaluated,
        precision=precision,
        recall=recall,
        f1=f1,
        source=source,
        missing_predictions=current.missing_predictions + incoming.missing_predictions,
    )


def _build_gate_summary_for_series(
    stream: str,
    series: Sequence[float],
    *,
    gate_low_threshold: float = 0.25,
    gate_high_threshold: float = 0.75,
    entropy_bins: int = 20,
) -> GateStreamSummary:
    if not series:
        return GateStreamSummary(
            stream=stream,
            count=0,
            mean=None,
            stddev=None,
            minimum=None,
            maximum=None,
            entropy=None,
            high_fraction=None,
            low_fraction=None,
            dwell_high_mean=None,
            dwell_low_mean=None,
            dwell_mid_mean=None,
            oscillation_count=0,
            oscillation_rate=None,
            thresholds=(gate_low_threshold, gate_high_threshold),
            series=list(series),
        )
    values = [float(value) for value in series]
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count if count > 0 else 0.0
    stddev = math.sqrt(variance)
    entropy = _shannon_entropy(values, bins=max(2, entropy_bins))
    states = [_gate_state(value, gate_low_threshold, gate_high_threshold) for value in values]
    high_fraction = states.count("high") / count
    low_fraction = states.count("low") / count
    dwell_stats = _dwell_statistics(states)
    oscillations = _oscillation_count(states)
    oscillation_rate = oscillations / max(1, count - 1)
    return GateStreamSummary(
        stream=stream,
        count=count,
        mean=mean,
        stddev=stddev,
        minimum=min(values),
        maximum=max(values),
        entropy=entropy,
        high_fraction=high_fraction,
        low_fraction=low_fraction,
        dwell_high_mean=dwell_stats.get("high"),
        dwell_low_mean=dwell_stats.get("low"),
        dwell_mid_mean=dwell_stats.get("mid"),
        oscillation_count=oscillations,
        oscillation_rate=oscillation_rate,
        thresholds=(gate_low_threshold, gate_high_threshold),
        series=list(values),
    )


def _compute_observed_coverage(
    plan_entries: Mapping[str, List[PlanEntry]],
    notes_text: Mapping[str, List[str]],
    *,
    overlap_threshold: float,
) -> Dict[Tuple[str, int], float]:
    observed: Dict[Tuple[str, int], float] = {}
    for stream, entries in plan_entries.items():
        stream_notes = notes_text.get(stream, [])
        if not stream_notes:
            for entry in entries:
                observed[entry.key()] = 0.0
            continue
        combined = " ".join(stream_notes).lower()
        for entry in entries:
            plan_text = entry.text.lower()
            score = 0.0
            if plan_text and plan_text in combined:
                score = 1.0
            else:
                overlap = _token_overlap_ratio(stream_notes, entry.text)
                if overlap >= 1.0:
                    score = 1.0
                elif overlap >= overlap_threshold:
                    score = 0.5
            observed[entry.key()] = score
    return observed


def _score_stream_coverage(
    stream: str,
    entries: Sequence[PlanEntry],
    observed: Mapping[Tuple[str, int], float],
    predictions: Dict[str, Dict[str, Dict[Tuple[str, int], float]]],
    *,
    coverage_threshold: float,
) -> CoverageStreamSummary:
    tp = fp = fn = 0.0
    support = len(entries)
    evaluated = 0
    missing_predictions = 0
    source = "text"
    for entry in entries:
        obs = observed.get(entry.key())
        if obs is None:
            continue
        evaluated += 1
        obs_positive = obs >= coverage_threshold
        pred_value = _lookup_prediction(entry, predictions)
        if pred_value is None:
            missing_predictions += 1
            pred_positive = None
        else:
            pred_positive = pred_value >= coverage_threshold
            source = "logits"
        if pred_positive is None:
            pred_positive = obs_positive
            if source != "logits":
                source = "text"
        if obs_positive and pred_positive:
            tp += 1.0
        elif obs_positive and not pred_positive:
            fn += 1.0
        elif (not obs_positive) and pred_positive:
            fp += 1.0
    precision, recall, f1 = _compute_prf(tp, fp, fn)
    return CoverageStreamSummary(
        stream=stream,
        tp=tp,
        fp=fp,
        fn=fn,
        support=support,
        evaluated=evaluated,
        precision=precision,
        recall=recall,
        f1=f1,
        source=source,
        missing_predictions=missing_predictions,
    )


def _lookup_prediction(
    entry: PlanEntry,
    predictions: Dict[str, Dict[str, Dict[Tuple[str, int], float]]],
) -> Optional[float]:
    stream = entry.stream
    index_key = entry.key()
    plan_id_key = (stream, entry.plan_item_id) if entry.plan_item_id is not None else None
    position_key = (stream, entry.position)
    for group in ("index", "plan_id", "position"):
        candidate_map = predictions.get(group, {})
        if group == "index":
            value = candidate_map.get(index_key)
        elif group == "plan_id" and plan_id_key is not None:
            value = candidate_map.get(plan_id_key)
        else:
            value = candidate_map.get(position_key)
        if value is not None:
            return value
    return None


def _collect_coverage_predictions(
    manifest: Mapping[str, Any],
) -> Dict[str, Dict[Tuple[str, int], float]]:
    predictions: Dict[str, Dict[Tuple[str, int], float]] = {
        "index": {},
        "plan_id": {},
        "position": {},
    }
    streams = manifest.get("streams", {})
    for stream_name, payload in streams.items():
        if not isinstance(payload, Mapping):
            continue
        stream = _normalize_stream(stream_name)
        coverage = payload.get("coverage", {})
        if not isinstance(coverage, Mapping):
            continue
        plan_items = coverage.get("plan_items")
        if isinstance(plan_items, Sequence):
            for position, item in enumerate(plan_items):
                if not isinstance(item, Mapping):
                    continue
                predicted = _coerce_status_value(item)
                if predicted is None:
                    continue
                idx = item.get("index")
                plan_item_id = item.get("plan_item_id")
                target_stream = _normalize_stream(item.get("stream", stream))
                key_index = (target_stream, int(idx)) if idx is not None else None
                key_plan = (target_stream, int(plan_item_id)) if plan_item_id is not None else None
                key_position = (target_stream, position)
                if key_index is not None:
                    predictions["index"][key_index] = predicted
                if key_plan is not None:
                    predictions["plan_id"][key_plan] = predicted
                predictions["position"][key_position] = predicted
    return predictions


def _coerce_status_value(item: Mapping[str, Any]) -> Optional[float]:
    status = str(item.get("status", "")).strip().lower()
    if status == "covered":
        return 1.0
    if status == "partial":
        return 0.5
    if status == "missing":
        return 0.0
    probability = item.get("probability")
    if isinstance(probability, (int, float)):
        prob = float(probability)
        if 0.0 <= prob <= 1.0:
            return prob
    return None


def _collect_plan_entries(
    manifest: Mapping[str, Any],
    *,
    override: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, List[PlanEntry]]:
    if override:
        plan_entries: Dict[str, List[PlanEntry]] = {}
        for stream, values in override.items():
            normalized_stream = _normalize_stream(stream)
            plan_entries[normalized_stream] = [
                PlanEntry(
                    stream=normalized_stream,
                    text=str(text),
                    index=idx,
                    plan_item_id=None,
                    position=idx,
                )
                for idx, text in enumerate(values)
                if isinstance(text, str) and text.strip()
            ]
        return plan_entries
    plan_section = manifest.get("plan", {})
    catalog = plan_section.get("catalog")
    entries: Dict[str, List[PlanEntry]] = defaultdict(list)
    if isinstance(catalog, Sequence):
        ordered = sorted(
            (
                item
                for item in catalog
                if isinstance(item, Mapping) and isinstance(item.get("text"), str)
            ),
            key=lambda item: int(item.get("index", 0)),
        )
        per_stream_position: Dict[str, int] = defaultdict(int)
        for item in ordered:
            stream = _normalize_stream(item.get("stream", ""))
            if not stream:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            index_value = item.get("index")
            plan_item_id = item.get("plan_item_id")
            position = per_stream_position[stream]
            per_stream_position[stream] += 1
            entries[stream].append(
                PlanEntry(
                    stream=stream,
                    text=text,
                    index=int(index_value) if index_value is not None else None,
                    plan_item_id=int(plan_item_id) if plan_item_id is not None else None,
                    position=position,
                )
            )
        if entries:
            return dict(entries)
    streams = manifest.get("streams", {})
    fallback: Dict[str, List[PlanEntry]] = defaultdict(list)
    for stream_name, payload in streams.items():
        if not isinstance(payload, Mapping):
            continue
        stream = _normalize_stream(stream_name)
        coverage = payload.get("coverage", {})
        plan_items = coverage.get("plan_items")
        if not isinstance(plan_items, Sequence):
            continue
        for position, item in enumerate(plan_items):
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            index_value = item.get("index")
            plan_item_id = item.get("plan_item_id")
            fallback[stream].append(
                PlanEntry(
                    stream=stream,
                    text=text,
                    index=int(index_value) if index_value is not None else None,
                    plan_item_id=int(plan_item_id) if plan_item_id is not None else None,
                    position=position,
                )
            )
    return dict(fallback)


def _collect_stream_notes(manifest: Mapping[str, Any]) -> Dict[str, List[str]]:
    streams_section = manifest.get("streams", {})
    notes: Dict[str, List[str]] = defaultdict(list)
    if isinstance(streams_section, Mapping):
        for stream_name, payload in streams_section.items():
            if not isinstance(payload, Mapping):
                continue
            stream = _normalize_stream(stream_name)
            text_value = payload.get("text")
            if isinstance(text_value, str) and text_value.strip():
                notes[stream].append(text_value.strip())
            reference = payload.get("reference_notes")
            if isinstance(reference, Sequence):
                for item in reference:
                    if isinstance(item, str) and item.strip():
                        notes[stream].append(item.strip())
    reference_root = manifest.get("reference_notes")
    if isinstance(reference_root, Mapping):
        for stream_name, entries in reference_root.items():
            stream = _normalize_stream(stream_name)
            if isinstance(entries, Sequence):
                for item in entries:
                    if isinstance(item, str) and item.strip():
                        notes[stream].append(item.strip())
    return dict(notes)


def _resolve_stride_durations(timings: Mapping[str, Any]) -> List[float]:
    durations = timings.get("stride_durations")
    if isinstance(durations, Sequence) and durations:
        resolved: List[float] = []
        for value in durations:
            try:
                resolved.append(float(value))
            except (TypeError, ValueError):
                continue
        if resolved:
            return resolved
    per_token = timings.get("per_token")
    if not isinstance(per_token, Sequence):
        return []
    aggregates: Dict[int, float] = defaultdict(float)
    for entry in per_token:
        if not isinstance(entry, Mapping):
            continue
        try:
            stride_index = int(entry.get("stride_index"))
        except (TypeError, ValueError):
            continue
        try:
            aggregates[stride_index] += float(entry.get("duration_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return [aggregates[key] for key in sorted(aggregates)]


def _count_tokens(manifest: Mapping[str, Any]) -> int:
    streams = manifest.get("streams", {})
    if not isinstance(streams, Mapping):
        return 0
    total = 0
    for payload in streams.values():
        if not isinstance(payload, Mapping):
            continue
        tokens = payload.get("token_ids", [])
        if isinstance(tokens, Sequence):
            total += len(tokens)
    return total


def _collect_agreement_series(
    manifest: Mapping[str, Any],
) -> Tuple[List[float], List[float], AgreementStats]:
    events = manifest.get("events", [])
    agreements: List[float] = []
    flags: List[float] = []
    if isinstance(events, Sequence):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            try:
                agreement = float(event.get("agreement"))
            except (TypeError, ValueError):
                continue
            rollback_performed = bool(event.get("rollback_performed"))
            agreements.append(agreement)
            flags.append(1.0 if rollback_performed else 0.0)
    stats = AgreementStats(
        count=len(agreements),
        sum_agreement=sum(agreements),
        sum_agreement_sq=sum(value * value for value in agreements),
        sum_flag=sum(flags),
        sum_flag_sq=sum(value * value for value in flags),
        sum_cross=sum(a * b for a, b in zip(agreements, flags)),
    )
    return agreements, flags, stats


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty sequence.")
    if q <= 0.0:
        return float(min(values))
    if q >= 100.0:
        return float(max(values))
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * (q / 100.0)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return float(lower_value)
    weight = position - lower_index
    return float(lower_value * (1.0 - weight) + upper_value * weight)


def _build_histogram(lengths: Sequence[int], bins: Sequence[int]) -> Dict[str, int]:
    if not lengths:
        return {}
    histogram: Dict[str, int] = {}
    counts = Counter(lengths)
    cumulative = 0
    for edge in bins:
        bucket_total = sum(count for length, count in counts.items() if length <= edge)
        histogram[f"<= {edge}"] = bucket_total - cumulative
        cumulative = bucket_total
    histogram[f"> {bins[-1]}"] = sum(count for length, count in counts.items() if length > bins[-1])
    return histogram


def _pearson_from_samples(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    var_x = sum((x - mean_x) ** 2 for x in x_values)
    var_y = sum((y - mean_y) ** 2 for y in y_values)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _pearson_from_stats(stats: AgreementStats) -> Optional[float]:
    if stats.count < 2:
        return None
    mean_x = stats.sum_agreement / stats.count
    mean_y = stats.sum_flag / stats.count
    cov = stats.sum_cross - stats.count * mean_x * mean_y
    var_x = stats.sum_agreement_sq - stats.count * mean_x * mean_x
    var_y = stats.sum_flag_sq - stats.count * mean_y * mean_y
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _downsample_series(series: Sequence[float], *, max_points: int) -> List[float]:
    if max_points <= 0 or len(series) <= max_points:
        return list(series)
    step = len(series) / max_points
    return [series[int(i * step)] for i in range(max_points)]


def _shannon_entropy(values: Sequence[float], *, bins: int) -> Optional[float]:
    if not values or bins <= 0:
        return None
    histogram = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int(value * bins)))
        histogram[index] += 1
    total = sum(histogram)
    if total == 0:
        return None
    entropy = 0.0
    for count in histogram:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return entropy


def _gate_state(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return "mid"


def _dwell_statistics(states: Sequence[str]) -> Dict[str, Optional[float]]:
    if not states:
        return {"high": None, "low": None, "mid": None}
    dwell_lengths: Dict[str, List[int]] = defaultdict(list)
    current_state = states[0]
    current_length = 1
    for state in states[1:]:
        if state == current_state:
            current_length += 1
            continue
        dwell_lengths[current_state].append(current_length)
        current_state = state
        current_length = 1
    dwell_lengths[current_state].append(current_length)
    return {
        state: (sum(lengths) / len(lengths) if lengths else None)
        for state, lengths in (
            ("high", dwell_lengths.get("high", [])),
            ("low", dwell_lengths.get("low", [])),
            ("mid", dwell_lengths.get("mid", [])),
        )
    }


def _oscillation_count(states: Sequence[str]) -> int:
    transitions = 0
    for prev, current in zip(states, states[1:]):
        if prev in {"high", "low"} and current in {"high", "low"} and prev != current:
            transitions += 1
    return transitions


def _token_overlap_ratio(notes: Sequence[str], plan_text: str) -> float:
    plan_tokens = set(_TOKEN_PATTERN.findall(plan_text.lower()))
    if not plan_tokens:
        return 0.0
    note_tokens: set[str] = set()
    for text in notes:
        note_tokens.update(_TOKEN_PATTERN.findall(text.lower()))
    if not note_tokens:
        return 0.0
    return len(plan_tokens & note_tokens) / len(plan_tokens)


def _normalize_stream(stream: Any) -> str:
    return str(stream or "").strip().lower()


def _resolve_coverage_threshold(manifest: Mapping[str, Any]) -> float:
    config = manifest.get("config", {})
    threshold = config.get("coverage_threshold")
    try:
        value = float(threshold)
        if 0.0 < value < 1.0:
            return value
    except (TypeError, ValueError):
        pass
    return 0.5


def _compute_prf(
    tp: float, fp: float, fn: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0.0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


__all__ = [
    "AggregateMetrics",
    "CoverageStreamSummary",
    "CoverageSummary",
    "GateStreamSummary",
    "GateSummary",
    "ManifestMetrics",
    "PlanEntry",
    "RollbackSummary",
    "aggregate_metrics",
    "compute_coverage_summary",
    "compute_gate_summary",
    "compute_rollback_summary",
    "summarize_manifest",
]
