#!/usr/bin/env python
"""CPU-only logit replay entrypoint for reproducing orchestrator telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from parallel_decoder_transformer.inference import MultiStreamOrchestrator, StepOutcome
from parallel_decoder_transformer.inference.replay import LogitReplayArtifact, ReplayModel
from parallel_decoder_transformer.data.tokenizer import resolve_tokenizer
from parallel_decoder_transformer.utils import configure_logging, get_git_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Path to a replay artifact directory containing manifest.json and tensor files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/replay/run_manifest.json"),
        help="Where to write the replay manifest JSON.",
    )
    parser.add_argument(
        "--stream-jsonl",
        action="store_true",
        help="Stream per-step telemetry as JSON for quick inspection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step summaries to the console.",
    )
    parser.add_argument(
        "--include-events",
        action="store_true",
        help="Include the per-token event log in the output manifest (disabled by default for compact artifacts).",
    )
    return parser.parse_args()


def format_event(event: StepOutcome) -> str:
    token_repr = event.token_text.replace("\n", "\\n")
    cf_part = f" cf={','.join(event.counterfactuals)}" if event.counterfactuals else ""
    margin = f" margin={event.top2_margin:.3f}" if event.top2_margin is not None else ""
    return (
        f"[stream={event.stream} stride={event.stride_index}] token={event.token_id} text='{token_repr}' "
        f"agree={event.agreement:.3f}{margin}{cf_part} notes={'Y' if event.notes_emitted else 'N'} "
        f"rollback={'Y' if event.rollback_performed else 'N'}"
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging(
        name="parallel decoder transformer.cli.logit_replay",
        extra_loggers=["parallel decoder transformer.inference"],
    )
    artifact = LogitReplayArtifact.load(args.artifact)
    inference_cfg = artifact.build_inference_config()
    tokenizer, tokenizer_manifest = resolve_tokenizer(artifact.tokenizer_cfg)
    model = ReplayModel(artifact)
    orchestrator = MultiStreamOrchestrator(
        model,
        tokenizer,
        inference_cfg,
        log_margins=args.verbose,
    )

    orchestrator.start(
        artifact.prompt,
        planner_notes=artifact.planner_payload(),
    )

    events: List[StepOutcome] = []
    while True:
        outcome = orchestrator.step()
        if outcome is None:
            break
        events.append(outcome)
        if args.verbose:
            logger.info(format_event(outcome))
        if args.stream_jsonl:
            payload = {
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
                "coverage_logits": outcome.coverage_logits,
                "counterfactuals": outcome.counterfactuals,
                "top2_margin": outcome.top2_margin,
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    manifest = orchestrator.finalize()
    git_meta = get_git_metadata()
    manifest["git_sha"] = git_meta.sha
    manifest["git_dirty"] = git_meta.dirty
    manifest["prompt"] = artifact.prompt
    manifest["tokenizer"] = tokenizer_manifest.to_dict()
    if args.include_events:
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
                "coverage_logits": event.coverage_logits,
                "counterfactuals": event.counterfactuals,
                "top2_margin": event.top2_margin,
            }
            for event in events
        ]
    manifest["replay_artifact"] = str(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("logit_replay_complete | artifact=%s | manifest=%s", args.artifact, args.output)


if __name__ == "__main__":
    main()
