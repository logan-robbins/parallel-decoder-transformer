"""Build an immutable accepted historical-source bundle from pinned revisions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from transformers import AutoTokenizer

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES
from pdt.datasets.historical_source import (
    HISTORICAL_FILTER_FAILURE_SCHEMA,
    HISTORICAL_REJECTION_SCHEMA,
    HistoricalFilterFailureManifest,
    HistoricalRejection,
    HistoricalRejectionManifest,
    HistoricalSource,
    RawWikimediaRevisionBundle,
    canonical_json_bytes,
    cluster_historical_families,
    evaluate_historical_eligibility,
    manifest_for_source_file,
    materialize_historical_source,
    parse_historical_dom,
    select_balanced_historical_sources,
    sha256_file,
    verify_filter_failure_bundle,
)
from pdt.datasets.immutable_io import write_bytes_new, write_jsonl_new


ACCEPTED_FILE_NAME = "accepted_sources.jsonl"
ACCEPTED_MANIFEST_NAME = "accepted_manifest.json"
REJECTIONS_FILE_NAME = "rejections.json"
SELECTION_FILE_NAME = "selection.json"
FAILURE_FILE_NAME = "failure.json"
ELIGIBLE_FILE_NAME = "eligible_sources.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, required=True)
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=DEFAULT_TRUNK_PROFILE,
    )
    args = parser.parse_args()
    build_historical_sources(
        input_path=args.input,
        output_dir=args.output_dir,
        trunk_profile=args.trunk_profile,
        examples_per_category=args.examples_per_category,
    )


def build_historical_sources(
    *,
    input_path: Path,
    output_dir: Path,
    trunk_profile: str,
    examples_per_category: int,
) -> None:
    """Validate every raw row and atomically publish one immutable bundle directory."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Pinned Wikimedia revision JSONL does not exist: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace historical-source bundle: {output_dir}")
    profile = TRUNK_PROFILES.get(trunk_profile)
    if profile is None:
        raise ValueError(
            f"Unknown trunk profile {trunk_profile!r}; expected one of {tuple(TRUNK_PROFILES)}."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        profile.base_model,
        revision=profile.revision,
        use_fast=True,
        local_files_only=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Historical source validation requires the pinned fast tokenizer.")

    bundles = _load_raw_bundles(input_path)
    parsed_rows = [
        (bundle, parse_historical_dom(bundle))
        for bundle in bundles
    ]
    eligible_rows = []
    rejections: list[HistoricalRejection] = []
    eligibility_by_id = {}
    for bundle, parsed in parsed_rows:
        eligibility = evaluate_historical_eligibility(
            bundle,
            parsed,
            tokenizer=tokenizer,
        )
        eligibility_by_id[bundle.source_id] = eligibility
        if eligibility.accepted:
            eligible_rows.append((bundle, parsed))
        else:
            rejections.append(
                HistoricalRejection(
                    source_id=bundle.source_id,
                    page_id=bundle.page_id,
                    revision_id=bundle.revision_id,
                    reasons=eligibility.reasons,
                    details=eligibility.details,
                )
            )
    rejection_manifest = HistoricalRejectionManifest(
        schema_version=HISTORICAL_REJECTION_SCHEMA,
        records=tuple(rejections),
    )
    if not eligible_rows:
        error = (
            "No raw revision bundle passed the fixed historical-source contract; "
            "inspect candidates rather than weakening admission thresholds."
        )
        _publish_failure_bundle(
            output_dir=output_dir,
            input_path=input_path,
            profile_name=profile.base_model,
            profile_revision=profile.revision,
            examples_per_category=examples_per_category,
            candidate_count=len(bundles),
            eligible_sources=(),
            rejection_manifest=rejection_manifest,
            failure_stage="eligibility",
            error=error,
        )
        raise ValueError(f"{error} Failure evidence was published to {output_dir}.")
    families = cluster_historical_families(eligible_rows)
    eligible_sources = [
        materialize_historical_source(
            bundle,
            parsed,
            eligibility_by_id[bundle.source_id],
            family_id=families[bundle.source_id],
            tokenizer_name=profile.base_model,
            tokenizer_revision=profile.revision,
        )
        for bundle, parsed in eligible_rows
    ]
    try:
        sources, selection_manifest = select_balanced_historical_sources(
            eligible_sources,
            examples_per_category=examples_per_category,
        )
    except ValueError as exc:
        error = str(exc)
        _publish_failure_bundle(
            output_dir=output_dir,
            input_path=input_path,
            profile_name=profile.base_model,
            profile_revision=profile.revision,
            examples_per_category=examples_per_category,
            candidate_count=len(bundles),
            eligible_sources=tuple(eligible_sources),
            rejection_manifest=rejection_manifest,
            failure_stage="selection",
            error=error,
        )
        raise ValueError(
            f"{error} Failure evidence was published to {output_dir}."
        ) from exc

    def write_success_bundle(temporary: Path) -> None:
        accepted_path = temporary / ACCEPTED_FILE_NAME
        write_jsonl_new(
            accepted_path,
            (source.model_dump(mode="json") for source in sources),
        )
        manifest = manifest_for_source_file(
            accepted_path,
            sources,
            input_path=input_path,
            tokenizer_name=profile.base_model,
            tokenizer_revision=profile.revision,
        )
        write_bytes_new(
            temporary / ACCEPTED_MANIFEST_NAME,
            canonical_json_bytes(manifest),
        )
        write_bytes_new(
            temporary / REJECTIONS_FILE_NAME,
            canonical_json_bytes(rejection_manifest),
        )
        write_bytes_new(
            temporary / SELECTION_FILE_NAME,
            canonical_json_bytes(selection_manifest),
        )

    _publish_output_directory(output_dir, write_success_bundle)

    split_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for source in sources:
        split_counts[source.split.value] = split_counts.get(source.split.value, 0) + 1
        category = source.historical_category.value
        category_counts[category] = category_counts.get(category, 0) + 1
    print(
        json.dumps(
            {
                "accepted": len(sources),
                "eligible": len(eligible_sources),
                "rejected": len(rejections),
                "splits": split_counts,
                "categories": category_counts,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


def _publish_failure_bundle(
    *,
    output_dir: Path,
    input_path: Path,
    profile_name: str,
    profile_revision: str,
    examples_per_category: int,
    candidate_count: int,
    eligible_sources: tuple[HistoricalSource, ...],
    rejection_manifest: HistoricalRejectionManifest,
    failure_stage: str,
    error: str,
) -> None:
    def write_failure_bundle(temporary: Path) -> None:
        eligible_path = temporary / ELIGIBLE_FILE_NAME
        if eligible_sources:
            write_jsonl_new(
                eligible_path,
                (source.model_dump(mode="json") for source in eligible_sources),
            )
        failure = HistoricalFilterFailureManifest(
            schema_version=HISTORICAL_FILTER_FAILURE_SCHEMA,
            failure_stage=failure_stage,
            raw_input_sha256=sha256_file(input_path),
            tokenizer=profile_name,
            tokenizer_revision=profile_revision,
            requested_examples_per_category=examples_per_category,
            candidate_count=candidate_count,
            eligible_source_ids=tuple(source.source_id for source in eligible_sources),
            eligible_source_file_name=ELIGIBLE_FILE_NAME if eligible_sources else None,
            eligible_source_file_sha256=(
                sha256_file(eligible_path) if eligible_sources else None
            ),
            eligible_source_file_bytes=(
                eligible_path.stat().st_size if eligible_sources else None
            ),
            rejected_count=len(rejection_manifest.records),
            error=error,
        )
        write_bytes_new(
            temporary / REJECTIONS_FILE_NAME,
            canonical_json_bytes(rejection_manifest),
        )
        write_bytes_new(
            temporary / FAILURE_FILE_NAME,
            canonical_json_bytes(failure),
        )

    _publish_output_directory(output_dir, write_failure_bundle)
    verify_filter_failure_bundle(output_dir)


def _publish_output_directory(
    output_dir: Path,
    writer: Callable[[Path], None],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        writer(temporary)
        os.rename(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_raw_bundles(path: Path) -> tuple[RawWikimediaRevisionBundle, ...]:
    bundles: list[RawWikimediaRevisionBundle] = []
    seen_sources: set[str] = set()
    seen_revisions: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                bundle = RawWikimediaRevisionBundle.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} violates the pinned revision bundle schema."
                ) from exc
            revision_identity = (bundle.page_id, bundle.revision_id)
            if bundle.source_id in seen_sources:
                raise ValueError(f"{path}:{line_number} repeats source_id={bundle.source_id!r}.")
            if revision_identity in seen_revisions:
                raise ValueError(
                    f"{path}:{line_number} repeats page/revision={revision_identity!r}."
                )
            seen_sources.add(bundle.source_id)
            seen_revisions.add(revision_identity)
            bundles.append(bundle)
    if not bundles:
        raise ValueError(f"{path} contains no pinned Wikimedia revision bundles.")
    return tuple(bundles)


if __name__ == "__main__":
    main()
