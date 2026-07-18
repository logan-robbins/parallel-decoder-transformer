"""Two-stage OpenAI Batch pipeline for source-grounded real-plan data."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Literal, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from pdt.datasets.historical_source import (
    DatasetSplit,
    HistoricalSource,
    HistoricalSourceManifest,
    canonical_json_bytes,
    load_historical_sources,
    sha256_file,
    verify_accepted_manifest,
)
from pdt.datasets.immutable_io import write_bytes_new, write_jsonl_new
from pdt.datasets.real_plan_schema import (
    FactExtractionOutput,
    JointPlanOutput,
    MAX_EXTRACTED_FACTS,
    MIN_EXTRACTED_FACTS,
    REAL_PLAN_SCHEMA,
    RealPlanExample,
    validate_fact_extraction,
    validate_real_plan_example,
)


CANONICAL_BATCH_MODEL = "gpt-5.4-mini-2026-03-17"
RESPONSES_ENDPOINT: Literal["/v1/responses"] = "/v1/responses"
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200_000_000
REAL_PLAN_SPLIT_MANIFEST_SCHEMA = "pdt-real-plan-split-manifest-v1"

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RealPlanSplitRecord(_StrictModel):
    split: DatasetSplit
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    examples: int = Field(gt=0)
    source_ids: tuple[str, ...] = Field(min_length=1)
    family_ids: tuple[str, ...] = Field(min_length=1)


class RealPlanSplitManifest(_StrictModel):
    schema_version: str = Field(pattern=rf"^{REAL_PLAN_SPLIT_MANIFEST_SCHEMA}$")
    input_file: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_examples: int = Field(gt=0)
    splits: tuple[RealPlanSplitRecord, RealPlanSplitRecord, RealPlanSplitRecord]


def build_fact_requests(
    source_path: Path,
    accepted_manifest_path: Path,
    accepted_manifest_sha256: str,
    output_path: Path,
) -> int:
    sources, _ = _load_accepted_sources(
        source_path,
        accepted_manifest_path,
        accepted_manifest_sha256,
    )
    requests = (
        _batch_request(
            custom_id=f"facts:{source.source_id}",
            schema_name="pdt_atomic_facts",
            schema=FactExtractionOutput.model_json_schema(),
            instructions=_FACT_SYSTEM_PROMPT,
            user_text=_fact_user_prompt(source),
            max_output_tokens=16_000,
            accepted_manifest_sha256=accepted_manifest_sha256,
        )
        for source in sources
    )
    return write_jsonl_new(output_path, requests)


def parse_fact_results(
    source_path: Path,
    accepted_manifest_path: Path,
    accepted_manifest_sha256: str,
    result_path: Path,
    output_path: Path,
) -> int:
    sources, _ = _load_accepted_sources(
        source_path,
        accepted_manifest_path,
        accepted_manifest_sha256,
    )
    source_ids = {source.source_id for source in sources}
    source_by_id = {source.source_id: source for source in sources}
    parsed: dict[str, FactExtractionOutput] = {}
    for row in _read_jsonl(result_path):
        custom_id, payload = _extract_batch_payload(row, prefix="facts:")
        source_id = custom_id.removeprefix("facts:")
        facts = FactExtractionOutput.model_validate_json(payload)
        if facts.source_id != source_id:
            raise ValueError(
                f"Batch result {custom_id!r} returned source_id={facts.source_id!r}."
            )
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Batch fact result names unknown source_id={source_id!r}.")
        validate_fact_extraction(source, facts)
        if source_id in parsed:
            raise ValueError(f"Duplicate Batch result for {source_id!r}.")
        parsed[source_id] = facts
    if set(parsed) != source_ids:
        raise ValueError(
            "Fact Batch results do not exactly cover source packets; "
            f"missing={sorted(source_ids - set(parsed))}, "
            f"extra={sorted(set(parsed) - source_ids)}."
        )
    ordered = (parsed[source.source_id].model_dump(mode="json") for source in sources)
    return write_jsonl_new(output_path, ordered)


def build_joint_requests(
    source_path: Path,
    accepted_manifest_path: Path,
    accepted_manifest_sha256: str,
    facts_path: Path,
    output_path: Path,
) -> int:
    sources, _ = _load_accepted_sources(
        source_path,
        accepted_manifest_path,
        accepted_manifest_sha256,
    )
    facts = _index_models(
        _load_models(facts_path, FactExtractionOutput),
        key="source_id",
    )
    source_ids = {source.source_id for source in sources}
    if set(facts) != source_ids:
        raise ValueError(
            "Fact rows must exactly cover sources before joint generation; "
            f"missing={sorted(source_ids - set(facts))}, "
            f"extra={sorted(set(facts) - source_ids)}."
        )
    for source in sources:
        validate_fact_extraction(source, facts[source.source_id])
    requests = (
        _batch_request(
            custom_id=f"joint:{source.source_id}",
            schema_name="pdt_three_lane_plan",
            schema=JointPlanOutput.model_json_schema(),
            instructions=_JOINT_SYSTEM_PROMPT,
            user_text=_joint_user_prompt(source, facts[source.source_id]),
            max_output_tokens=32_000,
            accepted_manifest_sha256=accepted_manifest_sha256,
        )
        for source in sources
    )
    return write_jsonl_new(output_path, requests)


def parse_joint_results(
    source_path: Path,
    accepted_manifest_path: Path,
    accepted_manifest_sha256: str,
    facts_path: Path,
    result_path: Path,
    output_path: Path,
) -> int:
    sources, _ = _load_accepted_sources(
        source_path,
        accepted_manifest_path,
        accepted_manifest_sha256,
    )
    source_by_id = {source.source_id: source for source in sources}
    facts = _index_models(
        _load_models(facts_path, FactExtractionOutput),
        key="source_id",
    )
    if set(facts) != set(source_by_id):
        raise ValueError(
            "Fact rows must exactly cover sources before joint parsing; "
            f"missing={sorted(set(source_by_id) - set(facts))}, "
            f"extra={sorted(set(facts) - set(source_by_id))}."
        )
    for source in sources:
        validate_fact_extraction(source, facts[source.source_id])
    parsed: dict[str, JointPlanOutput] = {}
    for row in _read_jsonl(result_path):
        custom_id, payload = _extract_batch_payload(row, prefix="joint:")
        source_id = custom_id.removeprefix("joint:")
        teacher = JointPlanOutput.model_validate_json(payload)
        if teacher.source_id != source_id:
            raise ValueError(
                f"Batch result {custom_id!r} returned source_id={teacher.source_id!r}."
            )
        if source_id in parsed:
            raise ValueError(f"Duplicate Batch result for {source_id!r}.")
        parsed[source_id] = _randomize_teacher_plan_ids(teacher, source_id=source_id)
    expected = set(source_by_id)
    if set(parsed) != expected:
        raise ValueError(
            "Joint Batch results do not exactly cover source packets; "
            f"missing={sorted(expected - set(parsed))}, "
            f"extra={sorted(set(parsed) - expected)}."
        )

    examples: list[dict[str, object]] = []
    for source in sources:
        fact_row = facts.get(source.source_id)
        if fact_row is None:
            raise ValueError(f"Missing facts for source {source.source_id!r}.")
        example = RealPlanExample(
            schema_version=REAL_PLAN_SCHEMA,
            source=source,
            facts=fact_row,
            teacher=parsed[source.source_id],
        )
        validate_real_plan_example(example)
        examples.append(example.model_dump(mode="json"))
    return write_jsonl_new(output_path, examples)


def split_real_plan_examples(input_path: Path, output_dir: Path) -> int:
    """Atomically partition validated raw examples by their source-family split."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Validated real-plan JSONL does not exist: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace real-plan split bundle: {output_dir}")
    examples = _load_models(input_path, RealPlanExample)
    grouped: dict[DatasetSplit, list[RealPlanExample]] = {
        split: [] for split in DatasetSplit
    }
    family_splits: dict[str, DatasetSplit] = {}
    source_ids: set[str] = set()
    for example in examples:
        validate_real_plan_example(example)
        source = example.source
        if source.source_id in source_ids:
            raise ValueError(f"Real-plan examples repeat source_id={source.source_id!r}.")
        source_ids.add(source.source_id)
        previous = family_splits.setdefault(source.family_id, source.split)
        if previous is not source.split:
            raise ValueError(
                f"Source family {source.family_id!r} crosses {previous.value} and "
                f"{source.split.value}."
            )
        grouped[source.split].append(example)
    empty = [split.value for split, rows in grouped.items() if not rows]
    if empty:
        raise ValueError(
            "A real-plan split bundle requires non-empty train, validation, and test "
            f"partitions; empty={empty}."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    records: list[RealPlanSplitRecord] = []
    try:
        for split in DatasetSplit:
            rows = grouped[split]
            destination = temporary / f"{split.value}.jsonl"
            write_jsonl_new(
                destination,
                (row.model_dump(mode="json") for row in rows),
            )
            records.append(
                RealPlanSplitRecord(
                    split=split,
                    file=destination.name,
                    sha256=sha256_file(destination),
                    examples=len(rows),
                    source_ids=tuple(row.source.source_id for row in rows),
                    family_ids=tuple(sorted({row.source.family_id for row in rows})),
                )
            )
        manifest = RealPlanSplitManifest(
            schema_version=REAL_PLAN_SPLIT_MANIFEST_SCHEMA,
            input_file=input_path.name,
            input_sha256=sha256_file(input_path),
            total_examples=len(examples),
            splits=(records[0], records[1], records[2]),
        )
        write_bytes_new(
            temporary / "manifest.json",
            canonical_json_bytes(manifest),
        )
        os.rename(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return len(examples)


def submit_batch(request_path: Path) -> Mapping[str, object]:
    """Upload one immutable request file and launch a 24-hour Responses batch."""

    if not request_path.is_file():
        raise FileNotFoundError(f"Batch request JSONL does not exist: {request_path}")
    _validate_batch_request_file(request_path)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set before submitting a Batch job.")
    from openai import OpenAI

    client = OpenAI()
    with request_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=RESPONSES_ENDPOINT,
        completion_window="24h",
        metadata={
            "pipeline": REAL_PLAN_SCHEMA,
            "request_file": request_path.name,
        },
    )
    return {
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "status": batch.status,
    }


def retrieve_batch(batch_id: str) -> Mapping[str, object]:
    if not batch_id:
        raise ValueError("batch_id must be non-empty.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set before retrieving a Batch job.")
    from openai import OpenAI

    batch = OpenAI().batches.retrieve(batch_id)
    counts = batch.request_counts
    if counts is None:
        raise RuntimeError(f"Batch {batch_id!r} did not return request_counts.")
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": {
            "total": counts.total,
            "completed": counts.completed,
            "failed": counts.failed,
        },
    }


def download_batch_results(batch_id: str, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace immutable Batch output: {output_path}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set before downloading Batch results.")
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch_id!r} is {batch.status!r}; results require status='completed'."
        )
    if not batch.output_file_id:
        raise RuntimeError(f"Completed Batch {batch_id!r} has no output_file_id.")
    if batch.request_counts is None:
        raise RuntimeError(f"Completed Batch {batch_id!r} has no request counts.")
    if batch.request_counts.failed:
        raise RuntimeError(
            f"Batch {batch_id!r} completed with {batch.request_counts.failed} failed "
            f"requests; inspect error_file_id={batch.error_file_id!r}."
        )
    content = client.files.content(batch.output_file_id)
    write_bytes_new(output_path, content.read())


def _batch_request(
    *,
    custom_id: str,
    schema_name: str,
    schema: Mapping[str, object],
    instructions: str,
    user_text: str,
    max_output_tokens: int,
    accepted_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": RESPONSES_ENDPOINT,
        "body": {
            "model": CANONICAL_BATCH_MODEL,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
            "metadata": {
                "pipeline": REAL_PLAN_SCHEMA,
                "accepted_manifest_sha256": accepted_manifest_sha256,
            },
            "store": False,
        },
    }


def _fact_user_prompt(source: HistoricalSource) -> str:
    return (
        "Extract the source-grounded fact inventory for this packet. Use only exact "
        "claims supported by the supplied paragraph text. Character offsets are zero-based "
        "Python string offsets within one paragraph and exact_quote must equal the indicated "
        f"slice. Produce {MIN_EXTRACTED_FACTS}-{MAX_EXTRACTED_FACTS} atomic facts spanning "
        "at least 12 source paragraphs and at least four source sections. Every provenance "
        "span must name one or more reference IDs attached to that paragraph. Prioritize "
        "facts useful in a broad expository answer, and create one plausible but contradicted "
        "hard negative for every fact. Return the exact source ID, revision ID, and "
        "model-visible SHA-256 supplied below.\n\n"
        f"SOURCE_ID: {source.source_id}\n"
        f"SOURCE_REVISION_ID: {source.revision_id}\n"
        f"SOURCE_MODEL_VISIBLE_SHA256: {source.model_visible_sha256}\n"
        f"TITLE: {source.title}\n\n"
        f"{_source_text(source)}"
    )


def _joint_user_prompt(
    source: HistoricalSource,
    facts: FactExtractionOutput,
) -> str:
    return (
        "Create one natural long-form expository prompt and exactly three complementary, "
        "unordered plans that jointly answer it from the source. Generate all three target "
        "sections in this one response. Each section must contain roughly 600-750 English "
        "words so the Qwen tokenizer yields 700-1000 tokens, across at least four connected "
        "paragraphs. Use developed, cohesive sentences rather than short or choppy sentences; "
        "do not write short answers, QA, bullet lists, or a fourth synthesis. Each fact must "
        "be OWNER in exactly one section and must be "
        "labeled REFERENCE or ABSENT in the other two. OWNER/REFERENCE evidence quotes must "
        "be exact substrings of the corresponding target section. Plans need 4-8 ordered "
        "nodes. Every section must reference at least two facts realized by its siblings, "
        "keep at least two other positive facts ABSENT, "
        "receive at least one dependency from each sibling, and cover every REFERENCE fact "
        "exactly once in a dependency. Each dependency must include exact evidence quotes "
        "from the earlier sibling paragraph and the later receiver paragraph. Cross-plan "
        "dependencies may only point from an earlier paragraph index to a later paragraph "
        "index. Put source evidence early enough and receiver evidence late enough that, "
        "after Qwen tokenization into 32-token blocks, the complete source-evidence block "
        "precedes the first receiver-evidence block; same-round communication is forbidden. "
        "plan_0, plan_1, and plan_2 "
        "are temporary unordered identifiers, not semantic roles.\n\n"
        "Return the exact source ID, revision ID, and model-visible SHA-256 supplied below."
        f"\n\nSOURCE_ID: {source.source_id}\n"
        f"SOURCE_REVISION_ID: {source.revision_id}\n"
        f"SOURCE_MODEL_VISIBLE_SHA256: {source.model_visible_sha256}\n"
        f"TITLE: {source.title}\n\n"
        f"{_source_text(source)}\n\nATOMIC_FACTS:\n"
        f"{facts.model_dump_json(indent=2)}"
    )


def _source_text(source: HistoricalSource) -> str:
    parts: list[str] = []
    previous_path: tuple[str, ...] = ()
    for section in source.sections:
        common = 0
        for left, right in zip(previous_path, section.heading_path):
            if left != right:
                break
            common += 1
        parts.extend(f"HEADING: {heading}" for heading in section.heading_path[common:])
        for paragraph in section.paragraphs:
            references = ",".join(paragraph.reference_ids)
            parts.append(
                f"[PARAGRAPH {paragraph.paragraph_id}; REFERENCES {references}]\n"
                f"{paragraph.text}"
            )
        previous_path = section.heading_path
    if not parts:
        raise ValueError(f"Historical source {source.source_id!r} has no renderable content.")
    return "\n\n".join(parts)


def _extract_batch_payload(
    row: Mapping[str, object],
    *,
    prefix: str,
) -> tuple[str, str]:
    custom_id = row.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id.startswith(prefix):
        raise ValueError(f"Batch result custom_id must start with {prefix!r}.")
    if row.get("error") is not None:
        raise RuntimeError(f"Batch request {custom_id!r} failed: {row['error']!r}")
    response = row.get("response")
    if not isinstance(response, Mapping):
        raise ValueError(f"Batch result {custom_id!r} has no response object.")
    if response.get("status_code") != 200:
        raise RuntimeError(
            f"Batch request {custom_id!r} returned HTTP {response.get('status_code')!r}."
        )
    body = response.get("body")
    if not isinstance(body, Mapping):
        raise ValueError(f"Batch response {custom_id!r} has no response body.")
    output = body.get("output")
    if not isinstance(output, list):
        raise ValueError(f"Batch response {custom_id!r} has no output array.")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
    if len(texts) != 1:
        raise ValueError(
            f"Batch response {custom_id!r} must contain exactly one output_text, "
            f"found {len(texts)}."
        )
    return custom_id, texts[0]


def _randomize_teacher_plan_ids(
    teacher: JointPlanOutput,
    *,
    source_id: str,
) -> JointPlanOutput:
    """Deterministically destroy any permanent meaning of teacher plan labels."""

    seed = int.from_bytes(hashlib.sha256(source_id.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    old_ids = ["plan_0", "plan_1", "plan_2"]
    new_ids = old_ids.copy()
    rng.shuffle(new_ids)
    mapping = dict(zip(old_ids, new_ids, strict=True))
    payload = teacher.model_dump(mode="json")
    for plan in payload["plans"]:
        plan["plan_id"] = mapping[plan["plan_id"]]
        for node in plan["nodes"]:
            for dependency in node["dependencies"]:
                dependency["source_plan_id"] = mapping[dependency["source_plan_id"]]
    for target in payload["target_sections"]:
        target["plan_id"] = mapping[target["plan_id"]]
    payload["presentation_order"] = [
        mapping[plan_id] for plan_id in payload["presentation_order"]
    ]
    rng.shuffle(payload["plans"])
    rng.shuffle(payload["target_sections"])
    return JointPlanOutput.model_validate(payload)


def _load_models(path: Path, model_type: type[_ModelT]) -> list[_ModelT]:
    rows = [model_type.model_validate(row) for row in _read_jsonl(path)]
    if not rows:
        raise ValueError(f"{path} contains no records.")
    return rows


def _load_accepted_sources(
    source_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[tuple[HistoricalSource, ...], HistoricalSourceManifest]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("Accepted manifest SHA-256 must be 64 lowercase hexadecimal digits.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Accepted source manifest does not exist: {manifest_path}")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Accepted manifest does not match the explicitly approved SHA-256: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}."
        )
    manifest = verify_accepted_manifest(source_path, manifest_path)
    sources = load_historical_sources(source_path)
    return sources, manifest


def _index_models(rows: Iterable[_ModelT], *, key: str) -> dict[str, _ModelT]:
    indexed: dict[str, _ModelT] = {}
    for row in rows:
        value = getattr(row, key)
        if not isinstance(value, str):
            raise TypeError(f"Index field {key!r} must be text.")
        if value in indexed:
            raise ValueError(f"Duplicate {key}={value!r}.")
        indexed[value] = row
    return indexed


def _read_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} must contain one JSON object.")
            yield row


def _validate_batch_request_file(path: Path) -> None:
    size = path.stat().st_size
    if size <= 0 or size > MAX_BATCH_FILE_BYTES:
        raise ValueError(
            f"Batch request file must be 1-{MAX_BATCH_FILE_BYTES} bytes; got {size}."
        )
    count = 0
    manifest_hashes: set[str] = set()
    for row in _read_jsonl(path):
        count += 1
        if row.get("method") != "POST" or row.get("url") != RESPONSES_ENDPOINT:
            raise ValueError("Batch request file contains a non-canonical request route.")
        body = row.get("body")
        if not isinstance(body, Mapping) or body.get("model") != CANONICAL_BATCH_MODEL:
            raise ValueError("Batch request file contains a non-canonical model.")
        metadata = body.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("pipeline") != REAL_PLAN_SCHEMA
        ):
            raise ValueError(
                f"Batch request file must be attested to pipeline {REAL_PLAN_SCHEMA!r}."
            )
        manifest_sha256 = metadata.get("accepted_manifest_sha256")
        if (
            not isinstance(manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
        ):
            raise ValueError(
                "Every Batch request must contain an accepted manifest SHA-256."
            )
        manifest_hashes.add(manifest_sha256)
    if not 1 <= count <= MAX_BATCH_REQUESTS:
        raise ValueError(
            f"Batch request file must contain 1-{MAX_BATCH_REQUESTS} requests; "
            f"got {count}."
        )
    if len(manifest_hashes) != 1:
        raise ValueError("One Batch request file cannot mix accepted source manifests.")


_FACT_SYSTEM_PROMPT = (
    "You are a meticulous source-grounded fact annotator. Never use outside knowledge. "
    "Return only the strict structured output. Preserve exact source character offsets."
)

_JOINT_SYSTEM_PROMPT = (
    "You design and jointly write synchronized long-form expository sections. The three "
    "plans are complementary and unordered. You must enforce exact fact ownership, delayed "
    "cross-section dependencies, and multi-paragraph prose using only the supplied source."
)
