"""Acquire exact English-Wikipedia revisions for the historical source gate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup, Tag
import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pdt.datasets.historical_source import (
    RAW_WIKIMEDIA_SCHEMA,
    AssessmentQuality,
    HistoricalCategory,
    HistoricalTopicKind,
    PageAssessment,
    RawWikimediaRevisionBundle,
    ReferenceKind,
    ReferenceRecord,
)


HISTORICAL_CANDIDATE_SCHEMA = "pdt-historical-candidate-v1"
ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
REST_REVISION_URL = "https://en.wikipedia.org/w/rest.php/v1/revision/{revision_id}/with_html"
_MAINTENANCE_TEMPLATE_NAMES = (
    "citation needed",
    "unreferenced",
    "more citations needed",
    "refimprove",
    "pov",
    "npov",
    "disputed",
    "original research",
    "or",
    "cleanup",
    "hoax",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HistoricalCandidate(_StrictModel):
    """A manually reviewed historical classification pinned to one revision."""

    schema_version: str = Field(pattern=rf"^{HISTORICAL_CANDIDATE_SCHEMA}$")
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=500)
    topic_kind: HistoricalTopicKind
    historical_category: HistoricalCategory
    event_end_year: int = Field(ge=1, le=2200)
    family_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=100)
    related_page_ids: tuple[int, ...] = Field(default_factory=tuple, max_length=5_000)

    @model_validator(mode="after")
    def validate_candidate_type(self) -> HistoricalCandidate:
        if self.topic_kind not in (
            HistoricalTopicKind.EVENT,
            HistoricalTopicKind.PROCESS,
        ):
            raise ValueError("Historical candidates must be an event or completed process.")
        return self


def load_historical_candidates(path: Path) -> tuple[HistoricalCandidate, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Historical candidate JSONL does not exist: {path}")
    rows: list[HistoricalCandidate] = []
    page_ids: set[int] = set()
    revision_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                candidate = HistoricalCandidate.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} violates {HISTORICAL_CANDIDATE_SCHEMA}."
                ) from exc
            if candidate.page_id in page_ids:
                raise ValueError(f"{path}:{line_number} repeats page_id={candidate.page_id}.")
            if candidate.revision_id in revision_ids:
                raise ValueError(
                    f"{path}:{line_number} repeats revision_id={candidate.revision_id}."
                )
            page_ids.add(candidate.page_id)
            revision_ids.add(candidate.revision_id)
            rows.append(candidate)
    if not rows:
        raise ValueError(f"{path} contains no historical candidates.")
    return tuple(rows)


def acquire_historical_revision(
    candidate: HistoricalCandidate,
    *,
    client: httpx.Client,
    acquisition_date: date,
) -> RawWikimediaRevisionBundle:
    """Fetch and cross-check Action API source, assessments, and REST Parsoid HTML."""

    action_response = client.get(
        ACTION_API_URL,
        params={
            "action": "query",
            "prop": "revisions|pageassessments|pageprops",
            "meta": "siteinfo",
            "siprop": "rightsinfo",
            "revids": str(candidate.revision_id),
            "rvprop": "ids|timestamp|sha1|content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        },
    )
    action_response.raise_for_status()
    action_payload = _mapping(action_response.json(), context="Action API response")
    page, revision = _extract_action_revision(action_payload)

    rest_response = client.get(
        REST_REVISION_URL.format(revision_id=candidate.revision_id),
    )
    rest_response.raise_for_status()
    rest_payload = _mapping(rest_response.json(), context="REST revision response")
    semantic_html = _required_text(rest_payload, "html", context="REST revision response")
    rest_page = _mapping(rest_payload.get("page"), context="REST revision page")

    page_id = _required_int(page, "pageid", context="Action API page")
    revision_id = _required_int(revision, "revid", context="Action API revision")
    title = _required_text(page, "title", context="Action API page")
    if (
        page_id != candidate.page_id
        or revision_id != candidate.revision_id
        or title != candidate.title
    ):
        raise ValueError(
            "Fetched Action API identity differs from the pinned candidate: "
            f"expected page={candidate.page_id}, revision={candidate.revision_id}, "
            f"title={candidate.title!r}; got page={page_id}, revision={revision_id}, "
            f"title={title!r}."
        )
    if (
        _required_int(rest_payload, "id", context="REST revision response") != revision_id
        or _required_int(rest_page, "id", context="REST revision page") != page_id
        or _required_text(rest_page, "title", context="REST revision page") != title
    ):
        raise ValueError("REST revision identity differs from the Action API identity.")

    slots = _mapping(revision.get("slots"), context="Action API revision slots")
    main_slot = _mapping(slots.get("main"), context="Action API main slot")
    raw_wikitext = _required_text(main_slot, "content", context="Action API main slot")
    revision_sha1 = _required_text(revision, "sha1", context="Action API revision")
    actual_sha1 = hashlib.sha1(raw_wikitext.encode("utf-8")).hexdigest()
    if actual_sha1 != revision_sha1:
        raise ValueError(
            f"Revision {revision_id} source SHA-1 mismatch: "
            f"API={revision_sha1}, computed={actual_sha1}."
        )
    namespace_value = page.get("ns")
    if type(namespace_value) is not int or namespace_value < 0:
        raise ValueError("Action API page.ns must be a non-negative integer.")
    namespace = namespace_value
    assessments = _parse_assessments(page.get("pageassessments"))
    pageprops = _mapping_or_empty(page.get("pageprops"), context="Action API pageprops")
    rest_license = _mapping(rest_payload.get("license"), context="REST revision license")
    license_name = _required_text(rest_license, "title", context="REST revision license")
    license_url = _required_text(rest_license, "url", context="REST revision license")
    references = parse_reference_records(semantic_html)
    action_json = action_response.text
    timestamp = _required_text(revision, "timestamp", context="Action API revision")

    return RawWikimediaRevisionBundle(
        schema_version=RAW_WIKIMEDIA_SCHEMA,
        source_id=f"enwiki-{page_id}-{revision_id}",
        page_id=page_id,
        revision_id=revision_id,
        revision_timestamp=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        dump_date=acquisition_date,
        language="en",
        namespace=namespace,
        title=title,
        source_url=f"https://en.wikipedia.org/w/index.php?oldid={revision_id}",
        license=f"{license_name} ({license_url})",
        revision_sha1=revision_sha1,
        is_redirect=bool(page.get("redirect")) or bool(
            re.match(r"^\s*#redirect\b", raw_wikitext, flags=re.IGNORECASE)
        ),
        is_disambiguation="disambiguation" in pageprops,
        content_complete=True,
        topic_kind=candidate.topic_kind,
        historical_category=candidate.historical_category,
        event_end_year=candidate.event_end_year,
        assessments=assessments,
        family_keys=candidate.family_keys,
        related_page_ids=candidate.related_page_ids,
        maintenance_templates=parse_maintenance_templates(semantic_html),
        references=references,
        action_api_response_json=action_json,
        action_api_response_sha256=_sha256(action_json),
        raw_wikitext=raw_wikitext,
        raw_wikitext_sha256=_sha256(raw_wikitext),
        semantic_html=semantic_html,
        semantic_html_sha256=_sha256(semantic_html),
    )


def acquire_candidate_file(
    candidate_path: Path,
    *,
    user_agent: str,
    acquisition_date: date,
) -> tuple[RawWikimediaRevisionBundle, ...]:
    """Acquire all candidates or fail without publishing a partial raw bundle."""

    if not user_agent.strip() or "@" not in user_agent:
        raise ValueError(
            "Wikimedia User-Agent must identify the project and include a contact email."
        )
    candidates = load_historical_candidates(candidate_path)
    headers = {
        "User-Agent": user_agent,
        "Api-User-Agent": user_agent,
        "Accept": "application/json",
    }
    with httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(60.0),
        follow_redirects=False,
    ) as client:
        return tuple(
            acquire_historical_revision(
                candidate,
                client=client,
                acquisition_date=acquisition_date,
            )
            for candidate in candidates
        )


def parse_reference_records(semantic_html: str) -> tuple[ReferenceRecord, ...]:
    """Parse bibliography records and explicit citation-template types from Parsoid HTML."""

    soup = BeautifulSoup(semantic_html, "html.parser")
    records: list[ReferenceRecord] = []
    seen: set[str] = set()
    for node in soup.find_all(id=re.compile(r"^cite_note-")):
        if not isinstance(node, Tag):
            continue
        reference_id = _normalize_reference_id(str(node.get("id", "")))
        if reference_id is None or reference_id in seen:
            continue
        citation_text = _reference_text(node)
        if len(citation_text) < 10:
            continue
        template_rows = tuple(_template_rows(node))
        source_type = _classify_reference(node, template_rows)
        records.append(
            ReferenceRecord(
                reference_id=reference_id,
                source_type=source_type,
                citation_text=citation_text,
            )
        )
        seen.add(reference_id)
    return tuple(records)


def parse_maintenance_templates(semantic_html: str) -> tuple[str, ...]:
    """Read maintenance transclusion names from Parsoid metadata, never prose."""

    soup = BeautifulSoup(semantic_html, "html.parser")
    observed: set[str] = set()
    for node in soup.find_all(attrs={"data-mw": True}):
        if not isinstance(node, Tag):
            continue
        raw = node.get("data-mw")
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Parsoid transclusion contains malformed data-mw JSON.") from exc
        for name, _ in _walk_templates(payload):
            if any(
                name == forbidden or name.startswith(f"{forbidden} ")
                for forbidden in _MAINTENANCE_TEMPLATE_NAMES
            ):
                observed.add(name)
    return tuple(sorted(observed))


def _extract_action_revision(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    query = _mapping(payload.get("query"), context="Action API query")
    pages = query.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise ValueError("Action API must return exactly one page for a pinned revision.")
    page = _mapping(pages[0], context="Action API page")
    if "missing" in page or "invalid" in page:
        raise ValueError("Pinned Wikimedia revision does not resolve to a valid page.")
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise ValueError("Action API must return exactly one pinned revision.")
    return page, _mapping(revisions[0], context="Action API revision")


def _parse_assessments(value: object) -> tuple[PageAssessment, ...]:
    if value is None:
        return ()
    rows = _mapping(value, context="PageAssessments response")
    assessments: list[PageAssessment] = []
    for project, raw_assessment in sorted(rows.items()):
        if not isinstance(project, str) or not project.strip():
            raise ValueError("PageAssessments project names must be non-empty text.")
        assessment = _mapping(
            raw_assessment,
            context=f"PageAssessments project {project!r}",
        )
        raw_quality = assessment.get("class")
        quality = (
            AssessmentQuality(raw_quality)
            if isinstance(raw_quality, str)
            and raw_quality in {item.value for item in AssessmentQuality}
            else AssessmentQuality.UNKNOWN
        )
        assessments.append(PageAssessment(project=project, quality=quality))
    return tuple(assessments)


def _template_rows(node: Tag) -> Iterable[tuple[str, Mapping[str, object]]]:
    for descendant in node.find_all(attrs={"data-mw": True}):
        if not isinstance(descendant, Tag):
            continue
        raw = descendant.get("data-mw")
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Parsoid citation contains malformed data-mw JSON.") from exc
        yield from _walk_templates(payload)


def _walk_templates(value: object) -> Iterable[tuple[str, Mapping[str, object]]]:
    if isinstance(value, Mapping):
        template = value.get("template")
        if isinstance(template, Mapping):
            target = template.get("target")
            params = template.get("params")
            if isinstance(target, Mapping) and isinstance(params, Mapping):
                name = target.get("wt")
                if isinstance(name, str) and name.strip():
                    yield _normalize_template_name(name), params
        for child in value.values():
            yield from _walk_templates(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_templates(child)


def _classify_reference(
    node: Tag,
    templates: Sequence[tuple[str, Mapping[str, object]]],
) -> ReferenceKind:
    classified: list[ReferenceKind] = []
    for name, params in templates:
        parameter_names = {str(key).strip().casefold() for key in params}
        if name in {"cite book", "book citation", "cite encyclopedia"}:
            classified.append(ReferenceKind.BOOK)
        elif name in {"cite journal", "cite magazine"}:
            classified.append(ReferenceKind.JOURNAL)
        elif name in {"cite thesis", "thesis"}:
            classified.append(ReferenceKind.THESIS)
        elif name in {"cite report", "cite technical report"}:
            classified.append(ReferenceKind.INSTITUTIONAL)
        elif name in {"cite archive", "cite archive document"}:
            classified.append(ReferenceKind.ARCHIVE)
        elif name == "cite news":
            classified.append(ReferenceKind.NEWS)
        elif name == "cite web":
            classified.append(ReferenceKind.WEB)
        elif name == "citation":
            if {"journal", "periodical"} & parameter_names:
                classified.append(ReferenceKind.JOURNAL)
            elif {"isbn", "chapter"} & parameter_names:
                classified.append(ReferenceKind.BOOK)
            elif {"degree", "thesis"} & parameter_names:
                classified.append(ReferenceKind.THESIS)
    priority = (
        ReferenceKind.JOURNAL,
        ReferenceKind.BOOK,
        ReferenceKind.THESIS,
        ReferenceKind.ARCHIVE,
        ReferenceKind.INSTITUTIONAL,
        ReferenceKind.NEWS,
        ReferenceKind.WEB,
    )
    for kind in priority:
        if kind in classified:
            return kind
    classes = {
        item.casefold()
        for descendant in node.find_all(class_=True)
        if isinstance(descendant, Tag)
        for item in _class_values(descendant)
    }
    for css_class, kind in (
        ("journal", ReferenceKind.JOURNAL),
        ("book", ReferenceKind.BOOK),
        ("thesis", ReferenceKind.THESIS),
        ("report", ReferenceKind.INSTITUTIONAL),
        ("news", ReferenceKind.NEWS),
        ("web", ReferenceKind.WEB),
    ):
        if css_class in classes:
            return kind
    return ReferenceKind.OTHER


def _reference_text(node: Tag) -> str:
    clone = BeautifulSoup(str(node), "html.parser")
    for backlink in clone.select(".mw-cite-backlink, .mw-linkback-text"):
        backlink.decompose()
    text = clone.get_text(" ", strip=True)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*(?:↑|\^|\d+(?:\.\d+)*)\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_reference_id(value: str) -> str | None:
    normalized = value.strip().removeprefix("cite_note-")
    normalized = re.sub(r"[^A-Za-z0-9_.:-]", "_", normalized)
    if not normalized or len(normalized) > 200:
        return None
    return normalized


def _normalize_template_name(value: str) -> str:
    value = value.strip().casefold().replace("_", " ")
    value = re.sub(r"^template\s*:\s*", "", value)
    return re.sub(r"\s+", " ", value)


def _class_values(node: Tag) -> tuple[str, ...]:
    value = node.get("class")
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object.")
    return value


def _mapping_or_empty(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return _mapping(value, context=context)


def _required_text(row: Mapping[str, object], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be non-empty text.")
    return value


def _required_int(row: Mapping[str, object], key: str, *, context: str) -> int:
    value = row.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer.")
    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ACTION_API_URL",
    "HISTORICAL_CANDIDATE_SCHEMA",
    "HistoricalCandidate",
    "REST_REVISION_URL",
    "acquire_candidate_file",
    "acquire_historical_revision",
    "load_historical_candidates",
    "parse_maintenance_templates",
    "parse_reference_records",
]
