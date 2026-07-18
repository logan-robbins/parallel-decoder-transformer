"""Pinned Wikimedia historical-source parsing, validation, and manifests.

This module is the only admissible source ingress for real-plan data. It
consumes immutable revision bundles, extracts model-visible prose from the
rendered semantic DOM, applies the complete historical/citation contract, and
publishes records whose exact byte identity can be verified before a paid
Batch request is constructed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Protocol

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, ConfigDict, Field, model_validator


RAW_WIKIMEDIA_SCHEMA = "pdt-wikimedia-revision-bundle-v1"
HISTORICAL_SOURCE_SCHEMA = "pdt-historical-source-v1"
HISTORICAL_MANIFEST_SCHEMA = "pdt-historical-source-manifest-v1"
HISTORICAL_REJECTION_SCHEMA = "pdt-historical-source-rejections-v1"
HISTORICAL_FILTER_FAILURE_SCHEMA = "pdt-historical-source-filter-failure-v1"
HISTORICAL_SELECTION_SCHEMA = "pdt-historical-source-selection-v1"
HISTORICAL_RENDERER_ID = "pdt-semantic-dom-allowlist-v1"

MIN_SOURCE_TOKENS = 3_000
MAX_SOURCE_TOKENS = 7_000
MIN_SUBSTANTIVE_SECTIONS = 6
MIN_SUBSTANTIVE_PARAGRAPHS = 12
MIN_PARAGRAPH_CHARACTERS = 80
MIN_PARAGRAPH_WORDS = 20
MIN_REFERENCE_OCCURRENCES = 30
MIN_DISTINCT_REFERENCES = 15
MIN_SCHOLARLY_REFERENCES = 8
MIN_CITED_PARAGRAPH_FRACTION = 0.70
MAX_REFERENCE_DOMINANCE = 0.25
MIN_EVENT_AGE_YEARS = 25


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HistoricalTopicKind(StrEnum):
    EVENT = "event"
    PROCESS = "process"
    BIOGRAPHY = "biography"
    LIST = "list"
    TIMELINE = "timeline"
    CATALOG = "catalog"
    CURRENT_EVENT = "current_event"
    OTHER = "other"


class HistoricalCategory(StrEnum):
    WARS_BATTLES = "wars_battles"
    REVOLUTIONS_TRANSITIONS = "revolutions_transitions"
    TREATIES_CRISES = "treaties_crises"
    SOCIAL_MOVEMENTS_REFORMS = "social_movements_reforms"
    EXPLORATION_MIGRATION = "exploration_migration"
    SCIENTIFIC_INDUSTRIAL = "scientific_industrial"
    DISASTERS_RECONSTRUCTION = "disasters_reconstruction"
    CULTURAL_INSTITUTIONAL = "cultural_institutional"


class AssessmentQuality(StrEnum):
    FA = "FA"
    GA = "GA"
    A = "A"
    B = "B"
    C = "C"
    START = "Start"
    STUB = "Stub"
    LIST = "List"
    UNKNOWN = "Unknown"


class ReferenceKind(StrEnum):
    BOOK = "book"
    JOURNAL = "journal"
    ARCHIVE = "archive"
    INSTITUTIONAL = "institutional"
    THESIS = "thesis"
    NEWS = "news"
    WEB = "web"
    OTHER = "other"


SCHOLARLY_REFERENCE_KINDS = frozenset(
    {
        ReferenceKind.BOOK,
        ReferenceKind.JOURNAL,
        ReferenceKind.ARCHIVE,
        ReferenceKind.INSTITUTIONAL,
        ReferenceKind.THESIS,
    }
)


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RejectionCode(StrEnum):
    RAW_DIGEST_MISMATCH = "raw_digest_mismatch"
    WRONG_LANGUAGE = "wrong_language"
    WRONG_NAMESPACE = "wrong_namespace"
    REDIRECT = "redirect"
    DISAMBIGUATION = "disambiguation"
    INADMISSIBLE_TOPIC_TYPE = "inadmissible_topic_type"
    INADMISSIBLE_TITLE = "inadmissible_title"
    EVENT_TOO_RECENT = "event_too_recent"
    MISSING_RELEVANT_ASSESSMENT = "missing_relevant_assessment"
    INCOMPLETE_CONTENT = "incomplete_content"
    MAINTENANCE_TEMPLATE = "maintenance_template"
    UNKNOWN_REFERENCE_ID = "unknown_reference_id"
    TOKEN_LENGTH = "token_length"
    INSUFFICIENT_SECTIONS = "insufficient_sections"
    INSUFFICIENT_PARAGRAPHS = "insufficient_paragraphs"
    INSUFFICIENT_REFERENCE_OCCURRENCES = "insufficient_reference_occurrences"
    INSUFFICIENT_DISTINCT_REFERENCES = "insufficient_distinct_references"
    INSUFFICIENT_SCHOLARLY_REFERENCES = "insufficient_scholarly_references"
    INSUFFICIENT_CITATION_COVERAGE = "insufficient_citation_coverage"
    REFERENCE_DOMINANCE = "reference_dominance"
    UNCLEAN_RENDER = "unclean_render"


class PageAssessment(_StrictModel):
    project: str = Field(min_length=1, max_length=300)
    quality: AssessmentQuality


class ReferenceRecord(_StrictModel):
    reference_id: str = Field(min_length=1, max_length=200)
    source_type: ReferenceKind
    citation_text: str = Field(min_length=10, max_length=4_000)
    title: str | None = Field(default=None, max_length=1_000)
    authors: tuple[str, ...] = Field(default_factory=tuple, max_length=30)
    publication: str | None = Field(default=None, max_length=1_000)
    year: int | None = Field(default=None, ge=1000, le=2200)
    identifiers: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference_id(self) -> ReferenceRecord:
        if any(ord(character) < 32 for character in self.reference_id):
            raise ValueError("Reference IDs cannot contain control characters.")
        return self


class RawWikimediaRevisionBundle(_StrictModel):
    """One complete immutable article revision plus non-visible audit metadata."""

    schema_version: str = Field(pattern=rf"^{RAW_WIKIMEDIA_SCHEMA}$")
    source_id: str = Field(min_length=1, max_length=200)
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    revision_timestamp: datetime
    dump_date: date
    language: str = Field(pattern=r"^en$")
    namespace: int
    title: str = Field(min_length=1, max_length=500)
    source_url: str = Field(min_length=1, max_length=2_000)
    license: str = Field(min_length=1, max_length=200)
    revision_sha1: str = Field(pattern=r"^[0-9a-f]{40}$")
    is_redirect: bool
    is_disambiguation: bool
    content_complete: bool
    topic_kind: HistoricalTopicKind
    historical_category: HistoricalCategory
    event_end_year: int = Field(ge=1, le=2200)
    assessments: tuple[PageAssessment, ...] = Field(default_factory=tuple, max_length=100)
    family_keys: tuple[str, ...] = Field(default_factory=tuple, max_length=100)
    related_page_ids: tuple[int, ...] = Field(default_factory=tuple, max_length=5_000)
    maintenance_templates: tuple[str, ...] = Field(default_factory=tuple, max_length=500)
    references: tuple[ReferenceRecord, ...] = Field(default_factory=tuple, max_length=5_000)
    action_api_response_json: str = Field(min_length=2)
    action_api_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_wikitext: str = Field(min_length=1)
    raw_wikitext_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_html: str = Field(min_length=1)
    semantic_html_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_and_collections(self) -> RawWikimediaRevisionBundle:
        if self.revision_timestamp.tzinfo is None:
            raise ValueError("revision_timestamp must be timezone-aware.")
        reference_ids = [record.reference_id for record in self.references]
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("Reference IDs must be unique within a revision bundle.")
        if len(set(self.related_page_ids)) != len(self.related_page_ids):
            raise ValueError("related_page_ids must not contain duplicates.")
        if self.page_id in self.related_page_ids:
            raise ValueError("related_page_ids cannot contain the article's own page_id.")
        normalized_family_keys = [key.strip().casefold() for key in self.family_keys]
        if any(not key for key in normalized_family_keys):
            raise ValueError("family_keys must contain non-empty text.")
        if len(set(normalized_family_keys)) != len(normalized_family_keys):
            raise ValueError("family_keys must be unique case-insensitively.")
        if hashlib.sha1(self.raw_wikitext.encode("utf-8")).hexdigest() != self.revision_sha1:
            raise ValueError("revision_sha1 does not match the complete raw wikitext.")
        return self


class HistoricalParagraph(_StrictModel):
    paragraph_id: str = Field(pattern=r"^p[0-9]{3}$")
    text: str = Field(min_length=MIN_PARAGRAPH_CHARACTERS)
    reference_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=500)


class HistoricalSection(_StrictModel):
    section_id: str = Field(pattern=r"^section_[0-9]{3}$")
    heading_path: tuple[str, ...] = Field(max_length=6)
    paragraphs: tuple[HistoricalParagraph, ...] = Field(min_length=1, max_length=200)


class HistoricalSource(_StrictModel):
    schema_version: str = Field(pattern=rf"^{HISTORICAL_SOURCE_SCHEMA}$")
    renderer_id: str = Field(pattern=rf"^{HISTORICAL_RENDERER_ID}$")
    source_id: str = Field(min_length=1, max_length=200)
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    revision_timestamp: datetime
    dump_date: date
    title: str = Field(min_length=1, max_length=500)
    source_url: str = Field(min_length=1, max_length=2_000)
    license: str = Field(min_length=1, max_length=200)
    revision_sha1: str = Field(pattern=r"^[0-9a-f]{40}$")
    historical_category: HistoricalCategory
    event_end_year: int
    assessments: tuple[PageAssessment, ...]
    family_id: str = Field(pattern=r"^family-[0-9a-f]{24}$")
    split: DatasetSplit
    tokenizer: str = Field(min_length=1, max_length=500)
    tokenizer_revision: str = Field(min_length=1, max_length=200)
    qwen_token_count: int = Field(ge=MIN_SOURCE_TOKENS, le=MAX_SOURCE_TOKENS)
    sections: tuple[HistoricalSection, ...] = Field(min_length=1, max_length=100)
    references: tuple[ReferenceRecord, ...] = Field(min_length=MIN_DISTINCT_REFERENCES)
    raw_wikitext_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_html_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_visible_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_paragraph_and_reference_identity(self) -> HistoricalSource:
        paragraphs = [
            paragraph
            for section in self.sections
            for paragraph in section.paragraphs
        ]
        paragraph_ids = [paragraph.paragraph_id for paragraph in paragraphs]
        if paragraph_ids != [f"p{index:03d}" for index in range(len(paragraphs))]:
            raise ValueError("Historical paragraph IDs must be consecutive in render order.")
        reference_ids = {record.reference_id for record in self.references}
        unknown = {
            reference_id
            for paragraph in paragraphs
            for reference_id in paragraph.reference_ids
            if reference_id not in reference_ids
        }
        if unknown:
            raise ValueError(f"Historical paragraphs name unknown references: {sorted(unknown)}.")
        rendered = render_historical_source(self)
        if _sha256_text(rendered) != self.model_visible_sha256:
            raise ValueError("model_visible_sha256 does not match the canonical renderer.")
        return self


class HistoricalManifestEntry(_StrictModel):
    source_id: str
    page_id: int
    revision_id: int
    family_id: str
    split: DatasetSplit
    historical_category: HistoricalCategory
    model_visible_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HistoricalSourceManifest(_StrictModel):
    schema_version: str = Field(pattern=rf"^{HISTORICAL_MANIFEST_SCHEMA}$")
    source_schema_version: str = Field(pattern=rf"^{HISTORICAL_SOURCE_SCHEMA}$")
    renderer_id: str = Field(pattern=rf"^{HISTORICAL_RENDERER_ID}$")
    tokenizer: str
    tokenizer_revision: str
    input_file_name: str
    input_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_file_bytes: int = Field(gt=0)
    source_file_name: str
    source_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_file_bytes: int = Field(gt=0)
    records: tuple[HistoricalManifestEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest_records(self) -> HistoricalSourceManifest:
        source_ids = [record.source_id for record in self.records]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Accepted manifest source IDs must be unique.")
        family_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
        for record in self.records:
            family_splits[record.family_id].add(record.split)
        leaked = {
            family_id: sorted(split.value for split in splits)
            for family_id, splits in family_splits.items()
            if len(splits) != 1
        }
        if leaked:
            raise ValueError(f"Historical families cross dataset splits: {leaked}.")
        return self


class HistoricalRejection(_StrictModel):
    source_id: str
    page_id: int
    revision_id: int
    reasons: tuple[RejectionCode, ...] = Field(min_length=1)
    details: Mapping[str, object] = Field(default_factory=dict)


class HistoricalRejectionManifest(_StrictModel):
    schema_version: str = Field(pattern=rf"^{HISTORICAL_REJECTION_SCHEMA}$")
    records: tuple[HistoricalRejection, ...]


class HistoricalFilterFailureManifest(_StrictModel):
    schema_version: str = Field(pattern=rf"^{HISTORICAL_FILTER_FAILURE_SCHEMA}$")
    failure_stage: str = Field(pattern=r"^(?:eligibility|selection)$")
    raw_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer: str = Field(min_length=1, max_length=500)
    tokenizer_revision: str = Field(min_length=1, max_length=200)
    requested_examples_per_category: int = Field(gt=0)
    candidate_count: int = Field(gt=0)
    eligible_source_ids: tuple[str, ...]
    eligible_source_file_name: str | None = Field(
        default=None,
        pattern=r"^eligible_sources\.jsonl$",
    )
    eligible_source_file_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    eligible_source_file_bytes: int | None = Field(default=None, gt=0)
    rejected_count: int = Field(ge=0)
    error: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_counts(self) -> HistoricalFilterFailureManifest:
        if len(set(self.eligible_source_ids)) != len(self.eligible_source_ids):
            raise ValueError("eligible_source_ids must be unique.")
        if len(self.eligible_source_ids) + self.rejected_count != self.candidate_count:
            raise ValueError(
                "Filter failure eligible and rejected counts must cover every candidate."
            )
        eligible_file_fields = (
            self.eligible_source_file_name,
            self.eligible_source_file_sha256,
            self.eligible_source_file_bytes,
        )
        if self.eligible_source_ids and any(value is None for value in eligible_file_fields):
            raise ValueError(
                "A filter failure with eligible sources must bind eligible_sources.jsonl."
            )
        if not self.eligible_source_ids and any(
            value is not None for value in eligible_file_fields
        ):
            raise ValueError(
                "A filter failure without eligible sources cannot name an eligible file."
            )
        return self


class HistoricalSelectionEntry(_StrictModel):
    source_id: str
    family_id: str
    historical_category: HistoricalCategory
    selected: bool
    deterministic_rank: str = Field(pattern=r"^[0-9a-f]{64}$")


class HistoricalSelectionManifest(_StrictModel):
    schema_version: str = Field(pattern=rf"^{HISTORICAL_SELECTION_SCHEMA}$")
    selection_rule: str = Field(pattern=r"^one-per-family-sha256-rank-v1$")
    examples_per_category: int = Field(gt=0)
    eligible_count: int = Field(gt=0)
    selected_count: int = Field(gt=0)
    records: tuple[HistoricalSelectionEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> HistoricalSelectionManifest:
        selected = [record for record in self.records if record.selected]
        if len(selected) != self.selected_count:
            raise ValueError("Historical selection selected_count is inconsistent.")
        if len(self.records) != self.eligible_count:
            raise ValueError("Historical selection eligible_count is inconsistent.")
        category_counts = Counter(record.historical_category for record in selected)
        expected = set(HistoricalCategory)
        if set(category_counts) != expected or any(
            count != self.examples_per_category for count in category_counts.values()
        ):
            raise ValueError("Historical selection is not exactly balanced by category.")
        family_ids = [record.family_id for record in selected]
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("Historical selection may include at most one article per family.")
        return self


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


class ParsedHistoricalArticle(_StrictModel):
    sections: tuple[HistoricalSection, ...]
    rendered_text: str
    unknown_reference_ids: tuple[str, ...] = ()


class EligibilityResult(_StrictModel):
    accepted: bool
    token_count: int
    reasons: tuple[RejectionCode, ...]
    details: Mapping[str, object]


_EXCLUDED_SECTION_NAMES = frozenset(
    {
        "references",
        "notes",
        "citations",
        "sources",
        "bibliography",
        "further reading",
        "external links",
        "see also",
        "gallery",
        "works cited",
        "footnotes",
        "literature",
    }
)
_FORBIDDEN_TAGS = frozenset(
    {
        "aside",
        "audio",
        "button",
        "canvas",
        "code",
        "figcaption",
        "figure",
        "form",
        "img",
        "input",
        "li",
        "map",
        "math",
        "nav",
        "noscript",
        "ol",
        "pre",
        "script",
        "style",
        "svg",
        "table",
        "template",
        "textarea",
        "ul",
        "video",
    }
)
_FORBIDDEN_ATTRIBUTE_FRAGMENTS = (
    "authority-control",
    "audiolink",
    "catlinks",
    "citation",
    "coordinates",
    "editsection",
    "gallery",
    "hatnote",
    "infobox",
    "metadata",
    "mw-edit",
    "mw-ref",
    "mw-references",
    "navbox",
    "noprint",
    "portalbox",
    "pronunciation",
    "reference",
    "reflist",
    "sidebar",
    "shortdescription",
    "thumb",
    "toc",
)
_CITATION_CLASS_FRAGMENTS = ("mw-ref", "reference")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_REFERENCE_MARKER_PATTERN = re.compile(r"\[(?:\d+|[a-z])\]")
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WORD_PATTERN = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)
_SPACE_PATTERN = re.compile(r"\s+")
_INADMISSIBLE_TITLE_PATTERN = re.compile(
    r"^(?:list|timeline|chronology|index|outline) of\b|"
    r"^(?:\d{1,4}(?:s)?|[a-z]+ century)$",
    flags=re.IGNORECASE,
)
_MAINTENANCE_NAMES = (
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
_HISTORY_PROJECT_FRAGMENTS = (
    "history",
    "military history",
    "wars",
    "revolutions",
    "social movements",
    "exploration",
    "disasters",
)
_ACCEPTED_QUALITIES = frozenset(
    {
        AssessmentQuality.FA,
        AssessmentQuality.GA,
        AssessmentQuality.A,
        AssessmentQuality.B,
    }
)


def verify_raw_bundle_digests(bundle: RawWikimediaRevisionBundle) -> None:
    """Fail if immutable wikitext or semantic HTML bytes do not match metadata."""

    mismatches: list[str] = []
    if _sha256_text(bundle.raw_wikitext) != bundle.raw_wikitext_sha256:
        mismatches.append("raw_wikitext_sha256")
    if _sha256_text(bundle.semantic_html) != bundle.semantic_html_sha256:
        mismatches.append("semantic_html_sha256")
    if _sha256_text(bundle.action_api_response_json) != bundle.action_api_response_sha256:
        mismatches.append("action_api_response_sha256")
    if mismatches:
        raise ValueError(f"Raw Wikimedia bundle digest mismatch: {mismatches}.")


def parse_historical_dom(bundle: RawWikimediaRevisionBundle) -> ParsedHistoricalArticle:
    """Extract only hierarchical headings, ordinary prose, and citation IDs."""

    verify_raw_bundle_digests(bundle)
    soup = BeautifulSoup(bundle.semantic_html, "html.parser")
    root = soup.find("main") or soup.find("body") or soup
    reference_ids = {record.reference_id for record in bundle.references}
    heading_stack: list[str] = []
    excluded_heading_level: int | None = None
    section_rows: list[tuple[tuple[str, ...], list[HistoricalParagraph]]] = [
        (tuple(), [])
    ]
    paragraph_counter = 0
    unknown_references: set[str] = set()

    for node in root.find_all(("h2", "h3", "h4", "h5", "h6", "p")):
        if not isinstance(node, Tag) or _has_forbidden_ancestor(node, stop=root):
            continue
        if node.name and node.name.startswith("h"):
            level = int(node.name[1])
            heading = _visible_text(node)
            if not heading:
                continue
            normalized_heading = _normalize_heading(heading)
            if excluded_heading_level is not None:
                if level > excluded_heading_level:
                    continue
                excluded_heading_level = None
            if _is_excluded_section_heading(normalized_heading):
                excluded_heading_level = level
                continue
            depth = level - 2
            heading_stack = heading_stack[:depth]
            heading_stack.append(heading)
            section_rows.append((tuple(heading_stack), []))
            continue

        if excluded_heading_level is not None or _is_forbidden_node(node):
            continue
        text = _visible_text(node)
        if not _is_substantive_paragraph(text):
            continue
        occurrences = tuple(_paragraph_reference_occurrences(node))
        for reference_id in occurrences:
            if reference_id not in reference_ids:
                unknown_references.add(reference_id)
        paragraph = HistoricalParagraph(
            paragraph_id=f"p{paragraph_counter:03d}",
            text=text,
            reference_ids=occurrences,
        )
        paragraph_counter += 1
        section_rows[-1][1].append(paragraph)

    sections = tuple(
        HistoricalSection(
            section_id=f"section_{section_index:03d}",
            heading_path=heading_path,
            paragraphs=tuple(paragraphs),
        )
        for section_index, (heading_path, paragraphs) in enumerate(
            row for row in section_rows if row[1]
        )
    )
    rendered = _render_parts(bundle.title, sections)
    _validate_model_visible_text(rendered)
    return ParsedHistoricalArticle(
        sections=sections,
        rendered_text=rendered,
        unknown_reference_ids=tuple(sorted(unknown_references)),
    )


def evaluate_historical_eligibility(
    bundle: RawWikimediaRevisionBundle,
    parsed: ParsedHistoricalArticle,
    *,
    tokenizer: TokenizerLike,
) -> EligibilityResult:
    """Apply every fixed historical, structural, and citation admission gate."""

    reasons: set[RejectionCode] = set()
    details: dict[str, object] = {}
    try:
        verify_raw_bundle_digests(bundle)
    except ValueError:
        reasons.add(RejectionCode.RAW_DIGEST_MISMATCH)
    if bundle.language != "en":
        reasons.add(RejectionCode.WRONG_LANGUAGE)
    if bundle.namespace != 0:
        reasons.add(RejectionCode.WRONG_NAMESPACE)
    if bundle.is_redirect:
        reasons.add(RejectionCode.REDIRECT)
    if bundle.is_disambiguation:
        reasons.add(RejectionCode.DISAMBIGUATION)
    if bundle.topic_kind not in (HistoricalTopicKind.EVENT, HistoricalTopicKind.PROCESS):
        reasons.add(RejectionCode.INADMISSIBLE_TOPIC_TYPE)
    if _INADMISSIBLE_TITLE_PATTERN.search(bundle.title.strip()):
        reasons.add(RejectionCode.INADMISSIBLE_TITLE)
    if bundle.event_end_year > bundle.dump_date.year - MIN_EVENT_AGE_YEARS:
        reasons.add(RejectionCode.EVENT_TOO_RECENT)
    if not any(
        assessment.quality in _ACCEPTED_QUALITIES
        and any(
            fragment in assessment.project.casefold()
            for fragment in _HISTORY_PROJECT_FRAGMENTS
        )
        for assessment in bundle.assessments
    ):
        reasons.add(RejectionCode.MISSING_RELEVANT_ASSESSMENT)
    if not bundle.content_complete:
        reasons.add(RejectionCode.INCOMPLETE_CONTENT)
    maintenance = _maintenance_templates(bundle)
    if maintenance:
        reasons.add(RejectionCode.MAINTENANCE_TEMPLATE)
        details["maintenance_templates"] = maintenance
    if parsed.unknown_reference_ids:
        reasons.add(RejectionCode.UNKNOWN_REFERENCE_ID)
        details["unknown_reference_ids"] = parsed.unknown_reference_ids
    try:
        _validate_model_visible_text(parsed.rendered_text)
    except ValueError as exc:
        reasons.add(RejectionCode.UNCLEAN_RENDER)
        details["render_error"] = str(exc)

    token_ids = tokenizer.encode(parsed.rendered_text, add_special_tokens=False)
    if not isinstance(token_ids, Sequence) or any(type(token) is not int for token in token_ids):
        raise TypeError("Pinned tokenizer.encode must return a sequence of integer token IDs.")
    token_count = len(token_ids)
    if not MIN_SOURCE_TOKENS <= token_count <= MAX_SOURCE_TOKENS:
        reasons.add(RejectionCode.TOKEN_LENGTH)
    paragraphs = [
        paragraph
        for section in parsed.sections
        for paragraph in section.paragraphs
    ]
    substantive_sections = sum(bool(section.heading_path) for section in parsed.sections)
    if substantive_sections < MIN_SUBSTANTIVE_SECTIONS:
        reasons.add(RejectionCode.INSUFFICIENT_SECTIONS)
    if len(paragraphs) < MIN_SUBSTANTIVE_PARAGRAPHS:
        reasons.add(RejectionCode.INSUFFICIENT_PARAGRAPHS)

    occurrences = [
        reference_id
        for paragraph in paragraphs
        for reference_id in paragraph.reference_ids
    ]
    occurrence_counts = Counter(occurrences)
    distinct_references = set(occurrences)
    scholarly = {
        record.reference_id
        for record in bundle.references
        if record.reference_id in distinct_references
        and record.source_type in SCHOLARLY_REFERENCE_KINDS
    }
    cited_paragraphs = sum(bool(paragraph.reference_ids) for paragraph in paragraphs)
    coverage = cited_paragraphs / len(paragraphs) if paragraphs else 0.0
    dominance = (
        max(occurrence_counts.values()) / len(occurrences)
        if occurrence_counts
        else math.inf
    )
    if len(occurrences) < MIN_REFERENCE_OCCURRENCES:
        reasons.add(RejectionCode.INSUFFICIENT_REFERENCE_OCCURRENCES)
    if len(distinct_references) < MIN_DISTINCT_REFERENCES:
        reasons.add(RejectionCode.INSUFFICIENT_DISTINCT_REFERENCES)
    if len(scholarly) < MIN_SCHOLARLY_REFERENCES:
        reasons.add(RejectionCode.INSUFFICIENT_SCHOLARLY_REFERENCES)
    if coverage < MIN_CITED_PARAGRAPH_FRACTION:
        reasons.add(RejectionCode.INSUFFICIENT_CITATION_COVERAGE)
    if dominance > MAX_REFERENCE_DOMINANCE:
        reasons.add(RejectionCode.REFERENCE_DOMINANCE)
    details.update(
        {
            "qwen_token_count": token_count,
            "substantive_sections": substantive_sections,
            "substantive_paragraphs": len(paragraphs),
            "reference_occurrences": len(occurrences),
            "distinct_references": len(distinct_references),
            "scholarly_references": len(scholarly),
            "cited_paragraph_fraction": coverage,
            "maximum_reference_dominance": dominance,
        }
    )
    ordered_reasons = tuple(sorted(reasons, key=lambda reason: reason.value))
    return EligibilityResult(
        accepted=not ordered_reasons,
        token_count=token_count,
        reasons=ordered_reasons,
        details=details,
    )


def cluster_historical_families(
    rows: Sequence[tuple[RawWikimediaRevisionBundle, ParsedHistoricalArticle]],
) -> Mapping[str, str]:
    """Cluster explicit relations and near-duplicate article signatures."""

    if not rows:
        raise ValueError("Historical family clustering requires at least one article.")
    source_ids = [bundle.source_id for bundle, _ in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Historical family clustering requires unique source IDs.")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    page_index = {bundle.page_id: index for index, (bundle, _) in enumerate(rows)}
    key_index: dict[str, int] = {}
    title_index: dict[str, int] = {}
    simhashes: list[int] = []
    for index, (bundle, parsed) in enumerate(rows):
        for page_id in bundle.related_page_ids:
            related = page_index.get(page_id)
            if related is not None:
                union(index, related)
        for family_key in bundle.family_keys:
            normalized = family_key.strip().casefold()
            previous = key_index.setdefault(normalized, index)
            union(index, previous)
        normalized_title = _normalized_title(bundle.title)
        previous_title = title_index.setdefault(normalized_title, index)
        union(index, previous_title)
        simhashes.append(_simhash(f"{bundle.title}\n{parsed.rendered_text[:4_000]}"))
    # Four disjoint 16-bit bands are an exact candidate index for Hamming
    # radius three: with at most three differing bits, at least one band must
    # be identical. The final distance check prevents false-positive unions.
    band_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()
    for index, simhash in enumerate(simhashes):
        for band in range(4):
            key = (band, (simhash >> (band * 16)) & 0xFFFF)
            for previous in band_buckets[key]:
                candidate_pairs.add((previous, index))
            band_buckets[key].append(index)
    for left, right in sorted(candidate_pairs):
        if (simhashes[left] ^ simhashes[right]).bit_count() <= 3:
            union(left, right)

    members: dict[int, list[str]] = defaultdict(list)
    for index, source_id in enumerate(source_ids):
        members[find(index)].append(source_id)
    family_by_root = {
        root: "family-" + hashlib.sha256(
            "\n".join(sorted(group)).encode("utf-8")
        ).hexdigest()[:24]
        for root, group in members.items()
    }
    return {
        source_id: family_by_root[find(index)]
        for index, source_id in enumerate(source_ids)
    }


def split_for_family(family_id: str) -> DatasetSplit:
    """Assign one deterministic 90/5/5 split at family granularity."""

    if not re.fullmatch(r"family-[0-9a-f]{24}", family_id):
        raise ValueError(f"Malformed family_id {family_id!r}.")
    bucket = int.from_bytes(hashlib.sha256(family_id.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < 90:
        return DatasetSplit.TRAIN
    if bucket < 95:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def select_balanced_historical_sources(
    sources: Sequence[HistoricalSource],
    *,
    examples_per_category: int,
) -> tuple[tuple[HistoricalSource, ...], HistoricalSelectionManifest]:
    """Select an exact category balance without duplicating event families."""

    if examples_per_category <= 0:
        raise ValueError("examples_per_category must be positive.")
    if not sources:
        raise ValueError("Historical source selection requires eligible sources.")
    source_ids = [source.source_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Historical source selection requires unique source IDs.")
    categories_by_family: dict[str, set[HistoricalCategory]] = defaultdict(set)
    for source in sources:
        categories_by_family[source.family_id].add(source.historical_category)
    inconsistent = {
        family_id: sorted(category.value for category in categories)
        for family_id, categories in categories_by_family.items()
        if len(categories) != 1
    }
    if inconsistent:
        raise ValueError(
            "Related historical families must have one reconciled category before "
            f"selection: {inconsistent}."
        )

    rank_by_id = {
        source.source_id: hashlib.sha256(
            (
                f"{source.source_id}\n{source.revision_id}\n"
                f"{source.model_visible_sha256}"
            ).encode("utf-8")
        ).hexdigest()
        for source in sources
    }
    selected_ids: set[str] = set()
    for category in HistoricalCategory:
        candidates = sorted(
            (
                source
                for source in sources
                if source.historical_category is category
            ),
            key=lambda source: (rank_by_id[source.source_id], source.source_id),
        )
        unique_families: set[str] = set()
        for source in candidates:
            if source.family_id in unique_families:
                continue
            unique_families.add(source.family_id)
            selected_ids.add(source.source_id)
            if len(unique_families) == examples_per_category:
                break
        if len(unique_families) != examples_per_category:
            raise ValueError(
                f"Category {category.value!r} has only {len(unique_families)} eligible "
                f"families; {examples_per_category} are required. Search more candidates "
                "without weakening admission thresholds."
            )

    selected = tuple(source for source in sources if source.source_id in selected_ids)
    manifest = HistoricalSelectionManifest(
        schema_version=HISTORICAL_SELECTION_SCHEMA,
        selection_rule="one-per-family-sha256-rank-v1",
        examples_per_category=examples_per_category,
        eligible_count=len(sources),
        selected_count=len(selected),
        records=tuple(
            HistoricalSelectionEntry(
                source_id=source.source_id,
                family_id=source.family_id,
                historical_category=source.historical_category,
                selected=source.source_id in selected_ids,
                deterministic_rank=rank_by_id[source.source_id],
            )
            for source in sorted(sources, key=lambda row: row.source_id)
        ),
    )
    return selected, manifest


def materialize_historical_source(
    bundle: RawWikimediaRevisionBundle,
    parsed: ParsedHistoricalArticle,
    eligibility: EligibilityResult,
    *,
    family_id: str,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> HistoricalSource:
    if not eligibility.accepted:
        raise ValueError(
            f"Cannot materialize rejected historical source {bundle.source_id!r}: "
            f"{[reason.value for reason in eligibility.reasons]}."
        )
    if not tokenizer_name or not tokenizer_revision:
        raise ValueError("Tokenizer identity and revision must be non-empty.")
    used_reference_ids = {
        reference_id
        for section in parsed.sections
        for paragraph in section.paragraphs
        for reference_id in paragraph.reference_ids
    }
    references = tuple(
        record for record in bundle.references if record.reference_id in used_reference_ids
    )
    return HistoricalSource(
        schema_version=HISTORICAL_SOURCE_SCHEMA,
        renderer_id=HISTORICAL_RENDERER_ID,
        source_id=bundle.source_id,
        page_id=bundle.page_id,
        revision_id=bundle.revision_id,
        revision_timestamp=bundle.revision_timestamp,
        dump_date=bundle.dump_date,
        title=bundle.title,
        source_url=bundle.source_url,
        license=bundle.license,
        revision_sha1=bundle.revision_sha1,
        historical_category=bundle.historical_category,
        event_end_year=bundle.event_end_year,
        assessments=bundle.assessments,
        family_id=family_id,
        split=split_for_family(family_id),
        tokenizer=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        qwen_token_count=eligibility.token_count,
        sections=parsed.sections,
        references=references,
        raw_wikitext_sha256=bundle.raw_wikitext_sha256,
        semantic_html_sha256=bundle.semantic_html_sha256,
        model_visible_sha256=_sha256_text(parsed.rendered_text),
    )


def render_historical_source(source: HistoricalSource) -> str:
    """Render only the title, hierarchical headings, and ordinary prose."""

    rendered = render_historical_sections(source.title, source.sections)
    _validate_model_visible_text(rendered)
    return rendered


def render_historical_sections(
    title: str,
    sections: Sequence[HistoricalSection],
) -> str:
    """Render a validated title/section sequence before source construction."""

    return _render_parts(title, sections)


def manifest_for_source_file(
    source_path: Path,
    sources: Sequence[HistoricalSource],
    *,
    input_path: Path,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> HistoricalSourceManifest:
    if not source_path.is_file():
        raise FileNotFoundError(f"Accepted historical source file does not exist: {source_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw revision bundle file does not exist: {input_path}")
    if not sources:
        raise ValueError("Accepted historical manifest requires at least one source.")
    return HistoricalSourceManifest(
        schema_version=HISTORICAL_MANIFEST_SCHEMA,
        source_schema_version=HISTORICAL_SOURCE_SCHEMA,
        renderer_id=HISTORICAL_RENDERER_ID,
        tokenizer=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        input_file_name=input_path.name,
        input_file_sha256=sha256_file(input_path),
        input_file_bytes=input_path.stat().st_size,
        source_file_name=source_path.name,
        source_file_sha256=sha256_file(source_path),
        source_file_bytes=source_path.stat().st_size,
        records=tuple(
            HistoricalManifestEntry(
                source_id=source.source_id,
                page_id=source.page_id,
                revision_id=source.revision_id,
                family_id=source.family_id,
                split=source.split,
                historical_category=source.historical_category,
                model_visible_sha256=source.model_visible_sha256,
            )
            for source in sources
        ),
    )


def verify_accepted_manifest(
    source_path: Path,
    manifest_path: Path,
) -> HistoricalSourceManifest:
    """Require an exact accepted-manifest match before any paid request."""

    if not source_path.is_file():
        raise FileNotFoundError(f"Historical source file does not exist: {source_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Accepted source manifest does not exist: {manifest_path}")
    try:
        manifest = HistoricalSourceManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError(f"Accepted source manifest is invalid: {manifest_path}") from exc
    if manifest.source_file_name != source_path.name:
        raise ValueError(
            "Accepted source manifest names a different source file: "
            f"expected {source_path.name!r}, got {manifest.source_file_name!r}."
        )
    actual_size = source_path.stat().st_size
    actual_digest = sha256_file(source_path)
    if actual_size != manifest.source_file_bytes or actual_digest != manifest.source_file_sha256:
        raise ValueError(
            "Historical source file does not match its accepted manifest: "
            f"expected bytes={manifest.source_file_bytes}, sha256={manifest.source_file_sha256}; "
            f"got bytes={actual_size}, sha256={actual_digest}."
        )
    sources = load_historical_sources(source_path)
    manifest_by_id = {record.source_id: record for record in manifest.records}
    if {source.source_id for source in sources} != set(manifest_by_id):
        raise ValueError("Accepted manifest does not exactly cover historical source rows.")
    for source in sources:
        record = manifest_by_id[source.source_id]
        if (
            source.page_id != record.page_id
            or source.revision_id != record.revision_id
            or source.family_id != record.family_id
            or source.split != record.split
            or source.historical_category != record.historical_category
            or source.model_visible_sha256 != record.model_visible_sha256
        ):
            raise ValueError(
                f"Accepted manifest metadata differs for source {source.source_id!r}."
            )
    return manifest


def verify_filter_failure_bundle(
    bundle_dir: Path,
) -> HistoricalFilterFailureManifest:
    """Verify a failed screen and any retained eligible source pool."""

    failure_path = bundle_dir / "failure.json"
    rejections_path = bundle_dir / "rejections.json"
    if not failure_path.is_file() or not rejections_path.is_file():
        raise FileNotFoundError(
            f"Filter failure bundle is incomplete: {bundle_dir}."
        )
    try:
        failure = HistoricalFilterFailureManifest.model_validate_json(
            failure_path.read_text(encoding="utf-8")
        )
        rejections = HistoricalRejectionManifest.model_validate_json(
            rejections_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise ValueError(f"Filter failure bundle is invalid: {bundle_dir}.") from exc
    rejected_ids = [record.source_id for record in rejections.records]
    if len(set(rejected_ids)) != len(rejected_ids):
        raise ValueError("Filter failure rejection source IDs must be unique.")
    if len(rejected_ids) != failure.rejected_count:
        raise ValueError("Filter failure rejected_count does not match rejections.json.")
    if set(rejected_ids) & set(failure.eligible_source_ids):
        raise ValueError("A source cannot be both eligible and rejected.")

    eligible_path = bundle_dir / "eligible_sources.jsonl"
    if failure.eligible_source_ids:
        if failure.eligible_source_file_name != eligible_path.name:
            raise ValueError("Filter failure names a non-canonical eligible source file.")
        if not eligible_path.is_file():
            raise FileNotFoundError(
                f"Filter failure eligible source file does not exist: {eligible_path}."
            )
        if (
            eligible_path.stat().st_size != failure.eligible_source_file_bytes
            or sha256_file(eligible_path) != failure.eligible_source_file_sha256
        ):
            raise ValueError(
                "Filter failure eligible source file differs from its integrity binding."
            )
        eligible_sources = load_historical_sources(eligible_path)
        if tuple(source.source_id for source in eligible_sources) != (
            failure.eligible_source_ids
        ):
            raise ValueError(
                "Filter failure eligible source IDs differ from eligible_sources.jsonl."
            )
    elif eligible_path.exists():
        raise ValueError(
            "Filter failure without eligible source IDs cannot contain eligible_sources.jsonl."
        )
    for forbidden_name in ("accepted_manifest.json", "accepted_sources.jsonl"):
        if (bundle_dir / forbidden_name).exists():
            raise ValueError(
                f"Filter failure bundle cannot contain {forbidden_name}."
            )
    return failure


def load_historical_sources(path: Path) -> tuple[HistoricalSource, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Historical source JSONL does not exist: {path}")
    sources: list[HistoricalSource] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                source = HistoricalSource.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} violates {HISTORICAL_SOURCE_SCHEMA}."
                ) from exc
            if source.source_id in seen:
                raise ValueError(f"{path}:{line_number} repeats {source.source_id!r}.")
            seen.add(source.source_id)
            sources.append(source)
    if not sources:
        raise ValueError(f"{path} contains no historical sources.")
    return tuple(sources)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: BaseModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _render_parts(title: str, sections: Sequence[HistoricalSection]) -> str:
    parts = [title.strip()]
    previous_path: tuple[str, ...] = tuple()
    for section in sections:
        common = 0
        for left, right in zip(previous_path, section.heading_path):
            if left != right:
                break
            common += 1
        parts.extend(section.heading_path[common:])
        parts.extend(paragraph.text for paragraph in section.paragraphs)
        previous_path = section.heading_path
    rendered = "\n\n".join(part for part in parts if part).strip()
    if not rendered:
        raise ValueError("Historical source renderer produced empty text.")
    return rendered


def _visible_text(node: Tag) -> str:
    pieces: list[str] = []

    def visit(value: Tag | NavigableString) -> None:
        if isinstance(value, NavigableString):
            pieces.append(str(value))
            return
        if _is_forbidden_node(value):
            return
        if value.name == "a":
            display = _SPACE_PATTERN.sub(" ", value.get_text(" ", strip=True)).strip()
            if display and not _URL_PATTERN.fullmatch(display):
                pieces.append(display)
            return
        for child in value.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(child)

    visit(node)
    text = _SPACE_PATTERN.sub(" ", " ".join(pieces)).strip()
    text = _URL_PATTERN.sub("", text)
    text = _REFERENCE_MARKER_PATTERN.sub("", text)
    return _SPACE_PATTERN.sub(" ", text).strip()


def _is_forbidden_node(node: Tag) -> bool:
    if node.name in _FORBIDDEN_TAGS:
        return True
    attributes = " ".join(
        [
            str(node.get("id", "")),
            " ".join(_attribute_values(node, "class")),
            str(node.get("role", "")),
            str(node.get("typeof", "")),
        ]
    ).casefold()
    return any(fragment in attributes for fragment in _FORBIDDEN_ATTRIBUTE_FRAGMENTS)


def _has_forbidden_ancestor(node: Tag, *, stop: Tag) -> bool:
    parent = node.parent
    while isinstance(parent, Tag) and parent is not stop:
        if _is_forbidden_node(parent):
            return True
        parent = parent.parent
    return False


def _paragraph_reference_occurrences(paragraph: Tag) -> Iterable[str]:
    for node in paragraph.find_all(True):
        if not isinstance(node, Tag):
            continue
        attributes = " ".join(
            (
                str(node.get("id", "")),
                " ".join(_attribute_values(node, "class")),
                str(node.get("typeof", "")),
            )
        ).casefold()
        if not (
            any(fragment in attributes for fragment in _CITATION_CLASS_FRAGMENTS)
            or "mw:extension/ref" in attributes
        ):
            continue
        anchor = node.find("a", href=True)
        if not isinstance(anchor, Tag):
            continue
        reference_id = _reference_id_from_href(str(anchor.get("href", "")))
        if reference_id:
            yield reference_id


def _reference_id_from_href(href: str) -> str | None:
    fragment = href.rsplit("#", 1)[-1]
    if not fragment or fragment == href:
        return None
    normalized = fragment.strip().removeprefix("cite_note-")
    normalized = re.sub(r"[^A-Za-z0-9_.:-]", "_", normalized)
    return normalized or None


def _attribute_values(node: Tag, name: str) -> tuple[str, ...]:
    value = node.get(name)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _normalize_heading(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", value).strip().casefold().rstrip(":")


def _is_excluded_section_heading(normalized_heading: str) -> bool:
    if normalized_heading in _EXCLUDED_SECTION_NAMES:
        return True
    parts = tuple(
        part.strip()
        for part in re.split(r"\s+(?:and|&)\s+", normalized_heading)
        if part.strip()
    )
    return len(parts) > 1 and all(part in _EXCLUDED_SECTION_NAMES for part in parts)


def _is_substantive_paragraph(text: str) -> bool:
    return (
        len(text) >= MIN_PARAGRAPH_CHARACTERS
        and len(_WORD_PATTERN.findall(text)) >= MIN_PARAGRAPH_WORDS
    )


def _maintenance_templates(bundle: RawWikimediaRevisionBundle) -> tuple[str, ...]:
    declared = {name.strip().casefold() for name in bundle.maintenance_templates}
    return tuple(
        sorted(
            name
            for name in declared
            if any(
                name == forbidden or name.startswith(f"{forbidden} ")
                for forbidden in _MAINTENANCE_NAMES
            )
        )
    )


def _validate_model_visible_text(text: str) -> None:
    if not text.strip():
        raise ValueError("Model-visible historical text is empty.")
    if _TAG_PATTERN.search(text):
        raise ValueError("Model-visible historical text contains markup tags.")
    if _URL_PATTERN.search(text):
        raise ValueError("Model-visible historical text contains a URL.")
    if _REFERENCE_MARKER_PATTERN.search(text):
        raise ValueError("Model-visible historical text contains a citation marker.")


def _normalized_title(title: str) -> str:
    title = re.sub(r"\([^)]*\)", " ", title.casefold())
    return _SPACE_PATTERN.sub(" ", re.sub(r"[^\w]+", " ", title)).strip()


def _simhash(text: str) -> int:
    features = set(_WORD_PATTERN.findall(text.casefold()))
    if not features:
        return 0
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "AssessmentQuality",
    "DatasetSplit",
    "EligibilityResult",
    "HISTORICAL_FILTER_FAILURE_SCHEMA",
    "HISTORICAL_MANIFEST_SCHEMA",
    "HISTORICAL_RENDERER_ID",
    "HISTORICAL_REJECTION_SCHEMA",
    "HISTORICAL_SELECTION_SCHEMA",
    "HISTORICAL_SOURCE_SCHEMA",
    "HistoricalCategory",
    "HistoricalFilterFailureManifest",
    "HistoricalManifestEntry",
    "HistoricalParagraph",
    "HistoricalRejection",
    "HistoricalRejectionManifest",
    "HistoricalSection",
    "HistoricalSelectionEntry",
    "HistoricalSelectionManifest",
    "HistoricalSource",
    "HistoricalSourceManifest",
    "HistoricalTopicKind",
    "PageAssessment",
    "ParsedHistoricalArticle",
    "RAW_WIKIMEDIA_SCHEMA",
    "RawWikimediaRevisionBundle",
    "ReferenceKind",
    "ReferenceRecord",
    "RejectionCode",
    "canonical_json_bytes",
    "cluster_historical_families",
    "evaluate_historical_eligibility",
    "load_historical_sources",
    "manifest_for_source_file",
    "materialize_historical_source",
    "parse_historical_dom",
    "render_historical_sections",
    "render_historical_source",
    "select_balanced_historical_sources",
    "sha256_file",
    "split_for_family",
    "verify_accepted_manifest",
    "verify_filter_failure_bundle",
    "verify_raw_bundle_digests",
]
