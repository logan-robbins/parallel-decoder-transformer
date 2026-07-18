from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ConfigDict, Field, model_validator


EXCLUDED_SECTION_TITLES = frozenset(
    {
        "bibliography",
        "citations",
        "external links",
        "further reading",
        "gallery",
        "notes",
        "notes and citations",
        "notes and references",
        "notes and sources",
        "notes, citations and sources",
        "notes, references and sources",
        "references",
        "see also",
        "sources",
    }
)
HEADING_TAGS = ("h2", "h3", "h4", "h5", "h6")
MIN_PARAGRAPH_CHARACTERS = 80
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceIdentity(StrictModel):
    source_id: str
    title: str
    page_id: int
    revision_id: int
    revision_timestamp: str
    source_url: str
    language: Literal["en"]
    license: str
    raw_wikitext_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_html_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceCitation(StrictModel):
    reference_id: str
    citation_text: str
    title: str | None
    authors: tuple[str, ...]
    publication: str | None
    year: int | str | None
    identifiers: dict[str, str]
    source_type: str


class SourceParagraph(StrictModel):
    paragraph_id: str = Field(pattern=r"^p[0-9]{3}$")
    section_path: tuple[str, ...]
    text: str = Field(min_length=MIN_PARAGRAPH_CHARACTERS)
    citation_ids: tuple[str, ...]


class WikipediaSourceDocument(StrictModel):
    schema_version: Literal["model-intrinsic-parallel-wikipedia-source-v1"]
    source: SourceIdentity
    headings: tuple[str, ...]
    paragraphs: tuple[SourceParagraph, ...]
    citations: tuple[SourceCitation, ...]
    model_visible_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_document(self) -> WikipediaSourceDocument:
        paragraph_ids = [paragraph.paragraph_id for paragraph in self.paragraphs]
        expected_ids = [f"p{index:03d}" for index in range(len(self.paragraphs))]
        if paragraph_ids != expected_ids:
            raise ValueError("paragraph IDs must be consecutive in source order")

        citation_ids = [citation.reference_id for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("source citation IDs must be unique")
        known_citations = set(citation_ids)
        for paragraph in self.paragraphs:
            unknown = set(paragraph.citation_ids) - known_citations
            if unknown:
                raise ValueError(
                    f"{paragraph.paragraph_id} references unknown citation IDs: {sorted(unknown)}"
                )

        visible_text = render_model_visible_text(self.headings, self.paragraphs)
        actual_hash = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()
        if actual_hash != self.model_visible_text_sha256:
            raise ValueError(
                "model-visible source hash does not match headings and paragraphs"
            )
        return self


def load_pinned_revision(
    raw_jsonl_path: Path,
    *,
    title: str,
    revision_id: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with raw_jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{raw_jsonl_path}:{line_number} is not valid JSON: {exc}"
                ) from exc
            if (
                record.get("title") == title
                and record.get("revision_id") == revision_id
            ):
                matches.append(record)

    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one raw revision for title={title!r}, revision_id={revision_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def extract_reference_records(semantic_html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(semantic_html, "html.parser")
    prefix = "mw-reference-text-cite_note-"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in soup.find_all(
        id=lambda value: isinstance(value, str) and value.startswith(prefix)
    ):
        element_id = str(element.get("id"))
        reference_id = element_id.removeprefix(prefix)
        if not reference_id or reference_id in seen:
            continue
        citation_text = _clean_text(element)
        if not citation_text:
            raise ValueError(f"reference {reference_id!r} has no citation text")
        records.append(
            {
                "reference_id": reference_id,
                "citation_text": citation_text,
                "title": None,
                "authors": [],
                "publication": None,
                "year": None,
                "identifiers": {},
                "source_type": "unclassified",
            }
        )
        seen.add(reference_id)
    if not records:
        raise ValueError("semantic HTML contains no resolvable reference records")
    return records


def extract_wikipedia_source(raw_record: dict[str, Any]) -> WikipediaSourceDocument:
    semantic_html = _required_string(raw_record, "semantic_html")
    expected_html_hash = _required_string(raw_record, "semantic_html_sha256")
    actual_html_hash = hashlib.sha256(semantic_html.encode("utf-8")).hexdigest()
    if actual_html_hash != expected_html_hash:
        raise ValueError(
            "raw semantic_html bytes do not match semantic_html_sha256; refusing extraction"
        )

    soup = BeautifulSoup(semantic_html, "html.parser")
    body = soup.body
    if body is None:
        raise ValueError("raw semantic HTML has no body")

    headings: list[str] = []
    paragraphs: list[SourceParagraph] = []
    heading_stack: list[str] = []

    for section in body.find_all("section"):
        heading = section.find(HEADING_TAGS, recursive=False)
        if heading is None:
            section_path: tuple[str, ...] = ()
        else:
            heading_text = _clean_text(heading)
            if not heading_text:
                raise ValueError("encountered an empty content heading")
            heading_level = int(heading.name[1])
            stack_index = heading_level - 2
            heading_stack = heading_stack[:stack_index]
            heading_stack.append(heading_text)
            section_path = tuple(heading_stack)
            if _normalized_heading(heading_text) in EXCLUDED_SECTION_TITLES:
                continue
            headings.append(heading_text)

        if section_path and any(
            _normalized_heading(part) in EXCLUDED_SECTION_TITLES
            for part in section_path
        ):
            continue

        for paragraph_tag in section.find_all("p", recursive=False):
            if _is_non_content_paragraph(paragraph_tag):
                continue
            citation_ids = tuple(_citation_ids(paragraph_tag))
            paragraph_text = _paragraph_text(paragraph_tag)
            if _introduces_excluded_block(paragraph_tag):
                paragraph_text = _remove_trailing_block_introduction(paragraph_text)
            if len(paragraph_text) < MIN_PARAGRAPH_CHARACTERS:
                continue
            paragraphs.append(
                SourceParagraph(
                    paragraph_id=f"p{len(paragraphs):03d}",
                    section_path=section_path,
                    text=paragraph_text,
                    citation_ids=citation_ids,
                )
            )

    if not paragraphs:
        raise ValueError("no ordinary prose paragraphs survived source extraction")

    used_citation_ids = {
        citation_id
        for paragraph in paragraphs
        for citation_id in paragraph.citation_ids
    }
    raw_citations = raw_record.get("references")
    if not isinstance(raw_citations, list):
        raise ValueError("raw revision references must be a list")
    raw_citation_by_id = {
        str(citation["reference_id"]): citation
        for citation in raw_citations
        if isinstance(citation, dict) and "reference_id" in citation
    }
    missing_citations = used_citation_ids - raw_citation_by_id.keys()
    if missing_citations:
        raise ValueError(
            f"content paragraphs cite references absent from raw metadata: {sorted(missing_citations)}"
        )

    citations = tuple(
        SourceCitation.model_validate(raw_citation_by_id[citation_id])
        for citation_id in sorted(used_citation_ids, key=_citation_sort_key)
    )
    visible_text = render_model_visible_text(tuple(headings), tuple(paragraphs))
    language = _required_string(raw_record, "language")
    if language != "en":
        raise ValueError(
            f"expected English Wikipedia source, found language={language!r}"
        )

    return WikipediaSourceDocument(
        schema_version="model-intrinsic-parallel-wikipedia-source-v1",
        source=SourceIdentity(
            source_id=_required_string(raw_record, "source_id"),
            title=_required_string(raw_record, "title"),
            page_id=_required_int(raw_record, "page_id"),
            revision_id=_required_int(raw_record, "revision_id"),
            revision_timestamp=_required_string(raw_record, "revision_timestamp"),
            source_url=_required_string(raw_record, "source_url"),
            language="en",
            license=_required_string(raw_record, "license"),
            raw_wikitext_sha256=_required_string(raw_record, "raw_wikitext_sha256"),
            semantic_html_sha256=expected_html_hash,
        ),
        headings=tuple(headings),
        paragraphs=tuple(paragraphs),
        citations=citations,
        model_visible_text_sha256=hashlib.sha256(
            visible_text.encode("utf-8")
        ).hexdigest(),
    )


def render_model_visible_text(
    headings: tuple[str, ...],
    paragraphs: tuple[SourceParagraph, ...],
) -> str:
    del headings
    lines: list[str] = []
    prior_path: tuple[str, ...] = ()
    for paragraph in paragraphs:
        common_prefix = 0
        for prior_heading, current_heading in zip(
            prior_path, paragraph.section_path, strict=False
        ):
            if prior_heading != current_heading:
                break
            common_prefix += 1
        for depth, heading in enumerate(
            paragraph.section_path[common_prefix:], start=common_prefix
        ):
            lines.append(f"{'#' * (depth + 2)} {heading}")
        lines.append(paragraph.text)
        prior_path = paragraph.section_path
    return "\n\n".join(lines)


def write_source_document(document: WikipediaSourceDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _paragraph_text(paragraph_tag: Tag) -> str:
    paragraph_copy = BeautifulSoup(str(paragraph_tag), "html.parser").find("p")
    if paragraph_copy is None:
        raise ValueError("failed to clone paragraph")
    for selector in (
        "sup.reference",
        "style",
        "script",
        "link",
        "meta",
        ".mw-editsection",
        "[style*='display:none']",
    ):
        for element in paragraph_copy.select(selector):
            element.decompose()
    return _clean_text(paragraph_copy)


def _citation_ids(paragraph_tag: Tag) -> Iterable[str]:
    seen: set[str] = set()
    for citation in paragraph_tag.select("sup.reference"):
        link = citation.find("a", href=True)
        if link is None:
            continue
        href = str(link["href"])
        marker = "#cite_note-"
        if marker not in href:
            continue
        reference_id = href.split(marker, maxsplit=1)[1]
        if not reference_id:
            raise ValueError(f"could not resolve citation metadata from href={href!r}")
        if reference_id not in seen:
            seen.add(reference_id)
            yield reference_id


def _is_non_content_paragraph(paragraph_tag: Tag) -> bool:
    classes = {str(value) for value in paragraph_tag.get_attribute_list("class")}
    if "mw-empty-elt" in classes:
        return True
    forbidden_ancestor_names = {"aside", "figure", "footer", "nav", "table"}
    parent = paragraph_tag.parent
    while isinstance(parent, Tag) and parent.name != "section":
        if parent.name in forbidden_ancestor_names:
            return True
        parent = parent.parent
    return False


def _introduces_excluded_block(paragraph_tag: Tag) -> bool:
    if not _clean_text(paragraph_tag).endswith(":"):
        return False
    next_element = paragraph_tag.find_next_sibling()
    return isinstance(next_element, Tag) and next_element.name in {
        "blockquote",
        "dl",
        "figure",
        "ol",
        "pre",
        "table",
        "ul",
    }


def _remove_trailing_block_introduction(paragraph_text: str) -> str:
    retained_text, separator, _ = paragraph_text.rpartition(". ")
    if not separator:
        return ""
    return f"{retained_text}."


def _clean_text(tag: Tag) -> str:
    text = tag.get_text(" ", strip=True).replace("\u00a0", " ")
    text = _WHITESPACE.sub(" ", text)
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text).strip()


def _normalized_heading(heading: str) -> str:
    return _WHITESPACE.sub(" ", heading).strip().casefold()


def _citation_sort_key(reference_id: str) -> tuple[int, str]:
    final_component = reference_id.rsplit("-", maxsplit=1)[-1]
    if final_component.isdigit():
        return (int(final_component), reference_id)
    return (2**31 - 1, reference_id)


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"raw revision field {key!r} must be a non-empty string")
    return value


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise ValueError(f"raw revision field {key!r} must be an integer")
    return value
