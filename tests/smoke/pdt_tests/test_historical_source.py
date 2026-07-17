from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from pdt.datasets.historical_source import (
    HISTORICAL_SOURCE_SCHEMA,
    RAW_WIKIMEDIA_SCHEMA,
    AssessmentQuality,
    HistoricalCategory,
    HistoricalSource,
    HistoricalTopicKind,
    PageAssessment,
    RawWikimediaRevisionBundle,
    ReferenceKind,
    ReferenceRecord,
    RejectionCode,
    canonical_json_bytes,
    cluster_historical_families,
    evaluate_historical_eligibility,
    load_historical_sources,
    manifest_for_source_file,
    materialize_historical_source,
    parse_historical_dom,
    select_balanced_historical_sources,
    verify_accepted_manifest,
)
from pdt.datasets.immutable_io import write_jsonl_new
from pdt.datasets.wikimedia_ingest import (
    HISTORICAL_CANDIDATE_SCHEMA,
    HistoricalCandidate,
    acquire_historical_revision,
    parse_maintenance_templates,
    parse_reference_records,
)


class WordContractTokenizer:
    """Deterministic tokenizer contract double for boundary-only tests."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _paragraph(section: int, paragraph: int, references: tuple[str, str, str]) -> str:
    words = " ".join(
        f"historical{section}_{paragraph}_{index}"
        for index in range(258)
    )
    citations = "".join(
        f'<sup class="mw-ref reference"><a href="./History#cite_note-{reference}">'
        f"[{index}]</a></sup>"
        for index, reference in enumerate(references, start=1)
    )
    return (
        "<p>"
        f"{words} "
        '<a href="./Institution">named institution</a> '
        '<span class="mw-editsection">forbidden edit label</span>'
        f"{citations}"
        "</p>"
    )


def _bundle(
    *,
    source_id: str = "history-1",
    page_id: int = 101,
    title: str = "The Example Historical Transition",
    family_keys: tuple[str, ...] = (),
    related_page_ids: tuple[int, ...] = (),
) -> RawWikimediaRevisionBundle:
    references = tuple(
        ReferenceRecord(
            reference_id=f"r{index:02d}",
            source_type=(
                ReferenceKind.BOOK if index < 9 else ReferenceKind.JOURNAL
            ),
            citation_text=f"Author {index}. A substantial historical source title.",
            title=f"Historical source {index}",
            authors=(f"Author {index}",),
            publication="University Press",
            year=1900 + index,
            identifiers={"isbn": f"978000000{index:02d}"},
        )
        for index in range(18)
    )
    body: list[str] = [
        "<html><body><main>",
        '<div class="hatnote"><p>forbidden navigation preface with enough words '
        "to look substantive but it must never become model-visible text.</p></div>",
    ]
    for section in range(6):
        body.append(f"<h2>Historical Phase {section + 1}</h2>")
        for paragraph in range(2):
            offset = (section * 2 + paragraph) * 3
            ids = tuple(f"r{(offset + item) % 18:02d}" for item in range(3))
            body.append(_paragraph(section, paragraph, ids))
        body.append(
            "<table><tr><td>forbidden table cell with extensive historical prose "
            "that cannot enter the source.</td></tr></table>"
        )
        body.append(
            "<ul><li>forbidden list item with extensive historical prose that "
            "cannot enter the source.</li></ul>"
        )
    body.extend(
        [
            "<h2>References</h2>",
            "<p>forbidden bibliography prose that is long enough to pass the "
            "paragraph length threshold and must be excluded as a subtree.</p>",
            "<h3>Printed sources</h3>",
            "<p>forbidden nested bibliography prose that is long enough to pass "
            "the paragraph threshold and must also be excluded.</p>",
            "</main></body></html>",
        ]
    )
    semantic_html = "".join(body)
    wikitext = "Complete pinned historical article source without maintenance templates."
    action_api_response = '{"query":{"pages":[]}}'
    return RawWikimediaRevisionBundle(
        schema_version=RAW_WIKIMEDIA_SCHEMA,
        source_id=source_id,
        page_id=page_id,
        revision_id=page_id * 10,
        revision_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        dump_date=date(2026, 1, 1),
        language="en",
        namespace=0,
        title=title,
        source_url=f"https://en.wikipedia.org/?curid={page_id}",
        license="CC BY-SA 4.0",
        revision_sha1=hashlib.sha1(wikitext.encode("utf-8")).hexdigest(),
        is_redirect=False,
        is_disambiguation=False,
        content_complete=True,
        topic_kind=HistoricalTopicKind.EVENT,
        historical_category=HistoricalCategory.REVOLUTIONS_TRANSITIONS,
        event_end_year=1900,
        assessments=(
            PageAssessment(project="WikiProject History", quality=AssessmentQuality.GA),
        ),
        family_keys=family_keys,
        related_page_ids=related_page_ids,
        maintenance_templates=(),
        references=references,
        action_api_response_json=action_api_response,
        action_api_response_sha256=_digest(action_api_response),
        raw_wikitext=wikitext,
        raw_wikitext_sha256=_digest(wikitext),
        semantic_html=semantic_html,
        semantic_html_sha256=_digest(semantic_html),
    )


def _accepted_source(bundle: RawWikimediaRevisionBundle) -> HistoricalSource:
    parsed = parse_historical_dom(bundle)
    eligibility = evaluate_historical_eligibility(
        bundle,
        parsed,
        tokenizer=WordContractTokenizer(),
    )
    assert eligibility.accepted, eligibility
    family_id = cluster_historical_families(((bundle, parsed),))[bundle.source_id]
    return materialize_historical_source(
        bundle,
        parsed,
        eligibility,
        family_id=family_id,
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_revision="pinned-revision",
    )


def test_semantic_dom_allowlist_preserves_only_headings_prose_links_and_citations() -> None:
    bundle = _bundle()
    parsed = parse_historical_dom(bundle)

    assert len(parsed.sections) == 6
    assert sum(len(section.paragraphs) for section in parsed.sections) == 12
    assert parsed.sections[0].heading_path == ("Historical Phase 1",)
    assert parsed.sections[0].paragraphs[0].reference_ids == ("r00", "r01", "r02")
    assert "named institution" in parsed.rendered_text
    for forbidden in (
        "forbidden edit label",
        "forbidden navigation preface",
        "forbidden table cell",
        "forbidden list item",
        "forbidden bibliography prose",
        "forbidden nested bibliography prose",
        "https://",
        "<p>",
        "[1]",
    ):
        assert forbidden not in parsed.rendered_text


@pytest.mark.parametrize(
    "forbidden_wrapper",
    [
        "nav",
        "aside",
        "figure",
        "figcaption",
        "table",
        "ul",
        "ol",
        "li",
        "pre",
        "code",
        "math",
        "template",
    ],
)
def test_forbidden_dom_ancestors_never_serialize(forbidden_wrapper: str) -> None:
    bundle = _bundle()
    forbidden = (
        f"<{forbidden_wrapper}><p>PROPERTY FORBIDDEN TEXT contains enough ordinary "
        "words and characters to look like a substantive historical paragraph but "
        f"must never survive extraction.</p></{forbidden_wrapper}>"
    )
    semantic_html = bundle.semantic_html.replace("</main>", forbidden + "</main>")
    bundle = bundle.model_copy(
        update={
            "semantic_html": semantic_html,
            "semantic_html_sha256": _digest(semantic_html),
        }
    )

    parsed = parse_historical_dom(bundle)

    assert "PROPERTY FORBIDDEN TEXT" not in parsed.rendered_text


@pytest.mark.parametrize(
    "forbidden_class",
    [
        "infobox",
        "hatnote",
        "navbox",
        "sidebar",
        "thumb",
        "coordinates",
        "authority-control",
        "pronunciation",
        "mw-editsection",
        "mw-references",
        "gallery",
    ],
)
def test_forbidden_dom_classes_never_serialize(forbidden_class: str) -> None:
    bundle = _bundle()
    forbidden = (
        f'<div class="{forbidden_class}"><p>CLASS FORBIDDEN TEXT contains enough '
        "ordinary words and characters to look like a substantive historical "
        "paragraph but must never survive extraction.</p></div>"
    )
    semantic_html = bundle.semantic_html.replace("</main>", forbidden + "</main>")
    bundle = bundle.model_copy(
        update={
            "semantic_html": semantic_html,
            "semantic_html_sha256": _digest(semantic_html),
        }
    )

    parsed = parse_historical_dom(bundle)

    assert "CLASS FORBIDDEN TEXT" not in parsed.rendered_text


def test_equivalent_excluded_section_name_removes_complete_subtree() -> None:
    bundle = _bundle()
    forbidden = (
        "<h2>Notes and References</h2>"
        "<p>EQUIVALENT REFERENCES TEXT contains enough ordinary words and characters "
        "to look substantive but belongs to an excluded reference subtree.</p>"
        "<h3>Archival works</h3>"
        "<p>EQUIVALENT NESTED TEXT contains enough ordinary words and characters to "
        "look substantive but belongs to the same excluded reference subtree.</p>"
    )
    semantic_html = bundle.semantic_html.replace("</main>", forbidden + "</main>")
    bundle = bundle.model_copy(
        update={
            "semantic_html": semantic_html,
            "semantic_html_sha256": _digest(semantic_html),
        }
    )

    parsed = parse_historical_dom(bundle)

    assert "EQUIVALENT REFERENCES TEXT" not in parsed.rendered_text
    assert "EQUIVALENT NESTED TEXT" not in parsed.rendered_text


def test_fixed_historical_eligibility_contract_accepts_complete_fixture() -> None:
    bundle = _bundle()
    parsed = parse_historical_dom(bundle)

    result = evaluate_historical_eligibility(
        bundle,
        parsed,
        tokenizer=WordContractTokenizer(),
    )

    assert result.accepted
    assert result.reasons == ()
    assert 3_000 <= result.token_count <= 7_000
    assert result.details["substantive_sections"] == 6
    assert result.details["substantive_paragraphs"] == 12
    assert result.details["reference_occurrences"] == 36
    assert result.details["distinct_references"] == 18


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"namespace": 1}, RejectionCode.WRONG_NAMESPACE),
        ({"is_redirect": True}, RejectionCode.REDIRECT),
        ({"is_disambiguation": True}, RejectionCode.DISAMBIGUATION),
        ({"topic_kind": HistoricalTopicKind.LIST}, RejectionCode.INADMISSIBLE_TOPIC_TYPE),
        ({"title": "List of historical transitions"}, RejectionCode.INADMISSIBLE_TITLE),
        ({"event_end_year": 2010}, RejectionCode.EVENT_TOO_RECENT),
        ({"content_complete": False}, RejectionCode.INCOMPLETE_CONTENT),
        ({"maintenance_templates": ("Unreferenced",)}, RejectionCode.MAINTENANCE_TEMPLATE),
    ],
)
def test_fixed_article_gates_reject_inadmissible_inputs(
    updates: dict[str, object],
    reason: RejectionCode,
) -> None:
    bundle = _bundle().model_copy(update=updates)
    parsed = parse_historical_dom(bundle)

    result = evaluate_historical_eligibility(
        bundle,
        parsed,
        tokenizer=WordContractTokenizer(),
    )

    assert not result.accepted
    assert reason in result.reasons


def test_raw_bundle_digest_mismatch_fails_before_parsing() -> None:
    bundle = _bundle().model_copy(update={"semantic_html_sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest mismatch"):
        parse_historical_dom(bundle)


def test_family_relations_are_transitive_and_never_cross_splits() -> None:
    left = _bundle(
        source_id="left",
        page_id=201,
        family_keys=("shared-campaign",),
    )
    middle = _bundle(
        source_id="middle",
        page_id=202,
        family_keys=("shared-campaign",),
        related_page_ids=(203,),
    )
    right = _bundle(
        source_id="right",
        page_id=203,
        title="A Distinct Reconstruction",
    )
    rows = tuple((bundle, parse_historical_dom(bundle)) for bundle in (left, middle, right))

    families = cluster_historical_families(rows)

    assert len(set(families.values())) == 1
    sources = tuple(
        materialize_historical_source(
            bundle,
            parsed,
            evaluate_historical_eligibility(
                bundle,
                parsed,
                tokenizer=WordContractTokenizer(),
            ),
            family_id=families[bundle.source_id],
            tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer_revision="pinned-revision",
        )
        for bundle, parsed in rows
    )
    assert len({source.split for source in sources}) == 1


def test_balanced_selection_is_exact_deterministic_and_one_per_family() -> None:
    base = _accepted_source(_bundle())
    candidates: list[HistoricalSource] = []
    identity = 0
    for category in HistoricalCategory:
        for _ in range(2):
            identity += 1
            candidates.append(
                base.model_copy(
                    update={
                        "source_id": f"candidate-{identity:03d}",
                        "page_id": 1_000 + identity,
                        "revision_id": 2_000 + identity,
                        "historical_category": category,
                        "family_id": f"family-{identity:024x}",
                    }
                )
            )

    selected, manifest = select_balanced_historical_sources(
        candidates,
        examples_per_category=1,
    )
    repeated, repeated_manifest = select_balanced_historical_sources(
        tuple(reversed(candidates)),
        examples_per_category=1,
    )

    assert len(selected) == len(HistoricalCategory)
    assert {source.historical_category for source in selected} == set(HistoricalCategory)
    assert len({source.family_id for source in selected}) == len(selected)
    assert {source.source_id for source in selected} == {
        source.source_id for source in repeated
    }
    assert manifest.records == repeated_manifest.records
    with pytest.raises(ValueError, match="Search more candidates"):
        select_balanced_historical_sources(candidates, examples_per_category=3)


def test_exact_accepted_manifest_detects_any_source_mutation(tmp_path: Path) -> None:
    source = _accepted_source(_bundle())
    input_path = tmp_path / "raw.jsonl"
    source_path = tmp_path / "accepted.jsonl"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text('{"immutable":"raw"}\n', encoding="utf-8")
    write_jsonl_new(source_path, (source.model_dump(mode="json"),))
    manifest = manifest_for_source_file(
        source_path,
        (source,),
        input_path=input_path,
        tokenizer_name=source.tokenizer,
        tokenizer_revision=source.tokenizer_revision,
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    verified = verify_accepted_manifest(source_path, manifest_path)

    assert verified.source_file_sha256 == manifest.source_file_sha256
    assert load_historical_sources(source_path) == (source,)
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        verify_accepted_manifest(source_path, manifest_path)


def test_historical_source_schema_rejects_unexpected_model_visible_fields() -> None:
    payload = _accepted_source(_bundle()).model_dump(mode="json")
    payload["summary"] = "A second unreviewed model-visible route."
    with pytest.raises(ValidationError):
        HistoricalSource.model_validate(payload)
    assert payload["schema_version"] == HISTORICAL_SOURCE_SCHEMA


def _parsoid_reference_html() -> str:
    data_mw = json.dumps(
        {
            "parts": [
                {
                    "template": {
                        "target": {"wt": "Cite book"},
                        "params": {
                            "title": {"wt": "A documented history"},
                            "isbn": {"wt": "9780000000000"},
                        },
                    }
                }
            ]
        }
    )
    return (
        "<html><body><main>"
        "<p>A sufficiently long historical paragraph records a documented event "
        "with ordinary prose and explicit evidence."
        '<sup class="mw-ref reference"><a href="./Article#cite_note-r00">[1]</a></sup>'
        "</p><h2>References</h2><ol class=\"mw-references\">"
        '<li id="cite_note-r00"><span class="mw-cite-backlink">↑</span>'
        f"<span data-mw='{data_mw}'><cite class=\"citation book\">"
        "Historian. A documented history. University Press, 1999."
        "</cite></span></li></ol></main></body></html>"
    )


def test_parsoid_reference_parser_uses_explicit_citation_template_type() -> None:
    records = parse_reference_records(_parsoid_reference_html())

    assert len(records) == 1
    assert records[0].reference_id == "r00"
    assert records[0].source_type is ReferenceKind.BOOK
    assert "documented history" in records[0].citation_text
    assert "↑" not in records[0].citation_text


def test_parsoid_maintenance_parser_reads_transclusion_metadata_only() -> None:
    data_mw = json.dumps(
        {
            "parts": [
                {
                    "template": {
                        "target": {"wt": "More citations needed section"},
                        "params": {},
                    }
                },
                {
                    "template": {
                        "target": {"wt": "Cite book"},
                        "params": {},
                    }
                },
            ]
        }
    )
    html = (
        "<html><body><p>The prose says cleanup and citation needed as ordinary words.</p>"
        f"<span typeof='mw:Transclusion' data-mw='{data_mw}'></span>"
        "</body></html>"
    )

    assert parse_maintenance_templates(html) == (
        "more citations needed section",
    )


def test_wikimedia_acquisition_cross_checks_exact_revision_and_retains_raw_response() -> None:
    wikitext = "A complete pinned revision with a cited historical account."
    revision_sha1 = hashlib.sha1(wikitext.encode("utf-8")).hexdigest()
    action_payload = {
        "batchcomplete": True,
        "query": {
            "rightsinfo": {
                "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
                "text": "Creative Commons Attribution-Share Alike 4.0",
            },
            "pages": [
                {
                    "pageid": 101,
                    "ns": 0,
                    "title": "Pinned Historical Event",
                    "revisions": [
                        {
                            "revid": 202,
                            "parentid": 201,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "sha1": revision_sha1,
                            "slots": {"main": {"content": wikitext}},
                        }
                    ],
                    "pageassessments": {
                        "Military history": {"class": "GA", "importance": "High"}
                    },
                    "pageprops": {},
                }
            ],
        },
    }
    rest_payload = {
        "id": 202,
        "page": {"id": 101, "key": "Pinned_Historical_Event", "title": "Pinned Historical Event"},
        "license": {
            "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
            "title": "Creative Commons Attribution-Share Alike 4.0",
        },
        "html": _parsoid_reference_html(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/w/api.php":
            return httpx.Response(200, json=action_payload, request=request)
        if request.url.path == "/w/rest.php/v1/revision/202/with_html":
            return httpx.Response(200, json=rest_payload, request=request)
        return httpx.Response(404, request=request)

    candidate = HistoricalCandidate(
        schema_version=HISTORICAL_CANDIDATE_SCHEMA,
        page_id=101,
        revision_id=202,
        title="Pinned Historical Event",
        topic_kind=HistoricalTopicKind.EVENT,
        historical_category=HistoricalCategory.WARS_BATTLES,
        event_end_year=1900,
        family_keys=("example-war",),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        bundle = acquire_historical_revision(
            candidate,
            client=client,
            acquisition_date=date(2026, 1, 1),
        )

    assert bundle.page_id == 101
    assert bundle.revision_id == 202
    assert bundle.revision_sha1 == revision_sha1
    assert bundle.raw_wikitext == wikitext
    assert bundle.references[0].source_type is ReferenceKind.BOOK
    assert bundle.assessments[0].project == "Military history"
    assert bundle.action_api_response_sha256 == _digest(
        bundle.action_api_response_json
    )
