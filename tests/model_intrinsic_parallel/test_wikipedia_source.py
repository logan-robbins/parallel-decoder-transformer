from __future__ import annotations

import hashlib

import pytest

from model_intrinsic_parallel.wikipedia_source import (
    extract_reference_records,
    extract_wikipedia_source,
)


def _raw_record(semantic_html: str) -> dict[str, object]:
    return {
        "source_id": "enwiki-1-2",
        "title": "Example event",
        "page_id": 1,
        "revision_id": 2,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "source_url": "https://en.wikipedia.org/w/index.php?oldid=2",
        "language": "en",
        "license": "CC BY-SA 4.0",
        "raw_wikitext_sha256": "a" * 64,
        "semantic_html": semantic_html,
        "semantic_html_sha256": hashlib.sha256(
            semantic_html.encode("utf-8")
        ).hexdigest(),
        "references": [
            {
                "reference_id": "book-4",
                "citation_text": "A real source",
                "title": "A title",
                "authors": ["An Author"],
                "publication": "A Press",
                "year": 2020,
                "identifiers": {},
                "source_type": "book",
            }
        ],
    }


def test_extracts_only_headings_and_ordinary_prose() -> None:
    kept = (
        "This ordinary paragraph is long enough to survive extraction and contains "
        "substantive prose about the historical event"
    )
    semantic_html = f"""
    <html><body>
      <section data-mw-section-id="0">
        <table><tr><td><p>This table paragraph must never be visible to the model even
        though it is deliberately long enough to cross the character threshold.</p></td></tr></table>
        <ul><li>List garbage</li></ul>
        <p>{kept}.<sup class="reference"><a href="./Example#cite_note-book-4">[4]</a></sup></p>
      </section>
      <section data-mw-section-id="1">
        <h2>Consequences</h2>
        <p>The consequences paragraph is also substantive ordinary prose and must remain
        in the model-visible document after its citation marker is removed.</p>
        <p>The article introduces an excluded list whose detached lead-in must not remain:</p>
        <ul><li>A long list item that is never ordinary model-visible prose.</li></ul>
        <p>This first sentence contains substantive historical prose that must remain
        intact in the model-visible source document.
        The next sentence only introduces an excluded quotation:</p>
        <blockquote><p>A quotation outside the ordinary prose allowlist.</p></blockquote>
      </section>
      <section data-mw-section-id="2">
        <h2>Notes, citations and sources</h2>
        <p>This reference prose is deliberately long but the complete excluded subtree
        must never become model-visible source content under any circumstances.</p>
      </section>
    </body></html>
    """

    document = extract_wikipedia_source(_raw_record(semantic_html))

    assert document.headings == ("Consequences",)
    assert len(document.paragraphs) == 3
    assert document.paragraphs[0].text == f"{kept}."
    assert document.paragraphs[0].citation_ids == ("book-4",)
    assert "table paragraph" not in " ".join(p.text for p in document.paragraphs)
    assert "detached lead-in" not in " ".join(p.text for p in document.paragraphs)
    assert document.paragraphs[2].text == (
        "This first sentence contains substantive historical prose that must remain "
        "intact in the model-visible source document."
    )
    assert "introduces an excluded quotation" not in " ".join(
        p.text for p in document.paragraphs
    )
    assert "reference prose" not in " ".join(p.text for p in document.paragraphs)


def test_rejects_mutated_raw_semantic_html() -> None:
    semantic_html = """
    <html><body><section><p>This paragraph contains enough ordinary prose to be accepted
    as source content by the strict single-article extraction path.</p></section></body></html>
    """
    record = _raw_record(semantic_html)
    record["semantic_html_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="do not match"):
        extract_wikipedia_source(record)


def test_extracts_reference_identity_from_parsoid_metadata() -> None:
    semantic_html = """
    <html><body><section><ol>
      <li><span id="mw-reference-text-cite_note-book-name-7">
        Author, <i>A Reliable Book</i>, University Press, 2020.
      </span></li>
    </ol></section></body></html>
    """

    records = extract_reference_records(semantic_html)

    assert records == [
        {
            "reference_id": "book-name-7",
            "citation_text": "Author, A Reliable Book, University Press, 2020.",
            "title": None,
            "authors": [],
            "publication": None,
            "year": None,
            "identifiers": {},
            "source_type": "unclassified",
        }
    ]
