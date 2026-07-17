from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from model_intrinsic_parallel.wikipedia_source import extract_reference_records


ACTION_API = "https://en.wikipedia.org/w/api.php"
REVISION_HTML_URL = "https://en.wikipedia.org/w/rest.php/v1/revision/{revision_id}/html"
CONTENT_LICENSE = (
    "Creative Commons Attribution-ShareAlike 4.0 International "
    "(https://creativecommons.org/licenses/by-sa/4.0/)"
)


def acquire_revisions(
    titles: tuple[str, ...],
    *,
    output_path: Path,
    user_agent: str,
) -> tuple[dict[str, Any], ...]:
    if not titles:
        raise ValueError("at least one Wikipedia title is required")
    if len(titles) != len(set(titles)):
        raise ValueError("Wikipedia title list contains duplicates")
    if output_path.exists():
        raise FileExistsError(
            f"raw output already exists and is immutable: {output_path}"
        )
    if "@" not in user_agent and "http" not in user_agent:
        raise ValueError("user agent must contain a contact email or URL")

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    acquired: list[dict[str, Any]] = []
    for title in titles:
        acquired.append(_acquire_one(session, title))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("x", encoding="utf-8") as handle:
        for record in acquired:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_path.replace(output_path)
    return tuple(acquired)


def load_titles(path: Path) -> tuple[str, ...]:
    titles = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not titles:
        raise ValueError(f"title file is empty: {path}")
    return titles


def _acquire_one(session: requests.Session, requested_title: str) -> dict[str, Any]:
    started = time.perf_counter()
    action_response = session.get(
        ACTION_API,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "prop": "revisions|pageassessments",
            "palimit": "max",
            "rvprop": "ids|timestamp|sha1|content",
            "rvslots": "main",
            "titles": requested_title,
        },
        timeout=60,
    )
    action_response.raise_for_status()
    action_payload = action_response.json()
    page = _single_page(action_payload, requested_title)
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise ValueError(
            f"{requested_title!r} did not return exactly one current revision"
        )
    revision = revisions[0]
    revision_id = _required_int(revision, "revid")
    slots = revision.get("slots")
    if not isinstance(slots, dict) or not isinstance(slots.get("main"), dict):
        raise ValueError(f"{requested_title!r} revision has no main content slot")
    raw_wikitext = _required_string(slots["main"], "content")

    html_response = session.get(
        REVISION_HTML_URL.format(revision_id=revision_id),
        headers={"Accept": "text/html"},
        timeout=60,
    )
    html_response.raise_for_status()
    header_revision_id = html_response.headers.get("content-revision-id")
    if header_revision_id != str(revision_id):
        raise ValueError(
            f"{requested_title!r} HTML revision mismatch: "
            f"expected {revision_id}, received {header_revision_id!r}"
        )
    semantic_html = html_response.text
    references = extract_reference_records(semantic_html)
    elapsed_seconds = time.perf_counter() - started
    transferred_bytes = len(action_response.content) + len(html_response.content)
    canonical_title = _required_string(page, "title")
    page_id = _required_int(page, "pageid")
    assessments = page.get("pageassessments", {})

    record = {
        "schema_version": "model-intrinsic-parallel-wikimedia-revision-v1",
        "source_id": f"enwiki-{page_id}-{revision_id}",
        "title": canonical_title,
        "requested_title": requested_title,
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": _required_string(revision, "timestamp"),
        "revision_sha1": _required_string(revision, "sha1"),
        "source_url": (
            "https://en.wikipedia.org/w/index.php?title="
            f"{quote(canonical_title.replace(' ', '_'))}&oldid={revision_id}"
        ),
        "language": "en",
        "license": CONTENT_LICENSE,
        "raw_wikitext": raw_wikitext,
        "raw_wikitext_sha256": hashlib.sha256(raw_wikitext.encode("utf-8")).hexdigest(),
        "semantic_html": semantic_html,
        "semantic_html_sha256": hashlib.sha256(
            semantic_html.encode("utf-8")
        ).hexdigest(),
        "references": references,
        "assessments": assessments,
        "action_api_response": action_payload,
        "transfer": {
            "bytes": transferred_bytes,
            "seconds": elapsed_seconds,
            "mebibytes_per_second": transferred_bytes / (1024**2) / elapsed_seconds,
        },
    }
    print(
        f"acquired title={canonical_title!r} revision={revision_id} "
        f"bytes={transferred_bytes} seconds={elapsed_seconds:.3f} "
        f"MiB/s={record['transfer']['mebibytes_per_second']:.3f} "
        f"references={len(references)}",
        flush=True,
    )
    return record


def _single_page(payload: Any, requested_title: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Action API response is not an object")
    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError("Action API response has no query object")
    pages = query.get("pages")
    if not isinstance(pages, list) or len(pages) != 1:
        raise ValueError(f"{requested_title!r} did not resolve to exactly one page")
    page = pages[0]
    if not isinstance(page, dict) or page.get("missing") is True:
        raise ValueError(f"Wikipedia page is missing: {requested_title!r}")
    return page


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise ValueError(f"field {key!r} must be an integer")
    return value
