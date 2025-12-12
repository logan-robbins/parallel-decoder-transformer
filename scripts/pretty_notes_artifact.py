from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _truncate_str(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head = value[: max(0, max_chars - 24)]
    return head + f" … <{len(value)} chars total>"


def _summarize(value: Any, *, max_list: int, max_str: int) -> Any:
    if isinstance(value, str):
        # Keep newlines readable in JSON dumps for model cards.
        return _truncate_str(value.replace("\n", "\\n"), max_chars=max_str)
    if isinstance(value, list):
        out = [_summarize(v, max_list=max_list, max_str=max_str) for v in value[:max_list]]
        if len(value) > max_list:
            out.append(f"... <{len(value) - max_list} more items>")
        return out
    if isinstance(value, dict):
        return {k: _summarize(v, max_list=max_list, max_str=max_str) for k, v in value.items()}
    return value


def build_excerpt(obj: dict[str, Any], *, max_list: int, max_str: int) -> dict[str, Any]:
    # Keep this stable and high-signal for HF README.
    keep: dict[str, Any] = {
        "sample_id": obj.get("sample_id"),
        "domain": obj.get("domain"),
        "plan_path": obj.get("plan_path"),
        "sectional_independence": obj.get("sectional_independence"),
        "lag_delta": obj.get("lag_delta"),
        "note_cadence_M": obj.get("note_cadence_M"),
    }

    # Show one stream worth of teacher notes (true_notes)
    true_notes = obj.get("true_notes")
    if isinstance(true_notes, list) and true_notes:
        keep["true_notes_example"] = _summarize(true_notes[0], max_list=max_list, max_str=max_str)

    # Show one speculative variant summary (noise_config + one stream)
    speculative_notes = obj.get("speculative_notes")
    if isinstance(speculative_notes, list) and speculative_notes:
        v0 = speculative_notes[0] if isinstance(speculative_notes[0], dict) else None
        if isinstance(v0, dict):
            spec_view: dict[str, Any] = {
                "variant_id": v0.get("variant_id"),
                "noise_config": v0.get("noise_config"),
                "lag_delta": v0.get("lag_delta"),
            }
            notes = v0.get("notes")
            if isinstance(notes, list) and notes:
                spec_view["notes_example"] = _summarize(notes[0], max_list=max_list, max_str=max_str)
            keep["speculative_variant_example"] = _summarize(
                spec_view, max_list=max_list, max_str=max_str
            )

    # Show one notes-bus snapshot metadata + one stream entry
    versioned_notes = obj.get("versioned_notes")
    if isinstance(versioned_notes, list) and versioned_notes:
        snap0 = versioned_notes[0] if isinstance(versioned_notes[0], dict) else None
        if isinstance(snap0, dict):
            snap_view: dict[str, Any] = {
                "snapshot_id": snap0.get("snapshot_id"),
                "source": snap0.get("source"),
                "lag_delta": snap0.get("lag_delta"),
                "note_cadence_M": snap0.get("note_cadence_M"),
                "ent_count": snap0.get("ent_count"),
                "fact_count": snap0.get("fact_count"),
            }
            notes = snap0.get("notes")
            if isinstance(notes, list) and notes:
                snap_view["notes_example"] = _summarize(notes[0], max_list=max_list, max_str=max_str)
            keep["versioned_notes_snapshot_0"] = _summarize(
                snap_view, max_list=max_list, max_str=max_str
            )

    # Rollback metadata is tiny; include it if present.
    if "rollback" in obj:
        keep["rollback"] = _summarize(obj["rollback"], max_list=max_list, max_str=max_str)

    return keep


def main() -> None:
    p = argparse.ArgumentParser(description="Pretty-print a truncated PDT notes artifact JSON.")
    p.add_argument("path", type=str, help="Path to notes artifact JSON (survey_*.json)")
    p.add_argument("--max-list", type=int, default=3, help="Max list items to show per list")
    p.add_argument("--max-str", type=int, default=240, help="Max string chars to show")
    args = p.parse_args()

    path = Path(args.path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("Expected top-level JSON object")

    excerpt = build_excerpt(obj, max_list=args.max_list, max_str=args.max_str)
    print(json.dumps(excerpt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

