"""Dataset processing helpers for plan items and notes extraction."""

from __future__ import annotations

from typing import Any, Dict, List

from .example import ExampleNotes, PlanPayload


class ArticleProcessor:
    """Utility methods for extracting structured data from DatasetExample components."""

    def _plan_items_by_stream(self, plan_payload: PlanPayload) -> Dict[str, List[str]]:
        """Return the plan item summaries grouped by stream id."""
        result: Dict[str, List[str]] = {}
        for section in plan_payload.sections:
            result.setdefault(section.stream, []).append(section.summary)
        return result

    def _notes_strings_by_stream(
        self,
        notes: ExampleNotes,
        plan_payload: PlanPayload,
    ) -> Dict[str, List[str]]:
        """Return human-readable strings from true_notes, keyed by stream id.

        All streams defined in ``plan_payload`` are guaranteed to appear in the
        output, even if they have no notes content (they get an empty list).
        """
        result: Dict[str, List[str]] = {}
        for notes_obj in notes.true_notes:
            strings: List[str] = []
            for entity in notes_obj.entities:
                strings.append(entity.name)
            for fact in notes_obj.facts:
                obj_str = str(fact.object) if not isinstance(fact.object, str) else fact.object
                strings.append(f"{fact.subj_id} {fact.predicate} {obj_str}")
            for cov in notes_obj.coverage:
                strings.append(f"{cov.status.value} {cov.plan_item_id}")
            result[notes_obj.stream_id] = strings
        for section in plan_payload.sections:
            if section.stream not in result:
                result[section.stream] = []
        return result
