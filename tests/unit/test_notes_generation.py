from __future__ import annotations

import json
from types import SimpleNamespace

from parallel_decoder_transformer.datasets.config import (
    GenerationConfig,
    LLMConfig,
    SpeculativeNotesNoiseConfig,
)
from parallel_decoder_transformer.datasets.notes_generation import (
    NotesGenerationConfig,
    NotesGenerator,
    PlanDocument,
)


def _stub_generator(monkeypatch, notes_dir, *, resume: bool) -> NotesGenerator:
    import parallel_decoder_transformer.utils.llm_client_factory as llm_factory
    import parallel_decoder_transformer.datasets.async_llm as async_llm_module

    dummy_client = SimpleNamespace(model="test-model", api_key="test-key")

    monkeypatch.setattr(llm_factory, "create_llm_client", lambda cfg: dummy_client)
    monkeypatch.setattr(
        async_llm_module,
        "AsyncStructuredLLMClient",
        lambda client: SimpleNamespace(submit_batch=lambda *args, **kwargs: []),
    )
    llm_cfg = LLMConfig()
    true_cfg = GenerationConfig()
    spec_cfg = GenerationConfig()
    noise_cfg = SpeculativeNotesNoiseConfig()
    notes_cfg = NotesGenerationConfig(output_root=notes_dir, resume_existing=resume)
    return NotesGenerator(
        llm_config=llm_cfg,
        true_cfg=true_cfg,
        speculative_cfg=spec_cfg,
        noise_cfg=noise_cfg,
        notes_cfg=notes_cfg,
    )


def _sample_plan(plan_dir, *, sample_id: str = "sample-1", domain: str = "qa") -> PlanDocument:
    plan_payload = {
        "sample_id": sample_id,
        "domain": domain,
        "input_text": "",
        "streams": [],
    }
    plan_path = plan_dir / domain / f"{sample_id}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    return PlanDocument(sample_id=sample_id, domain=domain, path=plan_path, payload=plan_payload)


def test_pending_plans_skip_existing_artifacts(tmp_path, monkeypatch) -> None:
    plan = _sample_plan(tmp_path / "plans")
    notes_dir = tmp_path / "notes"
    existing_path = notes_dir / plan.domain / f"{plan.sample_id}.json"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text("{}", encoding="utf-8")

    generator_resume = _stub_generator(monkeypatch, notes_dir, resume=True)
    pending = generator_resume._pending_plans([plan])
    assert pending == []

    generator_force = _stub_generator(monkeypatch, notes_dir, resume=False)
    pending_force = generator_force._pending_plans([plan])
    assert len(pending_force) == 1
    assert pending_force[0].sample_id == plan.sample_id


def test_log_failure_writes_entry(tmp_path, monkeypatch) -> None:
    plan = _sample_plan(tmp_path / "plans")
    notes_dir = tmp_path / "notes"
    generator = _stub_generator(monkeypatch, notes_dir, resume=True)
    generator._log_failure(
        sample_id=plan.sample_id,
        domain=plan.domain,
        stage="true_notes",
        error_type="test_error",
        message="failure reason",
        plan_path=plan.path,
        metadata={"extra": "value"},
    )
    log_path = generator._failure_log_path
    lines = [
        line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert lines, "Expected at least one failure log entry"
    entry = json.loads(lines[-1])
    assert entry["sample_id"] == plan.sample_id
    assert entry["stage"] == "true_notes"
    assert entry["error_type"] == "test_error"
    assert entry["extra"] == "value"
