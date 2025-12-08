from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from parallel_decoder_transformer.datasets.async_llm import (
    StructuredOutputResult,
    StructuredRequestError,
)
from parallel_decoder_transformer.datasets.config import GenerationConfig, LLMConfig
from parallel_decoder_transformer.datasets.plan_generation import (
    PlanGenerationConfig,
    PlanGenerator,
)
from parallel_decoder_transformer.datasets.preflight import (
    PreflightRunner,
    PreflightSettings,
    ReasoningGymPreflightConfig,
    SquadPreflightConfig,
    WikipediaPreflightConfig,
)


class _DummyEncoding:
    name = "dummy"

    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(char) for char in text]


def _preflight_runner(language_threshold: float = 0.3) -> PreflightRunner:
    runner = object.__new__(PreflightRunner)
    runner._plan_cfg = PlanGenerationConfig(total_per_domain={}, seed=0)  # type: ignore[attr-defined]
    runner._settings = PreflightSettings(
        squad=SquadPreflightConfig(
            max_question_tokens=10_000,
            max_context_tokens=10_000,
            max_total_tokens=50_000,
            min_context_tokens=1,
            min_question_tokens=1,
            max_non_latin_ratio=language_threshold,
        ),
        wikipedia=WikipediaPreflightConfig(),
        reasoning_gym=ReasoningGymPreflightConfig(),
        per_message_overhead=0,
        max_json_bytes=10**9,
        target_model="test-model",
        language_ratio_threshold=language_threshold,
    )
    runner._encoding = _DummyEncoding()
    runner._max_output_tokens = 16_384
    runner._plan_max_tokens = 256
    return runner


def test_squad_language_ratio_rejects_with_metadata() -> None:
    runner = _preflight_runner(language_threshold=0.2)
    record = {"question": "What is the term?", "context": "ASCII あいうえお"}
    outcome = runner._validate_squad_record(0, record, set())  # type: ignore[arg-type]
    assert not outcome["accepted"]
    reject = outcome["record"]
    assert reject["reason"] == "non_latin_ratio_exceeded"
    assert reject["metadata"]["language_ratio"] > 0.2


def test_squad_language_ratio_in_accept_record() -> None:
    runner = _preflight_runner(language_threshold=0.2)
    english_context = "The Continental Congress ratified the Articles."
    record = {"question": "Who ratified the Articles?", "context": english_context}
    outcome = runner._validate_squad_record(1, record, set())  # type: ignore[arg-type]
    assert outcome["accepted"]
    accept = outcome["record"]
    assert accept["source_metadata"]["language_ratio"] == 0.0


def test_plan_generator_logs_async_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "accepted.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "qa_sample_1",
                "domain": "qa",
                "prompt": "Plan prompt",
                "messages": [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "USR"},
                ],
                "source_metadata": {"source": "unit_test", "input_text": "context"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    plan_cfg = PlanGenerationConfig(
        total_per_domain={"qa": 1},
        output_root=tmp_path / "plans",
        preflight_manifest=manifest,
        use_async_client=True,
        batch_size=1,
        concurrency=1,
    )

    dummy_llm = types.SimpleNamespace(
        api_key="sk-test",
        model="stub-model",
        org_id=None,
        api_base="https://api.example.com",
    )

    monkeypatch.setattr(
        "parallel_decoder_transformer.datasets.plan_generation.create_llm_client",
        lambda cfg: dummy_llm,
    )

    class _StubAsyncClient:
        async def submit_batch(self, requests, *, concurrency, max_retries, retry_backoff=1.5):
            error = StructuredRequestError(
                "rate limited",
                status_code=429,
                error_type="http_429",
                retryable=False,
            )
            return [
                StructuredOutputResult(
                    request=requests[0],
                    response=None,
                    error=error,
                )
            ]

    monkeypatch.setattr(
        "parallel_decoder_transformer.datasets.plan_generation.AsyncStructuredLLMClient",
        lambda *args, **kwargs: _StubAsyncClient(),
    )

    generator = PlanGenerator(
        llm_config=LLMConfig(),
        generation_cfg=GenerationConfig(),
        plan_cfg=plan_cfg,
    )

    tasks = generator.build_tasks()
    assert len(tasks) == 1
    generator.generate(tasks)

    failure_files = sorted(plan_cfg.output_root.glob("plan_generation_failures_*.jsonl"))
    assert failure_files, "expected at least one timestamped failure log"
    entry = json.loads(failure_files[0].read_text(encoding="utf-8").strip())
    assert entry["sample_id"] == "qa_sample_1"
    assert entry["error_type"] == "http_429"
    assert entry["model_id"] == "stub-model"
