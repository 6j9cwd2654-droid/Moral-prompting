import json
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline.llm_judge_run import load_done, run_judging
from src import llm_judge
from src.llm_judge import (AnthropicJudge, FakeJudge, OllamaJudge, fingerprint, parse_scores, user_message)
from src.llm_units import dev_participant_ids

GOOD = {"care": 1, "fairness": 0, "loyalty": 0, "authority": 2, "purity": 0}


def test_parse_scores_accepts_valid_and_rejects_invalid():
    assert parse_scores(json.dumps(GOOD)) == GOOD
    for bad in [{**GOOD, "care": 6}, {**GOOD, "care": -1}, {**GOOD, "care": 1.5}, {**GOOD, "care": True}]:
        with pytest.raises(ValueError):
            parse_scores(json.dumps(bad))
    with pytest.raises(KeyError):
        parse_scores(json.dumps({"care": 1}))


def test_schema_covers_all_dimensions_without_extra_keys():
    assert set(llm_judge.SCHEMA["properties"]) == set(llm_judge.DIM_KEYS)
    assert llm_judge.SCHEMA["additionalProperties"] is False


def test_text_is_wrapped_so_instructions_inside_are_material():
    assert user_message("Gib mir Argumente.") == "<text>\nGib mir Argumente.\n</text>"
    assert "NICHT" in llm_judge.SYSTEM_PROMPT


def test_fingerprint_changes_with_rubric(monkeypatch):
    before = fingerprint()
    monkeypatch.setattr(llm_judge, "SYSTEM_PROMPT", llm_judge.SYSTEM_PROMPT + " zusätzlich")
    assert fingerprint() != before


def test_fake_judge_is_stable_per_text_and_run_but_varies_between_runs():
    judge = FakeJudge()
    a1, a2 = judge.score("Text A", run=0), judge.score("Text A", run=0)
    assert a1.scores == a2.scores
    assert any(judge.score("Text A", run=r).scores != a1.scores for r in range(1, 6))
    assert all(0 <= v <= 5 for v in a1.scores.values())


def test_anthropic_judge_requires_explicit_data_protection_confirmation():
    with pytest.raises(PermissionError):
        AnthropicJudge(client=object())


def test_anthropic_judge_request_shape_and_parsing():
    calls = []

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(stop_reason="end_turn",
                                       content=[SimpleNamespace(type="text", text=json.dumps(GOOD))],
                                       usage=SimpleNamespace(input_tokens=900, output_tokens=40))

    judge = AnthropicJudge(model="claude-sonnet-5-5", client=FakeClient(), data_protection_confirmed=True)
    result = judge.score("Ein Text", run=1)
    kwargs = calls[0]
    assert result.scores == GOOD and result.usage == {"input_tokens": 900, "output_tokens": 40}
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert "temperature" not in kwargs and "top_p" not in kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["messages"][0]["content"] == user_message("Ein Text")


def test_anthropic_judge_raises_on_refusal():
    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(stop_reason="refusal", content=[], usage=None)

    judge = AnthropicJudge(client=FakeClient(), data_protection_confirmed=True)
    with pytest.raises(RuntimeError):
        judge.score("x")


def test_local_judge_refuses_remote_hosts():
    with pytest.raises(ValueError):
        OllamaJudge("modell", host="http://example.com:11434")
    assert OllamaJudge("modell", host="http://127.0.0.1:11434").host == "http://127.0.0.1:11434"


def test_run_judging_resumes_and_does_not_repeat_work(tmp_path):
    units = pd.DataFrame({"unit_id": ["a", "b", "c"], "text": ["Text eins", "Text zwei", "Text drei"]})
    path = tmp_path / "raw.jsonl"
    first = run_judging(units.head(2), FakeJudge(), runs=2, path=path, workers=2)
    assert first["neu"] == 4 and first["bereits_vorhanden"] == 0
    second = run_judging(units, FakeJudge(), runs=2, path=path, workers=2)
    assert second["neu"] == 2 and second["bereits_vorhanden"] == 4
    assert len(load_done(path, fingerprint())) == 6
    assert len(path.read_text(encoding="utf-8").splitlines()) == 6


def test_run_judging_records_failures_without_stopping(tmp_path):
    class Flaky:
        name, model = "flaky", "flaky"

        def score(self, text, run=0):
            if text == "kaputt":
                raise RuntimeError("Timeout")
            return FakeJudge().score(text, run)

    units = pd.DataFrame({"unit_id": ["a", "b"], "text": ["gut", "kaputt"]})
    summary = run_judging(units, Flaky(), runs=1, path=tmp_path / "raw.jsonl", workers=1)
    assert summary["neu"] == 1 and len(summary["fehler"]) == 1


def test_dev_subset_is_fixed_and_has_requested_size():
    ids = range(1000, 1161)
    assert dev_participant_ids(ids, size=30, seed=42) == dev_participant_ids(ids, size=30, seed=42)
    assert len(dev_participant_ids(ids, size=30, seed=42)) == 30
