import json
from types import SimpleNamespace

import openai
import pytest

from feak_tc.mvp.llm import request_json


def test_gpt5_json_request_uses_responses_api_and_strict_schema(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text=json.dumps({"value": "ok"}))

    class FakeChatCompletions:
        def create(self, **kwargs):
            pytest.fail("GPT-5 request must not use Chat Completions")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    result = request_json(
        system="system",
        user="user",
        model="gpt-5-mini-2025-08-07",
        temperature=None,
        reasoning_effort="minimal",
        verbosity="low",
        json_schema=schema,
        response_format_name="unit_test",
    )

    assert result == {"value": "ok"}
    assert captured["instructions"] == "system"
    assert captured["input"] == "user"
    assert captured["reasoning"] == {"effort": "minimal"}
    assert captured["text"]["verbosity"] == "low"
    assert captured["text"]["format"] == {
        "type": "json_schema",
        "name": "unit_test",
        "schema": schema,
        "strict": True,
    }
    assert "temperature" not in captured


def test_gpt4o_json_request_keeps_chat_completions_path(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            pytest.fail("GPT-4o request must keep the existing Chat Completions path")

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content=json.dumps({"value": "ok"}))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    result = request_json(
        system="system",
        user="user",
        model="gpt-4o-mini",
        temperature=0.2,
    )

    assert result == {"value": "ok"}
    assert captured["temperature"] == 0.2
    assert captured["response_format"] == {"type": "json_object"}
