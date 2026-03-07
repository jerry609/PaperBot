from __future__ import annotations

from types import SimpleNamespace

from paperbot.infrastructure.llm.providers.openai_provider import OpenAIProvider


class _FakeBadRequest(Exception):
    pass


class _FakeRateLimit(Exception):
    pass


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise _FakeBadRequest("Developer instruction is not enabled for model")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="HELLO", reasoning_content=None))
            ]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_merge_system_into_user_promotes_system_text():
    merged = OpenAIProvider._merge_system_into_user(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ]
    )

    assert merged == [
        {
            "role": "user",
            "content": "System instruction:\nBe concise.\n\nUser request:\nSay hello.",
        }
    ]


def test_invoke_retries_without_system_when_provider_rejects_system_role():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model_name = "dummy"
    provider.timeout = 30.0
    provider.client = _FakeClient()

    result = provider.invoke(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ]
    )

    calls = provider.client.chat.completions.calls
    assert result == "HELLO"
    assert len(calls) == 2
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[1]["messages"][0]["role"] == "user"
    assert "System instruction:\nBe concise." in calls[1]["messages"][0]["content"]


def test_invoke_retries_transient_rate_limit(monkeypatch):
    class _RetryingCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise _FakeRateLimit("Error code: 429 - temporarily rate-limited upstream")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="HELLO", reasoning_content=None))
                ]
            )

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model_name = "dummy"
    provider.timeout = 30.0
    provider.max_transient_retries = 1
    provider.retry_backoff_sec = 0.0
    provider._provider_name = "openrouter"
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=_RetryingCompletions()))

    sleeps = []
    monkeypatch.setattr(
        "paperbot.infrastructure.llm.providers.openai_provider.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = provider.invoke([{"role": "user", "content": "Say hello."}])

    calls = provider.client.chat.completions.calls
    assert result == "HELLO"
    assert len(calls) == 2
    assert calls[0]["messages"][0]["role"] == "user"
    assert sleeps == []
