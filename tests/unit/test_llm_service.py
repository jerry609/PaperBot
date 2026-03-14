import json

from paperbot.application.services.llm_service import LLMService, _estimate_cost_usd


class _FakeProvider:
    def __init__(self, response: str = "ok"):
        self.response = response
        self.calls = 0

    def invoke_simple(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        return self.response

    def stream_invoke(self, messages, **kwargs):
        yield "A"
        yield "B"

    @property
    def info(self):
        class _Info:
            provider_name = "fake"
            model_name = "fake-model"
            cost_tier = 1

        return _Info()


class _FakeRouter:
    def __init__(self, provider: _FakeProvider):
        self.provider = provider
        self.task_types = []

    def get_provider(self, task_type: str = "default"):
        self.task_types.append(task_type)
        return self.provider


class _FakeResolver:
    def __init__(self, provider: _FakeProvider):
        self.provider = provider
        self.task_types = []

    def get_provider(self, task_type: str = "default"):
        self.task_types.append(task_type)
        return self.provider


class _FakeUsageStore:
    def __init__(self):
        self.rows = []

    def record_usage(self, **kwargs):
        self.rows.append(kwargs)
        return {"id": len(self.rows)}


def test_complete_uses_cache_for_same_request():
    provider = _FakeProvider(response="cached")
    service = LLMService(router=_FakeRouter(provider))

    out1 = service.complete(task_type="summary", system="s", user="u")
    out2 = service.complete(task_type="summary", system="s", user="u")

    assert out1 == "cached"
    assert out2 == "cached"
    assert provider.calls == 1


def test_business_methods_route_to_expected_task_types():
    provider = _FakeProvider(response=json.dumps(dict(score=77, reason="good")))
    router = _FakeRouter(provider)
    service = LLMService(router=router)

    service.summarize_paper("T", "A")
    service.analyze_trends(topic="icl", papers=[{"title": "p1"}])
    relevance = service.assess_relevance(paper={"title": "p1", "keywords": ["icl"]}, query="icl")
    service.generate_daily_insight({"title": "d", "queries": [], "stats": {}})

    assert relevance["score"] == 77
    assert "summary" in router.task_types
    assert "reasoning" in router.task_types
    assert "extraction" in router.task_types


def test_assess_relevance_fallback_when_non_json_response():
    provider = _FakeProvider(response="not-json")
    service = LLMService(router=_FakeRouter(provider))

    relevance = service.assess_relevance(
        paper={"title": "KV Cache Acceleration", "snippet": "cache speedup"},
        query="cache acceleration",
    )

    assert isinstance(relevance["score"], int)
    assert relevance["fallback"] is True
    assert "Fallback" in relevance["reason"]


def test_describe_task_provider_returns_model_metadata():
    provider = _FakeProvider(response="cached")
    service = LLMService(router=_FakeRouter(provider))

    info = service.describe_task_provider("analysis")

    assert info["provider_name"] == "fake"
    assert info["model_name"] == "fake-model"


def test_provider_resolver_can_be_injected():
    provider = _FakeProvider(response="resolver-ok")
    resolver = _FakeResolver(provider)
    service = LLMService(provider_resolver=resolver)

    output = service.complete(task_type="chat", system="sys", user="hello")

    assert output == "resolver-ok"
    assert resolver.task_types == ["chat"]


def test_complete_records_usage():
    provider = _FakeProvider(response="usage")
    usage_store = _FakeUsageStore()
    service = LLMService(router=_FakeRouter(provider), usage_store=usage_store)

    result = service.complete(task_type="summary", system="system prompt", user="user prompt")

    assert result == "usage"
    assert len(usage_store.rows) == 1
    row = usage_store.rows[0]
    assert row["task_type"] == "summary"
    assert row["provider_name"] == "fake"
    assert row["prompt_tokens"] >= 1
    assert row["completion_tokens"] >= 1
    assert row["estimated_cost_usd"] >= 0.0


def test_estimate_cost_usd_supports_openrouter_prefixed_openai_models():
    cost = _estimate_cost_usd(
        provider_name="openrouter",
        model_name="openai/gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    assert cost > 0.0


def test_complete_records_nonzero_cost_for_openrouter_prefixed_openai_models():
    class _OpenRouterOpenAIProvider(_FakeProvider):
        @property
        def info(self):
            class _Info:
                provider_name = "openrouter"
                model_name = "openai/gpt-4o-mini"
                cost_tier = 1

            return _Info()

    provider = _OpenRouterOpenAIProvider(response="usage")
    usage_store = _FakeUsageStore()
    service = LLMService(router=_FakeRouter(provider), usage_store=usage_store)

    service.complete(task_type="summary", system="system prompt", user="user prompt")

    assert usage_store.rows[0]["estimated_cost_usd"] > 0.0
