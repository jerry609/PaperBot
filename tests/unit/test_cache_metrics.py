"""Tests for CacheMetrics -- KV-cache hit rate tracker."""

from paperbot.infrastructure.swarm.codex_dispatcher import CacheMetrics


class _FakeUsage:
    def __init__(self, prompt_tokens, cached_tokens=None):
        self.prompt_tokens = prompt_tokens
        if cached_tokens is not None:
            self.prompt_tokens_details = type("D", (), {"cached_tokens": cached_tokens})()


class _FakeUsageNoCacheDetails:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens


def test_empty_metrics():
    m = CacheMetrics()
    assert m.hit_rate == 0.0
    assert m.total_prompt_tokens == 0
    assert "0/0" in m.report()


def test_record_with_cache():
    m = CacheMetrics()
    m.record(_FakeUsage(1000, 700))
    assert m.total_prompt_tokens == 1000
    assert m.cached_prompt_tokens == 700
    assert m.hit_rate == 0.7


def test_record_without_cache_details():
    m = CacheMetrics()
    m.record(_FakeUsageNoCacheDetails(500))
    assert m.total_prompt_tokens == 500
    assert m.cached_prompt_tokens == 0
    assert m.hit_rate == 0.0


def test_record_none():
    m = CacheMetrics()
    m.record(None)
    assert m.total_prompt_tokens == 0


def test_cumulative():
    m = CacheMetrics()
    m.record(_FakeUsage(1000, 800))
    m.record(_FakeUsage(1000, 600))
    assert m.total_prompt_tokens == 2000
    assert m.cached_prompt_tokens == 1400
    assert m.hit_rate == 0.7


def test_report_format():
    m = CacheMetrics()
    m.record(_FakeUsage(100, 50))
    r = m.report()
    assert "50/100" in r
    assert "50%" in r
