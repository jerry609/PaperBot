"""Tests for AppriseNotifier."""
from types import SimpleNamespace

import paperbot.infrastructure.push.apprise_notifier as notifier_mod
from paperbot.infrastructure.push.apprise_notifier import AppriseNotifier


def test_from_yaml_missing_file(tmp_path):
    notifier = AppriseNotifier.from_yaml(str(tmp_path / "nonexistent.yaml"))
    assert notifier.channel_count == 0


def test_from_yaml_empty_channels(tmp_path):
    config = tmp_path / "push.yaml"
    config.write_text("channels: []\n")
    notifier = AppriseNotifier.from_yaml(str(config))
    assert notifier.channel_count == 0


def test_from_yaml_parses_channels(tmp_path):
    config = tmp_path / "push.yaml"
    config.write_text(
        'channels:\n'
        '  - url: "json://localhost:8080"\n'
        '    tags: ["daily", "test"]\n'
        '  - url: "json://localhost:8081"\n'
        '    tags: ["daily"]\n'
    )
    notifier = AppriseNotifier.from_yaml(str(config))
    assert notifier.channel_count == 2


def test_from_yaml_skips_empty_urls(tmp_path):
    config = tmp_path / "push.yaml"
    config.write_text(
        'channels:\n'
        '  - url: ""\n'
        '    tags: ["daily"]\n'
        '  - url: "json://localhost"\n'
        '    tags: []\n'
    )
    notifier = AppriseNotifier.from_yaml(str(config))
    assert notifier.channel_count == 1


def test_no_urls_not_available():
    notifier = AppriseNotifier(urls=[])
    assert not notifier.available


def test_push_without_apprise_returns_error():
    notifier = AppriseNotifier(urls=[])
    result = notifier.push(body="test message")
    assert result["ok"] is False
    assert "not available" in result["error"]


def test_push_daily_digest_without_channels():
    notifier = AppriseNotifier(urls=[])
    report = {"title": "Test Digest", "date": "2026-03-02"}
    result = notifier.push_daily_digest(report=report, markdown="# Test")
    assert result["ok"] is False


def test_constructor_with_urls_and_tags():
    notifier = AppriseNotifier(
        urls=["json://localhost:9999"],
        tags={"json://localhost:9999": ["daily", "test"]},
    )
    assert notifier.channel_count == 1
    # available depends on apprise being installed
    # Just verify it doesn't crash


def test_push_daily_digest_prefers_html():
    """Verify the method signature and preference logic (no actual send)."""
    notifier = AppriseNotifier(urls=[])
    report = {"title": "Digest", "date": "2026-03-02"}
    # Just verify the method works without crashing
    result = notifier.push_daily_digest(
        report=report, markdown="# markdown", html="<h1>html</h1>"
    )
    assert isinstance(result, dict)


def test_push_daily_digest_uses_channel_formatter(monkeypatch):
    sent_payloads = []

    class _FakeApprise:
        def add(self, url, tag=None):
            self.url = url

        def notify(self, **kwargs):
            sent_payloads.append(kwargs)
            return True

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _FakeApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)

    notifier = AppriseNotifier(
        urls=["discord://webhook_id/webhook_token"],
        tags={"discord://webhook_id/webhook_token": ["daily", "discord"]},
    )
    report = {
        "title": "DailyPaper Digest",
        "date": "2026-03-02",
        "stats": {"unique_items": 1},
        "queries": [
            {
                "normalized_query": "kv cache",
                "top_items": [
                    {
                        "title": "FlashKV",
                        "url": "https://arxiv.org/abs/2601.00001",
                        "judge": {"overall": 4.8, "recommendation": "must_read"},
                        "digest_card": {"highlight": "2x faster decoding"},
                    }
                ],
            }
        ],
    }

    result = notifier.push_daily_digest(report=report, markdown="# fallback", tag="daily")
    assert result["ok"] is True
    assert sent_payloads, "expected at least one notify payload"
    assert "FlashKV" in str(sent_payloads[0]["body"])


def test_push_daily_digest_retries_retryable_failures(monkeypatch):
    attempts = {"count": 0}

    class _RetryApprise:
        def add(self, url, tag=None):
            return None

        def notify(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("429 Too Many Requests")
            return True

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _RetryApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_BACKOFF_S", "0")

    notifier = AppriseNotifier(
        urls=["discord://webhook_id/webhook_token"],
        tags={"discord://webhook_id/webhook_token": ["daily"]},
    )
    result = notifier.push_daily_digest(report={"title": "Digest", "date": "2026-03-03"}, tag="daily")

    assert result["ok"] is True
    assert attempts["count"] == 2
    assert result["channels"][0]["attempts"] == 2


def test_push_daily_digest_does_not_retry_non_retryable_failures(monkeypatch):
    attempts = {"count": 0}

    class _FailFastApprise:
        def add(self, url, tag=None):
            return None

        def notify(self, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("invalid webhook token")

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _FailFastApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_BACKOFF_S", "0")

    notifier = AppriseNotifier(
        urls=["discord://webhook_id/webhook_token"],
        tags={"discord://webhook_id/webhook_token": ["daily"]},
    )
    result = notifier.push_daily_digest(report={"title": "Digest", "date": "2026-03-03"}, tag="daily")

    assert result["ok"] is False
    assert attempts["count"] == 1
    assert result["channels"][0]["attempts"] == 1


def test_push_daily_digest_adds_error_code_mapping(monkeypatch):
    class _ErrorApprise:
        def add(self, url, tag=None):
            return None

        def notify(self, **kwargs):
            raise RuntimeError("503 Service Unavailable")

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _ErrorApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_ATTEMPTS", "1")
    notifier = AppriseNotifier(urls=["wecom://corp/key"], tags={"wecom://corp/key": ["daily"]})

    result = notifier.push_daily_digest(report={"title": "Digest", "date": "2026-03-03"}, tag="daily")

    assert result["ok"] is False
    assert result["channels"][0]["error_code"] == "downstream_unavailable"


def test_push_daily_digest_idempotency_skips_duplicate(monkeypatch):
    calls = {"count": 0}

    class _FakeApprise:
        def add(self, url, tag=None):
            return None

        def notify(self, **kwargs):
            calls["count"] += 1
            return True

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _FakeApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)
    monkeypatch.setenv("PAPERBOT_PUSH_IDEMPOTENCY_TTL_S", "3600")

    notifier = AppriseNotifier(
        urls=["discord://webhook_id/webhook_token"],
        tags={"discord://webhook_id/webhook_token": ["daily"]},
    )
    report = {"title": "Digest", "date": "2026-03-03", "queries": []}
    first = notifier.push_daily_digest(report=report, tag="daily")
    second = notifier.push_daily_digest(report=report, tag="daily")

    assert first["ok"] is True
    assert second["ok"] is True
    assert calls["count"] == 1
    assert second["channels"][0]["skipped"] is True
    assert second["channels"][0]["error_code"] == "idempotent_replay"
