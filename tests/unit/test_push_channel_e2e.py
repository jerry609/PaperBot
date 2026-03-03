from __future__ import annotations

from types import SimpleNamespace

import paperbot.infrastructure.push.apprise_notifier as notifier_mod
from paperbot.infrastructure.push.apprise_notifier import AppriseNotifier


def _sample_report():
    return {
        "title": "DailyPaper Digest",
        "date": "2026-03-03",
        "stats": {"unique_items": 2},
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


def test_mock_e2e_daily_digest_all_channels_green(monkeypatch):
    class _GreenApprise:
        def add(self, url, tag=None):
            self.url = url

        def notify(self, **kwargs):
            return True

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _GreenApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_BACKOFF_S", "0")

    urls = [
        "tgram://token/chat",
        "discord://webhook_id/webhook_token",
        "wecom://corp/key",
        "feishu://token",
    ]
    notifier = AppriseNotifier(urls=urls, tags={url: ["daily"] for url in urls})
    result = notifier.push_daily_digest(report=_sample_report(), tag="daily")

    assert result["ok"] is True
    assert len(result["channels"]) == 4
    assert all(item["ok"] for item in result["channels"])
    assert all(item["attempts"] == 1 for item in result["channels"])


def test_mock_e2e_daily_digest_failure_injection(monkeypatch):
    attempts_by_url = {}

    class _InjectedApprise:
        def add(self, url, tag=None):
            self.url = url

        def notify(self, **kwargs):
            attempts_by_url[self.url] = attempts_by_url.get(self.url, 0) + 1
            if "ratelimit" in self.url and attempts_by_url[self.url] == 1:
                raise RuntimeError("429 Too Many Requests")
            if "timeout" in self.url:
                raise RuntimeError("request timeout")
            if "authfail" in self.url:
                raise RuntimeError("401 invalid webhook token")
            return True

    fake_apprise = SimpleNamespace(
        Apprise=lambda: _InjectedApprise(),
        NotifyType=SimpleNamespace(INFO="info"),
        NotifyFormat=SimpleNamespace(TEXT="text", HTML="html", MARKDOWN="markdown"),
        common=SimpleNamespace(MATCH_ALL_TAG="*"),
    )

    monkeypatch.setattr(notifier_mod, "_HAS_APPRISE", True)
    monkeypatch.setattr(notifier_mod, "apprise", fake_apprise)
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PAPERBOT_PUSH_RETRY_BACKOFF_S", "0")

    urls = [
        "discord://ratelimit/success",
        "wecom://timeout/path",
        "feishu://authfail/path",
    ]
    notifier = AppriseNotifier(urls=urls, tags={url: ["daily"] for url in urls})
    result = notifier.push_daily_digest(report=_sample_report(), tag="daily")

    channels = {item["url"]: item for item in result["channels"]}
    assert result["ok"] is False
    assert channels["discord://ratelimit/success"]["ok"] is True
    assert channels["discord://ratelimit/success"]["attempts"] == 2
    assert channels["wecom://timeout/path"]["ok"] is False
    assert channels["wecom://timeout/path"]["error_code"] == "timeout"
    assert channels["feishu://authfail/path"]["ok"] is False
    assert channels["feishu://authfail/path"]["error_code"] == "auth_failed"
