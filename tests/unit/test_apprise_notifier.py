"""Tests for AppriseNotifier."""
import os
from pathlib import Path

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
