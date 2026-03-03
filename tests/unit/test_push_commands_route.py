from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import push_commands as push_commands_route


def _post_command(client: TestClient, *, chat_id: str, text: str):
    return client.post(
        "/api/push/telegram/command",
        json={"chat_id": chat_id, "text": text},
    )


def test_telegram_subscribe_keyword_and_list(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PAPERBOT_TELEGRAM_SUBS_PATH", str(tmp_path / "telegram_subs.json"))

    with TestClient(api_main.app) as client:
        resp = _post_command(client, chat_id="1001", text="/subscribe kv cache")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ok"] is True
        assert payload["subscriptions"]["keyword"] == ["kv cache"]
        assert payload["subscriptions"]["track"] == []

        listed = _post_command(client, chat_id="1001", text="/list")
        assert listed.status_code == 200
        assert "Keywords: kv cache" in listed.json()["reply"]


def test_telegram_subscribe_track_and_unsubscribe(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PAPERBOT_TELEGRAM_SUBS_PATH", str(tmp_path / "telegram_subs.json"))

    with TestClient(api_main.app) as client:
        created = _post_command(client, chat_id="1002", text="/subscribe track:Agents")
        assert created.status_code == 200
        assert created.json()["subscriptions"]["track"] == ["Agents"]

        removed = _post_command(client, chat_id="1002", text="/unsubscribe track:Agents")
        assert removed.status_code == 200
        payload = removed.json()
        assert payload["ok"] is True
        assert payload["subscriptions"]["track"] == []


def test_telegram_today_without_reports(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PAPERBOT_TELEGRAM_SUBS_PATH", str(tmp_path / "telegram_subs.json"))
    monkeypatch.setattr(push_commands_route, "_REPORTS_DIR", tmp_path / "missing_reports")

    with TestClient(api_main.app) as client:
        resp = _post_command(client, chat_id="1003", text="/today")

    assert resp.status_code == 200
    assert resp.json()["reply"] == "No daily digest available yet."


def test_telegram_today_with_report_titles(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PAPERBOT_TELEGRAM_SUBS_PATH", str(tmp_path / "telegram_subs.json"))
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "2026-03-03-digest.json"
    report_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "top_items": [
                            {"title": "FlashKV"},
                            {"title": "GraphRAG"},
                            {"title": "FlashKV"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(push_commands_route, "_REPORTS_DIR", reports_dir)

    with TestClient(api_main.app) as client:
        resp = _post_command(client, chat_id="1004", text="/today")

    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "Today's picks:" in reply
    assert "- FlashKV" in reply
    assert "- GraphRAG" in reply
