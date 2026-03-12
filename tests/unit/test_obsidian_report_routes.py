from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from paperbot.api.main import app
from paperbot.api.routes import obsidian as obsidian_route


def test_export_obsidian_report_endpoint_uses_settings_default_vault(
    monkeypatch,
    tmp_path: Path,
):
    vault = tmp_path / "vault"
    vault.mkdir()

    monkeypatch.setattr(
        obsidian_route,
        "create_settings",
        lambda: SimpleNamespace(
            obsidian=SimpleNamespace(
                vault_path=str(vault),
                root_dir="PaperBot",
                track_moc_filename="_MOC.md",
                group_tracks_in_folders=True,
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/obsidian/export-report",
            json={
                "title": "Graph Compression Report",
                "track_name": "ICL Compression",
                "summary": "A concise report for Obsidian export.",
                "sections": [
                    {
                        "title": "Findings",
                        "content": "Compression quality improves with retrieval-aware selection.",
                        "cited_papers": [
                            {
                                "title": "UniICL",
                                "year": 2026,
                                "semantic_scholar_id": "S2-UNIICL",
                                "relevant_finding": "Defines a unified benchmark.",
                            }
                        ],
                    }
                ],
                "references": [
                    {
                        "title": "UniICL",
                        "year": 2026,
                        "semantic_scholar_id": "S2-UNIICL",
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    note_path = Path(payload["note_path"])
    assert note_path.exists()
    assert note_path == vault / "PaperBot" / "Reports" / "graph-compression-report.md"

    body = note_path.read_text(encoding="utf-8")
    assert "# Graph Compression Report" in body
    assert "> [!abstract] 研究概述" in body
    assert "[[PaperBot/Papers/2026-uniicl-s2-uniicl|UniICL]]" in body


def test_export_obsidian_report_endpoint_requires_vault_path(monkeypatch):
    monkeypatch.setattr(
        obsidian_route,
        "create_settings",
        lambda: SimpleNamespace(
            obsidian=SimpleNamespace(
                vault_path="",
                root_dir="PaperBot",
                track_moc_filename="_MOC.md",
                group_tracks_in_folders=True,
            )
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/obsidian/export-report",
            json={"title": "Missing Vault"},
        )

    assert response.status_code == 400
    assert "vault_path is required" in response.json()["detail"]


class _FakeSyncResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class _FakeSyncService:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.scan_calls = 0

    def get_status(self):
        return _FakeSyncResult(
            {
                "last_synced_at": "2026-03-12T10:00:00+00:00",
                "pending_count": 1,
                "tracked_note_count": 4,
                "conflict_count": 1,
                "state_path": str(self.root_path / ".paperbot-sync-state.json"),
                "pending_dir": str(self.root_path / ".paperbot-pending"),
            }
        )

    def scan(self):
        self.scan_calls += 1
        return _FakeSyncResult(
            {
                "last_synced_at": "2026-03-12T10:05:00+00:00",
                "scanned_notes": 4,
                "changed_notes": 2,
                "memories_created": 3,
                "memories_skipped": 0,
                "tag_updates": 1,
                "wikilink_updates": 1,
                "note_updates": 1,
                "conflicts_detected": 1,
                "pending_count": 2,
            }
        )

    def sync_paths(self, paths):
        return self.scan()


class _FakeWatcher:
    def __init__(self, *, root_path: Path, on_paths_changed, debounce_seconds: float):
        self.root_path = root_path
        self._on_paths_changed = on_paths_changed
        self.debounce_seconds = debounce_seconds
        self.is_running = False

    def start(self):
        self.is_running = True
        return True

    def stop(self):
        self.is_running = False


def test_obsidian_sync_routes_expose_status_scan_and_watch_controls(monkeypatch, tmp_path: Path):
    root_path = tmp_path / "vault" / "PaperBot"
    root_path.mkdir(parents=True)
    fake_sync_service = _FakeSyncService(root_path)

    monkeypatch.setattr(
        obsidian_route,
        "_build_obsidian_sync_service",
        lambda request=None: fake_sync_service,
    )
    monkeypatch.setattr(
        obsidian_route,
        "_get_obsidian_config",
        lambda: SimpleNamespace(sync_debounce_seconds=0.5),
    )
    monkeypatch.setattr(obsidian_route, "ObsidianVaultWatcher", _FakeWatcher)
    monkeypatch.setattr(obsidian_route, "WATCHDOG_AVAILABLE", True)
    obsidian_route._vault_watcher = None

    with TestClient(app) as client:
        status_response = client.get("/api/obsidian/sync/status")
        scan_response = client.post("/api/obsidian/sync/scan")
        start_response = client.post("/api/obsidian/sync/watch/start")
        stop_response = client.post("/api/obsidian/sync/watch/stop")

    assert status_response.status_code == 200
    assert status_response.json()["tracked_note_count"] == 4
    assert status_response.json()["watchdog_available"] is True

    assert scan_response.status_code == 200
    assert scan_response.json()["memories_created"] == 3

    assert start_response.status_code == 200
    assert start_response.json()["watching"] is True
    assert start_response.json()["mode"] == "watchdog"

    assert stop_response.status_code == 200
    assert stop_response.json()["watching"] is False
    assert fake_sync_service.scan_calls >= 2
