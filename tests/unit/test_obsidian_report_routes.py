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
