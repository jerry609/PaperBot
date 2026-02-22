from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main


def test_prepare_project_dir_creates_missing_directory_under_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "workspace" / "paper-a"

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/runbook/project-dir/prepare",
            json={"project_dir": str(target), "create_if_missing": True},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["project_dir"] == str(target.resolve())
    assert payload["created"] is True
    assert target.is_dir()


def test_prepare_project_dir_rejects_path_outside_allowed_prefixes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    disallowed = Path.home().parent / "paperbot-runbook-disallowed"

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/runbook/project-dir/prepare",
            json={"project_dir": str(disallowed), "create_if_missing": False},
        )

    assert resp.status_code == 403
    assert "not allowed" in resp.json()["detail"]


def test_prepare_project_dir_rejects_file_path(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("hello", encoding="utf-8")

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/runbook/project-dir/prepare",
            json={"project_dir": str(file_path), "create_if_missing": True},
        )

    assert resp.status_code == 400
    assert "must be a directory" in resp.json()["detail"]
