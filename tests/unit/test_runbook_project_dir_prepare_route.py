from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbot.api import main as api_main
from paperbot.api.routes import runbook as runbook_route


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


def test_add_allowed_dir_accepts_existing_directory_under_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPERBOT_RUNBOOK_ALLOWLIST_MUTATION", "true")

    target = tmp_path / "workspace" / "team-a"
    target.mkdir(parents=True, exist_ok=True)

    with TestClient(api_main.app) as client:
        resp = client.post("/api/runbook/allowed-dirs", json={"directory": str(target)})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["directory"] == str(target.resolve())


def test_add_allowed_dir_rejects_outside_allowed_prefixes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPERBOT_RUNBOOK_ALLOWLIST_MUTATION", "true")

    outside = Path.home().resolve()

    with TestClient(api_main.app) as client:
        resp = client.post("/api/runbook/allowed-dirs", json={"directory": str(outside)})

    assert resp.status_code == 403
    assert "not allowed" in resp.json()["detail"]


def test_create_snapshot_bootstraps_sqlite_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PAPERBOT_DB_URL", f"sqlite:///{tmp_path / 'runbook.db'}")
    monkeypatch.setenv("PAPERBOT_RUNBOOK_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(runbook_route, "_provider", None)

    project_dir = tmp_path / "workspace"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "notes.md").write_text("hello runbook\n", encoding="utf-8")

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/api/runbook/snapshots",
            json={"project_dir": str(project_dir), "label": "bootstrap-check"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["file_count"] == 1
    assert payload["snapshot_id"] > 0
    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
