from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from paperbot.api.routes.studio_chat import _resolve_cli_project_dir, get_model_id, studio_cwd


def test_resolve_cli_project_dir_defaults_to_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_cli_project_dir(None) == tmp_path.resolve()


def test_resolve_cli_project_dir_accepts_path_under_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "workspace" / "paper-a"
    target.mkdir(parents=True, exist_ok=True)
    assert _resolve_cli_project_dir(str(target)) == target.resolve()


def test_resolve_cli_project_dir_rejects_outside_allowed_prefixes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    disallowed = Path(tempfile.gettempdir()).resolve().parent

    with pytest.raises(ValueError, match="not allowed"):
        _resolve_cli_project_dir(str(disallowed))


def test_get_model_id_maps_legacy_ui_value_to_cli_alias():
    assert get_model_id("claude-sonnet-4-5", for_cli=True) == "sonnet"


def test_get_model_id_passes_through_raw_claude_code_model_name():
    assert get_model_id("claude-sonnet-4-6", for_cli=True) == "claude-sonnet-4-6"


def test_get_model_id_maps_alias_for_api_fallback():
    assert get_model_id("sonnet", for_cli=False) == "claude-sonnet-4-5-20250514"


def test_studio_cwd_stays_within_allowed_prefixes_when_running_inside_repo(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "PaperBot"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES", raising=False)
    monkeypatch.setenv("PAPERBOT_RUNBOOK_ALLOWLIST_MUTATION", "false")

    payload = asyncio.run(studio_cwd())
    assert payload["cwd"] == str(repo_root.resolve())
    assert payload["actual_cwd"] == str(repo_root.resolve())
    assert payload["allowlist_mutation_enabled"] is False
    assert str(repo_root.resolve()) in payload["allowed_prefixes"]


def test_studio_cwd_prefers_existing_allowed_prefix_over_repo_root(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "PaperBot"
    repo_root.mkdir(parents=True, exist_ok=True)
    allowed_root = tmp_path / "Documents"
    allowed_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("PAPERBOT_RUNBOOK_ALLOW_DIR_PREFIXES", str(allowed_root))
    monkeypatch.setenv("PAPERBOT_RUNBOOK_ALLOWLIST_MUTATION", "true")

    payload = asyncio.run(studio_cwd())
    assert payload["cwd"] == str(allowed_root.resolve())
    assert payload["actual_cwd"] == str(repo_root.resolve())
    assert payload["allowlist_mutation_enabled"] is True
    assert str(allowed_root.resolve()) in payload["allowed_prefixes"]
