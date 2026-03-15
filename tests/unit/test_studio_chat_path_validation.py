from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from paperbot.api.routes.studio_chat import _resolve_cli_project_dir, get_model_id


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
