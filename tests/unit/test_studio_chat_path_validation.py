from __future__ import annotations

from pathlib import Path

import pytest

from paperbot.api.routes.studio_chat import _resolve_cli_project_dir


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
    disallowed = Path.home().resolve()

    with pytest.raises(ValueError, match="not allowed"):
        _resolve_cli_project_dir(str(disallowed))
