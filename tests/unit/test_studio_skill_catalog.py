from __future__ import annotations

import subprocess
from pathlib import Path

from paperbot.application.services import studio_skill_catalog


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return (result.stdout or result.stderr).strip()


def _write_skill_repo(repo_dir: Path, *, description: str) -> None:
    skill_dir = repo_dir / ".claude" / "skills" / "paper-reproduction"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: paper-reproduction",
                f"description: {description}",
                "tools:",
                "  - paper_search",
                "---",
                "",
                "# Paper Reproduction",
                "",
                "Use this skill to reproduce the selected paper.",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "skill.json").write_text(
        """
{
  "title": "Paper Reproduction",
  "slash_command": "/paper-reproduction",
  "recommended_for": ["paper", "context_pack"]
}
""".strip(),
        encoding="utf-8",
    )


def _commit_all(repo_dir: Path, message: str) -> str:
    _git(repo_dir, "add", ".")
    _git(repo_dir, "commit", "-m", message)
    return _git(repo_dir, "rev-parse", "HEAD")


def _init_source_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "source-skills"
    repo_dir.mkdir()
    _git(repo_dir, "init")
    _git(repo_dir, "config", "user.email", "test@example.com")
    _git(repo_dir, "config", "user.name", "PaperBot Test")
    _write_skill_repo(repo_dir, description="Initial Git-backed skill.")
    _commit_all(repo_dir, "initial")
    return repo_dir


def test_install_studio_skill_repo_clones_local_git_repo(tmp_path: Path):
    repo_root = tmp_path / "workspace"
    repo_root.mkdir()
    source_repo = _init_source_repo(tmp_path)

    payload = studio_skill_catalog.install_studio_skill_repo(str(source_repo), repo_root=repo_root)

    assert payload["installed"] is True
    assert payload["repo"]["repo_url"] == str(source_repo)
    assert payload["repo"]["slug"] == "source-skills"
    assert payload["skills"][0]["scope"] == "installed"
    assert payload["skills"][0]["repo_slug"] == "source-skills"
    assert payload["skills"][0]["repo_url"] == str(source_repo)
    assert (repo_root / ".paperbot" / "studio" / "installed-skills.json").is_file()


def test_list_and_update_studio_skill_catalog_for_installed_repo(tmp_path: Path):
    repo_root = tmp_path / "workspace"
    repo_root.mkdir()
    source_repo = _init_source_repo(tmp_path)

    install_payload = studio_skill_catalog.install_studio_skill_repo(str(source_repo), repo_root=repo_root)
    first_commit = install_payload["repo"]["last_known_commit"]

    _write_skill_repo(source_repo, description="Updated Git-backed skill.")
    second_commit = _commit_all(source_repo, "update")
    assert second_commit != first_commit

    catalog = studio_skill_catalog.list_studio_skill_catalog(repo_root)
    assert catalog["summary"]["installed_skill_count"] == 1
    assert catalog["summary"]["installed_repo_count"] == 1
    assert catalog["updates"][0]["slug"] == "source-skills"
    assert catalog["updates"][0]["remote_commit"] == second_commit

    updated = studio_skill_catalog.update_studio_skill_repo("source-skills", repo_root=repo_root)
    assert updated["updated"] is True
    assert updated["repo"]["last_known_commit"] == second_commit

    detail = studio_skill_catalog.get_studio_skill_detail("installed--source-skills--paper-reproduction", repo_root)
    assert detail.skill.scope == "installed"
    assert "Updated Git-backed skill." in detail.skill.description
    assert "Use this skill to reproduce the selected paper." in detail.readme
