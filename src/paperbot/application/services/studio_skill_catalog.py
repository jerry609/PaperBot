from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from paperbot.application.services.studio_skill_registry import (
    StudioSkillSummary,
    discover_studio_skills,
    discover_studio_skills_in_root,
)

_MARKETPLACE_ENV_KEY = "PAPERBOT_STUDIO_SKILL_MARKETPLACE"
_MARKETPLACE_FILE = ".paperbot/studio/marketplace.json"
_INSTALLED_REPOS_FILE = ".paperbot/studio/installed-skills.json"
_INSTALLED_REPOS_DIR = ".paperbot/studio/skill-repos"
_DEFAULT_GIT_TIMEOUT = 20
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


class StudioSkillCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class StudioInstalledSkillRepo:
    slug: str
    title: str
    description: str
    repo_url: str
    repo_ref: Optional[str]
    install_path: str
    installed_at: str
    last_known_commit: Optional[str] = None
    remote_commit: Optional[str] = None
    update_available: bool = False
    skills: List[StudioSkillSummary] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["skills"] = [skill.to_payload() for skill in self.skills]
        return payload


@dataclass(frozen=True)
class StudioMarketplaceRepo:
    slug: str
    title: str
    description: str
    repo_url: str
    repo_ref: Optional[str] = None
    installed: bool = False
    install_path: Optional[str] = None
    installed_repo_slug: Optional[str] = None
    installed_skill_count: int = 0
    update_available: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudioSkillDetail:
    skill: StudioSkillSummary
    readme: str
    requires_workspace: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "skill": self.skill.to_payload(),
            "readme": self.readme,
            "setup": {
                "requires_workspace": self.requires_workspace,
                "context_modules": list(self.skill.context_modules),
                "recommended_for": list(self.skill.recommended_for),
            },
        }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "skill-repo"


def _installed_repos_file(repo_root: Path) -> Path:
    return repo_root / _INSTALLED_REPOS_FILE


def _installed_repos_dir(repo_root: Path) -> Path:
    return repo_root / _INSTALLED_REPOS_DIR


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_git(
    args: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = _DEFAULT_GIT_TIMEOUT,
    allow_failure: bool = False,
) -> Optional[str]:
    command = ["git", *args]
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if allow_failure:
            return None
        raise StudioSkillCatalogError(f"Git command failed: {' '.join(command)} ({exc})") from exc

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        if allow_failure:
            return None
        raise StudioSkillCatalogError(stderr or stdout or f"Git command failed: {' '.join(command)}")

    return stdout or stderr or ""


def _load_installed_repo_records(repo_root: Path) -> List[Dict[str, Any]]:
    payload = _read_json_file(_installed_repos_file(repo_root))
    if isinstance(payload, dict):
        records = payload.get("repos")
    else:
        records = payload

    if not isinstance(records, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        repo_url = _normalize_text(item.get("repo_url"))
        slug = _normalize_text(item.get("slug"))
        install_path = _normalize_text(item.get("install_path"))
        if not repo_url or not slug or not install_path:
            continue
        normalized.append(
            {
                "slug": slug,
                "title": _normalize_text(item.get("title")) or slug.replace("-", " ").title(),
                "description": _normalize_text(item.get("description")) or "Installed Git-backed skill repository.",
                "repo_url": repo_url,
                "repo_ref": _normalize_text(item.get("repo_ref")),
                "install_path": install_path,
                "installed_at": _normalize_text(item.get("installed_at")) or _utc_now_iso(),
                "last_known_commit": _normalize_text(item.get("last_known_commit")),
            }
        )
    return normalized


def _save_installed_repo_records(repo_root: Path, records: List[Dict[str, Any]]) -> None:
    _write_json_file(_installed_repos_file(repo_root), {"repos": records})


def _repo_title_from_url(repo_url: str) -> str:
    tail = repo_url.rstrip("/").rsplit("/", 1)[-1]
    tail = tail[:-4] if tail.endswith(".git") else tail
    return tail.replace("-", " ").replace("_", " ").title() or "Skill Repository"


def _repo_slug_from_url(repo_url: str) -> str:
    tail = repo_url.rstrip("/").rsplit("/", 1)[-1]
    tail = tail[:-4] if tail.endswith(".git") else tail
    return _slugify(tail)


def _unique_repo_slug(repo_root: Path, repo_url: str, existing_records: List[Dict[str, Any]]) -> str:
    existing_slugs = {record["slug"] for record in existing_records}
    base = _repo_slug_from_url(repo_url)
    if base not in existing_slugs:
        return base

    suffix = 2
    while True:
        candidate = f"{base}-{suffix}"
        if candidate not in existing_slugs:
            return candidate
        suffix += 1


def _load_marketplace_entries(repo_root: Path) -> List[StudioMarketplaceRepo]:
    entries: List[Dict[str, Any]] = []
    env_value = _normalize_text(os.getenv(_MARKETPLACE_ENV_KEY))
    if env_value:
        try:
            parsed = json.loads(env_value)
            if isinstance(parsed, list):
                entries.extend(item for item in parsed if isinstance(item, dict))
            elif isinstance(parsed, dict) and isinstance(parsed.get("repos"), list):
                entries.extend(item for item in parsed["repos"] if isinstance(item, dict))
        except json.JSONDecodeError:
            pass

    file_payload = _read_json_file(repo_root / _MARKETPLACE_FILE)
    if isinstance(file_payload, list):
        entries.extend(item for item in file_payload if isinstance(item, dict))
    elif isinstance(file_payload, dict) and isinstance(file_payload.get("repos"), list):
        entries.extend(item for item in file_payload["repos"] if isinstance(item, dict))

    marketplace: List[StudioMarketplaceRepo] = []
    seen: set[str] = set()
    for item in entries:
        repo_url = _normalize_text(item.get("repo_url"))
        if not repo_url or repo_url in seen:
            continue
        seen.add(repo_url)
        slug = _normalize_text(item.get("slug")) or _repo_slug_from_url(repo_url)
        marketplace.append(
            StudioMarketplaceRepo(
                slug=slug,
                title=_normalize_text(item.get("title")) or _repo_title_from_url(repo_url),
                description=_normalize_text(item.get("description")) or "Git-backed Studio skill pack.",
                repo_url=repo_url,
                repo_ref=_normalize_text(item.get("repo_ref")),
            )
        )
    return marketplace


def _resolve_repo_path(repo_root: Path, record: Dict[str, Any]) -> Path:
    install_path = Path(record["install_path"])
    return install_path if install_path.is_absolute() else repo_root / install_path


def _safe_git_output(args: List[str], *, cwd: Path, timeout: int = _DEFAULT_GIT_TIMEOUT) -> Optional[str]:
    return _run_git(args, cwd=cwd, timeout=timeout, allow_failure=True)


def _resolve_local_commit(repo_path: Path) -> Optional[str]:
    return _safe_git_output(["rev-parse", "HEAD"], cwd=repo_path)


def _resolve_remote_commit(repo_path: Path, repo_url: str, repo_ref: Optional[str]) -> Optional[str]:
    requested_ref = _normalize_text(repo_ref)
    if requested_ref and not _GIT_SHA_PATTERN.fullmatch(requested_ref):
        output = _safe_git_output(["ls-remote", repo_url, f"refs/heads/{requested_ref}"], cwd=repo_path, timeout=8)
        if output:
            return output.split()[0]
        output = _safe_git_output(["ls-remote", repo_url, requested_ref], cwd=repo_path, timeout=8)
        if output:
            return output.split()[0]

    output = _safe_git_output(["ls-remote", repo_url, "HEAD"], cwd=repo_path, timeout=8)
    if not output:
        return None
    return output.split()[0]


def _discover_installed_repo_skills(repo_root: Path, record: Dict[str, Any]) -> List[StudioSkillSummary]:
    repo_path = _resolve_repo_path(repo_root, record)
    if not repo_path.is_dir():
        return []

    return discover_studio_skills_in_root(
        repo_path,
        repo_root=repo_root,
        scope="installed",
        repo_slug=record["slug"],
        repo_url=record["repo_url"],
        repo_label=record["title"],
        repo_ref=record.get("repo_ref"),
        repo_commit=record.get("last_known_commit"),
    )


def _build_installed_repo(repo_root: Path, record: Dict[str, Any]) -> StudioInstalledSkillRepo:
    repo_path = _resolve_repo_path(repo_root, record)
    current_commit = _resolve_local_commit(repo_path) if repo_path.is_dir() else None
    remote_commit = None
    update_available = False
    if current_commit and repo_path.is_dir():
        remote_commit = _resolve_remote_commit(repo_path, record["repo_url"], record.get("repo_ref"))
        update_available = bool(remote_commit and remote_commit != current_commit)

    skills = _discover_installed_repo_skills(
        repo_root,
        {
            **record,
            "last_known_commit": current_commit or record.get("last_known_commit"),
        },
    )

    return StudioInstalledSkillRepo(
        slug=record["slug"],
        title=record["title"],
        description=record["description"],
        repo_url=record["repo_url"],
        repo_ref=record.get("repo_ref"),
        install_path=_relative_to_repo(repo_path, repo_root),
        installed_at=record["installed_at"],
        last_known_commit=current_commit or record.get("last_known_commit"),
        remote_commit=remote_commit,
        update_available=update_available,
        skills=skills,
    )


def list_available_studio_skills(repo_root: Optional[Path] = None) -> List[StudioSkillSummary]:
    root = (repo_root or _repository_root()).resolve()
    project_skills = discover_studio_skills(root)
    installed_repos = [
        _build_installed_repo(root, record)
        for record in _load_installed_repo_records(root)
    ]
    installed_skills = [skill for repo in installed_repos for skill in repo.skills]
    return sorted(
        [*project_skills, *installed_skills],
        key=lambda skill: (skill.scope != "project", skill.title.lower(), skill.id.lower()),
    )


def list_studio_skill_catalog(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = (repo_root or _repository_root()).resolve()
    project_skills = discover_studio_skills(root)
    installed_records = _load_installed_repo_records(root)
    installed_repos = [_build_installed_repo(root, record) for record in installed_records]

    marketplace_by_url = {
        entry.repo_url: entry for entry in _load_marketplace_entries(root)
    }
    for repo in installed_repos:
        existing = marketplace_by_url.get(repo.repo_url)
        marketplace_by_url[repo.repo_url] = StudioMarketplaceRepo(
            slug=existing.slug if existing else repo.slug,
            title=existing.title if existing else repo.title,
            description=existing.description if existing else repo.description,
            repo_url=repo.repo_url,
            repo_ref=existing.repo_ref if existing else repo.repo_ref,
            installed=True,
            install_path=repo.install_path,
            installed_repo_slug=repo.slug,
            installed_skill_count=len(repo.skills),
            update_available=repo.update_available,
        )

    updates = [repo.to_payload() for repo in installed_repos if repo.update_available]
    return {
        "project_skills": [skill.to_payload() for skill in project_skills],
        "installed_skills": [skill.to_payload() for repo in installed_repos for skill in repo.skills],
        "installed_repos": [repo.to_payload() for repo in installed_repos],
        "marketplace_repos": [repo.to_payload() for repo in sorted(marketplace_by_url.values(), key=lambda item: item.title.lower())],
        "updates": updates,
        "summary": {
            "project_skill_count": len(project_skills),
            "installed_skill_count": sum(len(repo.skills) for repo in installed_repos),
            "installed_repo_count": len(installed_repos),
            "update_count": len(updates),
            "marketplace_repo_count": len(marketplace_by_url),
        },
    }


def _resolve_skill_markdown(repo_root: Path, skill: StudioSkillSummary) -> str:
    skill_path = Path(skill.path)
    if not skill_path.is_absolute():
        skill_path = repo_root / skill_path
    skill_file = skill_path / "SKILL.md"
    try:
        raw = skill_file.read_text(encoding="utf-8")
    except OSError:
        return ""

    if raw.startswith("---\n"):
        closing = raw.find("\n---", 4)
        if closing != -1:
            remainder = raw[closing + 4 :]
            return remainder.lstrip("\n").strip()
    return raw.strip()


def get_studio_skill_detail(skill_key: str, repo_root: Optional[Path] = None) -> StudioSkillDetail:
    root = (repo_root or _repository_root()).resolve()
    normalized_key = skill_key.strip().lower()
    for skill in list_available_studio_skills(root):
        if skill.key.strip().lower() != normalized_key:
            continue
        return StudioSkillDetail(
            skill=skill,
            readme=_resolve_skill_markdown(root, skill),
            requires_workspace=True,
        )
    raise StudioSkillCatalogError(f"Unknown Studio skill: {skill_key}")


def install_studio_skill_repo(
    repo_url: str,
    *,
    repo_ref: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = (repo_root or _repository_root()).resolve()
    normalized_url = _normalize_text(repo_url)
    if not normalized_url:
        raise StudioSkillCatalogError("repo_url is required")

    normalized_ref = _normalize_text(repo_ref)
    records = _load_installed_repo_records(root)
    existing = next((record for record in records if record["repo_url"] == normalized_url), None)
    if existing is not None:
        if normalized_ref and existing.get("repo_ref") != normalized_ref:
            existing["repo_ref"] = normalized_ref
            _save_installed_repo_records(root, records)
            return update_studio_skill_repo(existing["slug"], repo_ref=normalized_ref, repo_root=root)
        installed_repo = _build_installed_repo(root, existing)
        return {
            "repo": installed_repo.to_payload(),
            "skills": [skill.to_payload() for skill in installed_repo.skills],
            "installed": False,
        }

    slug = _unique_repo_slug(root, normalized_url, records)
    target_dir = _installed_repos_dir(root) / slug
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    clone_args = ["clone", "--depth", "1"]
    if normalized_ref and not _GIT_SHA_PATTERN.fullmatch(normalized_ref):
        clone_args.extend(["--branch", normalized_ref])
    clone_args.extend([normalized_url, str(target_dir)])
    _run_git(clone_args)

    try:
        if normalized_ref and _GIT_SHA_PATTERN.fullmatch(normalized_ref):
            _run_git(["checkout", normalized_ref], cwd=target_dir)

        current_commit = _resolve_local_commit(target_dir)
        effective_ref = normalized_ref or _safe_git_output(["branch", "--show-current"], cwd=target_dir)
        new_record = {
            "slug": slug,
            "title": _repo_title_from_url(normalized_url),
            "description": "Installed Git-backed Studio skill repository.",
            "repo_url": normalized_url,
            "repo_ref": effective_ref if effective_ref and effective_ref != "HEAD" else None,
            "install_path": _relative_to_repo(target_dir, root),
            "installed_at": _utc_now_iso(),
            "last_known_commit": current_commit,
        }
        skills = _discover_installed_repo_skills(root, new_record)
        if not skills:
            raise StudioSkillCatalogError(
                "The repository installed successfully, but no supported skill directories were found."
            )

        records.append(new_record)
        _save_installed_repo_records(root, records)
        installed_repo = _build_installed_repo(root, new_record)
        return {
            "repo": installed_repo.to_payload(),
            "skills": [skill.to_payload() for skill in installed_repo.skills],
            "installed": True,
        }
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise


def update_studio_skill_repo(
    repo_slug: str,
    *,
    repo_ref: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = (repo_root or _repository_root()).resolve()
    records = _load_installed_repo_records(root)
    record = next((item for item in records if item["slug"] == repo_slug), None)
    if record is None:
        raise StudioSkillCatalogError(f"Unknown installed skill repository: {repo_slug}")

    repo_path = _resolve_repo_path(root, record)
    if not repo_path.is_dir():
        raise StudioSkillCatalogError(f"Installed repository is missing: {repo_path}")

    effective_ref = _normalize_text(repo_ref) or _normalize_text(record.get("repo_ref"))
    _run_git(["fetch", "--tags", "origin"], cwd=repo_path)

    if effective_ref:
        _run_git(["checkout", effective_ref], cwd=repo_path)
        if not _GIT_SHA_PATTERN.fullmatch(effective_ref):
            _run_git(["pull", "--ff-only", "origin", effective_ref], cwd=repo_path)
    else:
        current_branch = _safe_git_output(["branch", "--show-current"], cwd=repo_path)
        if current_branch:
            _run_git(["pull", "--ff-only", "origin", current_branch], cwd=repo_path)
        else:
            _run_git(["pull", "--ff-only"], cwd=repo_path)

    record["repo_ref"] = effective_ref
    record["last_known_commit"] = _resolve_local_commit(repo_path)
    _save_installed_repo_records(root, records)

    installed_repo = _build_installed_repo(root, record)
    return {
        "repo": installed_repo.to_payload(),
        "skills": [skill.to_payload() for skill in installed_repo.skills],
        "updated": True,
    }
