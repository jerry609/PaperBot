from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class StudioSkillSummary:
    id: str
    title: str
    description: str
    slash_command: str
    scope: str
    tools: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)
    ecosystems: List[str] = field(default_factory=list)
    primary_ecosystem: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    manifest_source: str = "frontmatter"
    path: str = ""
    prompt_hint: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


_FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$", re.IGNORECASE)
_SKILL_DISCOVERY_ROOTS = (
    ("claude_code", ".claude/skills"),
    ("opencode", ".opencode/skills"),
    ("github_copilot", ".github/skills"),
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _normalize_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        item = _normalize_text(value)
        return [item] if item else []
    if not isinstance(value, list):
        return []

    normalized: List[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _normalize_text(item)
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _merge_string_lists(*groups: List[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            cleaned = _normalize_text(item)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            merged.append(cleaned)
    return merged


def _humanize_skill_id(value: str) -> str:
    words = re.sub(r"[_-]+", " ", value).strip().split()
    if not words:
        return "Skill"
    return " ".join(word[:1].upper() + word[1:] for word in words)


def _read_frontmatter(skill_path: Path) -> Dict[str, Any]:
    if not skill_path.is_file():
        return {}

    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}

    try:
        payload = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _read_skill_manifest(skill_path: Path) -> Dict[str, Any]:
    if not skill_path.is_file():
        return {}

    try:
        payload = json.loads(skill_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _resolve_skill_title(skill_id: str, manifest: Dict[str, Any], frontmatter: Dict[str, Any]) -> str:
    candidates = [
        manifest.get("title"),
        frontmatter.get("title"),
        manifest.get("name"),
        frontmatter.get("name"),
    ]

    for candidate in candidates:
        cleaned = _normalize_text(candidate)
        if not cleaned:
            continue
        if _SLUG_PATTERN.fullmatch(cleaned):
            return _humanize_skill_id(cleaned)
        return cleaned

    return _humanize_skill_id(skill_id)


def _resolve_skill_description(manifest: Dict[str, Any], frontmatter: Dict[str, Any]) -> str:
    return _normalize_text(manifest.get("description")) or _normalize_text(frontmatter.get("description")) or ""


def _resolve_skill_id(skill_dir: Path, manifest: Dict[str, Any], frontmatter: Dict[str, Any]) -> str:
    for candidate in (manifest.get("id"), manifest.get("name"), frontmatter.get("name")):
        cleaned = _normalize_text(candidate)
        if cleaned:
            return cleaned
    return skill_dir.name


def _resolve_slash_command(skill_id: str, manifest: Dict[str, Any]) -> str:
    explicit = _normalize_text(manifest.get("slash_command"))
    if explicit:
        return explicit if explicit.startswith("/") else f"/{explicit}"

    entrypoints = manifest.get("entrypoints")
    if isinstance(entrypoints, dict):
        slash_values = entrypoints.get("slash")
        if isinstance(slash_values, list):
            for item in slash_values:
                cleaned = _normalize_text(item)
                if cleaned:
                    return cleaned if cleaned.startswith("/") else f"/{cleaned}"
        elif isinstance(slash_values, str):
            cleaned = _normalize_text(slash_values)
            if cleaned:
                return cleaned if cleaned.startswith("/") else f"/{cleaned}"

    return f"/{skill_id}"


def _resolve_skill_path(skill_dir: Path, repo_root: Path) -> str:
    try:
        return str(skill_dir.relative_to(repo_root))
    except ValueError:
        return str(skill_dir)


def _load_skill_summary(skill_dir: Path, repo_root: Path, ecosystem: str) -> Optional[StudioSkillSummary]:
    manifest = _read_skill_manifest(skill_dir / "skill.json")
    frontmatter = _read_frontmatter(skill_dir / "SKILL.md")
    if not manifest and not frontmatter:
        return None

    skill_id = _resolve_skill_id(skill_dir, manifest, frontmatter)
    return StudioSkillSummary(
        id=skill_id,
        title=_resolve_skill_title(skill_id, manifest, frontmatter),
        description=_resolve_skill_description(manifest, frontmatter),
        slash_command=_resolve_slash_command(skill_id, manifest),
        scope="project",
        tools=_normalize_string_list(manifest.get("tools")) or _normalize_string_list(frontmatter.get("tools")),
        recommended_for=_normalize_string_list(manifest.get("recommended_for")),
        ecosystems=[ecosystem],
        primary_ecosystem=ecosystem,
        paths=[_resolve_skill_path(skill_dir, repo_root)],
        manifest_source="skill.json" if manifest else "frontmatter",
        path=_resolve_skill_path(skill_dir, repo_root),
        prompt_hint=_normalize_text(manifest.get("prompt_hint")),
    )


def _merge_skill_summaries(existing: StudioSkillSummary, incoming: StudioSkillSummary) -> StudioSkillSummary:
    return StudioSkillSummary(
        id=existing.id,
        title=existing.title or incoming.title,
        description=existing.description or incoming.description,
        slash_command=existing.slash_command or incoming.slash_command,
        scope=existing.scope or incoming.scope,
        tools=_merge_string_lists(existing.tools, incoming.tools),
        recommended_for=_merge_string_lists(existing.recommended_for, incoming.recommended_for),
        ecosystems=_merge_string_lists(existing.ecosystems, incoming.ecosystems),
        primary_ecosystem=existing.primary_ecosystem or incoming.primary_ecosystem,
        paths=_merge_string_lists(existing.paths, incoming.paths),
        manifest_source=existing.manifest_source or incoming.manifest_source,
        path=existing.path or incoming.path,
        prompt_hint=existing.prompt_hint or incoming.prompt_hint,
    )


def discover_studio_skills(repo_root: Optional[Path] = None) -> List[StudioSkillSummary]:
    root = (repo_root or _repository_root()).resolve()
    discovered_by_key: Dict[str, StudioSkillSummary] = {}

    for ecosystem, relative_root in _SKILL_DISCOVERY_ROOTS:
        skills_root = root / relative_root
        if not skills_root.is_dir():
            continue

        for skill_dir in sorted(path for path in skills_root.iterdir() if path.is_dir()):
            summary = _load_skill_summary(skill_dir, root, ecosystem)
            if summary is None:
                continue

            key = summary.id.strip().lower()
            existing = discovered_by_key.get(key)
            if existing is None:
                discovered_by_key[key] = summary
            else:
                discovered_by_key[key] = _merge_skill_summaries(existing, summary)

    return sorted(discovered_by_key.values(), key=lambda skill: (skill.title.lower(), skill.id.lower()))
