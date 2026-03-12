from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import yaml

_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n?", flags=re.DOTALL)
_SECTION_PATTERN = re.compile(r"(?m)^##\s+(.+?)\s*$")
_TITLE_PATTERN = re.compile(r"(?m)^#\s+(.+?)\s*$")
_INLINE_TAG_PATTERN = re.compile(r"(?<![\w/])#([A-Za-z][\w/-]*)")
_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass(frozen=True)
class MarkdownSection:
    heading: str
    markdown: str
    is_managed: bool


@dataclass(frozen=True)
class ParsedObsidianNote:
    frontmatter: Dict[str, Any]
    body: str
    title: str
    managed_body: str
    user_sections: List[MarkdownSection]
    all_tags: List[str]
    user_tags: List[str]
    all_wikilinks: List[str]
    user_wikilinks: List[str]


def hash_markdown(text: str) -> str:
    normalized_lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    normalized = "\n".join(normalized_lines).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_note_text(
    text: str,
    *,
    managed_headings: Iterable[str] | None = None,
) -> ParsedObsidianNote:
    frontmatter, body = split_frontmatter(text)
    managed_heading_set = {
        value.strip().casefold() for value in (managed_headings or ()) if value.strip()
    }
    prefix, sections = split_sections(body, managed_headings=managed_heading_set)
    managed_chunks: List[str] = []
    if prefix.strip():
        managed_chunks.append(prefix.strip())

    user_sections: List[MarkdownSection] = []
    for section in sections:
        if section.is_managed:
            managed_chunks.append(section.markdown.strip())
        else:
            user_sections.append(section)

    managed_body = _join_markdown_chunks(managed_chunks)
    title = extract_title(body)
    all_tags = extract_tags(body=body, frontmatter=frontmatter)
    managed_tags = {
        normalize_tag(tag)
        for tag in _coerce_string_list(frontmatter.get("paperbot_managed_tags"))
        if normalize_tag(tag)
    }
    user_tags = [tag for tag in all_tags if tag not in managed_tags]

    all_wikilinks = extract_wikilinks(body)
    managed_wikilinks = {
        normalize_wikilink_target(target)
        for target in _coerce_string_list(frontmatter.get("paperbot_managed_links"))
        if normalize_wikilink_target(target)
    }
    user_wikilinks = [target for target in all_wikilinks if target not in managed_wikilinks]

    return ParsedObsidianNote(
        frontmatter=frontmatter,
        body=body,
        title=title,
        managed_body=managed_body,
        user_sections=user_sections,
        all_tags=all_tags,
        user_tags=user_tags,
        all_wikilinks=all_wikilinks,
        user_wikilinks=user_wikilinks,
    )


def split_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text

    match = _FRONTMATTER_PATTERN.match(text)
    if match is None:
        return {}, text

    payload = yaml.safe_load(match.group(1)) or {}
    if not isinstance(payload, dict):
        raise ValueError("Obsidian frontmatter must be a mapping")
    return dict(payload), text[match.end() :]


def split_sections(
    body: str,
    *,
    managed_headings: Iterable[str] | None = None,
) -> tuple[str, List[MarkdownSection]]:
    managed_heading_set = {
        value.strip().casefold() for value in (managed_headings or ()) if value.strip()
    }
    matches = list(_SECTION_PATTERN.finditer(body))
    if not matches:
        return body, []

    prefix = body[: matches[0].start()]
    sections: List[MarkdownSection] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        heading = match.group(1).strip()
        markdown = body[start:end].strip()
        sections.append(
            MarkdownSection(
                heading=heading,
                markdown=markdown,
                is_managed=heading.casefold() in managed_heading_set,
            )
        )
    return prefix, sections


def extract_title(body: str) -> str:
    match = _TITLE_PATTERN.search(body)
    if match is None:
        return ""
    return match.group(1).strip()


def extract_tags(*, body: str, frontmatter: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for raw_tag in _coerce_string_list(frontmatter.get("tags")):
        normalized = normalize_tag(raw_tag)
        if normalized:
            values.append(normalized)
    for match in _INLINE_TAG_PATTERN.finditer(body):
        normalized = normalize_tag(match.group(1))
        if normalized:
            values.append(normalized)
    return _dedupe(values)


def extract_wikilinks(body: str) -> List[str]:
    return _dedupe(
        normalize_wikilink_target(match.group(1))
        for match in _WIKILINK_PATTERN.finditer(body)
        if normalize_wikilink_target(match.group(1))
    )


def merge_user_sections(managed_body: str, user_sections: Sequence[MarkdownSection]) -> str:
    chunks: List[str] = []
    if managed_body.strip():
        chunks.append(managed_body.strip())
    chunks.extend(section.markdown.strip() for section in user_sections if section.markdown.strip())
    return _join_markdown_chunks(chunks)


def user_sections_hash(user_sections: Sequence[MarkdownSection]) -> str:
    return hash_markdown(_join_markdown_chunks(section.markdown for section in user_sections))


def normalize_tag(value: str) -> str:
    normalized = str(value or "").strip()
    while normalized.startswith("#"):
        normalized = normalized[1:]
    return normalized.strip().lower()


def normalize_wikilink_target(value: str) -> str:
    return str(value or "").strip()


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: Iterable[str]) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(value)
    return items


def _join_markdown_chunks(chunks: Iterable[str]) -> str:
    values = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
    if not values:
        return ""
    return "\n\n".join(values).rstrip() + "\n"
