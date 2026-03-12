from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from paperbot.application.ports.vault_exporter_port import VaultExporterPort


DEFAULT_PAPER_TEMPLATE = """# {{ title }}

## Summary
{{ abstract }}

## Metadata
{% for row in metadata_rows -%}
- {{ row }}
{% endfor %}
{% if track_link %}

## Tracks
- {{ track_link }}
{% endif %}
{% if related_links %}

## Related Papers
{% for link in related_links -%}
- {{ link }}
{% endfor %}
{% endif %}
{% if reference_links %}

## References
{% for link in reference_links -%}
- {{ link }}
{% endfor %}
{% endif %}
{% if cited_by_links %}

## Cited By
{% for link in cited_by_links -%}
- {{ link }}
{% endfor %}
{% endif %}
{% if external_links %}

## Links
{% for link in external_links -%}
- {{ link }}
{% endfor %}
{% endif %}
"""

_FRONTMATTER_EXPR = re.compile(r"\{\{\-?\s*frontmatter\s*\-?\}\}")


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    tokens: list[str] = []
    token: list[str] = []
    for char in normalized:
        if char.isalnum():
            token.append(char)
            continue
        if token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))
    return "-".join(tokens) or "untitled"


def _compact_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


def _yaml_frontmatter(payload: Dict[str, Any]) -> str:
    body = yaml.safe_dump(
        _compact_dict(payload),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return f"---\n{body}\n---\n"


class ObsidianFilesystemExporter(VaultExporterPort):
    """Write PaperBot artifacts directly into an Obsidian-compatible vault."""

    def export_library_snapshot(
        self,
        *,
        vault_path: Path,
        saved_items: List[Dict[str, Any]],
        track: Optional[Dict[str, Any]] = None,
        root_dir: str = "PaperBot",
        paper_template_path: Optional[Path] = None,
        track_moc_filename: str = "_MOC.md",
        group_tracks_in_folders: bool = True,
    ) -> Dict[str, Any]:
        vault_dir = Path(vault_path).expanduser().resolve()
        if not vault_dir.exists() or not vault_dir.is_dir():
            raise ValueError("vault_path must be an existing directory")
        template_path = (
            Path(paper_template_path).expanduser() if paper_template_path is not None else None
        )

        root_path = vault_dir / root_dir
        papers_dir = root_path / "Papers"
        tracks_dir = root_path / "Tracks"
        papers_dir.mkdir(parents=True, exist_ok=True)
        tracks_dir.mkdir(parents=True, exist_ok=True)

        paper_refs: List[Dict[str, str]] = []
        for item in saved_items:
            paper = item.get("paper") or {}
            if not isinstance(paper, dict) or not paper:
                continue
            paper_refs.append(
                self._write_paper_note(
                    papers_dir=papers_dir,
                    root_dir=root_dir,
                    paper=paper,
                    track=track,
                    saved_at=item.get("saved_at"),
                    template_path=template_path,
                    track_moc_filename=track_moc_filename,
                    group_tracks_in_folders=group_tracks_in_folders,
                )
            )

        self._sync_paper_citation_backlinks(papers_dir=papers_dir, paper_refs=paper_refs)

        track_ref: Optional[Dict[str, str]] = None
        if track is not None:
            track_ref = self._write_track_note(
                tracks_dir=tracks_dir,
                root_dir=root_dir,
                track=track,
                paper_refs=paper_refs,
                track_moc_filename=track_moc_filename,
                group_tracks_in_folders=group_tracks_in_folders,
            )

        moc_path = self._write_moc_note(
            root_path=root_path,
            paper_refs=paper_refs,
            track_ref=track_ref,
        )

        return {
            "vault_path": str(vault_dir),
            "root_dir": root_dir,
            "paper_count": len(paper_refs),
            "paper_notes": [ref["path"] for ref in paper_refs],
            "track_note": track_ref["path"] if track_ref else None,
            "moc_note": str(moc_path),
        }

    def _write_paper_note(
        self,
        *,
        papers_dir: Path,
        root_dir: str,
        paper: Dict[str, Any],
        track: Optional[Dict[str, Any]],
        saved_at: Optional[str],
        template_path: Optional[Path],
        track_moc_filename: str,
        group_tracks_in_folders: bool,
    ) -> Dict[str, str]:
        note_stem = self._paper_note_stem(paper)
        note_path = papers_dir / f"{note_stem}.md"
        note_title = str(paper.get("title") or "Untitled Paper")
        note_link = self._wikilink(
            root_dir=root_dir,
            section="Papers",
            note_stem=note_stem,
            label=note_title,
        )
        track_link = (
            self._track_wikilink(
                root_dir=root_dir,
                label=str(track.get("name") or "Track"),
                track=track,
                track_moc_filename=track_moc_filename,
                group_tracks_in_folders=group_tracks_in_folders,
            )
            if track is not None
            else None
        )
        related_links = self._paper_related_links(paper=paper, root_dir=root_dir)
        related_titles = self._paper_related_titles(paper)
        reference_entries = self._paper_reference_entries(paper)
        citation_entries = self._paper_citation_entries(paper)
        reference_links = self._paper_entry_links(entries=reference_entries, root_dir=root_dir)
        cited_by_links = self._paper_entry_links(entries=citation_entries, root_dir=root_dir)

        frontmatter = _yaml_frontmatter(
            {
                "paperbot_type": "paper",
                "paperbot_id": paper.get("id"),
                "title": paper.get("title"),
                "authors": list(paper.get("authors") or []),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
                "semantic_scholar_id": paper.get("semantic_scholar_id"),
                "openalex_id": paper.get("openalex_id"),
                "citation_count": paper.get("citation_count"),
                "saved_at": saved_at,
                "track": track.get("name") if track else None,
                "tags": self._paper_tags(paper, track),
                "related_papers": related_titles,
                "cites": reference_links,
                "cited_by": cited_by_links,
            }
        )

        body = self._render_paper_note(
            template_path=template_path,
            frontmatter=frontmatter,
            title=note_title,
            abstract=str(paper.get("abstract") or "_No abstract available._"),
            metadata_rows=[
                f"Authors: {', '.join(paper.get('authors') or []) or 'Unknown'}",
                f"Venue: {paper.get('venue') or 'Unknown'}",
                f"Year: {paper.get('year') or 'Unknown'}",
                f"Citations: {int(paper.get('citation_count') or 0)}",
            ],
            track_link=track_link,
            external_links=self._paper_links(paper),
            related_links=related_links,
            reference_links=reference_links,
            cited_by_links=cited_by_links,
            paper=paper,
            track=track,
            related_titles=related_titles,
        )

        note_path.write_text(body.rstrip() + "\n", encoding="utf-8")
        return {
            "title": note_title,
            "path": str(note_path),
            "link": note_link,
            "stem": note_stem,
            "references": reference_entries,
            "citations": citation_entries,
        }

    def _write_track_note(
        self,
        *,
        tracks_dir: Path,
        root_dir: str,
        track: Dict[str, Any],
        paper_refs: List[Dict[str, str]],
        track_moc_filename: str,
        group_tracks_in_folders: bool,
    ) -> Dict[str, str]:
        note_stem = self._track_note_stem(track)
        note_dir = tracks_dir / note_stem if group_tracks_in_folders else tracks_dir
        note_dir.mkdir(parents=True, exist_ok=True)
        note_filename = track_moc_filename if group_tracks_in_folders else f"{note_stem}.md"
        note_path = note_dir / note_filename
        note_link = self._track_wikilink(
            root_dir=root_dir,
            label=str(track.get("name") or "Track"),
            track=track,
            track_moc_filename=track_moc_filename,
            group_tracks_in_folders=group_tracks_in_folders,
        )

        frontmatter = _yaml_frontmatter(
            {
                "paperbot_type": "track",
                "track_id": track.get("id"),
                "user_id": track.get("user_id"),
                "name": track.get("name"),
                "moc_note": note_filename,
                "paper_count": len(paper_refs),
                "keywords": list(track.get("keywords") or []),
                "methods": list(track.get("methods") or []),
                "venues": list(track.get("venues") or []),
                "is_active": track.get("is_active"),
            }
        )

        lines = [
            frontmatter,
            f"# {track.get('name') or 'Untitled Track'}",
            "",
            str(track.get("description") or "_No description provided._"),
            "",
            "## Focus",
            f"- Keywords: {', '.join(track.get('keywords') or []) or 'None'}",
            f"- Methods: {', '.join(track.get('methods') or []) or 'None'}",
            f"- Venues: {', '.join(track.get('venues') or []) or 'None'}",
            "",
            "## Research Tasks",
        ]
        tasks = list(track.get("tasks") or [])
        if tasks:
            for task in tasks:
                title = str(task.get("title") or "Untitled task").strip()
                status = str(task.get("status") or "todo").strip()
                lines.append(f"- [{status}] {title}")
        else:
            lines.append("- _No tracked tasks._")

        lines.extend([
            "",
            "## Milestones",
        ])
        milestones = list(track.get("milestones") or [])
        if milestones:
            for milestone in milestones:
                name = str(milestone.get("name") or "Untitled milestone").strip()
                status = str(milestone.get("status") or "todo").strip()
                lines.append(f"- [{status}] {name}")
        else:
            lines.append("- _No milestones defined._")

        lines.extend([
            "",
            "## Tracked Scholars",
        ])
        scholars = list(track.get("scholars") or track.get("tracked_scholars") or [])
        if scholars:
            for scholar in scholars:
                name = str(scholar.get("name") or scholar.get("scholar_name") or "Unknown scholar").strip()
                affiliation = str(scholar.get("affiliation") or "").strip()
                if affiliation:
                    lines.append(f"- {name} ({affiliation})")
                else:
                    lines.append(f"- {name}")
        else:
            lines.append("- _No linked scholars._")

        lines.extend([
            "",
            "## Saved Papers",
        ])
        if paper_refs:
            lines.extend([f"- {ref['link']}" for ref in paper_refs])
        else:
            lines.append("- _No saved papers exported._")

        note_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return {
            "title": str(track.get("name") or "Track"),
            "path": str(note_path),
            "link": note_link,
            "stem": note_stem,
        }

    def _write_moc_note(
        self,
        *,
        root_path: Path,
        paper_refs: List[Dict[str, str]],
        track_ref: Optional[Dict[str, str]],
    ) -> Path:
        moc_path = root_path / "MOC.md"
        lines = [
            "# PaperBot MOC",
            "",
            "## Tracks",
        ]
        if track_ref is not None:
            lines.append(f"- {track_ref['link']}")
        else:
            lines.append("- _No track note exported._")

        lines.extend(["", "## Papers"])
        if paper_refs:
            lines.extend([f"- {ref['link']}" for ref in paper_refs])
        else:
            lines.append("- _No papers exported._")

        moc_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return moc_path

    @staticmethod
    def _paper_note_stem(paper: Dict[str, Any]) -> str:
        year = str(paper.get("year") or "").strip()
        title_slug = _slugify(str(paper.get("title") or "untitled"))
        suffix = (
            str(paper.get("arxiv_id") or "").strip()
            or str(paper.get("doi") or "").strip()
            or str(paper.get("semantic_scholar_id") or "").strip()
            or str(paper.get("id") or "").strip()
        )
        suffix_slug = _slugify(suffix) if suffix else ""
        parts = [part for part in [year, title_slug, suffix_slug] if part]
        return "-".join(parts)[:120]

    @staticmethod
    def _track_note_stem(track: Dict[str, Any]) -> str:
        return _slugify(str(track.get("name") or "track"))[:120]

    def _track_wikilink(
        self,
        *,
        root_dir: str,
        label: str,
        track: Dict[str, Any],
        track_moc_filename: str,
        group_tracks_in_folders: bool,
    ) -> str:
        note_stem = self._track_note_stem(track)
        if group_tracks_in_folders:
            return self._wikilink(
                root_dir=root_dir,
                section=f"Tracks/{note_stem}",
                note_stem=Path(track_moc_filename).stem,
                label=label,
            )
        return self._wikilink(
            root_dir=root_dir,
            section="Tracks",
            note_stem=note_stem,
            label=label,
        )

    @staticmethod
    def _paper_tags(paper: Dict[str, Any], track: Optional[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for value in list(paper.get("keywords") or []) + list(paper.get("fields_of_study") or []):
            tag = _slugify(str(value))
            if tag and tag not in values:
                values.append(tag)
        if track is not None:
            tag = _slugify(str(track.get("name") or ""))
            if tag and tag not in values:
                values.append(tag)
        return values[:10]

    @staticmethod
    def _paper_links(paper: Dict[str, Any]) -> List[str]:
        links: List[str] = []
        if paper.get("url"):
            links.append(f"[Paper URL]({paper['url']})")
        doi = str(paper.get("doi") or "").strip()
        if doi:
            links.append(f"[DOI](https://doi.org/{doi})")
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if arxiv_id:
            links.append(f"[arXiv](https://arxiv.org/abs/{arxiv_id})")
        return links

    @staticmethod
    def _build_template_environment(*, loader: Optional[FileSystemLoader] = None) -> Environment:
        return Environment(
            loader=loader,
            autoescape=select_autoescape(
                enabled_extensions=("html", "htm", "xml"),
                default_for_string=True,
                default=True,
            ),
            keep_trailing_newline=True,
            trim_blocks=False,
            lstrip_blocks=False,
        )

    @staticmethod
    def _strip_frontmatter_expression(template_source: str) -> str:
        stripped = _FRONTMATTER_EXPR.sub("", template_source)
        return stripped.lstrip("\n")

    def _render_paper_note(
        self,
        *,
        template_path: Optional[Path],
        frontmatter: str,
        title: str,
        abstract: str,
        metadata_rows: List[str],
        track_link: Optional[str],
        external_links: List[str],
        related_links: List[str],
        reference_links: List[str],
        cited_by_links: List[str],
        paper: Dict[str, Any],
        track: Optional[Dict[str, Any]],
        related_titles: List[str],
    ) -> str:
        template_source = DEFAULT_PAPER_TEMPLATE
        environment = self._build_template_environment()
        if template_path:
            resolved = template_path.expanduser().resolve()
            environment = self._build_template_environment(loader=FileSystemLoader(str(resolved.parent)))
            template_source = resolved.read_text(encoding="utf-8")

        body = environment.from_string(
            self._strip_frontmatter_expression(template_source)
        ).render(
            title=title,
            abstract=abstract,
            metadata_rows=metadata_rows,
            track_link=track_link,
            external_links=external_links,
            related_links=related_links,
            reference_links=reference_links,
            cited_by_links=cited_by_links,
            paper=paper,
            track=track,
            related_titles=related_titles,
        )
        return f"{frontmatter}{body}"

    def _sync_paper_citation_backlinks(
        self,
        *,
        papers_dir: Path,
        paper_refs: List[Dict[str, Any]],
    ) -> None:
        for paper_ref in paper_refs:
            note_path = Path(str(paper_ref.get("path") or "")).expanduser()
            if not note_path:
                continue
            current_link = str(paper_ref.get("link") or "").strip()
            if not current_link:
                continue

            for entry in list(paper_ref.get("references") or []):
                target_path = papers_dir / f"{self._paper_note_stem(entry)}.md"
                if target_path == note_path:
                    continue
                self._update_note_link_index(
                    note_path=target_path,
                    frontmatter_key="cited_by",
                    heading="Cited By",
                    link=current_link,
                )

            for entry in list(paper_ref.get("citations") or []):
                target_path = papers_dir / f"{self._paper_note_stem(entry)}.md"
                if target_path == note_path:
                    continue
                self._update_note_link_index(
                    note_path=target_path,
                    frontmatter_key="cites",
                    heading="References",
                    link=current_link,
                )

    def _update_note_link_index(
        self,
        *,
        note_path: Path,
        frontmatter_key: str,
        heading: str,
        link: str,
    ) -> None:
        if not note_path.exists() or not note_path.is_file():
            return

        frontmatter, body = self._read_note(note_path)
        values = [str(item).strip() for item in list(frontmatter.get(frontmatter_key) or []) if str(item).strip()]
        if link not in values:
            values.append(link)
        frontmatter[frontmatter_key] = values

        updated_body = self._upsert_markdown_section(body=body, heading=heading, links=values)
        note_path.write_text(
            f"{_yaml_frontmatter(frontmatter)}{updated_body.rstrip()}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_note(note_path: Path) -> tuple[Dict[str, Any], str]:
        text = note_path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            match = re.match(r"^---\n(.*?)\n---\n?", text, flags=re.DOTALL)
            if match:
                payload = yaml.safe_load(match.group(1)) or {}
                if isinstance(payload, dict):
                    return payload, text[match.end():]
        return {}, text

    @staticmethod
    def _upsert_markdown_section(
        *,
        body: str,
        heading: str,
        links: List[str],
    ) -> str:
        section_lines = [f"## {heading}"]
        section_lines.extend([f"- {link}" for link in links])
        section_text = "\n".join(section_lines).rstrip()

        trimmed = body.rstrip()
        pattern = re.compile(
            rf"(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)"
        )
        if pattern.search(trimmed):
            updated = pattern.sub(section_text + "\n", trimmed)
        else:
            separator = "\n\n" if trimmed else ""
            updated = f"{trimmed}{separator}{section_text}\n"
        return updated.rstrip() + "\n"

    def _paper_related_links(self, *, paper: Dict[str, Any], root_dir: str) -> List[str]:
        links: List[str] = []
        seen: set[str] = set()
        for entry in self._paper_related_entries(paper):
            title = str(entry.get("title") or "").strip()
            if not title or title.casefold() in seen:
                continue
            seen.add(title.casefold())
            note_stem = self._paper_note_stem(entry)
            links.append(
                self._wikilink(
                    root_dir=root_dir,
                    section="Papers",
                    note_stem=note_stem,
                    label=title,
                )
            )
        return links

    def _paper_entry_links(
        self,
        *,
        entries: List[Dict[str, Any]],
        root_dir: str,
    ) -> List[str]:
        links: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            title = str(entry.get("title") or "").strip()
            if not title:
                continue
            dedupe_key = str(
                entry.get("semantic_scholar_id")
                or entry.get("doi")
                or entry.get("arxiv_id")
                or entry.get("id")
                or title.casefold()
            ).strip()
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            links.append(
                self._wikilink(
                    root_dir=root_dir,
                    section="Papers",
                    note_stem=self._paper_note_stem(entry),
                    label=title,
                )
            )
        return links

    def _paper_related_titles(self, paper: Dict[str, Any]) -> List[str]:
        titles: List[str] = []
        seen: set[str] = set()
        for entry in self._paper_related_entries(paper):
            title = str(entry.get("title") or "").strip()
            if not title or title.casefold() in seen:
                continue
            seen.add(title.casefold())
            titles.append(title)
        return titles

    @classmethod
    def _paper_reference_entries(cls, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
        return cls._paper_relation_entries(
            [
                paper.get("references"),
                metadata.get("references"),
            ],
            nested_keys=["citedPaper", "paper"],
        )

    @classmethod
    def _paper_citation_entries(cls, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
        return cls._paper_relation_entries(
            [
                paper.get("citations"),
                metadata.get("citations"),
            ],
            nested_keys=["citingPaper", "paper"],
        )

    @classmethod
    def _paper_related_entries(cls, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
        return cls._paper_relation_entries(
            [
                paper.get("related_papers"),
                paper.get("related_titles"),
                paper.get("references"),
                metadata.get("related_papers"),
                metadata.get("references"),
            ],
            nested_keys=["citedPaper", "paper"],
        )

    @classmethod
    def _paper_relation_entries(
        cls,
        candidates: List[Any],
        *,
        nested_keys: List[str],
    ) -> List[Dict[str, Any]]:
        related: List[Dict[str, Any]] = []
        for bucket in candidates:
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                normalized = cls._normalize_paper_relation_entry(item, nested_keys=nested_keys)
                if normalized is not None:
                    related.append(normalized)
        return related

    @staticmethod
    def _normalize_paper_relation_entry(
        item: Any,
        *,
        nested_keys: List[str],
    ) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            title = str(item).strip()
            return {"title": title} if title else None
        if not isinstance(item, dict):
            return None

        payload: Dict[str, Any] = dict(item)
        for nested_key in nested_keys:
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                payload = dict(nested)
                break

        title = str(payload.get("title") or payload.get("name") or "").strip()
        if not title:
            return None

        return {
            "title": title,
            "year": payload.get("year"),
            "arxiv_id": payload.get("arxiv_id"),
            "doi": payload.get("doi"),
            "semantic_scholar_id": payload.get("semantic_scholar_id") or payload.get("paperId"),
            "id": payload.get("id") or payload.get("paper_id") or payload.get("paperId"),
        }

    @staticmethod
    def _wikilink(
        *,
        root_dir: str,
        section: str,
        note_stem: str,
        label: Optional[str] = None,
    ) -> str:
        target = f"{root_dir}/{section}/{note_stem}"
        if label:
            return f"[[{target}|{label}]]"
        return f"[[{target}]]"
