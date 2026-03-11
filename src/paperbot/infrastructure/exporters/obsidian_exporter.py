from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader

from paperbot.application.ports.vault_exporter_port import VaultExporterPort


DEFAULT_PAPER_TEMPLATE = """{{ frontmatter }}
# {{ title }}

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
{% if external_links %}

## Links
{% for link in external_links -%}
- {{ link }}
{% endfor %}
{% endif %}
"""


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

    def __init__(self, *, paper_template_path: Optional[Path] = None):
        self._paper_template_path = Path(paper_template_path).expanduser() if paper_template_path else None

    def export_library_snapshot(
        self,
        *,
        vault_path: Path,
        saved_items: List[Dict[str, Any]],
        track: Optional[Dict[str, Any]] = None,
        root_dir: str = "PaperBot",
        paper_template_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        vault_dir = Path(vault_path).expanduser().resolve()
        if not vault_dir.exists() or not vault_dir.is_dir():
            raise ValueError("vault_path must be an existing directory")
        template_path = self._resolve_paper_template_path(paper_template_path)

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
                )
            )

        track_ref: Optional[Dict[str, str]] = None
        if track is not None:
            track_ref = self._write_track_note(
                tracks_dir=tracks_dir,
                root_dir=root_dir,
                track=track,
                paper_refs=paper_refs,
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
            self._wikilink(
                root_dir=root_dir,
                section="Tracks",
                note_stem=self._track_note_stem(track),
                label=str(track.get("name") or "Track"),
            )
            if track is not None
            else None
        )
        related_links = self._paper_related_links(paper=paper, root_dir=root_dir)
        related_titles = self._paper_related_titles(paper)

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
        }

    def _write_track_note(
        self,
        *,
        tracks_dir: Path,
        root_dir: str,
        track: Dict[str, Any],
        paper_refs: List[Dict[str, str]],
    ) -> Dict[str, str]:
        note_stem = self._track_note_stem(track)
        note_path = tracks_dir / f"{note_stem}.md"
        note_link = self._wikilink(
            root_dir=root_dir,
            section="Tracks",
            note_stem=note_stem,
            label=str(track.get("name") or "Track"),
        )

        frontmatter = _yaml_frontmatter(
            {
                "paperbot_type": "track",
                "track_id": track.get("id"),
                "user_id": track.get("user_id"),
                "name": track.get("name"),
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
            "## Saved Papers",
        ]
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

    def _resolve_paper_template_path(self, paper_template_path: Optional[Path]) -> Optional[Path]:
        if paper_template_path is not None:
            return Path(paper_template_path).expanduser()
        return self._paper_template_path

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
        paper: Dict[str, Any],
        track: Optional[Dict[str, Any]],
        related_titles: List[str],
    ) -> str:
        if template_path:
            resolved = template_path.expanduser().resolve()
            environment = Environment(
                loader=FileSystemLoader(str(resolved.parent)),
                autoescape=False,
                keep_trailing_newline=True,
                trim_blocks=False,
                lstrip_blocks=False,
            )
            template = environment.get_template(resolved.name)
            return template.render(
                frontmatter=frontmatter,
                title=title,
                abstract=abstract,
                metadata_rows=metadata_rows,
                track_link=track_link,
                external_links=external_links,
                related_links=related_links,
                paper=paper,
                track=track,
                related_titles=related_titles,
            )

        return Environment(autoescape=False).from_string(DEFAULT_PAPER_TEMPLATE).render(
            frontmatter=frontmatter,
            title=title,
            abstract=abstract,
            metadata_rows=metadata_rows,
            track_link=track_link,
            external_links=external_links,
            related_links=related_links,
            paper=paper,
            track=track,
            related_titles=related_titles,
        )

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

    @staticmethod
    def _paper_related_entries(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        related: List[Dict[str, Any]] = []
        candidates = [
            paper.get("related_papers"),
            paper.get("related_titles"),
            paper.get("references"),
            (paper.get("metadata") or {}).get("related_papers") if isinstance(paper.get("metadata"), dict) else None,
            (paper.get("metadata") or {}).get("references") if isinstance(paper.get("metadata"), dict) else None,
        ]
        for bucket in candidates:
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if isinstance(item, dict):
                    title = str(item.get("title") or item.get("name") or "").strip()
                    if title:
                        related.append(
                            {
                                "title": title,
                                "year": item.get("year"),
                                "arxiv_id": item.get("arxiv_id"),
                                "doi": item.get("doi"),
                                "semantic_scholar_id": item.get("semantic_scholar_id"),
                                "id": item.get("id"),
                            }
                        )
                else:
                    title = str(item or "").strip()
                    if title:
                        related.append({"title": title})
        return related

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
