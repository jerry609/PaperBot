from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .obsidian_exporter import ObsidianFilesystemExporter, _slugify, _yaml_frontmatter


def _markdown_table_cell(value: str) -> str:
    return str(value or "").replace("\n", " ").replace("|", "\\|").strip()


class ObsidianReportExporter:
    """Render long-form research reports into Obsidian-friendly markdown notes."""

    def __init__(self) -> None:
        self._note_exporter = ObsidianFilesystemExporter()

    def export_report_note(
        self,
        *,
        vault_path: Path,
        report: Dict[str, Any],
        root_dir: str = "PaperBot",
        track_moc_filename: str = "_MOC.md",
        group_tracks_in_folders: bool = True,
    ) -> Dict[str, Any]:
        vault_dir = Path(vault_path).expanduser().resolve()
        if not vault_dir.exists() or not vault_dir.is_dir():
            raise ValueError("vault_path must be an existing directory")

        root_path = vault_dir / root_dir
        reports_dir = root_path / "Reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        title = str(report.get("title") or "Untitled Report").strip() or "Untitled Report"
        note_stem = _slugify(title)
        note_path = reports_dir / f"{note_stem}.md"
        track_name = str(report.get("track_name") or "").strip()
        tags = self._report_tags(report)

        frontmatter = _yaml_frontmatter(
            {
                "title": title,
                "type": "research-report",
                "track": (
                    self._track_link(
                        root_dir=root_dir,
                        track_name=track_name,
                        track_moc_filename=track_moc_filename,
                        group_tracks_in_folders=group_tracks_in_folders,
                    )
                    if track_name
                    else None
                ),
                "tags": tags,
                "created": datetime.now(timezone.utc).isoformat(),
                "agent_workflow": str(report.get("workflow_type") or "research"),
            }
        )

        lines = [
            frontmatter,
            f"# {title}",
        ]

        summary = str(report.get("summary") or "").strip()
        if summary:
            lines.extend(["", *self._callout("abstract", "研究概述", summary)])

        key_insight = str(report.get("key_insight") or "").strip()
        if key_insight:
            lines.extend(["", *self._callout("tip", "核心观点", key_insight)])

        for section in list(report.get("sections") or []):
            section_title = str(section.get("title") or "Untitled Section").strip() or "Untitled Section"
            content = str(section.get("content") or "").strip() or "_No section content provided._"
            lines.extend(["", f"## {section_title}", content])
            for cited_paper in list(section.get("cited_papers") or []):
                lines.extend([
                    "",
                    *self._callout(
                        "quote",
                        self._paper_link(root_dir=root_dir, paper=cited_paper),
                        str(cited_paper.get("relevant_finding") or "").strip() or "Referenced in this section.",
                    ),
                ])

        methods = list(report.get("methods") or [])
        if methods:
            lines.extend(["", "## 方法论对比", "", "| 方法 | 论文 | 优势 | 局限 |", "|---|---|---|---|"])
            for method in methods:
                paper_title = str(method.get("paper") or "").strip()
                paper_link = (
                    self._paper_link(root_dir=root_dir, paper={"title": paper_title})
                    if paper_title
                    else ""
                )
                lines.append(
                    "| {name} | {paper} | {pros} | {cons} |".format(
                        name=_markdown_table_cell(str(method.get("name") or "")),
                        paper=_markdown_table_cell(paper_link),
                        pros=_markdown_table_cell(str(method.get("pros") or "")),
                        cons=_markdown_table_cell(str(method.get("cons") or "")),
                    )
                )

        trends = str(report.get("trends") or "").strip()
        if trends:
            lines.extend(["", *self._callout("info", "趋势分析", trends)])

        future_directions = str(report.get("future_directions") or "").strip()
        if future_directions:
            lines.extend(["", "## 未来方向", future_directions])

        references = list(report.get("references") or [])
        lines.extend(["", "## 引用论文"])
        if references:
            for reference in references:
                authors = ", ".join([str(name).strip() for name in list(reference.get("authors") or []) if str(name).strip()])
                suffix_parts = [part for part in [authors, str(reference.get("year") or "").strip()] if part]
                suffix = f" — {', '.join(suffix_parts)}" if suffix_parts else ""
                lines.append(f"- {self._paper_link(root_dir=root_dir, paper=reference)}{suffix}")
        else:
            lines.append("- _No references linked._")

        note_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return {
            "vault_path": str(vault_dir),
            "root_dir": root_dir,
            "title": title,
            "note_path": str(note_path),
        }

    @staticmethod
    def _report_tags(report: Dict[str, Any]) -> List[str]:
        tags = ["report"]
        for raw_tag in list(report.get("tags") or []):
            tag = _slugify(str(raw_tag))
            if tag and tag not in tags:
                tags.append(tag)
        workflow_tag = _slugify(str(report.get("workflow_type") or "research"))
        if workflow_tag and workflow_tag not in tags:
            tags.append(workflow_tag)
        return tags

    def _track_link(
        self,
        *,
        root_dir: str,
        track_name: str,
        track_moc_filename: str,
        group_tracks_in_folders: bool,
    ) -> str:
        return self._note_exporter._track_wikilink(
            root_dir=root_dir,
            label=track_name,
            track={"name": track_name},
            track_moc_filename=track_moc_filename,
            group_tracks_in_folders=group_tracks_in_folders,
        )

    def _paper_link(self, *, root_dir: str, paper: Dict[str, Any]) -> str:
        title = str(paper.get("title") or "Untitled Paper").strip() or "Untitled Paper"
        return self._note_exporter._wikilink(
            root_dir=root_dir,
            section="Papers",
            note_stem=self._note_exporter._paper_note_stem(paper),
            label=title,
        )

    @staticmethod
    def _callout(callout_type: str, title: str, content: str) -> List[str]:
        lines = [f"> [!{callout_type}] {title}"]
        content_lines = str(content or "").splitlines() or [""]
        for line in content_lines:
            lines.append(f"> {line}".rstrip())
        return lines
