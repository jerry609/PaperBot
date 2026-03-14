"""export_to_obsidian MCP tool wrapping ObsidianFilesystemExporter.

Renders a paper as Obsidian-formatted markdown with YAML frontmatter.
No filesystem I/O — in-memory rendering only.
Uses anyio.to_thread.run_sync() to wrap the synchronous _render_paper_note() call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import anyio

from paperbot.mcp.tools._audit import log_tool_call

logger = logging.getLogger(__name__)

# Module-level lazy singleton for the exporter
_exporter = None


def _get_exporter():
    """Construct ObsidianFilesystemExporter on first call (lazy singleton)."""
    global _exporter
    if _exporter is None:
        from paperbot.infrastructure.exporters.obsidian_exporter import (
            ObsidianFilesystemExporter,
        )

        _exporter = ObsidianFilesystemExporter()
    return _exporter


async def _export_to_obsidian_impl(
    title: str,
    abstract: str,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    venue: str = "",
    arxiv_id: str = "",
    doi: str = "",
    _run_id: str = "",
) -> Dict[str, Any]:
    """Core implementation of export_to_obsidian, callable from both MCP registration and tests.

    Export a paper as Obsidian-formatted markdown with YAML frontmatter.

    Args:
        title: Paper title.
        abstract: Paper abstract text.
        authors: List of author name strings.
        year: Publication year.
        venue: Conference or journal name.
        arxiv_id: arXiv identifier (e.g. '1706.03762').
        doi: DOI identifier.
        _run_id: Optional run ID for event correlation.

    Returns:
        Dict with key 'markdown' containing the full rendered markdown string
        (YAML frontmatter + body). No filesystem writes are performed.
    """
    start = time.monotonic()
    normalized_authors = list(authors or [])
    args = {
        "title": title,
        "abstract_len": len(abstract),
        "author_count": len(normalized_authors),
        "year": year,
        "arxiv_id": arxiv_id or None,
    }

    try:
        # Build metadata rows for the template
        metadata_rows: List[str] = []
        if normalized_authors:
            metadata_rows.append(f"Authors: {', '.join(normalized_authors)}")
        if year:
            metadata_rows.append(f"Year: {year}")
        if venue:
            metadata_rows.append(f"Venue: {venue}")

        # Build external links from identifiers
        external_links: List[str] = []
        if arxiv_id:
            external_links.append(f"[arXiv](https://arxiv.org/abs/{arxiv_id})")
        if doi:
            external_links.append(f"[DOI](https://doi.org/{doi})")

        # Build paper dict for template rendering
        paper: Dict[str, Any] = {
            "title": title,
            "abstract": abstract,
            "authors": normalized_authors,
            "year": year,
            "venue": venue,
            "arxiv_id": arxiv_id,
            "doi": doi,
        }

        exporter = _get_exporter()
        body = await anyio.to_thread.run_sync(
            lambda: exporter._render_paper_note(
                template_path=None,
                title=title,
                abstract=abstract,
                metadata_rows=metadata_rows,
                track_link=None,
                external_links=external_links,
                related_links=[],
                reference_links=[],
                cited_by_links=[],
                paper=paper,
                track=None,
                related_titles=[],
            )
        )

        # Import _yaml_frontmatter to build YAML header
        from paperbot.infrastructure.exporters.obsidian_exporter import _yaml_frontmatter

        frontmatter = _yaml_frontmatter(
            {
                "title": title,
                "paperbot_type": "paper",
                "authors": normalized_authors,
                "year": year,
                "venue": venue or None,
                "arxiv_id": arxiv_id or None,
                "doi": doi or None,
            }
        )

        markdown = frontmatter + body

        output = {"markdown": markdown}

        log_tool_call(
            tool_name="export_to_obsidian",
            arguments=args,
            result_summary={"markdown_len": len(markdown)},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
        )
        return output

    except Exception as exc:
        log_tool_call(
            tool_name="export_to_obsidian",
            arguments=args,
            result_summary={},
            duration_ms=(time.monotonic() - start) * 1000,
            run_id=_run_id or None,
            error=str(exc),
        )
        raise


def register(mcp) -> None:
    """Register the export_to_obsidian tool on the given FastMCP instance."""

    @mcp.tool()
    async def export_to_obsidian(
        title: str,
        abstract: str,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        venue: str = "",
        arxiv_id: str = "",
        doi: str = "",
        _run_id: str = "",
    ) -> dict:
        """Export a paper as Obsidian-formatted markdown with YAML frontmatter.

        Returns in-memory rendered markdown (no filesystem writes). The markdown
        includes YAML frontmatter (title, paperbot_type, authors, year, venue,
        arxiv_id, doi) and a formatted body with Summary, Metadata, and Links sections.
        """
        return await _export_to_obsidian_impl(
            title, abstract, authors, year, venue, arxiv_id, doi, _run_id
        )
