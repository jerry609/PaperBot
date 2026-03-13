from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import RenderContext, Renderer

try:
    from weasyprint import HTML  # type: ignore
except (ImportError, OSError):  # pragma: no cover
    HTML = None


class PDFRenderer(Renderer):
    """Render HTML to PDF when WeasyPrint is available."""

    def render(
        self,
        ctx: RenderContext,
        html_content: str,
        output_path: Path,
    ) -> Optional[Path]:
        if HTML is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_content).write_pdf(str(output_path))
        return output_path
