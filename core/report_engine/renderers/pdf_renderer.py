"""
HTML -> PDF 渲染（依赖 weasyprint，可选）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from weasyprint import HTML  # type: ignore
except ImportError:  # pragma: no cover
    HTML = None


class PDFRenderer:
    def render(self, html_content: str, output_path: Path) -> Optional[Path]:
        if HTML is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_content).write_pdf(str(output_path))
        return output_path

