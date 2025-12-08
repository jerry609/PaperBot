"""
HTML -> PDF 渲染（依赖 weasyprint，可选）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from .base import RenderContext, Renderer

try:
    from weasyprint import HTML  # type: ignore
except ImportError:  # pragma: no cover
    HTML = None


class PDFRenderer(Renderer):
    """PDF 渲染器（依赖 weasyprint）。"""
    
    def render(self, ctx: RenderContext, html_content: str, output_path: Path) -> Optional[Path]:
        """
        渲染为 PDF。
        
        Args:
            ctx: 渲染上下文
            html_content: HTML 内容
            output_path: 输出路径
            
        Returns:
            写入的文件路径，如果 weasyprint 不可用则返回 None
        """
        if HTML is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_content).write_pdf(str(output_path))
        return output_path

