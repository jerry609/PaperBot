"""
报告渲染器模块。

提供多种输出格式的渲染能力：
- HTMLRenderer: HTML 渲染
- PDFRenderer: PDF 渲染（依赖 weasyprint）
"""

from .base import RenderContext, Renderer
from .html_renderer import HTMLRenderer
from .pdf_renderer import PDFRenderer

__all__ = [
    "RenderContext",
    "Renderer",
    "HTMLRenderer",
    "PDFRenderer",
]

