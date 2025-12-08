"""
IR -> HTML 渲染器（精简版）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from .base import RenderContext, Renderer

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.6; }}
    h1, h2, h3 {{ color: #1f2937; }}
    .toc {{ background: #f9fafb; padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; }}
    .toc a {{ color: #2563eb; text-decoration: none; }}
    pre {{ background: #f3f4f6; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <div class="meta">
    <p><strong>模型:</strong> {model_info} | <strong>抓取时间:</strong> {data_time} | <strong>环境:</strong> {env_info}</p>
  </div>
  <div class="toc">
    <strong>目录</strong>
    <ol>
      {toc_items}
    </ol>
  </div>
  {chapters_html}
</body>
</html>
"""


class HTMLRenderer(Renderer):
    """HTML 渲染器。"""
    
    def render(self, ctx: RenderContext, **kwargs) -> str:
        """
        渲染为 HTML。
        
        Args:
            ctx: 渲染上下文
            **kwargs: 额外参数
            
        Returns:
            HTML 字符串
        """
        toc_items = "\n".join(
            [f'<li><a href="#{c.get("slug", f"sec-{i}")}">{c.get("title","")}</a></li>' for i, c in enumerate(ctx.chapters)]
        )
        chapters_html = "\n".join([self._render_chapter(c, idx) for idx, c in enumerate(ctx.chapters)])
        return HTML_TEMPLATE.format(
            title=ctx.title,
            subtitle=ctx.subtitle,
            toc_items=toc_items,
            chapters_html=chapters_html,
            model_info=ctx.model_info or "n/a",
            data_time=ctx.data_time or "n/a",
            env_info=ctx.env_info or "n/a",
        )

    def _render_chapter(self, chapter: Dict[str, Any], idx: int) -> str:
        """渲染单个章节。"""
        slug = chapter.get("slug", f"sec-{idx}")
        title = chapter.get("title", "")
        blocks = chapter.get("blocks", [])
        parts = [f'<section id="{slug}"><h2>{title}</h2>']
        for b in blocks:
            btype = b.get("type")
            content = b.get("content", "")
            if btype == "heading":
                level = b.get("level", 2)
                parts.append(f"<h{level}>{content}</h{level}>")
            elif btype == "paragraph":
                parts.append(f"<p>{content}</p>")
            elif btype == "bullet_list":
                if isinstance(content, list):
                    items = "".join([f"<li>{c}</li>" for c in content])
                    parts.append(f"<ul>{items}</ul>")
            elif btype == "number_list":
                if isinstance(content, list):
                    items = "".join([f"<li>{c}</li>" for c in content])
                    parts.append(f"<ol>{items}</ol>")
            elif btype == "quote":
                parts.append(f"<blockquote>{content}</blockquote>")
            elif btype == "code":
                parts.append(f"<pre><code>{content}</code></pre>")
            else:
                parts.append(f"<p>{content}</p>")
        parts.append("</section>")
        return "\n".join(parts)

    def persist(self, html: str, path: Path) -> Path:
        """
        持久化 HTML 到文件。
        
        Args:
            html: HTML 内容
            path: 输出路径
            
        Returns:
            写入的文件路径
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path

