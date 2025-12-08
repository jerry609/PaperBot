"""
报告生成器

封装报告生成逻辑，支持多种输出格式。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    报告生成器
    
    支持 Markdown、HTML、PDF 格式输出。
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
    ):
        """
        初始化生成器
        
        Args:
            output_dir: 输出目录
            template_dir: 模板目录
        """
        self.output_dir = output_dir or Path("output/reports")
        self.template_dir = template_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        生成 Markdown 报告
        
        Args:
            data: 报告数据
            filename: 文件名（可选）
            
        Returns:
            输出文件路径
        """
        # 委托给 ReportWriter
        try:
            from paperbot.presentation.reports.writer import ReportWriter
            writer = ReportWriter(output_dir=self.output_dir)
            
            # 从数据中提取必要信息
            from src.paperbot.domain import PaperMeta, InfluenceResult
            
            paper = PaperMeta.from_dict(data.get("paper", {}))
            influence = InfluenceResult.from_dict(data.get("influence", {}))
            
            return writer.write_report(
                content=self._render_basic_report(data),
                paper=paper,
                scholar_name=data.get("scholar_name"),
            )
        except ImportError:
            # 降级到简单实现
            return self._write_simple_report(data, filename)
    
    def _render_basic_report(self, data: Dict[str, Any]) -> str:
        """渲染基础报告"""
        paper = data.get("paper", {})
        influence = data.get("influence", {})
        
        return f"""# {paper.get('title', 'Unknown')}

## 基本信息

- **作者**: {', '.join(paper.get('authors', []))}
- **年份**: {paper.get('year', 'N/A')}
- **发表于**: {paper.get('venue', 'N/A')}
- **引用数**: {paper.get('citation_count', 0)}

## 摘要

{paper.get('abstract', 'N/A')}

## 影响力评分

- **总分**: {influence.get('total_score', 0):.1f}/100
- **学术影响力**: {influence.get('academic_score', 0):.1f}/100
- **工程影响力**: {influence.get('engineering_score', 0):.1f}/100

{influence.get('explanation', '')}

---
*由 PaperBot 自动生成*
"""
    
    def _write_simple_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """简单报告写入"""
        content = self._render_basic_report(data)
        
        if not filename:
            paper_id = data.get("paper", {}).get("paper_id", "report")
            filename = f"{paper_id}.md"
        
        output_path = self.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        
        logger.info(f"Report written to: {output_path}")
        return output_path
    
    def generate_html(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        生成 HTML 报告
        
        Args:
            data: 报告数据
            filename: 文件名（可选）
            
        Returns:
            输出文件路径或 None
        """
        try:
            import markdown
            
            md_content = self._render_basic_report(data)
            html_content = markdown.markdown(md_content, extensions=['extra', 'toc'])
            
            if not filename:
                paper_id = data.get("paper", {}).get("paper_id", "report")
                filename = f"{paper_id}.html"
            
            output_path = self.output_dir / filename
            
            html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{data.get('paper', {}).get('title', 'Report')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }}
        h1 {{ color: #1a1a1a; }}
        h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
            
            output_path.write_text(html_doc, encoding="utf-8")
            logger.info(f"HTML report written to: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("markdown library not available, skipping HTML generation")
            return None

