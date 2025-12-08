# reports/writer.py
"""
报告写入器
负责将生成的报告写入文件系统
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

# 使用相对导入以支持新架构
try:
    from paperbot.domain.paper import PaperMeta
    from paperbot.domain.influence.result import InfluenceResult
except ImportError:
    # 回退到原始导入路径
    try:
        from paperbot.domain.paper import PaperMeta
        from paperbot.domain.influence.result import InfluenceResult
    except ImportError:
        PaperMeta = None
        InfluenceResult = None

logger = logging.getLogger(__name__)


class ReportWriter:
    """报告写入器"""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        template_name: str = "paper_report.md.j2",
    ):
        """
        初始化报告写入器
        
        Args:
            output_dir: 输出目录路径
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            output_dir = project_root / "output" / "reports"
        
        self.output_dir = Path(output_dir)
        self.template_name = template_name
        self._ensure_output_dir()
        
        # 初始化 Jinja2 环境（如果可用）
        self._jinja_env = None
        if HAS_JINJA2:
            templates_dir = Path(__file__).parent / "templates"
            if templates_dir.exists():
                self._jinja_env = Environment(
                    loader=FileSystemLoader(str(templates_dir)),
                    autoescape=select_autoescape(['html', 'xml']),
                )
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_filename(self, name: str) -> str:
        """
        清理文件名，移除非法字符
        
        Args:
            name: 原始名称
            
        Returns:
            清理后的文件名
        """
        # 替换非法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # 替换多个连续空格/下划线
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        # 移除首尾下划线
        sanitized = sanitized.strip('_')
        # 限制长度
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized or "unnamed"
    
    def _get_scholar_dir(self, scholar_name: str) -> Path:
        """获取学者的报告目录"""
        safe_name = self._sanitize_filename(scholar_name)
        scholar_dir = self.output_dir / safe_name
        scholar_dir.mkdir(parents=True, exist_ok=True)
        return scholar_dir
    
    def _generate_filename(
        self,
        paper: "PaperMeta",
        date: Optional[datetime] = None,
    ) -> str:
        """
        生成报告文件名
        
        格式: {YYYY-MM-DD}_{paper_id}.md
        """
        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        
        # 使用论文 ID（更短且唯一）
        paper_id = self._sanitize_filename(paper.paper_id[:20])
        
        return f"{date_str}_{paper_id}.md"
    
    def write_report(
        self,
        report_content: str,
        paper: "PaperMeta",
        scholar_name: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        写入报告到文件
        
        Args:
            report_content: 报告内容（Markdown）
            paper: 论文元数据
            scholar_name: 学者名称（用于目录分组）
            filename: 自定义文件名
            
        Returns:
            报告文件路径
        """
        # 确定输出目录
        if scholar_name:
            output_dir = self._get_scholar_dir(scholar_name)
        else:
            output_dir = self.output_dir
        
        # 确定文件名
        if filename is None:
            filename = self._generate_filename(paper)
        
        # 写入文件
        file_path = output_dir / filename
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            logger.info(f"Report written to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to write report: {e}")
            raise
    
    def render_template(
        self,
        paper: "PaperMeta",
        influence: "InfluenceResult",
        research_result: Optional[Dict[str, Any]] = None,
        code_analysis_result: Optional[Dict[str, Any]] = None,
        quality_result: Optional[Dict[str, Any]] = None,
        scholar_name: Optional[str] = None,
        repro_result: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        使用 Jinja2 模板渲染报告
        
        Args:
            paper: 论文元数据
            influence: 影响力评分结果
            research_result: 研究阶段结果
            code_analysis_result: 代码分析结果
            quality_result: 质量评估结果
            scholar_name: 学者名称
            
        Returns:
            渲染后的 Markdown 报告
        """
        # 准备模板数据，确保与 paper_report.md.j2 期望的字段一致
        template_data = {
            "paper": paper.to_dict(),
            "influence": influence.to_dict(),
            "research": research_result or {},
            "code_analysis": code_analysis_result or {},
            "quality": quality_result or {},
            "scholar_name": scholar_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "repro": repro_result or {},
            "meta": meta or {},
        }

        if not HAS_JINJA2 or not self._jinja_env:
            logger.warning("Jinja2 not available, using fallback template")
            return self._fallback_render(
                paper, influence, research_result,
                code_analysis_result, quality_result, scholar_name
            )
        
        try:
            template = self._jinja_env.get_template(self.template_name)
            return template.render(**template_data)
        except Exception as e:
            logger.warning(f"Template rendering failed: {e}, using fallback")
            return self._fallback_render(
                paper, influence, research_result,
                code_analysis_result, quality_result, scholar_name
            )
    
    def _fallback_render(
        self,
        paper: "PaperMeta",
        influence: "InfluenceResult",
        research_result: Optional[Dict[str, Any]] = None,
        code_analysis_result: Optional[Dict[str, Any]] = None,
        quality_result: Optional[Dict[str, Any]] = None,
        scholar_name: Optional[str] = None,
    ) -> str:
        """备用模板渲染"""
        authors = ", ".join(paper.authors) if paper.authors else "未知"
        
        report = f"""# {paper.title}

## 📋 元信息

| 属性 | 值 |
|------|-----|
| **作者** | {authors} |
| **年份** | {paper.year or '未知'} |
| **发表于** | {paper.venue or '未知'} |
| **引用数** | {paper.citation_count} |
| **Semantic Scholar ID** | {paper.paper_id} |
"""
        if scholar_name:
            report += f"\n> 📚 **追踪学者**: {scholar_name}\n"
        
        report += f"""
---

## 📝 摘要

{paper.tldr or paper.abstract or '暂无摘要'}

---

## 💻 代码信息

"""
        if paper.github_url:
            report += f"- **仓库地址**: [{paper.github_url}]({paper.github_url})\n"
            if code_analysis_result:
                if code_analysis_result.get("stars"):
                    report += f"- **Stars**: ⭐ {code_analysis_result['stars']}\n"
                if code_analysis_result.get("language"):
                    report += f"- **语言**: {code_analysis_result['language']}\n"
        else:
            report += "未发现公开代码仓库\n"
        
        report += f"""
---

## 📊 影响力评分 (PIS)

| 维度 | 分数 |
|------|------|
| **🎯 总分** | **{influence.total_score:.1f}/100** |
| 📚 学术影响力 | {influence.academic_score:.1f}/100 |
| 🔧 工程影响力 | {influence.engineering_score:.1f}/100 |

{influence.explanation}

---

## 🎯 推荐级别

**{influence.recommendation.value}**

---

*📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*🤖 由 PaperBot 自动生成*
"""
        return report
    
    def write_summary_report(
        self,
        scholar_name: str,
        papers_results: list,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        写入学者论文汇总报告
        
        Args:
            scholar_name: 学者名称
            papers_results: 论文分析结果列表
            output_filename: 输出文件名
            
        Returns:
            报告文件路径
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        if output_filename is None:
            output_filename = f"{date_str}_summary.md"
        
        # 构建汇总报告
        report = f"""# {scholar_name} - 论文追踪汇总报告

**生成日期**: {date_str}  
**论文数量**: {len(papers_results)}

---

## 📊 论文列表

| # | 论文标题 | 年份 | 引用 | PIS评分 | 推荐 |
|---|----------|------|------|---------|------|
"""
        for i, (_, influence, data) in enumerate(papers_results, 1):
            paper_data = data.get("paper", {})
            title = paper_data.get("title", "未知")[:50]
            year = paper_data.get("year", "-")
            citations = paper_data.get("citation_count", 0)
            score = influence.total_score
            rec = influence.recommendation.value[:4]
            
            report += f"| {i} | {title}... | {year} | {citations} | {score:.1f} | {rec} |\n"
        
        report += f"""
---

## 📈 统计信息

"""
        if papers_results:
            scores = [r[1].total_score for r in papers_results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            report += f"""- **平均 PIS 评分**: {avg_score:.1f}
- **最高评分**: {max_score:.1f}
- **最低评分**: {min_score:.1f}
"""
        
        report += f"""
---

*🤖 由 PaperBot 自动生成*
"""
        
        # 写入文件
        scholar_dir = self._get_scholar_dir(scholar_name)
        file_path = scholar_dir / output_filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Summary report written to: {file_path}")
        return file_path

