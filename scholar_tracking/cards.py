# scholar_tracking/cards.py
"""
信息卡片格式化 - 借鉴 JobLeap 的职位卡片设计
实现论文/学者信息卡片展示
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import textwrap

from .models import Scholar, PaperMeta
from .feed import FeedEvent, FeedEventType


class CardStyle(Enum):
    """卡片样式"""
    COMPACT = "compact"       # 紧凑模式
    DETAILED = "detailed"     # 详细模式
    MINIMAL = "minimal"       # 极简模式


class OutputFormat(Enum):
    """输出格式"""
    PLAIN_TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class CardTheme:
    """卡片主题配置"""
    # 边框字符
    top_left: str = "┌"
    top_right: str = "┐"
    bottom_left: str = "└"
    bottom_right: str = "┘"
    horizontal: str = "─"
    vertical: str = "│"
    
    # 分隔符
    separator: str = "├"
    separator_end: str = "┤"
    
    # 卡片宽度
    width: int = 60
    
    # 图标
    icons: Dict[str, str] = field(default_factory=lambda: {
        "paper": "📄",
        "scholar": "👤",
        "citation": "📊",
        "venue": "🏛️",
        "date": "📅",
        "code": "💻",
        "star": "⭐",
        "tag": "🏷️",
        "link": "🔗",
        "new": "🆕",
        "trending": "🔥",
        "affiliation": "🏢",
        "h_index": "📈",
    })


# 默认主题
DEFAULT_THEME = CardTheme()


class TextWrapper:
    """文本换行处理"""
    
    @staticmethod
    def wrap(text: str, width: int, prefix: str = "") -> List[str]:
        """将文本换行到指定宽度"""
        if not text:
            return []
        
        # 使用 textwrap 进行换行
        wrapper = textwrap.TextWrapper(
            width=width - len(prefix),
            initial_indent="",
            subsequent_indent="",
            break_long_words=True,
            break_on_hyphens=True,
        )
        
        lines = wrapper.wrap(text)
        return [prefix + line for line in lines]
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """截断文本"""
        if not text or len(text) <= max_length:
            return text or ""
        return text[:max_length - len(suffix)] + suffix


class PaperCard:
    """
    论文信息卡片
    
    设计灵感来自 JobLeap 的职位卡片:
    - 标题突出
    - 关键信息标签化
    - 简洁的描述
    - 时间戳
    """
    
    def __init__(self, paper: PaperMeta, theme: CardTheme = None):
        self.paper = paper
        self.theme = theme or DEFAULT_THEME
    
    def render(
        self,
        style: CardStyle = CardStyle.DETAILED,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
    ) -> str:
        """渲染卡片"""
        if output_format == OutputFormat.MARKDOWN:
            return self._render_markdown(style)
        elif output_format == OutputFormat.HTML:
            return self._render_html(style)
        elif output_format == OutputFormat.JSON:
            return self._render_json()
        else:
            return self._render_text(style)
    
    def _render_text(self, style: CardStyle) -> str:
        """渲染纯文本卡片"""
        t = self.theme
        w = t.width
        icons = t.icons
        
        lines = []
        
        # 顶部边框
        lines.append(t.top_left + t.horizontal * (w - 2) + t.top_right)
        
        # 标题
        title = TextWrapper.truncate(self.paper.title, w - 6)
        lines.append(f"{t.vertical} {icons['paper']} {title}".ljust(w - 1) + t.vertical)
        
        # 作者
        if self.paper.authors:
            authors = ", ".join(self.paper.authors[:3])
            if len(self.paper.authors) > 3:
                authors += f" +{len(self.paper.authors) - 3}"
            authors = TextWrapper.truncate(authors, w - 6)
            lines.append(f"{t.vertical}    {authors}".ljust(w - 1) + t.vertical)
        
        # 分隔线
        if style != CardStyle.MINIMAL:
            lines.append(t.separator + t.horizontal * (w - 2) + t.separator_end)
        
        # 详细模式显示摘要
        if style == CardStyle.DETAILED and self.paper.abstract:
            abstract = TextWrapper.truncate(self.paper.abstract, 150)
            wrapped = TextWrapper.wrap(abstract, w - 6, f"{t.vertical}    ")
            lines.extend(wrapped)
            lines.append(f"{t.vertical}".ljust(w - 1) + t.vertical)
        
        # 标签行
        tags = []
        if self.paper.venue:
            tags.append(f"{icons['venue']} {self.paper.venue}")
        if self.paper.year:
            tags.append(f"{self.paper.year}")
        
        if tags:
            tag_line = " | ".join(tags)
            tag_line = TextWrapper.truncate(tag_line, w - 6)
            lines.append(f"{t.vertical} {icons['tag']} {tag_line}".ljust(w - 1) + t.vertical)
        
        # 指标行
        metrics = []
        metrics.append(f"{icons['citation']} 引用: {self.paper.citation_count}")
        if self.paper.has_code:
            metrics.append(f"{icons['code']} 有代码")
        
        metric_line = " | ".join(metrics)
        lines.append(f"{t.vertical} {metric_line}".ljust(w - 1) + t.vertical)
        
        # 时间戳
        if self.paper.publication_date:
            date_str = self.paper.publication_date[:10]
            lines.append(f"{t.vertical} {icons['date']} 发布: {date_str}".ljust(w - 1) + t.vertical)
        
        # 底部边框
        lines.append(t.bottom_left + t.horizontal * (w - 2) + t.bottom_right)
        
        return "\n".join(lines)
    
    def _render_markdown(self, style: CardStyle) -> str:
        """渲染 Markdown 卡片"""
        icons = self.theme.icons
        lines = []
        
        # 标题
        lines.append(f"### {icons['paper']} {self.paper.title}")
        lines.append("")
        
        # 作者
        if self.paper.authors:
            authors = ", ".join(self.paper.authors[:5])
            if len(self.paper.authors) > 5:
                authors += f" +{len(self.paper.authors) - 5}"
            lines.append(f"**作者:** {authors}")
        
        # 摘要
        if style == CardStyle.DETAILED and self.paper.abstract:
            lines.append("")
            lines.append(f"> {TextWrapper.truncate(self.paper.abstract, 300)}")
        
        # 标签
        lines.append("")
        tags = []
        if self.paper.venue:
            tags.append(f"`{self.paper.venue}`")
        if self.paper.year:
            tags.append(f"`{self.paper.year}`")
        for field in self.paper.fields_of_study[:3]:
            tags.append(f"`{field}`")
        
        if tags:
            lines.append(" ".join(tags))
        
        # 指标
        lines.append("")
        lines.append(f"- {icons['citation']} **引用:** {self.paper.citation_count}")
        if self.paper.influential_citation_count:
            lines.append(f"- {icons['star']} **有影响力引用:** {self.paper.influential_citation_count}")
        if self.paper.has_code:
            lines.append(f"- {icons['code']} **代码:** [GitHub]({self.paper.github_url})")
        
        # 链接
        if self.paper.url:
            lines.append(f"- {icons['link']} [查看论文]({self.paper.url})")
        
        lines.append("")
        lines.append("---")
        
        return "\n".join(lines)
    
    def _render_html(self, style: CardStyle) -> str:
        """渲染 HTML 卡片"""
        icons = self.theme.icons
        
        code_badge = ""
        if self.paper.has_code:
            code_badge = f'<span class="badge badge-code">{icons["code"]} 有代码</span>'
        
        tags_html = ""
        if self.paper.fields_of_study:
            tags = [f'<span class="tag">{f}</span>' for f in self.paper.fields_of_study[:3]]
            tags_html = " ".join(tags)
        
        abstract_html = ""
        if style == CardStyle.DETAILED and self.paper.abstract:
            abstract_html = f'<p class="abstract">{TextWrapper.truncate(self.paper.abstract, 300)}</p>'
        
        return f"""
<div class="paper-card">
    <div class="card-header">
        <h3>{icons['paper']} {self.paper.title}</h3>
        <span class="authors">{', '.join(self.paper.authors[:3])}</span>
    </div>
    {abstract_html}
    <div class="card-meta">
        <span class="venue">{icons['venue']} {self.paper.venue or 'Unknown'}</span>
        <span class="year">{self.paper.year or 'N/A'}</span>
        {tags_html}
    </div>
    <div class="card-stats">
        <span class="citations">{icons['citation']} {self.paper.citation_count} 引用</span>
        {code_badge}
    </div>
    <div class="card-footer">
        <span class="date">{icons['date']} {self.paper.publication_date or 'N/A'}</span>
    </div>
</div>
"""
    
    def _render_json(self) -> str:
        """渲染 JSON"""
        import json
        return json.dumps(self.paper.to_dict(), ensure_ascii=False, indent=2)


class ScholarCard:
    """
    学者信息卡片
    
    类似 JobLeap 展示公司信息的方式
    """
    
    def __init__(self, scholar: Scholar, theme: CardTheme = None):
        self.scholar = scholar
        self.theme = theme or DEFAULT_THEME
    
    def render(
        self,
        style: CardStyle = CardStyle.DETAILED,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
        recent_papers: List[PaperMeta] = None,
    ) -> str:
        """渲染卡片"""
        if output_format == OutputFormat.MARKDOWN:
            return self._render_markdown(style, recent_papers)
        elif output_format == OutputFormat.HTML:
            return self._render_html(style, recent_papers)
        elif output_format == OutputFormat.JSON:
            return self._render_json()
        else:
            return self._render_text(style, recent_papers)
    
    def _render_text(self, style: CardStyle, recent_papers: List[PaperMeta] = None) -> str:
        """渲染纯文本卡片"""
        t = self.theme
        w = t.width
        icons = t.icons
        
        lines = []
        
        # 顶部边框
        lines.append(t.top_left + t.horizontal * (w - 2) + t.top_right)
        
        # 姓名
        lines.append(f"{t.vertical} {icons['scholar']} {self.scholar.name}".ljust(w - 1) + t.vertical)
        
        # 机构
        if self.scholar.affiliations:
            aff = TextWrapper.truncate(self.scholar.affiliations[0], w - 8)
            lines.append(f"{t.vertical}    {icons['affiliation']} {aff}".ljust(w - 1) + t.vertical)
        
        # 分隔线
        lines.append(t.separator + t.horizontal * (w - 2) + t.separator_end)
        
        # 指标
        metrics = []
        if self.scholar.h_index:
            metrics.append(f"{icons['h_index']} H指数: {self.scholar.h_index}")
        if self.scholar.citation_count:
            metrics.append(f"{icons['citation']} 引用: {self.scholar.citation_count}")
        if self.scholar.paper_count:
            metrics.append(f"{icons['paper']} 论文: {self.scholar.paper_count}")
        
        for metric in metrics:
            lines.append(f"{t.vertical} {metric}".ljust(w - 1) + t.vertical)
        
        # 研究关键词
        if style != CardStyle.MINIMAL and self.scholar.keywords:
            keywords = ", ".join(self.scholar.keywords[:5])
            keywords = TextWrapper.truncate(keywords, w - 8)
            lines.append(f"{t.vertical} {icons['tag']} {keywords}".ljust(w - 1) + t.vertical)
        
        # 最近论文
        if style == CardStyle.DETAILED and recent_papers:
            lines.append(t.separator + t.horizontal * (w - 2) + t.separator_end)
            lines.append(f"{t.vertical} 最新论文:".ljust(w - 1) + t.vertical)
            for paper in recent_papers[:3]:
                title = TextWrapper.truncate(paper.title, w - 10)
                lines.append(f"{t.vertical}   • {title}".ljust(w - 1) + t.vertical)
        
        # 底部边框
        lines.append(t.bottom_left + t.horizontal * (w - 2) + t.bottom_right)
        
        return "\n".join(lines)
    
    def _render_markdown(self, style: CardStyle, recent_papers: List[PaperMeta] = None) -> str:
        """渲染 Markdown 卡片"""
        icons = self.theme.icons
        lines = []
        
        # 标题
        lines.append(f"### {icons['scholar']} {self.scholar.name}")
        
        # 机构
        if self.scholar.affiliations:
            lines.append(f"*{self.scholar.affiliations[0]}*")
        
        lines.append("")
        
        # 指标
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        if self.scholar.h_index:
            lines.append(f"| {icons['h_index']} H指数 | {self.scholar.h_index} |")
        if self.scholar.citation_count:
            lines.append(f"| {icons['citation']} 总引用 | {self.scholar.citation_count:,} |")
        if self.scholar.paper_count:
            lines.append(f"| {icons['paper']} 论文数 | {self.scholar.paper_count} |")
        
        # 关键词
        if self.scholar.keywords:
            lines.append("")
            tags = [f"`{k}`" for k in self.scholar.keywords[:5]]
            lines.append(f"**研究领域:** {' '.join(tags)}")
        
        # 最近论文
        if style == CardStyle.DETAILED and recent_papers:
            lines.append("")
            lines.append("**最新论文:**")
            for paper in recent_papers[:5]:
                lines.append(f"- [{paper.title}]({paper.url or '#'}) ({paper.year})")
        
        # 链接
        if self.scholar.homepage:
            lines.append("")
            lines.append(f"{icons['link']} [主页]({self.scholar.homepage})")
        
        lines.append("")
        lines.append("---")
        
        return "\n".join(lines)
    
    def _render_html(self, style: CardStyle, recent_papers: List[PaperMeta] = None) -> str:
        """渲染 HTML 卡片"""
        icons = self.theme.icons
        
        keywords_html = ""
        if self.scholar.keywords:
            tags = [f'<span class="tag">{k}</span>' for k in self.scholar.keywords[:5]]
            keywords_html = f'<div class="keywords">{" ".join(tags)}</div>'
        
        papers_html = ""
        if style == CardStyle.DETAILED and recent_papers:
            paper_items = [
                f'<li><a href="{p.url or "#"}">{p.title}</a> ({p.year})</li>'
                for p in recent_papers[:3]
            ]
            papers_html = f'<div class="recent-papers"><h4>最新论文</h4><ul>{"".join(paper_items)}</ul></div>'
        
        return f"""
<div class="scholar-card">
    <div class="card-header">
        <h3>{icons['scholar']} {self.scholar.name}</h3>
        <span class="affiliation">{icons['affiliation']} {self.scholar.affiliations[0] if self.scholar.affiliations else 'Unknown'}</span>
    </div>
    <div class="card-stats">
        <div class="stat">
            <span class="label">{icons['h_index']} H指数</span>
            <span class="value">{self.scholar.h_index or 'N/A'}</span>
        </div>
        <div class="stat">
            <span class="label">{icons['citation']} 引用</span>
            <span class="value">{self.scholar.citation_count or 'N/A'}</span>
        </div>
        <div class="stat">
            <span class="label">{icons['paper']} 论文</span>
            <span class="value">{self.scholar.paper_count or 'N/A'}</span>
        </div>
    </div>
    {keywords_html}
    {papers_html}
</div>
"""
    
    def _render_json(self) -> str:
        """渲染 JSON"""
        import json
        return json.dumps(self.scholar.to_dict(), ensure_ascii=False, indent=2)


class FeedEventCard:
    """信息流事件卡片"""
    
    def __init__(self, event: FeedEvent, theme: CardTheme = None):
        self.event = event
        self.theme = theme or DEFAULT_THEME
    
    def render(self, output_format: OutputFormat = OutputFormat.PLAIN_TEXT) -> str:
        """渲染卡片"""
        if output_format == OutputFormat.MARKDOWN:
            return self._render_markdown()
        else:
            return self._render_text()
    
    def _render_text(self) -> str:
        """渲染纯文本"""
        t = self.theme
        w = t.width
        
        time_str = self.event.timestamp.strftime("%Y-%m-%d %H:%M")
        
        lines = []
        lines.append(t.top_left + t.horizontal * (w - 2) + t.top_right)
        lines.append(f"{t.vertical} [{time_str}] {self.event.title}".ljust(w - 1) + t.vertical)
        
        desc = TextWrapper.truncate(self.event.description, w - 6)
        lines.append(f"{t.vertical}   {desc}".ljust(w - 1) + t.vertical)
        
        if self.event.scholar:
            lines.append(f"{t.vertical}   👤 {self.event.scholar.name}".ljust(w - 1) + t.vertical)
        
        lines.append(t.bottom_left + t.horizontal * (w - 2) + t.bottom_right)
        
        return "\n".join(lines)
    
    def _render_markdown(self) -> str:
        """渲染 Markdown"""
        time_str = self.event.timestamp.strftime("%Y-%m-%d %H:%M")
        
        lines = []
        lines.append(f"#### {self.event.title}")
        lines.append(f"*{time_str}*")
        lines.append("")
        lines.append(self.event.description)
        
        if self.event.scholar:
            lines.append(f"- 👤 学者: {self.event.scholar.name}")
        if self.event.paper:
            lines.append(f"- 📄 论文: {self.event.paper.title}")
        
        lines.append("")
        
        return "\n".join(lines)


class CardRenderer:
    """
    卡片渲染服务
    
    统一管理各类卡片的渲染
    """
    
    def __init__(self, theme: CardTheme = None):
        self.theme = theme or DEFAULT_THEME
    
    def render_paper(
        self,
        paper: PaperMeta,
        style: CardStyle = CardStyle.DETAILED,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
    ) -> str:
        """渲染论文卡片"""
        card = PaperCard(paper, self.theme)
        return card.render(style, output_format)
    
    def render_scholar(
        self,
        scholar: Scholar,
        style: CardStyle = CardStyle.DETAILED,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
        recent_papers: List[PaperMeta] = None,
    ) -> str:
        """渲染学者卡片"""
        card = ScholarCard(scholar, self.theme)
        return card.render(style, output_format, recent_papers)
    
    def render_feed_event(
        self,
        event: FeedEvent,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
    ) -> str:
        """渲染信息流事件卡片"""
        card = FeedEventCard(event, self.theme)
        return card.render(output_format)
    
    def render_paper_list(
        self,
        papers: List[PaperMeta],
        style: CardStyle = CardStyle.COMPACT,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
        max_items: int = 10,
    ) -> str:
        """批量渲染论文列表"""
        results = []
        for paper in papers[:max_items]:
            results.append(self.render_paper(paper, style, output_format))
        
        separator = "\n\n" if output_format == OutputFormat.MARKDOWN else "\n"
        return separator.join(results)
    
    def render_scholar_list(
        self,
        scholars: List[Scholar],
        style: CardStyle = CardStyle.COMPACT,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
        papers_map: Dict[str, List[PaperMeta]] = None,
        max_items: int = 10,
    ) -> str:
        """批量渲染学者列表"""
        papers_map = papers_map or {}
        results = []
        
        for scholar in scholars[:max_items]:
            recent_papers = papers_map.get(scholar.semantic_scholar_id, [])[:3]
            results.append(self.render_scholar(scholar, style, output_format, recent_papers))
        
        separator = "\n\n" if output_format == OutputFormat.MARKDOWN else "\n"
        return separator.join(results)
    
    def render_feed(
        self,
        events: List[FeedEvent],
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT,
        max_items: int = 20,
    ) -> str:
        """渲染信息流"""
        results = []
        for event in events[:max_items]:
            results.append(self.render_feed_event(event, output_format))
        
        separator = "\n" if output_format == OutputFormat.MARKDOWN else "\n"
        return separator.join(results)


# CSS 样式表（用于 HTML 输出）
CARD_CSS = """
<style>
.paper-card, .scholar-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    background: #fff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    max-width: 600px;
}

.card-header h3 {
    margin: 0 0 8px 0;
    font-size: 18px;
    color: #333;
}

.card-header .authors,
.card-header .affiliation {
    color: #666;
    font-size: 14px;
}

.abstract {
    color: #555;
    font-size: 14px;
    line-height: 1.5;
    margin: 12px 0;
    padding: 10px;
    background: #f9f9f9;
    border-radius: 4px;
}

.card-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 12px 0;
}

.card-meta span {
    font-size: 13px;
    color: #666;
}

.tag {
    display: inline-block;
    padding: 2px 8px;
    background: #e3f2fd;
    color: #1976d2;
    border-radius: 4px;
    font-size: 12px;
    margin: 2px;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
}

.badge-code {
    background: #e8f5e9;
    color: #388e3c;
}

.card-stats {
    display: flex;
    gap: 20px;
    margin: 12px 0;
}

.stat {
    text-align: center;
}

.stat .label {
    display: block;
    font-size: 12px;
    color: #888;
}

.stat .value {
    display: block;
    font-size: 18px;
    font-weight: bold;
    color: #333;
}

.card-footer {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #eee;
    font-size: 12px;
    color: #999;
}

.recent-papers h4 {
    font-size: 14px;
    margin: 12px 0 8px 0;
}

.recent-papers ul {
    margin: 0;
    padding-left: 20px;
}

.recent-papers li {
    font-size: 13px;
    margin: 4px 0;
}

.keywords {
    margin: 12px 0;
}
</style>
"""
