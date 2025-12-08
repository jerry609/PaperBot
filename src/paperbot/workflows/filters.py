# scholar_tracking/filters.py
"""
多维度筛选器 - 借鉴 JobLeap 的筛选模式
实现学者类型、机构、研究领域、引用量等多维度筛选功能
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Set
from enum import Enum
from datetime import datetime, timedelta

from paperbot.domain.scholar import Scholar
from paperbot.domain.paper import PaperMeta


class ScholarType(Enum):
    """学者类型分类 - 类似 JobLeap 的公司类型筛选"""
    TOP_VENUE_REGULAR = "顶会常客"      # 经常在顶会发表
    HIGH_CITATION = "高引学者"           # 高引用量
    RISING_STAR = "新锐研究员"           # 近年崛起
    INDUSTRY_EXPERT = "业界大牛"         # 来自工业界
    PROLIFIC_AUTHOR = "高产作者"         # 发表数量多
    INTERDISCIPLINARY = "跨领域学者"     # 多领域交叉


class AffiliationType(Enum):
    """机构类型 - 类似 JobLeap 的互联网大厂/国央企等"""
    UNIVERSITY = "高校"
    TECH_COMPANY = "科技公司"
    RESEARCH_LAB = "研究院"
    STARTUP = "初创公司"
    GOVERNMENT = "政府机构"


class ResearchArea(Enum):
    """研究领域"""
    AI_SECURITY = "AI安全"
    LLM = "大语言模型"
    FEDERATED_LEARNING = "联邦学习"
    PRIVACY = "隐私计算"
    ADVERSARIAL_ML = "对抗机器学习"
    SYSTEMS_SECURITY = "系统安全"
    NETWORK_SECURITY = "网络安全"
    CRYPTOGRAPHY = "密码学"
    MALWARE = "恶意软件分析"
    SOFTWARE_SECURITY = "软件安全"


# 预定义的顶级会议列表
TOP_SECURITY_VENUES = {
    "S&P", "IEEE S&P", "Oakland",
    "CCS", "ACM CCS",
    "USENIX Security",
    "NDSS",
}

TOP_AI_VENUES = {
    "NeurIPS", "ICML", "ICLR",
    "CVPR", "ICCV", "ECCV",
    "ACL", "EMNLP", "NAACL",
    "AAAI", "IJCAI",
}

TOP_VENUES = TOP_SECURITY_VENUES | TOP_AI_VENUES

# 知名机构映射
KNOWN_AFFILIATIONS: Dict[str, AffiliationType] = {
    # 高校
    "UC Berkeley": AffiliationType.UNIVERSITY,
    "Stanford": AffiliationType.UNIVERSITY,
    "MIT": AffiliationType.UNIVERSITY,
    "CMU": AffiliationType.UNIVERSITY,
    "Carnegie Mellon": AffiliationType.UNIVERSITY,
    "Tsinghua": AffiliationType.UNIVERSITY,
    "Peking University": AffiliationType.UNIVERSITY,
    "Zhejiang University": AffiliationType.UNIVERSITY,
    
    # 科技公司
    "Google": AffiliationType.TECH_COMPANY,
    "Microsoft": AffiliationType.TECH_COMPANY,
    "Meta": AffiliationType.TECH_COMPANY,
    "Facebook": AffiliationType.TECH_COMPANY,
    "Amazon": AffiliationType.TECH_COMPANY,
    "Apple": AffiliationType.TECH_COMPANY,
    "OpenAI": AffiliationType.TECH_COMPANY,
    "Anthropic": AffiliationType.TECH_COMPANY,
    "DeepMind": AffiliationType.TECH_COMPANY,
    "Alibaba": AffiliationType.TECH_COMPANY,
    "Tencent": AffiliationType.TECH_COMPANY,
    "Baidu": AffiliationType.TECH_COMPANY,
    "ByteDance": AffiliationType.TECH_COMPANY,
    
    # 研究院
    "MSR": AffiliationType.RESEARCH_LAB,
    "Microsoft Research": AffiliationType.RESEARCH_LAB,
    "Google Research": AffiliationType.RESEARCH_LAB,
    "IBM Research": AffiliationType.RESEARCH_LAB,
}


@dataclass
class FilterCriteria:
    """筛选条件"""
    
    # 学者类型筛选
    scholar_types: List[ScholarType] = field(default_factory=list)
    
    # 机构筛选
    affiliation_types: List[AffiliationType] = field(default_factory=list)
    affiliation_names: List[str] = field(default_factory=list)
    
    # 研究领域筛选
    research_areas: List[ResearchArea] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # 数值筛选
    min_citations: Optional[int] = None
    max_citations: Optional[int] = None
    min_h_index: Optional[int] = None
    min_papers: Optional[int] = None
    
    # 时间筛选
    active_since: Optional[datetime] = None  # 近期有发表的
    
    # 自定义筛选函数
    custom_filters: List[Callable[[Scholar], bool]] = field(default_factory=list)


@dataclass
class PaperFilterCriteria:
    """论文筛选条件"""
    
    # 会议/期刊筛选
    venues: List[str] = field(default_factory=list)
    top_venues_only: bool = False
    
    # 年份筛选
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    
    # 引用筛选
    min_citations: Optional[int] = None
    max_citations: Optional[int] = None
    
    # 内容筛选
    title_contains: List[str] = field(default_factory=list)
    abstract_contains: List[str] = field(default_factory=list)
    fields_of_study: List[str] = field(default_factory=list)
    
    # 代码筛选
    has_code: Optional[bool] = None
    
    # 作者筛选
    author_names: List[str] = field(default_factory=list)
    
    # 自定义筛选
    custom_filters: List[Callable[[PaperMeta], bool]] = field(default_factory=list)


class ScholarClassifier:
    """学者分类器 - 自动识别学者类型"""
    
    @staticmethod
    def classify(scholar: Scholar, papers: Optional[List[PaperMeta]] = None) -> Set[ScholarType]:
        """
        根据学者信息和论文列表自动分类
        
        Args:
            scholar: 学者信息
            papers: 学者的论文列表
        
        Returns:
            学者类型集合
        """
        types = set()
        papers = papers or []
        
        # 高引学者: 总引用 > 10000
        if scholar.citation_count and scholar.citation_count > 10000:
            types.add(ScholarType.HIGH_CITATION)
        
        # 高产作者: 论文数 > 100
        if scholar.paper_count and scholar.paper_count > 100:
            types.add(ScholarType.PROLIFIC_AUTHOR)
        
        # 顶会常客: 近5年在顶会发表 > 5篇
        if papers:
            current_year = datetime.now().year
            recent_top_papers = [
                p for p in papers
                if p.year and p.year >= current_year - 5
                and p.venue and any(
                    top_venue.lower() in p.venue.lower()
                    for top_venue in TOP_VENUES
                )
            ]
            if len(recent_top_papers) >= 5:
                types.add(ScholarType.TOP_VENUE_REGULAR)
        
        # 新锐研究员: 近3年引用增长快
        if papers:
            recent_papers = [
                p for p in papers
                if p.year and p.year >= datetime.now().year - 3
            ]
            recent_citations = sum(p.citation_count for p in recent_papers)
            if recent_citations > 1000 and len(recent_papers) >= 3:
                types.add(ScholarType.RISING_STAR)
        
        # 业界大牛: 来自工业界
        for affiliation in scholar.affiliations:
            for name, aff_type in KNOWN_AFFILIATIONS.items():
                if name.lower() in affiliation.lower():
                    if aff_type == AffiliationType.TECH_COMPANY:
                        types.add(ScholarType.INDUSTRY_EXPERT)
                        break
        
        # 跨领域学者: 论文涉及多个领域
        if papers:
            all_fields = set()
            for p in papers:
                all_fields.update(p.fields_of_study)
            if len(all_fields) >= 3:
                types.add(ScholarType.INTERDISCIPLINARY)
        
        return types
    
    @staticmethod
    def get_affiliation_type(affiliation: str) -> Optional[AffiliationType]:
        """识别机构类型"""
        affiliation_lower = affiliation.lower()
        
        for name, aff_type in KNOWN_AFFILIATIONS.items():
            if name.lower() in affiliation_lower:
                return aff_type
        
        # 启发式规则
        if any(kw in affiliation_lower for kw in ["university", "大学", "institute"]):
            return AffiliationType.UNIVERSITY
        if any(kw in affiliation_lower for kw in ["inc", "corp", "ltd", "公司"]):
            return AffiliationType.TECH_COMPANY
        if any(kw in affiliation_lower for kw in ["lab", "research", "研究"]):
            return AffiliationType.RESEARCH_LAB
        
        return None


class ScholarFilter:
    """
    学者筛选器 - 主筛选类
    
    借鉴 JobLeap 的多维度筛选模式:
    - 社招/校招/实习 → 学者类型
    - 专精特新/上市公司/国央企 → 机构类型
    - 地点筛选 → 研究领域筛选
    """
    
    def __init__(self):
        self.classifier = ScholarClassifier()
    
    def filter_scholars(
        self,
        scholars: List[Scholar],
        criteria: FilterCriteria,
        papers_map: Optional[Dict[str, List[PaperMeta]]] = None,
    ) -> List[Scholar]:
        """
        筛选学者
        
        Args:
            scholars: 学者列表
            criteria: 筛选条件
            papers_map: 学者ID到论文列表的映射
        
        Returns:
            筛选后的学者列表
        """
        papers_map = papers_map or {}
        results = []
        
        for scholar in scholars:
            scholar_id = scholar.semantic_scholar_id or ""
            if self._match_scholar(scholar, criteria, papers_map.get(scholar_id, [])):
                results.append(scholar)
        
        return results
    
    def _match_scholar(
        self,
        scholar: Scholar,
        criteria: FilterCriteria,
        papers: List[PaperMeta],
    ) -> bool:
        """检查学者是否匹配筛选条件"""
        
        # 学者类型筛选
        if criteria.scholar_types:
            scholar_types = self.classifier.classify(scholar, papers)
            if not scholar_types.intersection(set(criteria.scholar_types)):
                return False
        
        # 机构类型筛选
        if criteria.affiliation_types:
            matched = False
            for aff in scholar.affiliations:
                aff_type = self.classifier.get_affiliation_type(aff)
                if aff_type in criteria.affiliation_types:
                    matched = True
                    break
            if not matched:
                return False
        
        # 机构名称筛选
        if criteria.affiliation_names:
            matched = False
            for aff in scholar.affiliations:
                for name in criteria.affiliation_names:
                    if name.lower() in aff.lower():
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                return False
        
        # 关键词筛选
        if criteria.keywords:
            scholar_keywords = set(k.lower() for k in scholar.keywords)
            filter_keywords = set(k.lower() for k in criteria.keywords)
            if not scholar_keywords.intersection(filter_keywords):
                return False
        
        # 引用量筛选
        if criteria.min_citations is not None:
            if not scholar.citation_count or scholar.citation_count < criteria.min_citations:
                return False
        
        if criteria.max_citations is not None:
            if scholar.citation_count and scholar.citation_count > criteria.max_citations:
                return False
        
        # H指数筛选
        if criteria.min_h_index is not None:
            if not scholar.h_index or scholar.h_index < criteria.min_h_index:
                return False
        
        # 论文数筛选
        if criteria.min_papers is not None:
            if not scholar.paper_count or scholar.paper_count < criteria.min_papers:
                return False
        
        # 活跃度筛选
        if criteria.active_since and papers:
            recent_papers = [
                p for p in papers
                if p.publication_date and
                datetime.fromisoformat(p.publication_date) >= criteria.active_since
            ]
            if not recent_papers:
                return False
        
        # 自定义筛选
        for custom_filter in criteria.custom_filters:
            if not custom_filter(scholar):
                return False
        
        return True


class PaperFilter:
    """论文筛选器"""
    
    def filter_papers(
        self,
        papers: List[PaperMeta],
        criteria: PaperFilterCriteria,
    ) -> List[PaperMeta]:
        """
        筛选论文
        
        Args:
            papers: 论文列表
            criteria: 筛选条件
        
        Returns:
            筛选后的论文列表
        """
        results = []
        
        for paper in papers:
            if self._match_paper(paper, criteria):
                results.append(paper)
        
        return results
    
    def _match_paper(self, paper: PaperMeta, criteria: PaperFilterCriteria) -> bool:
        """检查论文是否匹配筛选条件"""
        
        # 会议筛选
        if criteria.venues:
            if not paper.venue:
                return False
            venue_lower = paper.venue.lower()
            if not any(v.lower() in venue_lower for v in criteria.venues):
                return False
        
        # 顶会筛选
        if criteria.top_venues_only:
            if not paper.venue:
                return False
            if not any(v.lower() in paper.venue.lower() for v in TOP_VENUES):
                return False
        
        # 年份筛选
        if criteria.min_year is not None:
            if not paper.year or paper.year < criteria.min_year:
                return False
        
        if criteria.max_year is not None:
            if paper.year and paper.year > criteria.max_year:
                return False
        
        # 引用筛选
        if criteria.min_citations is not None:
            if paper.citation_count < criteria.min_citations:
                return False
        
        if criteria.max_citations is not None:
            if paper.citation_count > criteria.max_citations:
                return False
        
        # 标题包含
        if criteria.title_contains:
            title_lower = paper.title.lower()
            if not any(kw.lower() in title_lower for kw in criteria.title_contains):
                return False
        
        # 摘要包含
        if criteria.abstract_contains and paper.abstract:
            abstract_lower = paper.abstract.lower()
            if not any(kw.lower() in abstract_lower for kw in criteria.abstract_contains):
                return False
        
        # 领域筛选
        if criteria.fields_of_study:
            paper_fields = set(f.lower() for f in paper.fields_of_study)
            filter_fields = set(f.lower() for f in criteria.fields_of_study)
            if not paper_fields.intersection(filter_fields):
                return False
        
        # 代码筛选
        if criteria.has_code is not None:
            if paper.has_code != criteria.has_code:
                return False
        
        # 作者筛选
        if criteria.author_names:
            paper_authors = set(a.lower() for a in paper.authors)
            filter_authors = set(a.lower() for a in criteria.author_names)
            if not paper_authors.intersection(filter_authors):
                return False
        
        # 自定义筛选
        for custom_filter in criteria.custom_filters:
            if not custom_filter(paper):
                return False
        
        return True


class FilterPresets:
    """预设筛选条件 - 常用筛选模板"""
    
    @staticmethod
    def top_security_researchers() -> FilterCriteria:
        """顶级安全研究员"""
        return FilterCriteria(
            scholar_types=[ScholarType.TOP_VENUE_REGULAR, ScholarType.HIGH_CITATION],
            keywords=["security", "privacy", "cryptography", "malware"],
            min_citations=5000,
        )
    
    @staticmethod
    def rising_ml_stars() -> FilterCriteria:
        """新锐ML研究员"""
        return FilterCriteria(
            scholar_types=[ScholarType.RISING_STAR],
            keywords=["machine learning", "deep learning", "neural network", "LLM"],
            min_citations=1000,
            active_since=datetime.now() - timedelta(days=365),
        )
    
    @staticmethod
    def industry_researchers() -> FilterCriteria:
        """工业界研究员"""
        return FilterCriteria(
            scholar_types=[ScholarType.INDUSTRY_EXPERT],
            affiliation_types=[AffiliationType.TECH_COMPANY],
        )
    
    @staticmethod
    def prolific_authors() -> FilterCriteria:
        """高产作者"""
        return FilterCriteria(
            scholar_types=[ScholarType.PROLIFIC_AUTHOR],
            min_papers=50,
        )
    
    @staticmethod
    def top_venue_papers(year: Optional[int] = None) -> PaperFilterCriteria:
        """顶会论文"""
        return PaperFilterCriteria(
            top_venues_only=True,
            min_year=year or datetime.now().year - 2,
        )
    
    @staticmethod
    def highly_cited_papers(min_citations: int = 100) -> PaperFilterCriteria:
        """高引论文"""
        return PaperFilterCriteria(
            min_citations=min_citations,
        )
    
    @staticmethod
    def papers_with_code() -> PaperFilterCriteria:
        """带代码的论文"""
        return PaperFilterCriteria(
            has_code=True,
        )
    
    @staticmethod
    def llm_security_papers() -> PaperFilterCriteria:
        """LLM安全论文"""
        return PaperFilterCriteria(
            title_contains=[
                "LLM", "language model", "GPT", "ChatGPT",
                "jailbreak", "prompt injection", "adversarial"
            ],
            fields_of_study=["Computer Science", "Security"],
        )


class FilterService:
    """
    筛选服务 - 整合学者和论文筛选
    
    提供类似 JobLeap 的完整筛选体验
    """
    
    def __init__(self):
        self.scholar_filter = ScholarFilter()
        self.paper_filter = PaperFilter()
        self.presets = FilterPresets()
    
    def quick_filter(
        self,
        scholars: List[Scholar],
        preset_name: str,
        papers_map: Optional[Dict[str, List[PaperMeta]]] = None,
    ) -> List[Scholar]:
        """使用预设快速筛选学者"""
        preset_methods = {
            "top_security": self.presets.top_security_researchers,
            "rising_ml": self.presets.rising_ml_stars,
            "industry": self.presets.industry_researchers,
            "prolific": self.presets.prolific_authors,
        }
        
        if preset_name not in preset_methods:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        criteria = preset_methods[preset_name]()
        return self.scholar_filter.filter_scholars(scholars, criteria, papers_map)
    
    def advanced_filter(
        self,
        scholars: List[Scholar],
        scholar_types: Optional[List[str]] = None,
        affiliations: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        min_citations: Optional[int] = None,
        min_h_index: Optional[int] = None,
        papers_map: Optional[Dict[str, List[PaperMeta]]] = None,
    ) -> List[Scholar]:
        """高级筛选"""
        criteria = FilterCriteria(
            scholar_types=[ScholarType(t) for t in (scholar_types or [])],
            affiliation_names=affiliations or [],
            keywords=keywords or [],
            min_citations=min_citations,
            min_h_index=min_h_index,
        )
        return self.scholar_filter.filter_scholars(scholars, criteria, papers_map)
    
    def filter_papers_quick(
        self,
        papers: List[PaperMeta],
        preset_name: str,
    ) -> List[PaperMeta]:
        """使用预设快速筛选论文"""
        preset_methods = {
            "top_venue": self.presets.top_venue_papers,
            "highly_cited": self.presets.highly_cited_papers,
            "with_code": self.presets.papers_with_code,
            "llm_security": self.presets.llm_security_papers,
        }
        
        if preset_name not in preset_methods:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        criteria = preset_methods[preset_name]()
        return self.paper_filter.filter_papers(papers, criteria)
