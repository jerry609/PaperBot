"""
论文数据模型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class PaperMeta:
    """论文元数据模型"""
    
    # 核心标识
    paper_id: str  # Semantic Scholar Paper ID
    title: str
    
    # 作者信息
    authors: List[str] = field(default_factory=list)
    
    # 发表信息
    year: Optional[int] = None
    venue: Optional[str] = None
    
    # 引用与指标
    citation_count: int = 0
    influential_citation_count: int = 0
    
    # 内容
    abstract: Optional[str] = None
    tldr: Optional[str] = None  # Semantic Scholar 提供的 AI 摘要
    
    # 链接
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    
    # 代码相关
    github_url: Optional[str] = None
    has_code: bool = False
    
    # 分类与标签
    fields_of_study: List[str] = field(default_factory=list)
    
    # 元数据
    publication_date: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据校验与规范化"""
        if not self.paper_id:
            raise ValueError("Paper ID cannot be empty")
        if not self.title:
            raise ValueError("Paper title cannot be empty")
        
        # 确保 citation_count 非负
        self.citation_count = max(0, self.citation_count or 0)
        self.influential_citation_count = max(0, self.influential_citation_count or 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "abstract": self.abstract,
            "tldr": self.tldr,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "github_url": self.github_url,
            "has_code": self.has_code,
            "fields_of_study": self.fields_of_study,
            "publication_date": self.publication_date,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperMeta":
        """从字典创建"""
        created_at = datetime.now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        return cls(
            paper_id=data["paper_id"],
            title=data["title"],
            authors=data.get("authors", []),
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citation_count", 0),
            influential_citation_count=data.get("influential_citation_count", 0),
            abstract=data.get("abstract"),
            tldr=data.get("tldr"),
            url=data.get("url"),
            pdf_url=data.get("pdf_url"),
            doi=data.get("doi"),
            arxiv_id=data.get("arxiv_id"),
            github_url=data.get("github_url"),
            has_code=data.get("has_code", False),
            fields_of_study=data.get("fields_of_study", []),
            publication_date=data.get("publication_date"),
            created_at=created_at,
        )
    
    @classmethod
    def from_semantic_scholar(cls, data: Dict[str, Any]) -> "PaperMeta":
        """从 Semantic Scholar API 响应创建"""
        # 提取作者名称
        authors = []
        for author in data.get("authors", []):
            if isinstance(author, dict) and author.get("name"):
                authors.append(author["name"])
            elif isinstance(author, str):
                authors.append(author)
        
        # 提取 TLDR
        tldr = None
        if data.get("tldr") and isinstance(data["tldr"], dict):
            tldr = data["tldr"].get("text")
        
        # 提取外部 ID
        external_ids = data.get("externalIds", {}) or {}
        arxiv_id = external_ids.get("ArXiv")
        doi = external_ids.get("DOI")
        
        # 提取 URL
        url = data.get("url")
        
        # 检查是否有开源代码
        open_access_pdf = data.get("openAccessPdf", {}) or {}
        pdf_url = open_access_pdf.get("url")
        
        return cls(
            paper_id=data["paperId"],
            title=data["title"],
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citationCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            abstract=data.get("abstract"),
            tldr=tldr,
            url=url,
            pdf_url=pdf_url,
            doi=doi,
            arxiv_id=arxiv_id,
            fields_of_study=data.get("fieldsOfStudy", []) or [],
            publication_date=data.get("publicationDate"),
        )
    
    def __str__(self) -> str:
        return f"Paper({self.title[:50]}..., year={self.year}, citations={self.citation_count})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class CodeMeta:
    """代码仓库元数据模型"""
    
    # 仓库信息
    repo_url: str
    repo_name: Optional[str] = None
    repo_owner: Optional[str] = None
    
    # GitHub 指标
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    open_issues: int = 0
    
    # 代码信息
    language: Optional[str] = None
    languages: Dict[str, int] = field(default_factory=dict)  # {language: bytes}
    
    # 活跃度
    last_commit_date: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # 其他
    description: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    license: Optional[str] = None
    has_readme: bool = False
    has_docs: bool = False
    
    # 可复现性评估 (由 CodeAnalysisAgent 填充)
    reproducibility_score: Optional[float] = None
    quality_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "repo_url": self.repo_url,
            "repo_name": self.repo_name,
            "repo_owner": self.repo_owner,
            "stars": self.stars,
            "forks": self.forks,
            "watchers": self.watchers,
            "open_issues": self.open_issues,
            "language": self.language,
            "languages": self.languages,
            "last_commit_date": self.last_commit_date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "topics": self.topics,
            "license": self.license,
            "has_readme": self.has_readme,
            "has_docs": self.has_docs,
            "reproducibility_score": self.reproducibility_score,
            "quality_notes": self.quality_notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeMeta":
        """从字典创建"""
        return cls(
            repo_url=data["repo_url"],
            repo_name=data.get("repo_name"),
            repo_owner=data.get("repo_owner"),
            stars=data.get("stars", 0),
            forks=data.get("forks", 0),
            watchers=data.get("watchers", 0),
            open_issues=data.get("open_issues", 0),
            language=data.get("language"),
            languages=data.get("languages", {}),
            last_commit_date=data.get("last_commit_date"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            description=data.get("description"),
            topics=data.get("topics", []),
            license=data.get("license"),
            has_readme=data.get("has_readme", False),
            has_docs=data.get("has_docs", False),
            reproducibility_score=data.get("reproducibility_score"),
            quality_notes=data.get("quality_notes"),
        )
    
    @classmethod
    def from_github_api(cls, data: Dict[str, Any]) -> "CodeMeta":
        """从 GitHub API 响应创建"""
        return cls(
            repo_url=data.get("html_url", ""),
            repo_name=data.get("name"),
            repo_owner=data.get("owner", {}).get("login"),
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            watchers=data.get("watchers_count", 0),
            open_issues=data.get("open_issues_count", 0),
            language=data.get("language"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            description=data.get("description"),
            topics=data.get("topics", []),
            license=data.get("license", {}).get("spdx_id") if data.get("license") else None,
        )
    
    def __str__(self) -> str:
        return f"CodeMeta({self.repo_name}, stars={self.stars})"
    
    def __repr__(self) -> str:
        return self.__str__()

