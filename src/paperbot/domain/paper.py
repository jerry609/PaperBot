# src/paperbot/domain/paper.py
"""
论文相关领域模型。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class PaperMeta:
    """论文元数据模型。"""
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: int = 0
    github_url: Optional[str] = None
    has_code: bool = False
    url: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "github_url": self.github_url,
            "has_code": self.has_code,
            "url": self.url,
            "doi": self.doi,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperMeta":
        """从字典创建实例。"""
        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            abstract=data.get("abstract"),
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citation_count", 0),
            github_url=data.get("github_url"),
            has_code=data.get("has_code", False),
            url=data.get("url"),
            doi=data.get("doi"),
            keywords=data.get("keywords", []),
        )


@dataclass
class CodeMeta:
    """代码仓库元数据模型。"""
    repo_url: str
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    language: Optional[str] = None
    license: Optional[str] = None
    has_readme: bool = False
    has_docs: bool = False
    updated_at: Optional[str] = None
    last_commit_date: Optional[str] = None
    reproducibility_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "repo_url": self.repo_url,
            "stars": self.stars,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "language": self.language,
            "license": self.license,
            "has_readme": self.has_readme,
            "has_docs": self.has_docs,
            "updated_at": self.updated_at,
            "last_commit_date": self.last_commit_date,
            "reproducibility_score": self.reproducibility_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeMeta":
        """从字典创建实例。"""
        return cls(
            repo_url=data.get("repo_url", ""),
            stars=data.get("stars", 0),
            forks=data.get("forks", 0),
            open_issues=data.get("open_issues", 0),
            language=data.get("language"),
            license=data.get("license"),
            has_readme=data.get("has_readme", False),
            has_docs=data.get("has_docs", False),
            updated_at=data.get("updated_at"),
            last_commit_date=data.get("last_commit_date"),
            reproducibility_score=data.get("reproducibility_score"),
        )
