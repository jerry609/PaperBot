# src/paperbot/domain/paper.py
"""
论文相关领域模型。
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from paperbot.domain.identity import PaperIdentity


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
    fields_of_study: List[str] = field(default_factory=list)  # 研究领域
    publication_date: Optional[str] = None  # 发布日期 (ISO 格式)
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
            "fields_of_study": self.fields_of_study,
            "publication_date": self.publication_date,
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
            fields_of_study=data.get("fields_of_study", []),
            publication_date=data.get("publication_date"),
        )

    @classmethod
    def from_semantic_scholar(cls, data: Dict[str, Any]) -> "PaperMeta":
        """从 Semantic Scholar API 响应创建实例。"""
        # 提取作者名称
        authors = []
        if "authors" in data:
            for author in data["authors"]:
                if isinstance(author, dict):
                    authors.append(author.get("name", ""))
                elif isinstance(author, str):
                    authors.append(author)

        # 提取年份
        year = data.get("year")
        if not year and data.get("publicationDate"):
            try:
                year = int(data["publicationDate"][:4])
            except (ValueError, TypeError):
                pass

        # 提取 GitHub URL
        github_url = None
        has_code = False
        if data.get("openAccessPdf"):
            pdf_url = data["openAccessPdf"].get("url", "")
            if "github" in pdf_url.lower():
                github_url = pdf_url
                has_code = True

        # 检查 externalIds 中的 GitHub
        if data.get("externalIds"):
            for _, value in data["externalIds"].items():
                if "github" in str(value).lower():
                    github_url = value
                    has_code = True
                    break

        fields_of_study = data.get("fieldsOfStudy", []) or []

        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            authors=authors,
            abstract=data.get("abstract"),
            year=year,
            venue=data.get("venue") or data.get("publicationVenue", {}).get("name"),
            citation_count=data.get("citationCount", 0),
            github_url=github_url,
            has_code=has_code,
            url=data.get("url"),
            doi=data.get("externalIds", {}).get("DOI"),
            keywords=fields_of_study,
            fields_of_study=fields_of_study,
            publication_date=data.get("publicationDate"),
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


def _compute_title_hash(title: str) -> str:
    normalized = (title or "").lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class PaperCandidate:
    """Normalized paper from any external source (Anti-Corruption Layer output)."""

    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: int = 0
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    fields_of_study: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    identities: List[PaperIdentity] = field(default_factory=list)
    title_hash: str = ""
    canonical_id: Optional[int] = None
    retrieval_score: float = 0.0
    retrieval_sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title_hash and self.title:
            self.title_hash = _compute_title_hash(self.title)

    def get_identity(self, source: str) -> Optional[str]:
        for ident in self.identities:
            if ident.source == source:
                return ident.external_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "keywords": self.keywords,
            "fields_of_study": self.fields_of_study,
            "publication_date": self.publication_date,
            "identities": [
                {"source": i.source, "external_id": i.external_id} for i in self.identities
            ],
            "title_hash": self.title_hash,
            "canonical_id": self.canonical_id,
            "retrieval_score": float(self.retrieval_score or 0.0),
            "retrieval_sources": [str(x) for x in (self.retrieval_sources or []) if str(x)],
        }
