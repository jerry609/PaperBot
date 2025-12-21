# src/paperbot/domain/scholar.py
"""
学者相关领域模型。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Scholar:
    """学者模型。"""
    scholar_id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    research_fields: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # 研究关键词
    h_index: Optional[int] = None
    citation_count: int = 0
    paper_count: int = 0
    homepage: Optional[str] = None
    google_scholar_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    email: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None  # 最后检查时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "scholar_id": self.scholar_id,
            "name": self.name,
            "affiliations": self.affiliations,
            "research_fields": self.research_fields,
            "keywords": self.keywords,
            "h_index": self.h_index,
            "citation_count": self.citation_count,
            "paper_count": self.paper_count,
            "homepage": self.homepage,
            "google_scholar_id": self.google_scholar_id,
            "semantic_scholar_id": self.semantic_scholar_id,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scholar":
        """从字典创建实例。"""
        return cls(
            scholar_id=data.get("scholar_id", ""),
            name=data.get("name", ""),
            affiliations=data.get("affiliations", []),
            research_fields=data.get("research_fields", []),
            keywords=data.get("keywords", []),
            h_index=data.get("h_index"),
            citation_count=data.get("citation_count", 0),
            paper_count=data.get("paper_count", 0),
            homepage=data.get("homepage"),
            google_scholar_id=data.get("google_scholar_id"),
            semantic_scholar_id=data.get("semantic_scholar_id"),
            email=data.get("email"),
        )

    @classmethod
    def from_config(cls, data: Dict[str, Any]) -> "Scholar":
        """
        从订阅配置创建 Scholar 实例（用于 SubscriptionService）。

        订阅配置通常最少包含：
        - name
        - semantic_scholar_id
        """
        semantic_id = data.get("semantic_scholar_id") or data.get("semanticScholarId") or data.get("author_id")
        scholar_id = data.get("scholar_id") or data.get("id") or semantic_id or data.get("name") or ""

        def _to_int(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        affiliations = data.get("affiliations") or []
        research_fields = data.get("research_fields") or data.get("researchFields") or []
        keywords = data.get("keywords") or []

        return cls(
            scholar_id=str(scholar_id),
            name=str(data.get("name") or ""),
            affiliations=list(affiliations) if isinstance(affiliations, (list, tuple)) else [],
            research_fields=list(research_fields) if isinstance(research_fields, (list, tuple)) else [],
            keywords=list(keywords) if isinstance(keywords, (list, tuple)) else [],
            h_index=_to_int(data.get("h_index") or data.get("hIndex")),
            citation_count=int(data.get("citation_count") or data.get("citationCount") or 0),
            paper_count=int(data.get("paper_count") or data.get("paperCount") or 0),
            homepage=data.get("homepage"),
            google_scholar_id=data.get("google_scholar_id") or data.get("googleScholarId"),
            semantic_scholar_id=str(semantic_id) if semantic_id is not None else None,
            email=data.get("email"),
        )
