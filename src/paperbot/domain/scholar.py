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
    h_index: Optional[int] = None
    citation_count: int = 0
    paper_count: int = 0
    homepage: Optional[str] = None
    google_scholar_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    email: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "scholar_id": self.scholar_id,
            "name": self.name,
            "affiliations": self.affiliations,
            "research_fields": self.research_fields,
            "h_index": self.h_index,
            "citation_count": self.citation_count,
            "paper_count": self.paper_count,
            "homepage": self.homepage,
            "google_scholar_id": self.google_scholar_id,
            "semantic_scholar_id": self.semantic_scholar_id,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scholar":
        """从字典创建实例。"""
        return cls(
            scholar_id=data.get("scholar_id", ""),
            name=data.get("name", ""),
            affiliations=data.get("affiliations", []),
            research_fields=data.get("research_fields", []),
            h_index=data.get("h_index"),
            citation_count=data.get("citation_count", 0),
            paper_count=data.get("paper_count", 0),
            homepage=data.get("homepage"),
            google_scholar_id=data.get("google_scholar_id"),
            semantic_scholar_id=data.get("semantic_scholar_id"),
            email=data.get("email"),
        )
