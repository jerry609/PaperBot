"""
学者数据模型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Scholar:
    """学者数据模型"""
    
    # 基本信息
    name: str
    semantic_scholar_id: str
    
    # 可选信息
    keywords: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    
    # 学术指标 (可选，从 API 获取后填充)
    h_index: Optional[int] = None
    citation_count: Optional[int] = None
    paper_count: Optional[int] = None
    
    # 追踪元数据
    last_checked: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据校验"""
        if not self.name:
            raise ValueError("Scholar name cannot be empty")
        if not self.semantic_scholar_id:
            raise ValueError("Semantic Scholar ID cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "semantic_scholar_id": self.semantic_scholar_id,
            "keywords": self.keywords,
            "affiliations": self.affiliations,
            "homepage": self.homepage,
            "h_index": self.h_index,
            "citation_count": self.citation_count,
            "paper_count": self.paper_count,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scholar":
        """从字典创建"""
        # 处理日期字段
        last_checked = None
        if data.get("last_checked"):
            last_checked = datetime.fromisoformat(data["last_checked"])
        
        created_at = datetime.now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        return cls(
            name=data["name"],
            semantic_scholar_id=data["semantic_scholar_id"],
            keywords=data.get("keywords", []),
            affiliations=data.get("affiliations", []),
            homepage=data.get("homepage"),
            h_index=data.get("h_index"),
            citation_count=data.get("citation_count"),
            paper_count=data.get("paper_count"),
            last_checked=last_checked,
            created_at=created_at,
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Scholar":
        """从配置文件创建（简化版本）"""
        return cls(
            name=config["name"],
            semantic_scholar_id=config["semantic_scholar_id"],
            keywords=config.get("keywords", []),
        )
    
    def __str__(self) -> str:
        return f"Scholar({self.name}, id={self.semantic_scholar_id})"
    
    def __repr__(self) -> str:
        return self.__str__()

