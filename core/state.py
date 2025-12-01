"""
状态管理模块

来源: BettaFish/QueryEngine/state/state.py
适配: PaperBot 学者追踪系统

提供可序列化的状态管理，支持:
- 数据类定义
- JSON 序列化/反序列化
- 文件持久化
- 进度追踪
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime
from pathlib import Path


class TrackingStage(Enum):
    """追踪阶段枚举"""
    INIT = "initialization"
    FETCHING_SCHOLARS = "fetching_scholars"
    FETCHING_PAPERS = "fetching_papers"
    ANALYZING_CODE = "analyzing_code"
    CALCULATING_INFLUENCE = "calculating_influence"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScholarState:
    """学者状态"""
    scholar_id: str = ""
    name: str = ""
    semantic_scholar_id: str = ""
    affiliations: List[str] = field(default_factory=list)
    h_index: int = 0
    citation_count: int = 0
    paper_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scholar_id": self.scholar_id,
            "name": self.name,
            "semantic_scholar_id": self.semantic_scholar_id,
            "affiliations": self.affiliations,
            "h_index": self.h_index,
            "citation_count": self.citation_count,
            "paper_count": self.paper_count,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScholarState":
        return cls(
            scholar_id=data.get("scholar_id", ""),
            name=data.get("name", ""),
            semantic_scholar_id=data.get("semantic_scholar_id", ""),
            affiliations=data.get("affiliations", []),
            h_index=data.get("h_index", 0),
            citation_count=data.get("citation_count", 0),
            paper_count=data.get("paper_count", 0),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )


@dataclass
class PaperState:
    """论文状态"""
    paper_id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    venue: str = ""
    year: int = 0
    citation_count: int = 0
    abstract: str = ""
    url: str = ""
    arxiv_id: str = ""
    github_url: str = ""
    is_new: bool = False  # 是否是新发现的论文
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "venue": self.venue,
            "year": self.year,
            "citation_count": self.citation_count,
            "abstract": self.abstract,
            "url": self.url,
            "arxiv_id": self.arxiv_id,
            "github_url": self.github_url,
            "is_new": self.is_new,
            "discovered_at": self.discovered_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperState":
        return cls(
            paper_id=data.get("paper_id", ""),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            venue=data.get("venue", ""),
            year=data.get("year", 0),
            citation_count=data.get("citation_count", 0),
            abstract=data.get("abstract", ""),
            url=data.get("url", ""),
            arxiv_id=data.get("arxiv_id", ""),
            github_url=data.get("github_url", ""),
            is_new=data.get("is_new", False),
            discovered_at=data.get("discovered_at", datetime.now().isoformat()),
        )


@dataclass
class InfluenceState:
    """影响力计算状态"""
    scholar_id: str = ""
    academic_score: float = 0.0
    engineering_score: float = 0.0
    total_score: float = 0.0
    tier1_papers: int = 0
    tier2_papers: int = 0
    github_stars: int = 0
    github_forks: int = 0
    calculated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scholar_id": self.scholar_id,
            "academic_score": self.academic_score,
            "engineering_score": self.engineering_score,
            "total_score": self.total_score,
            "tier1_papers": self.tier1_papers,
            "tier2_papers": self.tier2_papers,
            "github_stars": self.github_stars,
            "github_forks": self.github_forks,
            "calculated_at": self.calculated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InfluenceState":
        return cls(
            scholar_id=data.get("scholar_id", ""),
            academic_score=data.get("academic_score", 0.0),
            engineering_score=data.get("engineering_score", 0.0),
            total_score=data.get("total_score", 0.0),
            tier1_papers=data.get("tier1_papers", 0),
            tier2_papers=data.get("tier2_papers", 0),
            github_stars=data.get("github_stars", 0),
            github_forks=data.get("github_forks", 0),
            calculated_at=data.get("calculated_at", datetime.now().isoformat()),
        )


@dataclass
class TrackingState:
    """
    学者追踪的完整状态
    
    包含所有学者、论文、影响力数据和执行进度
    """
    # 任务信息
    task_id: str = ""
    task_name: str = ""
    
    # 执行状态
    stage: TrackingStage = TrackingStage.INIT
    progress: float = 0.0
    is_completed: bool = False
    
    # 数据状态
    scholars: List[ScholarState] = field(default_factory=list)
    papers: List[PaperState] = field(default_factory=list)
    new_papers: List[PaperState] = field(default_factory=list)  # 新发现的论文
    influence_results: List[InfluenceState] = field(default_factory=list)
    
    # 错误信息
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # ============ 进度管理 ============
    
    def update_stage(self, stage: TrackingStage, progress: Optional[float] = None):
        """更新阶段和进度"""
        self.stage = stage
        if progress is not None:
            self.progress = progress
        self.update_timestamp()
    
    def mark_completed(self):
        """标记完成"""
        self.is_completed = True
        self.stage = TrackingStage.COMPLETED
        self.progress = 1.0
        self.update_timestamp()
    
    def mark_failed(self, error: str):
        """标记失败"""
        self.stage = TrackingStage.FAILED
        self.add_error(error, self.stage.value)
        self.update_timestamp()
    
    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now().isoformat()
    
    # ============ 数据操作 ============
    
    def add_scholar(self, scholar: ScholarState):
        """添加学者"""
        self.scholars.append(scholar)
        self.update_timestamp()
    
    def add_paper(self, paper: PaperState):
        """添加论文"""
        self.papers.append(paper)
        if paper.is_new:
            self.new_papers.append(paper)
        self.update_timestamp()
    
    def add_influence(self, influence: InfluenceState):
        """添加影响力结果"""
        self.influence_results.append(influence)
        self.update_timestamp()
    
    def add_error(self, message: str, stage: str, context: Dict[str, Any] = None):
        """添加错误"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "message": message,
            "context": context or {},
        }
        self.errors.append(error_info)
    
    # ============ 统计信息 ============
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        return {
            "task_id": self.task_id,
            "stage": self.stage.value,
            "progress": self.progress,
            "is_completed": self.is_completed,
            "scholars_count": len(self.scholars),
            "papers_count": len(self.papers),
            "new_papers_count": len(self.new_papers),
            "errors_count": len(self.errors),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def get_new_papers_by_scholar(self, scholar_id: str) -> List[PaperState]:
        """获取指定学者的新论文"""
        # 需要根据论文的作者来匹配
        return [p for p in self.new_papers if scholar_id in str(p.authors)]
    
    # ============ 序列化 ============
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "stage": self.stage.value,
            "progress": self.progress,
            "is_completed": self.is_completed,
            "scholars": [s.to_dict() for s in self.scholars],
            "papers": [p.to_dict() for p in self.papers],
            "new_papers": [p.to_dict() for p in self.new_papers],
            "influence_results": [i.to_dict() for i in self.influence_results],
            "errors": self.errors,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackingState":
        """从字典创建"""
        return cls(
            task_id=data.get("task_id", ""),
            task_name=data.get("task_name", ""),
            stage=TrackingStage(data.get("stage", "initialization")),
            progress=data.get("progress", 0.0),
            is_completed=data.get("is_completed", False),
            scholars=[ScholarState.from_dict(s) for s in data.get("scholars", [])],
            papers=[PaperState.from_dict(p) for p in data.get("papers", [])],
            new_papers=[PaperState.from_dict(p) for p in data.get("new_papers", [])],
            influence_results=[InfluenceState.from_dict(i) for i in data.get("influence_results", [])],
            errors=data.get("errors", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrackingState":
        """从 JSON 字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    # ============ 文件持久化 ============
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "TrackingState":
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return cls.from_json(json_str)
