"""
数据源抽象层
用于在学者追踪/分析流程中统一访问数据，无论是 API 还是本地数据集。
MVP：提供 BaseDataSource 抽象、LocalFileDataSource（CSV/JSON）、占位的 DBDataSource。
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# 支持多种导入路径
try:
    from paperbot.domain.paper import PaperMeta
    from paperbot.domain.scholar import Scholar
except ImportError:
    try:
        from paperbot.domain.paper import PaperMeta
        from paperbot.domain.scholar import Scholar
    except ImportError:
        PaperMeta = None
        Scholar = None

logger = logging.getLogger(__name__)


class BaseDataSource:
    """数据源抽象基类"""

    async def fetch_papers_by_author(self, scholar, limit: int = 20) -> List:
        raise NotImplementedError


class LocalFileDataSource(BaseDataSource):
    """
    本地文件数据源。
    支持 CSV（字段：paper_id,title,year,venue,abstract,author_id,url）和 JSON(list[dict])。
    若存在 author_id 字段则按 scholar.semantic_scholar_id 过滤，否则返回全量前 N 条。
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def _load_rows(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        if self.dataset_path.suffix.lower() == ".csv":
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        if self.dataset_path.suffix.lower() in {".json", ".jsonl"}:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        raise ValueError(f"Unsupported dataset format: {self.dataset_path}")

    async def fetch_papers_by_author(self, scholar, limit: int = 20) -> List:
        rows = self._load_rows()
        # 按 author_id 过滤；若不存在则返回全量
        scholar_id = getattr(scholar, 'semantic_scholar_id', str(scholar)) if scholar else ""
        filtered = [
            r for r in rows
            if str(r.get("author_id", "")).strip() == str(scholar_id).strip()
        ]
        if not filtered:
            filtered = rows

        papers = []
        for r in filtered[:limit]:
            if PaperMeta is not None:
                paper = PaperMeta(
                    paper_id=str(r.get("paper_id") or r.get("id") or "").strip() or None,
                    title=r.get("title") or "",
                    year=int(r.get("year")) if r.get("year") else None,
                    venue=r.get("venue"),
                    abstract=r.get("abstract"),
                    url=r.get("url"),
                    authors=r.get("authors"),
                )
                if paper.paper_id and paper.title:
                    papers.append(paper)
            else:
                # 返回原始字典
                papers.append(r)
        return papers


class DBDataSource(BaseDataSource):
    """占位 DB 数据源，后续可用 SQLAlchemy 实现"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def fetch_papers_by_author(self, scholar, limit: int = 20) -> List:
        logger.warning("DBDataSource not implemented, returning empty list")
        return []


def build_data_source(cfg: Dict[str, Any]) -> Optional[BaseDataSource]:
    """
    根据配置构建数据源实例。
    cfg: {"type": "api|local|hybrid", "dataset_path": "...", "dataset_name": "..."}
    """
    ds_type = (cfg or {}).get("type", "api").lower()
    if ds_type == "local":
        ds_path = cfg.get("dataset_path")
        if not ds_path and cfg.get("dataset_name"):
            ds_path = Path("datasets/processed") / f"{cfg['dataset_name']}.csv"
        if not ds_path:
            logger.warning("LocalFileDataSource requires dataset_path or dataset_name")
            return None
        return LocalFileDataSource(Path(ds_path))
    if ds_type == "db":
        return DBDataSource(cfg)
    # api / hybrid 默认返回 None，由上层调用远程 API
    return None

