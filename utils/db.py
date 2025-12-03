"""
数据库与缓存工具

参考: BettaFish/InsightEngine/utils/db.py
适配: PaperBot 学者追踪 - 本地缓存与状态持久化

提供:
- SQLite 本地缓存
- JSON 文件存储
- 异步数据库访问 (可选)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from loguru import logger


# ===== SQLite 本地缓存 =====

class LocalCache:
    """
    基于 SQLite 的本地缓存
    
    用于存储学者追踪的中间状态和历史数据
    """
    
    def __init__(self, db_path: str = "data/cache.db"):
        """
        初始化本地缓存
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 学者缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scholars (
                    scholar_id TEXT PRIMARY KEY,
                    name TEXT,
                    data TEXT,
                    updated_at TEXT
                )
            """)
            
            # 论文缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    scholar_id TEXT,
                    data TEXT,
                    updated_at TEXT
                )
            """)
            
            # 追踪状态表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracking_state (
                    scholar_id TEXT PRIMARY KEY,
                    last_checked TEXT,
                    known_paper_ids TEXT,
                    state_data TEXT
                )
            """)
            
            # 键值缓存表 (通用)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kv_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at TEXT
                )
            """)
            
            conn.commit()
    
    # ===== 学者缓存 =====
    
    def get_scholar(self, scholar_id: str) -> Optional[Dict[str, Any]]:
        """获取学者缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM scholars WHERE scholar_id = ?",
                (scholar_id,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def set_scholar(self, scholar_id: str, name: str, data: Dict[str, Any]) -> None:
        """设置学者缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO scholars (scholar_id, name, data, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (scholar_id, name, json.dumps(data, ensure_ascii=False), datetime.now().isoformat())
            )
            conn.commit()
    
    def list_scholars(self) -> List[Dict[str, Any]]:
        """列出所有缓存的学者"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT scholar_id, name, updated_at FROM scholars")
            rows = cursor.fetchall()
            return [
                {"scholar_id": row[0], "name": row[1], "updated_at": row[2]}
                for row in rows
            ]
    
    # ===== 论文缓存 =====
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """获取论文缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM papers WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def set_paper(self, paper_id: str, title: str, data: Dict[str, Any], scholar_id: Optional[str] = None) -> None:
        """设置论文缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO papers (paper_id, title, scholar_id, data, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (paper_id, title, scholar_id, json.dumps(data, ensure_ascii=False), datetime.now().isoformat())
            )
            conn.commit()
    
    def get_scholar_papers(self, scholar_id: str) -> List[Dict[str, Any]]:
        """获取学者的所有论文"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM papers WHERE scholar_id = ?",
                (scholar_id,)
            )
            rows = cursor.fetchall()
            return [json.loads(row[0]) for row in rows]
    
    # ===== 追踪状态 =====
    
    def get_tracking_state(self, scholar_id: str) -> Optional[Dict[str, Any]]:
        """获取追踪状态"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_checked, known_paper_ids, state_data FROM tracking_state WHERE scholar_id = ?",
                (scholar_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "last_checked": row[0],
                    "known_paper_ids": json.loads(row[1]) if row[1] else [],
                    "state_data": json.loads(row[2]) if row[2] else {},
                }
        return None
    
    def set_tracking_state(
        self,
        scholar_id: str,
        last_checked: Optional[str] = None,
        known_paper_ids: Optional[List[str]] = None,
        state_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """设置追踪状态"""
        existing = self.get_tracking_state(scholar_id) or {
            "last_checked": None,
            "known_paper_ids": [],
            "state_data": {},
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tracking_state (scholar_id, last_checked, known_paper_ids, state_data)
                VALUES (?, ?, ?, ?)
                """,
                (
                    scholar_id,
                    last_checked or existing["last_checked"] or datetime.now().isoformat(),
                    json.dumps(known_paper_ids if known_paper_ids is not None else existing["known_paper_ids"]),
                    json.dumps(state_data if state_data is not None else existing["state_data"], ensure_ascii=False),
                )
            )
            conn.commit()
    
    def add_known_papers(self, scholar_id: str, paper_ids: List[str]) -> None:
        """添加已知论文 ID"""
        state = self.get_tracking_state(scholar_id)
        known = set(state["known_paper_ids"]) if state else set()
        known.update(paper_ids)
        self.set_tracking_state(scholar_id, known_paper_ids=list(known))
    
    def get_known_paper_ids(self, scholar_id: str) -> Set[str]:
        """获取已知论文 ID 集合"""
        state = self.get_tracking_state(scholar_id)
        return set(state["known_paper_ids"]) if state else set()
    
    # ===== 通用键值缓存 =====
    
    def get(self, key: str) -> Optional[str]:
        """获取缓存值"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value, expires_at FROM kv_cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                expires_at = row[1]
                if expires_at:
                    if datetime.fromisoformat(expires_at) < datetime.now():
                        # 已过期，删除
                        cursor.execute("DELETE FROM kv_cache WHERE key = ?", (key,))
                        conn.commit()
                        return None
                return row[0]
        return None
    
    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        """设置缓存值"""
        expires_at = None
        if ttl_seconds:
            from datetime import timedelta
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO kv_cache (key, value, expires_at)
                VALUES (?, ?, ?)
                """,
                (key, value, expires_at)
            )
            conn.commit()
    
    def delete(self, key: str) -> None:
        """删除缓存值"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM kv_cache WHERE key = ?", (key,))
            conn.commit()
    
    def clear_expired(self) -> int:
        """清理过期缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM kv_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                (datetime.now().isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted


# ===== JSON 文件存储 =====

class JSONFileStorage:
    """
    基于 JSON 文件的简单存储
    
    适用于小规模数据的持久化
    """
    
    def __init__(self, base_dir: str = "data/json"):
        """
        初始化 JSON 存储
        
        Args:
            base_dir: 存储目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, collection: str, key: str) -> Path:
        """获取文件路径"""
        collection_dir = self.base_dir / collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        # 清理 key 中的非法字符
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return collection_dir / f"{safe_key}.json"
    
    def get(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """获取数据"""
        path = self._get_path(collection, key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"读取 JSON 失败: {path}, {e}")
        return None
    
    def set(self, collection: str, key: str, data: Dict[str, Any]) -> None:
        """保存数据"""
        path = self._get_path(collection, key)
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"写入 JSON 失败: {path}, {e}")
    
    def delete(self, collection: str, key: str) -> bool:
        """删除数据"""
        path = self._get_path(collection, key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_keys(self, collection: str) -> List[str]:
        """列出集合中的所有键"""
        collection_dir = self.base_dir / collection
        if not collection_dir.exists():
            return []
        return [f.stem for f in collection_dir.glob("*.json")]
    
    def list_all(self, collection: str) -> List[Dict[str, Any]]:
        """列出集合中的所有数据"""
        keys = self.list_keys(collection)
        return [self.get(collection, key) for key in keys if self.get(collection, key)]


# ===== 状态序列化助手 =====

@dataclass
class SerializableState:
    """可序列化的状态基类"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableState":
        """从字典创建"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SerializableState":
        """从 JSON 字符串创建"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ScholarTrackingCheckpoint(SerializableState):
    """
    学者追踪检查点
    
    用于保存和恢复追踪进度
    """
    scholar_id: str
    scholar_name: str = ""
    last_checked: str = ""
    papers_processed: int = 0
    papers_total: int = 0
    known_paper_ids: List[str] = field(default_factory=list)
    current_stage: str = "idle"  # idle, fetching, analyzing, complete
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self) -> None:
        """标记开始"""
        self.current_stage = "fetching"
        self.last_checked = datetime.now().isoformat()
    
    def mark_analyzing(self, total: int) -> None:
        """标记正在分析"""
        self.current_stage = "analyzing"
        self.papers_total = total
    
    def mark_progress(self, processed: int) -> None:
        """标记进度"""
        self.papers_processed = processed
    
    def mark_complete(self) -> None:
        """标记完成"""
        self.current_stage = "complete"
    
    def mark_error(self, message: str) -> None:
        """标记错误"""
        self.current_stage = "error"
        self.error_message = message
    
    def add_known_papers(self, paper_ids: List[str]) -> None:
        """添加已知论文"""
        existing = set(self.known_paper_ids)
        existing.update(paper_ids)
        self.known_paper_ids = list(existing)


# ===== 便捷函数 =====

_default_cache: Optional[LocalCache] = None
_default_storage: Optional[JSONFileStorage] = None


def get_cache(db_path: str = "data/cache.db") -> LocalCache:
    """获取默认缓存实例"""
    global _default_cache
    if _default_cache is None:
        _default_cache = LocalCache(db_path)
    return _default_cache


def get_storage(base_dir: str = "data/json") -> JSONFileStorage:
    """获取默认存储实例"""
    global _default_storage
    if _default_storage is None:
        _default_storage = JSONFileStorage(base_dir)
    return _default_storage


__all__ = [
    # 缓存类
    "LocalCache",
    "JSONFileStorage",
    # 状态类
    "SerializableState",
    "ScholarTrackingCheckpoint",
    # 便捷函数
    "get_cache",
    "get_storage",
]
