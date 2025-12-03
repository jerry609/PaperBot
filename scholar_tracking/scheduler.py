# scholar_tracking/scheduler.py
"""
定时收录调度器 - 借鉴 JobLeap 的信息收录机制
实现定期收录新论文，标注收录时间，支持用户通知
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from pathlib import Path

from .models import Scholar, PaperMeta
from .feed import FeedGenerator, FeedEventFactory, ScholarFeedService


class CollectionInterval(Enum):
    """收录间隔"""
    HOURLY = 3600           # 每小时
    DAILY = 86400           # 每天
    WEEKLY = 604800         # 每周
    CUSTOM = 0              # 自定义


class NotificationType(Enum):
    """通知类型"""
    NEW_PAPER = "new_paper"
    CITATION_MILESTONE = "citation_milestone"
    CONFERENCE_DEADLINE = "conference_deadline"
    WEEKLY_DIGEST = "weekly_digest"


@dataclass
class CollectionRecord:
    """收录记录 - 类似 JobLeap 的 "收录时间" 概念"""
    
    paper_id: str
    scholar_id: str
    collected_at: datetime
    source: str = "semantic_scholar"
    
    # 收录时的快照
    citation_count_at_collection: int = 0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "scholar_id": self.scholar_id,
            "collected_at": self.collected_at.isoformat(),
            "source": self.source,
            "citation_count_at_collection": self.citation_count_at_collection,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionRecord":
        return cls(
            paper_id=data["paper_id"],
            scholar_id=data["scholar_id"],
            collected_at=datetime.fromisoformat(data["collected_at"]),
            source=data.get("source", "semantic_scholar"),
            citation_count_at_collection=data.get("citation_count_at_collection", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NotificationConfig:
    """通知配置"""
    
    enabled: bool = True
    
    # 通知类型开关
    notify_new_papers: bool = True
    notify_citation_milestones: bool = True
    notify_conference_deadlines: bool = True
    notify_weekly_digest: bool = True
    
    # 引用里程碑阈值
    citation_milestones: List[int] = field(
        default_factory=lambda: [10, 50, 100, 500, 1000]
    )
    
    # 通知渠道
    channels: List[str] = field(default_factory=lambda: ["console"])
    
    # 静默时间段（不发送通知）
    quiet_hours: Optional[tuple] = None  # (start_hour, end_hour)


@dataclass
class SchedulerConfig:
    """调度器配置"""
    
    # 收录间隔
    collection_interval: CollectionInterval = CollectionInterval.DAILY
    custom_interval_seconds: int = 86400
    
    # 收录设置
    max_papers_per_scholar: int = 100  # 每个学者最多收录论文数
    collect_recent_days: int = 30       # 收录最近N天的论文
    
    # 存储路径
    data_dir: str = "./data/scheduler"
    
    # 通知配置
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    
    # 重试配置
    max_retries: int = 3
    retry_delay_seconds: int = 60


class NotificationHandler:
    """通知处理器"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self._handlers: Dict[str, Callable] = {
            "console": self._console_notify,
        }
    
    def register_handler(self, channel: str, handler: Callable):
        """注册通知处理器"""
        self._handlers[channel] = handler
    
    async def notify(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """发送通知"""
        if not self.config.enabled:
            return
        
        # 检查静默时间
        if self._is_quiet_hours():
            return
        
        # 检查通知类型是否启用
        if not self._is_notification_enabled(notification_type):
            return
        
        # 向所有配置的渠道发送通知
        for channel in self.config.channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    await handler(notification_type, title, message, data)
                except Exception as e:
                    print(f"Notification error on {channel}: {e}")
    
    def _is_quiet_hours(self) -> bool:
        """检查是否在静默时间段"""
        if not self.config.quiet_hours:
            return False
        
        start, end = self.config.quiet_hours
        current_hour = datetime.now().hour
        
        if start <= end:
            return start <= current_hour < end
        else:
            return current_hour >= start or current_hour < end
    
    def _is_notification_enabled(self, notification_type: NotificationType) -> bool:
        """检查通知类型是否启用"""
        type_map = {
            NotificationType.NEW_PAPER: self.config.notify_new_papers,
            NotificationType.CITATION_MILESTONE: self.config.notify_citation_milestones,
            NotificationType.CONFERENCE_DEADLINE: self.config.notify_conference_deadlines,
            NotificationType.WEEKLY_DIGEST: self.config.notify_weekly_digest,
        }
        return type_map.get(notification_type, True)
    
    async def _console_notify(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """控制台通知"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"📢 [{timestamp}] {notification_type.value.upper()}")
        print(f"   {title}")
        print(f"   {message}")
        if data:
            print(f"   详情: {data}")
        print(f"{'='*60}\n")


class CollectionStorage:
    """收录数据存储"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._records_file = self.data_dir / "collection_records.json"
        self._state_file = self.data_dir / "scheduler_state.json"
        
        self._records: Dict[str, CollectionRecord] = {}
        self._load_records()
    
    def _load_records(self):
        """加载收录记录"""
        if self._records_file.exists():
            try:
                with open(self._records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, record_data in data.items():
                        self._records[key] = CollectionRecord.from_dict(record_data)
            except Exception as e:
                print(f"Error loading collection records: {e}")
    
    def _save_records(self):
        """保存收录记录"""
        try:
            data = {k: v.to_dict() for k, v in self._records.items()}
            with open(self._records_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving collection records: {e}")
    
    def _make_key(self, paper_id: str, scholar_id: str) -> str:
        """生成记录键"""
        return f"{scholar_id}:{paper_id}"
    
    def add_record(self, record: CollectionRecord):
        """添加收录记录"""
        key = self._make_key(record.paper_id, record.scholar_id)
        self._records[key] = record
        self._save_records()
    
    def get_record(self, paper_id: str, scholar_id: str) -> Optional[CollectionRecord]:
        """获取收录记录"""
        key = self._make_key(paper_id, scholar_id)
        return self._records.get(key)
    
    def has_record(self, paper_id: str, scholar_id: str) -> bool:
        """检查是否已收录"""
        key = self._make_key(paper_id, scholar_id)
        return key in self._records
    
    def get_records_by_scholar(self, scholar_id: str) -> List[CollectionRecord]:
        """获取学者的所有收录记录"""
        return [
            r for r in self._records.values()
            if r.scholar_id == scholar_id
        ]
    
    def get_recent_records(self, days: int = 7) -> List[CollectionRecord]:
        """获取最近的收录记录"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            r for r in self._records.values()
            if r.collected_at >= cutoff
        ]
    
    def save_state(self, state: Dict[str, Any]):
        """保存调度器状态"""
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving scheduler state: {e}")
    
    def load_state(self) -> Dict[str, Any]:
        """加载调度器状态"""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading scheduler state: {e}")
        return {}


class PaperCollector:
    """
    论文收录器
    
    借鉴 JobLeap 的信息收录机制:
    - 定期从数据源收录新信息
    - 标注收录时间
    - 去重处理
    - 变更检测
    """
    
    def __init__(
        self,
        config: SchedulerConfig,
        paper_fetcher: Optional[Callable[[str], Awaitable[List[PaperMeta]]]] = None,
    ):
        self.config = config
        self.storage = CollectionStorage(config.data_dir)
        self.notification = NotificationHandler(config.notification)
        self.feed_service = ScholarFeedService()
        
        # 论文获取函数（需要外部注入）
        self._paper_fetcher = paper_fetcher
        
        # 追踪的学者
        self._tracked_scholars: Dict[str, Scholar] = {}
        
        # 上次引用计数缓存（用于检测变化）
        self._citation_cache: Dict[str, int] = {}
    
    def set_paper_fetcher(self, fetcher: Callable[[str], Awaitable[List[PaperMeta]]]):
        """设置论文获取函数"""
        self._paper_fetcher = fetcher
    
    def track_scholar(self, scholar: Scholar):
        """添加追踪学者"""
        self._tracked_scholars[scholar.semantic_scholar_id] = scholar
        self.feed_service.track_scholar(scholar)
    
    def untrack_scholar(self, scholar_id: str):
        """取消追踪学者"""
        self._tracked_scholars.pop(scholar_id, None)
        self.feed_service.untrack_scholar(scholar_id)
    
    async def collect_for_scholar(self, scholar: Scholar) -> List[PaperMeta]:
        """
        收录学者的论文
        
        Returns:
            新收录的论文列表
        """
        if not self._paper_fetcher:
            print(f"Warning: No paper fetcher configured")
            return []
        
        try:
            # 获取论文
            papers = await self._paper_fetcher(scholar.semantic_scholar_id)
            
            new_papers = []
            for paper in papers[:self.config.max_papers_per_scholar]:
                # 检查是否已收录
                if self.storage.has_record(paper.paper_id, scholar.semantic_scholar_id):
                    # 检查引用变化
                    await self._check_citation_change(scholar, paper)
                    continue
                
                # 创建收录记录
                record = CollectionRecord(
                    paper_id=paper.paper_id,
                    scholar_id=scholar.semantic_scholar_id,
                    collected_at=datetime.now(),
                    citation_count_at_collection=paper.citation_count,
                    metadata={
                        "title": paper.title,
                        "venue": paper.venue,
                        "year": paper.year,
                    }
                )
                
                self.storage.add_record(record)
                new_papers.append(paper)
                
                # 更新引用缓存
                self._citation_cache[paper.paper_id] = paper.citation_count
                
                # 添加到信息流
                self.feed_service.process_new_papers(scholar, [paper])
                
                # 发送通知
                await self.notification.notify(
                    NotificationType.NEW_PAPER,
                    f"🆕 {scholar.name} 的新论文",
                    f"{paper.title} ({paper.venue or 'Unknown'}, {paper.year or 'N/A'})",
                    {"paper_id": paper.paper_id, "citations": paper.citation_count}
                )
            
            return new_papers
            
        except Exception as e:
            print(f"Error collecting papers for {scholar.name}: {e}")
            return []
    
    async def _check_citation_change(self, scholar: Scholar, paper: PaperMeta):
        """检查引用变化"""
        old_count = self._citation_cache.get(paper.paper_id)
        
        if old_count is not None and paper.citation_count != old_count:
            # 更新缓存
            self._citation_cache[paper.paper_id] = paper.citation_count
            
            # 添加到信息流
            self.feed_service.process_citation_changes(scholar, paper, old_count)
            
            # 检查是否达到里程碑
            for milestone in self.config.notification.citation_milestones:
                if old_count < milestone <= paper.citation_count:
                    await self.notification.notify(
                        NotificationType.CITATION_MILESTONE,
                        f"🎉 引用里程碑: {milestone}",
                        f"《{paper.title}》达到 {paper.citation_count} 次引用！",
                        {"paper_id": paper.paper_id, "milestone": milestone}
                    )
                    break
    
    async def collect_all(self) -> Dict[str, List[PaperMeta]]:
        """收录所有追踪学者的论文"""
        results = {}
        
        for scholar_id, scholar in self._tracked_scholars.items():
            papers = await self.collect_for_scholar(scholar)
            results[scholar_id] = papers
            
            # 避免请求过快
            await asyncio.sleep(1)
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取收录统计"""
        recent_records = self.storage.get_recent_records(days=7)
        
        return {
            "tracked_scholars": len(self._tracked_scholars),
            "total_records": len(self.storage._records),
            "records_last_7_days": len(recent_records),
            "papers_by_scholar": {
                scholar_id: len(self.storage.get_records_by_scholar(scholar_id))
                for scholar_id in self._tracked_scholars
            }
        }


class Scheduler:
    """
    调度器主类
    
    管理定时收录任务
    """
    
    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        self.collector = PaperCollector(self.config)
        self.storage = CollectionStorage(self.config.data_dir)
        
        self._running = False
        self._last_collection: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
    
    def set_paper_fetcher(self, fetcher: Callable[[str], Awaitable[List[PaperMeta]]]):
        """设置论文获取函数"""
        self.collector.set_paper_fetcher(fetcher)
    
    def track_scholar(self, scholar: Scholar):
        """添加追踪学者"""
        self.collector.track_scholar(scholar)
    
    def track_scholars(self, scholars: List[Scholar]):
        """批量添加追踪学者"""
        for scholar in scholars:
            self.track_scholar(scholar)
    
    def get_interval_seconds(self) -> int:
        """获取收录间隔（秒）"""
        if self.config.collection_interval == CollectionInterval.CUSTOM:
            return self.config.custom_interval_seconds
        return self.config.collection_interval.value
    
    async def run_once(self) -> Dict[str, List[PaperMeta]]:
        """执行一次收录"""
        print(f"[{datetime.now()}] Starting collection...")
        
        results = await self.collector.collect_all()
        
        self._last_collection = datetime.now()
        self._save_state()
        
        # 统计
        total_new = sum(len(papers) for papers in results.values())
        print(f"[{datetime.now()}] Collection complete. New papers: {total_new}")
        
        return results
    
    async def start(self):
        """启动调度器"""
        if self._running:
            print("Scheduler is already running")
            return
        
        self._running = True
        self._load_state()
        
        print(f"Scheduler started. Interval: {self.get_interval_seconds()}s")
        
        # 启动后台任务
        self._task = asyncio.create_task(self._run_loop())
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._save_state()
        print("Scheduler stopped")
    
    async def _run_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 检查是否需要执行收录
                if self._should_collect():
                    await self.run_once()
                
                # 等待下一次检查
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)
    
    def _should_collect(self) -> bool:
        """检查是否应该执行收录"""
        if self._last_collection is None:
            return True
        
        elapsed = (datetime.now() - self._last_collection).total_seconds()
        return elapsed >= self.get_interval_seconds()
    
    def _save_state(self):
        """保存状态"""
        state = {
            "last_collection": self._last_collection.isoformat() if self._last_collection else None,
            "tracked_scholars": list(self.collector._tracked_scholars.keys()),
        }
        self.storage.save_state(state)
    
    def _load_state(self):
        """加载状态"""
        state = self.storage.load_state()
        if state.get("last_collection"):
            self._last_collection = datetime.fromisoformat(state["last_collection"])
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "running": self._running,
            "last_collection": self._last_collection.isoformat() if self._last_collection else None,
            "next_collection_in": self._get_next_collection_time(),
            "collection_stats": self.collector.get_collection_stats(),
        }
    
    def _get_next_collection_time(self) -> Optional[str]:
        """获取下次收录时间"""
        if not self._last_collection:
            return "Immediately"
        
        next_time = self._last_collection + timedelta(seconds=self.get_interval_seconds())
        if next_time <= datetime.now():
            return "Pending"
        
        remaining = (next_time - datetime.now()).total_seconds()
        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining / 60)}m"
        else:
            return f"{int(remaining / 3600)}h"
    
    def get_feed(self, limit: int = 20) -> str:
        """获取信息流"""
        return self.collector.feed_service.get_feed_formatted(limit)


class ConferenceTracker:
    """
    会议追踪器
    
    追踪重要会议的截止日期并发送提醒
    """
    
    def __init__(self, notification: NotificationHandler):
        self.notification = notification
        self._conferences: Dict[str, Dict[str, Any]] = {}
    
    def add_conference(
        self,
        name: str,
        deadline: datetime,
        url: Optional[str] = None,
        remind_days_before: List[int] = None,
    ):
        """添加会议"""
        self._conferences[name] = {
            "name": name,
            "deadline": deadline,
            "url": url,
            "remind_days_before": remind_days_before or [7, 3, 1],
            "reminded": set(),
        }
    
    async def check_deadlines(self):
        """检查截止日期"""
        now = datetime.now()
        
        for name, conf in self._conferences.items():
            deadline = conf["deadline"]
            days_left = (deadline - now).days
            
            for remind_day in conf["remind_days_before"]:
                if days_left <= remind_day and remind_day not in conf["reminded"]:
                    await self.notification.notify(
                        NotificationType.CONFERENCE_DEADLINE,
                        f"⏰ {name} 截止提醒",
                        f"距离截止日期还有 {days_left} 天！",
                        {"deadline": deadline.isoformat(), "url": conf.get("url")}
                    )
                    conf["reminded"].add(remind_day)
    
    def get_upcoming_conferences(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取即将到来的会议"""
        now = datetime.now()
        cutoff = now + timedelta(days=days)
        
        upcoming = []
        for name, conf in self._conferences.items():
            if now <= conf["deadline"] <= cutoff:
                days_left = (conf["deadline"] - now).days
                upcoming.append({
                    "name": name,
                    "deadline": conf["deadline"].isoformat(),
                    "days_left": days_left,
                    "url": conf.get("url"),
                })
        
        return sorted(upcoming, key=lambda x: x["days_left"])


# 便捷函数
def create_scheduler(
    interval: str = "daily",
    data_dir: str = "./data/scheduler",
    notify_console: bool = True,
) -> Scheduler:
    """
    创建调度器
    
    Args:
        interval: "hourly", "daily", "weekly"
        data_dir: 数据存储目录
        notify_console: 是否启用控制台通知
    
    Returns:
        配置好的调度器
    """
    interval_map = {
        "hourly": CollectionInterval.HOURLY,
        "daily": CollectionInterval.DAILY,
        "weekly": CollectionInterval.WEEKLY,
    }
    
    config = SchedulerConfig(
        collection_interval=interval_map.get(interval, CollectionInterval.DAILY),
        data_dir=data_dir,
        notification=NotificationConfig(
            enabled=True,
            channels=["console"] if notify_console else [],
        ),
    )
    
    return Scheduler(config)
