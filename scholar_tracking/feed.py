# scholar_tracking/feed.py
"""
信息流生成器 - 借鉴 JobLeap 的信息聚合模式
生成学者动态信息流，包括新论文、引用变化、会议通知等事件
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, Callable
from datetime import datetime, timedelta
from enum import Enum
import heapq

from .models import Scholar, PaperMeta


class FeedEventType(Enum):
    """信息流事件类型"""
    NEW_PAPER = "new_paper"           # 新论文发布
    CITATION_CHANGE = "citation"      # 引用变化
    CONFERENCE_NOTICE = "conference"  # 会议通知
    SCHOLAR_UPDATE = "scholar"        # 学者信息更新
    TRENDING = "trending"             # 热门论文
    RECOMMENDATION = "recommend"      # 推荐内容


@dataclass
class FeedEvent:
    """信息流事件"""
    
    event_type: FeedEventType
    title: str
    description: str
    timestamp: datetime
    
    # 关联数据
    scholar: Optional[Scholar] = None
    paper: Optional[PaperMeta] = None
    
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 优先级 (数字越小优先级越高)
    priority: int = 5
    
    # 是否已读
    is_read: bool = False
    
    def __lt__(self, other: "FeedEvent"):
        """用于优先队列排序：先按时间倒序，再按优先级"""
        if self.timestamp != other.timestamp:
            return self.timestamp > other.timestamp  # 新事件优先
        return self.priority < other.priority
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "scholar": self.scholar.to_dict() if self.scholar else None,
            "paper": self.paper.to_dict() if self.paper else None,
            "metadata": self.metadata,
            "priority": self.priority,
            "is_read": self.is_read,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedEvent":
        return cls(
            event_type=FeedEventType(data["event_type"]),
            title=data["title"],
            description=data["description"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            scholar=Scholar.from_dict(data["scholar"]) if data.get("scholar") else None,
            paper=PaperMeta.from_dict(data["paper"]) if data.get("paper") else None,
            metadata=data.get("metadata", {}),
            priority=data.get("priority", 5),
            is_read=data.get("is_read", False),
        )


class FeedGenerator:
    """
    信息流生成器
    
    借鉴 JobLeap 的信息聚合模式:
    - 多源数据聚合
    - 时间戳追踪
    - 分类筛选
    - 优先级排序
    """
    
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self._events: List[FeedEvent] = []
        self._event_sources: List[Callable[[], List[FeedEvent]]] = []
    
    def register_source(self, source: Callable[[], List[FeedEvent]]):
        """注册事件源"""
        self._event_sources.append(source)
    
    def add_event(self, event: FeedEvent):
        """添加事件到信息流"""
        heapq.heappush(self._events, event)
        
        # 维护最大事件数
        while len(self._events) > self.max_events:
            heapq.heappop(self._events)
    
    def add_events(self, events: List[FeedEvent]):
        """批量添加事件"""
        for event in events:
            self.add_event(event)
    
    def refresh(self):
        """从所有事件源刷新数据"""
        for source in self._event_sources:
            try:
                new_events = source()
                self.add_events(new_events)
            except Exception as e:
                print(f"Error refreshing from source: {e}")
    
    def get_feed(
        self,
        limit: int = 50,
        event_types: Optional[List[FeedEventType]] = None,
        since: Optional[datetime] = None,
        scholar_ids: Optional[List[str]] = None,
        unread_only: bool = False,
    ) -> List[FeedEvent]:
        """
        获取信息流
        
        Args:
            limit: 返回条目数
            event_types: 筛选事件类型
            since: 筛选时间范围
            scholar_ids: 筛选特定学者
            unread_only: 仅未读
        
        Returns:
            排序后的事件列表
        """
        # 排序所有事件
        sorted_events = sorted(self._events, reverse=True)
        
        # 应用筛选条件
        filtered = []
        for event in sorted_events:
            # 类型筛选
            if event_types and event.event_type not in event_types:
                continue
            
            # 时间筛选
            if since and event.timestamp < since:
                continue
            
            # 学者筛选
            if scholar_ids:
                if not event.scholar or event.scholar.semantic_scholar_id not in scholar_ids:
                    continue
            
            # 未读筛选
            if unread_only and event.is_read:
                continue
            
            filtered.append(event)
            
            if len(filtered) >= limit:
                break
        
        return filtered
    
    def get_today_feed(self, limit: int = 50) -> List[FeedEvent]:
        """获取今日信息流"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_feed(limit=limit, since=today)
    
    def get_weekly_digest(self) -> Dict[str, Any]:
        """获取周报摘要"""
        week_ago = datetime.now() - timedelta(days=7)
        events = self.get_feed(limit=500, since=week_ago)
        
        # 统计各类事件
        stats = {event_type: 0 for event_type in FeedEventType}
        scholars_active = set()
        new_papers = []
        
        for event in events:
            stats[event.event_type] += 1
            if event.scholar:
                scholars_active.add(event.scholar.name)
            if event.event_type == FeedEventType.NEW_PAPER and event.paper:
                new_papers.append(event.paper)
        
        return {
            "period": {
                "start": week_ago.isoformat(),
                "end": datetime.now().isoformat(),
            },
            "total_events": len(events),
            "event_breakdown": {k.value: v for k, v in stats.items()},
            "active_scholars": list(scholars_active),
            "new_papers_count": len(new_papers),
            "top_papers": sorted(
                new_papers,
                key=lambda p: p.citation_count,
                reverse=True
            )[:10],
        }
    
    def mark_as_read(self, events: List[FeedEvent]):
        """标记事件为已读"""
        event_ids = {id(e) for e in events}
        for event in self._events:
            if id(event) in event_ids:
                event.is_read = True
    
    def clear_old_events(self, days: int = 30):
        """清理旧事件"""
        cutoff = datetime.now() - timedelta(days=days)
        self._events = [e for e in self._events if e.timestamp > cutoff]
        heapq.heapify(self._events)


class FeedEventFactory:
    """事件工厂 - 创建各类信息流事件"""
    
    @staticmethod
    def new_paper_event(scholar: Scholar, paper: PaperMeta) -> FeedEvent:
        """创建新论文事件"""
        return FeedEvent(
            event_type=FeedEventType.NEW_PAPER,
            title=f"🆕 {scholar.name} 发表新论文",
            description=paper.title,
            timestamp=datetime.now(),
            scholar=scholar,
            paper=paper,
            priority=2,  # 新论文高优先级
            metadata={
                "venue": paper.venue,
                "year": paper.year,
            }
        )
    
    @staticmethod
    def citation_change_event(
        scholar: Scholar,
        paper: PaperMeta,
        old_count: int,
        new_count: int
    ) -> FeedEvent:
        """创建引用变化事件"""
        change = new_count - old_count
        emoji = "📈" if change > 0 else "📉"
        
        return FeedEvent(
            event_type=FeedEventType.CITATION_CHANGE,
            title=f"{emoji} {paper.title[:50]}... 引用变化",
            description=f"引用数从 {old_count} 变为 {new_count} ({'+' if change > 0 else ''}{change})",
            timestamp=datetime.now(),
            scholar=scholar,
            paper=paper,
            priority=4,
            metadata={
                "old_count": old_count,
                "new_count": new_count,
                "change": change,
            }
        )
    
    @staticmethod
    def conference_notice_event(
        conference_name: str,
        event_name: str,
        deadline: Optional[datetime] = None,
        url: Optional[str] = None,
    ) -> FeedEvent:
        """创建会议通知事件"""
        description = f"{conference_name}: {event_name}"
        if deadline:
            days_left = (deadline - datetime.now()).days
            description += f" (还剩 {days_left} 天)"
        
        return FeedEvent(
            event_type=FeedEventType.CONFERENCE_NOTICE,
            title=f"🎯 {conference_name} 通知",
            description=description,
            timestamp=datetime.now(),
            priority=3,
            metadata={
                "conference": conference_name,
                "event": event_name,
                "deadline": deadline.isoformat() if deadline else None,
                "url": url,
            }
        )
    
    @staticmethod
    def scholar_update_event(
        scholar: Scholar,
        update_type: str,
        details: str,
    ) -> FeedEvent:
        """创建学者更新事件"""
        return FeedEvent(
            event_type=FeedEventType.SCHOLAR_UPDATE,
            title=f"👤 {scholar.name} {update_type}",
            description=details,
            timestamp=datetime.now(),
            scholar=scholar,
            priority=5,
            metadata={"update_type": update_type}
        )
    
    @staticmethod
    def trending_paper_event(paper: PaperMeta, reason: str) -> FeedEvent:
        """创建热门论文事件"""
        return FeedEvent(
            event_type=FeedEventType.TRENDING,
            title=f"🔥 热门论文",
            description=f"{paper.title} - {reason}",
            timestamp=datetime.now(),
            paper=paper,
            priority=3,
            metadata={"reason": reason}
        )
    
    @staticmethod
    def recommendation_event(
        title: str,
        description: str,
        paper: Optional[PaperMeta] = None,
        scholar: Optional[Scholar] = None,
    ) -> FeedEvent:
        """创建推荐事件"""
        return FeedEvent(
            event_type=FeedEventType.RECOMMENDATION,
            title=f"💡 {title}",
            description=description,
            timestamp=datetime.now(),
            paper=paper,
            scholar=scholar,
            priority=6,
        )


class ScholarFeedService:
    """
    学者信息流服务
    
    整合信息流生成、筛选、格式化的完整服务
    """
    
    def __init__(self):
        self.generator = FeedGenerator()
        self.factory = FeedEventFactory()
        self._tracked_scholars: Dict[str, Scholar] = {}
        self._paper_cache: Dict[str, PaperMeta] = {}
    
    def track_scholar(self, scholar: Scholar):
        """添加追踪学者"""
        self._tracked_scholars[scholar.semantic_scholar_id] = scholar
    
    def untrack_scholar(self, scholar_id: str):
        """取消追踪学者"""
        self._tracked_scholars.pop(scholar_id, None)
    
    def process_new_papers(self, scholar: Scholar, papers: List[PaperMeta]):
        """处理学者的新论文"""
        for paper in papers:
            if paper.paper_id not in self._paper_cache:
                # 新论文
                event = self.factory.new_paper_event(scholar, paper)
                self.generator.add_event(event)
                self._paper_cache[paper.paper_id] = paper
    
    def process_citation_changes(
        self,
        scholar: Scholar,
        paper: PaperMeta,
        old_citation_count: int
    ):
        """处理引用变化"""
        if paper.citation_count != old_citation_count:
            event = self.factory.citation_change_event(
                scholar, paper, old_citation_count, paper.citation_count
            )
            self.generator.add_event(event)
    
    def add_conference_notice(
        self,
        conference: str,
        event_name: str,
        deadline: Optional[datetime] = None,
        url: Optional[str] = None,
    ):
        """添加会议通知"""
        event = self.factory.conference_notice_event(
            conference, event_name, deadline, url
        )
        self.generator.add_event(event)
    
    def get_feed(self, **kwargs) -> List[FeedEvent]:
        """获取信息流"""
        return self.generator.get_feed(**kwargs)
    
    def get_feed_formatted(self, limit: int = 20) -> str:
        """获取格式化的信息流文本"""
        events = self.get_feed(limit=limit)
        
        if not events:
            return "📭 暂无新动态"
        
        lines = ["📰 **学者动态 Feed**", ""]
        
        for event in events:
            time_str = event.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"├── [{time_str}] {event.title}")
            lines.append(f"│   {event.description[:80]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "tracked_scholars": len(self._tracked_scholars),
            "cached_papers": len(self._paper_cache),
            "total_events": len(self.generator._events),
            "weekly_digest": self.generator.get_weekly_digest(),
        }
