# Scholar Tracking Subsystem
# 学者追踪子系统
# 
# 包含以下核心模块:
# - Feed: 信息流生成器，借鉴 JobLeap 的信息聚合模式
# - Filters: 多维度筛选器，支持学者类型、机构、领域等筛选
# - Cards: 信息卡片格式化，类似 JobLeap 的职位卡片设计
# - Scheduler: 定时收录调度器，支持自动收录和通知

# 延迟导入以避免循环依赖
def __getattr__(name):
    # Agent 主类
    if name == "ScholarTrackingAgent":
        from .agent import ScholarTrackingAgent
        return ScholarTrackingAgent
    elif name == "create_agent":
        from .agent import create_agent
        return create_agent
    # 子 Agents
    elif name in ("ScholarProfileAgent", "SemanticScholarAgent", "PaperTrackerAgent"):
        from .agents import ScholarProfileAgent, SemanticScholarAgent, PaperTrackerAgent
        return locals()[name]
    # Models
    elif name in ("Scholar", "PaperMeta", "CodeMeta", "InfluenceResult"):
        from .models import Scholar, PaperMeta, CodeMeta, InfluenceResult
        return locals()[name]
    # Services
    elif name in ("SubscriptionService", "CacheService"):
        from .services import SubscriptionService, CacheService
        return locals()[name]
    # Feed 模块
    elif name in ("FeedGenerator", "FeedEvent", "FeedEventType", "FeedEventFactory", "ScholarFeedService"):
        from .feed import FeedGenerator, FeedEvent, FeedEventType, FeedEventFactory, ScholarFeedService
        return locals()[name]
    # Filters 模块
    elif name in ("ScholarFilter", "PaperFilter", "FilterCriteria", "PaperFilterCriteria", 
                  "FilterPresets", "FilterService", "ScholarType", "AffiliationType", "ResearchArea"):
        from .filters import (ScholarFilter, PaperFilter, FilterCriteria, PaperFilterCriteria,
                             FilterPresets, FilterService, ScholarType, AffiliationType, ResearchArea)
        return locals()[name]
    # Cards 模块
    elif name in ("PaperCard", "ScholarCard", "FeedEventCard", "CardRenderer", 
                  "CardStyle", "OutputFormat", "CardTheme"):
        from .cards import (PaperCard, ScholarCard, FeedEventCard, CardRenderer,
                           CardStyle, OutputFormat, CardTheme)
        return locals()[name]
    # Scheduler 模块
    elif name in ("Scheduler", "PaperCollector", "CollectionRecord", "SchedulerConfig",
                  "NotificationConfig", "ConferenceTracker", "create_scheduler"):
        from .scheduler import (Scheduler, PaperCollector, CollectionRecord, SchedulerConfig,
                               NotificationConfig, ConferenceTracker, create_scheduler)
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Agent 主类
    "ScholarTrackingAgent",
    "create_agent",
    # 子 Agents
    "ScholarProfileAgent",
    "SemanticScholarAgent",
    "PaperTrackerAgent",
    # Models
    "Scholar",
    "PaperMeta",
    "CodeMeta",
    "InfluenceResult",
    # Services
    "SubscriptionService",
    "CacheService",
    # Feed 模块 (借鉴 JobLeap)
    "FeedGenerator",
    "FeedEvent",
    "FeedEventType",
    "FeedEventFactory",
    "ScholarFeedService",
    # Filters 模块 (借鉴 JobLeap)
    "ScholarFilter",
    "PaperFilter",
    "FilterCriteria",
    "PaperFilterCriteria",
    "FilterPresets",
    "FilterService",
    "ScholarType",
    "AffiliationType",
    "ResearchArea",
    # Cards 模块 (借鉴 JobLeap)
    "PaperCard",
    "ScholarCard",
    "FeedEventCard",
    "CardRenderer",
    "CardStyle",
    "OutputFormat",
    "CardTheme",
    # Scheduler 模块
    "Scheduler",
    "PaperCollector",
    "CollectionRecord",
    "SchedulerConfig",
    "NotificationConfig",
    "ConferenceTracker",
    "create_scheduler",
]