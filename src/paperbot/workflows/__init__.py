"""
工作流定义层 - 业务流程编排。
"""

# 兼容层：也导出原有的 ScholarWorkflowCoordinator
try:
    from paperbot.core.workflow_coordinator import ScholarWorkflowCoordinator
except ImportError:
    ScholarWorkflowCoordinator = None

# Feed 模块
from .feed import (
    FeedGenerator,
    FeedEvent,
    FeedEventType,
    FeedEventFactory,
    ScholarFeedService,
)

# Filters 模块
from .filters import (
    ScholarFilter,
    PaperFilter,
    FilterCriteria,
    PaperFilterCriteria,
    FilterPresets,
    FilterService,
    ScholarType,
    AffiliationType,
    ResearchArea,
)

# Scheduler 模块
from .scheduler import (
    Scheduler,
    PaperCollector,
    CollectionRecord,
    SchedulerConfig,
    NotificationConfig,
    ConferenceTracker,
    create_scheduler,
)

# Nodes
from .nodes import (
    ScholarFetchNode,
    PaperDetectionNode,
    InfluenceCalculationNode,
    ReportGenerationNode,
    ReflectionSearchNode,
    ReflectionSummaryNode,
)

__all__ = [
    "ScholarWorkflowCoordinator",
    # Feed
    "FeedGenerator",
    "FeedEvent",
    "FeedEventType",
    "FeedEventFactory",
    "ScholarFeedService",
    # Filters
    "ScholarFilter",
    "PaperFilter",
    "FilterCriteria",
    "PaperFilterCriteria",
    "FilterPresets",
    "FilterService",
    "ScholarType",
    "AffiliationType",
    "ResearchArea",
    # Scheduler
    "Scheduler",
    "PaperCollector",
    "CollectionRecord",
    "SchedulerConfig",
    "NotificationConfig",
    "ConferenceTracker",
    "create_scheduler",
    # Nodes
    "ScholarFetchNode",
    "PaperDetectionNode",
    "InfluenceCalculationNode",
    "ReportGenerationNode",
    "ReflectionSearchNode",
    "ReflectionSummaryNode",
]
