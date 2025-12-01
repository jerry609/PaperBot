# Scholar Tracking Subsystem
# 学者追踪子系统

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
]