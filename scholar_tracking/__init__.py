# Scholar Tracking Subsystem
# 学者追踪子系统

# 延迟导入以避免循环依赖
def __getattr__(name):
    if name in ("ScholarProfileAgent", "SemanticScholarAgent", "PaperTrackerAgent"):
        from .agents import ScholarProfileAgent, SemanticScholarAgent, PaperTrackerAgent
        return locals()[name]
    elif name in ("Scholar", "PaperMeta", "CodeMeta", "InfluenceResult"):
        from .models import Scholar, PaperMeta, CodeMeta, InfluenceResult
        return locals()[name]
    elif name in ("SubscriptionService", "CacheService"):
        from .services import SubscriptionService, CacheService
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Agents
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
