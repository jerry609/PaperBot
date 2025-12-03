# PaperBot Tools
# 工具模块

from .search import (
    AuthorResult,
    PaperResult,
    SearchResponse,
    SemanticScholarSearch,
    SearchResultRanker,
)

from .keyword_optimizer import (
    OptimizedQuery,
    KeywordOptimizer,
    SecurityPaperQueryBuilder,
    SECURITY_SYNONYMS,
    ABBREVIATION_EXPANSIONS,
)

__all__ = [
    # 搜索
    "AuthorResult",
    "PaperResult",
    "SearchResponse",
    "SemanticScholarSearch",
    "SearchResultRanker",
    # 关键词优化
    "OptimizedQuery",
    "KeywordOptimizer",
    "SecurityPaperQueryBuilder",
    "SECURITY_SYNONYMS",
    "ABBREVIATION_EXPANSIONS",
]
