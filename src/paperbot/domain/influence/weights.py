# src/paperbot/domain/influence/weights.py
"""
影响力评分权重配置
"""

# 默认权重配置
INFLUENCE_WEIGHTS = {
    # 主权重: 学术影响力 vs 工程影响力
    "academic_weight": 0.6,      # w1
    "engineering_weight": 0.4,   # w2
    
    # 学术影响力内部权重
    "academic": {
        "citation_weight": 0.6,   # 引用数权重
        "venue_weight": 0.4,      # 发表渠道权重
    },
    
    # 工程影响力内部权重
    "engineering": {
        "code_availability_weight": 0.3,  # 代码可用性权重
        "stars_weight": 0.4,              # GitHub Stars 权重
        "reproducibility_weight": 0.3,    # 可复现性权重
    },
    
    # 时间衰减配置
    "recency_half_life_years": 5,  # 半衰期（年）
}

# 引用数评分映射
# (min_citations, max_citations, score)
CITATION_SCORE_RANGES = [
    (0, 0, 0),           # 0 引用: 0 分
    (1, 5, 10),          # 1-5 引用: 10 分
    (6, 20, 25),         # 6-20 引用: 25 分
    (21, 50, 40),        # 21-50 引用: 40 分
    (51, 100, 55),       # 51-100 引用: 55 分
    (101, 200, 70),      # 101-200 引用: 70 分
    (201, 500, 85),      # 201-500 引用: 85 分
    (501, float('inf'), 100),  # 500+ 引用: 100 分
]

# GitHub Stars 评分映射
STARS_SCORE_RANGES = [
    (0, 0, 0),           # 0 stars: 0 分
    (1, 10, 15),         # 1-10 stars: 15 分
    (11, 50, 30),        # 11-50 stars: 30 分
    (51, 100, 45),       # 51-100 stars: 45 分
    (101, 500, 60),      # 101-500 stars: 60 分
    (501, 1000, 75),     # 501-1000 stars: 75 分
    (1001, 5000, 90),    # 1001-5000 stars: 90 分
    (5001, float('inf'), 100),  # 5000+ stars: 100 分
]

# 顶会评分
VENUE_SCORES = {
    "tier1": 100,  # 顶会
    "tier2": 60,   # 优秀会议
    "other": 20,   # 其他
}

# 代码可用性评分
CODE_AVAILABILITY_SCORES = {
    "has_code": 100,    # 有代码
    "no_code": 0,       # 无代码
}


def get_citation_score(citation_count: int) -> float:
    """
    根据引用数获取评分
    
    Args:
        citation_count: 引用数
        
    Returns:
        评分 (0-100)
    """
    for min_c, max_c, score in CITATION_SCORE_RANGES:
        if min_c <= citation_count <= max_c:
            return score
    return 0


def get_stars_score(stars: int) -> float:
    """
    根据 GitHub Stars 获取评分
    
    Args:
        stars: Star 数量
        
    Returns:
        评分 (0-100)
    """
    for min_s, max_s, score in STARS_SCORE_RANGES:
        if min_s <= stars <= max_s:
            return score
    return 0

