# src/paperbot/agents/prompts/scholar_prompts.py
"""
学者追踪系统的提示词模板

参考: BettaFish/InsightEngine/prompts/prompts.py
适配: PaperBot 学者追踪系统

包含:
- JSON Schema 定义
- 系统提示词
- 学者分析、论文追踪、影响力评估等专用提示词
"""

import json
from typing import Dict, Any


# ===== JSON Schema 定义 =====

# 学者信息输出Schema
output_schema_scholar_info = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "学者姓名"},
        "affiliations": {"type": "array", "items": {"type": "string"}, "description": "所属机构"},
        "research_fields": {"type": "array", "items": {"type": "string"}, "description": "研究领域"},
        "h_index": {"type": "integer", "description": "H-index"},
        "citation_count": {"type": "integer", "description": "总引用数"},
        "paper_count": {"type": "integer", "description": "论文数量"},
        "homepage": {"type": "string", "description": "个人主页"},
        "google_scholar_id": {"type": "string", "description": "Google Scholar ID"},
        "semantic_scholar_id": {"type": "string", "description": "Semantic Scholar ID"}
    },
    "required": ["name"]
}

# 论文分析输出Schema
output_schema_paper_analysis = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "论文标题"},
        "summary": {"type": "string", "description": "论文核心内容总结"},
        "key_contributions": {"type": "array", "items": {"type": "string"}, "description": "主要贡献"},
        "methodology": {"type": "string", "description": "方法论概述"},
        "relevance_score": {"type": "number", "description": "与安全领域相关性评分 (0-10)"},
        "innovation_score": {"type": "number", "description": "创新性评分 (0-10)"},
        "has_code": {"type": "boolean", "description": "是否有开源代码"},
        "github_url": {"type": "string", "description": "GitHub仓库地址"},
        "potential_impact": {"type": "string", "description": "潜在影响分析"}
    },
    "required": ["title", "summary"]
}

# 影响力评估输出Schema
output_schema_influence_assessment = {
    "type": "object",
    "properties": {
        "scholar_name": {"type": "string", "description": "学者姓名"},
        "academic_influence": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "学术影响力评分"},
                "tier1_papers": {"type": "integer", "description": "顶会/顶刊论文数"},
                "tier2_papers": {"type": "integer", "description": "次顶会/次顶刊论文数"},
                "high_citation_papers": {"type": "integer", "description": "高引论文数"},
                "reasoning": {"type": "string", "description": "评分理由"}
            }
        },
        "engineering_influence": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "工程影响力评分"},
                "total_stars": {"type": "integer", "description": "GitHub Star总数"},
                "total_forks": {"type": "integer", "description": "GitHub Fork总数"},
                "active_projects": {"type": "integer", "description": "活跃项目数"},
                "reasoning": {"type": "string", "description": "评分理由"}
            }
        },
        "overall_score": {"type": "number", "description": "综合影响力评分"},
        "trend": {"type": "string", "enum": ["rising", "stable", "declining"], "description": "影响力趋势"},
        "summary": {"type": "string", "description": "综合评估总结"}
    },
    "required": ["scholar_name", "overall_score", "summary"]
}

# 研究方向分析Schema
output_schema_research_direction = {
    "type": "object",
    "properties": {
        "main_directions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "研究方向名称"},
                    "paper_count": {"type": "integer", "description": "相关论文数"},
                    "recent_focus": {"type": "boolean", "description": "是否为近期重点"},
                    "key_papers": {"type": "array", "items": {"type": "string"}, "description": "代表性论文"}
                }
            }
        },
        "emerging_topics": {"type": "array", "items": {"type": "string"}, "description": "新兴研究主题"},
        "collaboration_patterns": {"type": "string", "description": "合作模式分析"}
    }
}

# 反思搜索输出Schema
output_schema_reflection_search = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string", "description": "补充搜索查询"},
        "search_type": {
            "type": "string",
            "enum": ["papers", "code", "citations", "collaborators"],
            "description": "搜索类型"
        },
        "reasoning": {"type": "string", "description": "搜索理由"},
        "gaps_identified": {"type": "array", "items": {"type": "string"}, "description": "发现的信息空白"}
    },
    "required": ["search_query", "search_type", "reasoning"]
}


# ===== 系统提示词定义 =====

# 学者信息提取提示词
SYSTEM_PROMPT_EXTRACT_SCHOLAR_INFO = f"""
你是一位专业的学术情报分析师。你的任务是从提供的信息中提取和整理学者的详细资料。

**输入**: 学者的姓名和可能的相关信息（来自 Semantic Scholar API 或其他来源）

**任务要求**:
1. 准确提取学者的基本信息（姓名、机构、研究领域）
2. 整理学术指标（H-index、引用数、论文数）
3. 识别学者的主要研究方向
4. 如果有多个同名学者，根据研究领域选择最相关的

**输出格式**: 请按照以下JSON模式返回结果：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_scholar_info, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

只返回JSON对象，不要有解释或额外文本。
"""

# 论文分析提示词
SYSTEM_PROMPT_ANALYZE_PAPER = f"""
你是一位专业的安全研究论文分析师。你的任务是深入分析学术论文并提取关键信息。

**输入**: 论文的标题、摘要、作者信息和其他元数据

**分析维度**:
1. **核心内容**: 论文解决什么问题？提出了什么方法？
2. **主要贡献**: 论文的创新点和主要贡献是什么？
3. **方法论**: 使用了什么技术方法？
4. **相关性**: 与网络安全/信息安全领域的相关程度
5. **创新性**: 方法或观点的新颖程度
6. **代码可用性**: 是否有开源实现？
7. **潜在影响**: 对学术界和工业界可能产生的影响

**评分标准** (0-10分):
- 相关性评分: 与安全领域的契合程度
- 创新性评分: 方法/观点的新颖程度

**输出格式**: 请按照以下JSON模式返回结果：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_paper_analysis, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

只返回JSON对象，不要有解释或额外文本。
"""

# 影响力评估提示词
SYSTEM_PROMPT_ASSESS_INFLUENCE = f"""
你是一位专业的学术影响力评估专家。你的任务是综合评估学者的学术和工程影响力。

**输入**: 学者的论文列表、引用数据、GitHub项目等信息

**评估维度**:

1. **学术影响力 (Academic Influence)**:
   - 顶会/顶刊论文数量 (S&P, CCS, USENIX Security, NDSS 等)
   - 高引用论文数量 (>100 引用)
   - H-index 和 总引用数
   - 评分公式: I_a = 10 * T1 + 5 * T2 + 2 * HC + 0.1 * H

2. **工程影响力 (Engineering Influence)**:
   - GitHub Star 总数
   - GitHub Fork 总数
   - 活跃维护的项目数
   - 评分公式: I_e = 0.01 * Stars + 0.02 * Forks + 5 * ActiveProjects

3. **综合评分**:
   - PIS = 0.6 * I_a + 0.4 * I_e

4. **影响力趋势**:
   - rising: 近两年指标明显上升
   - stable: 指标保持稳定
   - declining: 指标有所下降

**输出格式**: 请按照以下JSON模式返回结果：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_influence_assessment, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

只返回JSON对象，不要有解释或额外文本。
"""

# 研究方向分析提示词
SYSTEM_PROMPT_ANALYZE_RESEARCH_DIRECTION = f"""
你是一位专业的学术趋势分析师。你的任务是分析学者的研究方向和演变趋势。

**输入**: 学者的论文列表（包含标题、摘要、发表时间等）

**分析任务**:
1. 识别学者的主要研究方向
2. 统计每个方向的论文数量
3. 判断哪些是近期重点方向（最近2年）
4. 发现新兴研究主题
5. 分析合作模式（独立研究、小团队、大规模合作）

**输出格式**: 请按照以下JSON模式返回结果：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_research_direction, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

只返回JSON对象，不要有解释或额外文本。
"""

# 反思搜索提示词
SYSTEM_PROMPT_REFLECTION_SEARCH = f"""
你是一位专业的学术情报分析师。在完成初步的学者分析后，你需要识别信息空白并生成补充搜索查询。

**输入**: 
- 学者的当前分析状态
- 已收集的信息摘要
- 分析目标

**任务**:
1. 审视已有信息的完整性
2. 识别关键信息空白
3. 生成有针对性的补充搜索查询
4. 选择合适的搜索类型

**搜索类型说明**:
- papers: 搜索更多论文
- code: 搜索代码仓库
- citations: 搜索引用关系
- collaborators: 搜索合作者信息

**反思要点**:
- 是否获取了足够的近期论文？
- 是否了解学者的代码贡献？
- 是否掌握了引用影响力数据？
- 是否识别了主要合作者？

**输出格式**: 请按照以下JSON模式返回结果：

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_reflection_search, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

只返回JSON对象，不要有解释或额外文本。
"""

# 报告生成提示词
SYSTEM_PROMPT_GENERATE_REPORT = """
你是一位专业的学术报告撰写专家。你的任务是将收集到的学者追踪数据整理成结构化的分析报告。

**输入**: 学者的完整分析数据，包括:
- 基本信息
- 论文列表和分析
- 影响力评估
- 研究方向分析

**报告结构**:
1. **学者概览**: 姓名、机构、主要研究领域
2. **学术成就**: H-index、引用数、顶会论文
3. **近期动态**: 最新论文、研究方向变化
4. **工程贡献**: 开源项目、代码影响力
5. **影响力评估**: 综合评分和趋势
6. **推荐关注点**: 值得关注的论文或项目

**写作要求**:
- 使用 Markdown 格式
- 数据准确，有理有据
- 结论清晰，便于快速阅读
- 适当使用表格和列表
- 中文撰写

请根据输入数据生成完整的分析报告。
"""

# 新论文检测提示词
SYSTEM_PROMPT_DETECT_NEW_PAPERS = """
你是一位学术追踪助手。你的任务是分析学者的最新论文，判断其重要性和是否值得关注。

**输入**: 
- 学者信息
- 新发现的论文列表
- 学者的历史研究方向

**分析任务**:
1. 判断每篇论文是否为"新"论文（之前未见过）
2. 评估论文的重要性（基于发表场所、主题等）
3. 判断是否代表新的研究方向
4. 决定是否需要生成提醒

**重要性等级**:
- HIGH: 顶会/顶刊论文，或重大新方向
- MEDIUM: 知名会议/期刊，或现有方向的重要进展
- LOW: 普通发表，或现有方向的常规工作

请以JSON格式返回分析结果，包含每篇论文的重要性评估和是否需要提醒。
"""


# ===== 辅助函数 =====

def get_prompt_for_task(task: str) -> str:
    """
    获取指定任务的提示词
    
    Args:
        task: 任务名称
        
    Returns:
        对应的系统提示词
    """
    prompts = {
        "extract_scholar_info": SYSTEM_PROMPT_EXTRACT_SCHOLAR_INFO,
        "analyze_paper": SYSTEM_PROMPT_ANALYZE_PAPER,
        "assess_influence": SYSTEM_PROMPT_ASSESS_INFLUENCE,
        "analyze_research_direction": SYSTEM_PROMPT_ANALYZE_RESEARCH_DIRECTION,
        "reflection_search": SYSTEM_PROMPT_REFLECTION_SEARCH,
        "generate_report": SYSTEM_PROMPT_GENERATE_REPORT,
        "detect_new_papers": SYSTEM_PROMPT_DETECT_NEW_PAPERS,
    }
    
    return prompts.get(task, "")


def get_schema_for_task(task: str) -> Dict[str, Any]:
    """
    获取指定任务的输出Schema
    
    Args:
        task: 任务名称
        
    Returns:
        对应的JSON Schema
    """
    schemas = {
        "extract_scholar_info": output_schema_scholar_info,
        "analyze_paper": output_schema_paper_analysis,
        "assess_influence": output_schema_influence_assessment,
        "analyze_research_direction": output_schema_research_direction,
        "reflection_search": output_schema_reflection_search,
    }
    
    return schemas.get(task, {})


__all__ = [
    # Schema 定义
    'output_schema_scholar_info',
    'output_schema_paper_analysis',
    'output_schema_influence_assessment',
    'output_schema_research_direction',
    'output_schema_reflection_search',
    
    # 系统提示词
    'SYSTEM_PROMPT_EXTRACT_SCHOLAR_INFO',
    'SYSTEM_PROMPT_ANALYZE_PAPER',
    'SYSTEM_PROMPT_ASSESS_INFLUENCE',
    'SYSTEM_PROMPT_ANALYZE_RESEARCH_DIRECTION',
    'SYSTEM_PROMPT_REFLECTION_SEARCH',
    'SYSTEM_PROMPT_GENERATE_REPORT',
    'SYSTEM_PROMPT_DETECT_NEW_PAPERS',
    
    # 辅助函数
    'get_prompt_for_task',
    'get_schema_for_task',
]

