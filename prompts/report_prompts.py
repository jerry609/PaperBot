# prompts/report_prompts.py
"""
报告生成相关 Prompt 模板
"""

# 论文分析报告生成 Prompt
PAPER_REPORT_PROMPT = """你是一位技术写作专家。请基于以下分析结果，生成一份专业的论文分析报告。

## 论文信息

**标题**: {title}
**作者**: {authors}
**年份**: {year}
**发表于**: {venue}
**引用数**: {citations}

## 摘要

{abstract}

## 研究分析

{research_analysis}

## 代码分析（如有）

{code_analysis}

## 质量评估

{quality_assessment}

## 影响力评分

- 总分: {total_score}/100
- 学术影响力: {academic_score}/100
- 工程影响力: {engineering_score}/100

## 任务

请生成一份结构化的 Markdown 报告，包含以下部分：

1. **执行摘要**：3-5段的核心内容总结
2. **核心贡献**：论文的主要贡献点（列表）
3. **技术分析**：关键技术和方法的解读
4. **代码评估**：代码质量和可复现性分析（如有代码）
5. **优势与不足**：客观的优缺点分析
6. **推荐阅读人群**：适合哪些读者
7. **阅读建议**：如何高效阅读这篇论文

请用中文撰写，保持专业性和可读性。使用 Markdown 格式。
"""

# 学者追踪摘要报告 Prompt
SCHOLAR_SUMMARY_PROMPT = """请为学者 {scholar_name} 的最新论文生成追踪摘要报告。

## 新发现的论文

{papers_list}

## 任务

生成一份简洁的摘要报告，包含：

1. **概览**：新论文数量和总体情况
2. **亮点论文**：重点推荐的论文（影响力评分最高的 1-3 篇）
3. **研究趋势**：基于新论文分析该学者近期的研究方向
4. **推荐阅读顺序**：按重要性排序的阅读建议

请用中文撰写，使用 Markdown 格式。
"""

# 对比分析报告 Prompt
COMPARISON_REPORT_PROMPT = """请生成以下两篇论文的对比分析报告。

## 论文 A

{paper_a_info}

## 论文 B

{paper_b_info}

## 任务

生成对比报告，包含：

1. **问题对比**：两篇论文解决的问题有何异同
2. **方法对比**：技术方法的差异
3. **结果对比**：实验结果和性能对比
4. **适用场景**：各自更适合的应用场景
5. **综合评价**：哪篇更值得深入研究（说明理由）

请用中文撰写，使用表格等方式清晰呈现对比。
"""

# 周报/月报生成 Prompt
PERIODIC_REPORT_PROMPT = """请生成 {period} 的论文追踪周期报告。

## 追踪学者

{scholars_list}

## 新发现论文

{papers_summary}

## 任务

生成周期报告，包含：

1. **概览统计**：本期追踪的学者数、新论文数
2. **重点论文**：本期最值得关注的论文（Top 5）
3. **趋势分析**：观察到的研究趋势
4. **下期展望**：预期关注的方向

请用中文撰写，使用 Markdown 格式。
"""

# Prompt 集合
REPORT_PROMPTS = {
    "paper_report": PAPER_REPORT_PROMPT,
    "scholar_summary": SCHOLAR_SUMMARY_PROMPT,
    "comparison_report": COMPARISON_REPORT_PROMPT,
    "periodic_report": PERIODIC_REPORT_PROMPT,
}
