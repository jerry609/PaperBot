# prompts/quality_prompts.py
"""
质量评估相关 Prompt 模板
"""

# 论文综合质量评估 Prompt
PAPER_QUALITY_PROMPT = """你是一位学术论文审稿专家。请对以下论文进行综合质量评估。

## 论文信息

**标题**: {title}
**作者**: {authors}
**年份**: {year}
**发表于**: {venue}
**引用数**: {citations}

## 摘要

{abstract}

## 代码分析结果（如有）

{code_analysis}

## 任务

请从以下维度评估论文质量：

1. **学术价值** (0-100)
   - 问题的重要性
   - 方法的创新性
   - 理论贡献

2. **实践价值** (0-100)
   - 实际应用潜力
   - 工业界采用可能性
   - 技术成熟度

3. **可信度** (0-100)
   - 实验设计的严谨性
   - 结果的可复现性
   - 作者声誉和发表渠道

4. **可读性** (0-100)
   - 写作质量
   - 论文结构
   - 图表清晰度

请以 JSON 格式输出：
```json
{{
  "academic_value": 85,
  "practical_value": 75,
  "credibility": 80,
  "readability": 90,
  "overall_score": 82,
  "overall_assessment": "整体评价（2-3段）",
  "strengths": ["优点1", "优点2", "优点3"],
  "weaknesses": ["不足1", "不足2"],
  "target_audience": ["目标读者群1", "目标读者群2"],
  "follow_up_suggestions": ["后续研究建议1", "建议2"]
}}
```
"""

# 论文影响力预测 Prompt
IMPACT_PREDICTION_PROMPT = """基于以下论文信息，预测其未来影响力。

## 论文信息

**标题**: {title}
**发表于**: {venue}
**年份**: {year}
**当前引用数**: {citations}

## 摘要

{abstract}

## 任务

请预测：
1. 短期影响力（1-2年内）
2. 长期影响力（5年以上）
3. 可能的应用领域
4. 潜在的研究热点衍生

以 JSON 格式输出预测结果。
"""

# 与类似工作对比 Prompt
COMPARISON_PROMPT = """请将以下论文与该领域的其他重要工作进行对比。

## 目标论文

**标题**: {title}
**摘要**: {abstract}

## 任务

1. 识别该论文所属的研究方向
2. 列出该方向的其他重要工作
3. 分析本论文相对于其他工作的创新点
4. 指出可能的改进空间

以结构化的方式输出分析结果。
"""

# Prompt 集合
QUALITY_PROMPTS = {
    "paper_quality": PAPER_QUALITY_PROMPT,
    "impact_prediction": IMPACT_PREDICTION_PROMPT,
    "comparison": COMPARISON_PROMPT,
}
