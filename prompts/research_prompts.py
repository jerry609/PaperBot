# prompts/research_prompts.py
"""
论文研究相关 Prompt 模板
"""

# 论文摘要扩展 Prompt
PAPER_SUMMARY_PROMPT = """你是一位资深的学术论文分析专家。请分析以下论文信息，并提供详细的摘要和见解。

## 论文信息

**标题**: {title}
**作者**: {authors}
**年份**: {year}
**发表于**: {venue}
**摘要**: {abstract}

## 任务

请完成以下分析：

1. **执行摘要**（3-5段）：
   - 论文要解决的核心问题是什么？
   - 采用了什么方法/技术？
   - 主要贡献和创新点是什么？
   - 实验结果如何？
   - 这项工作的意义和应用场景是什么？

2. **核心贡献**（列表形式）：
   - 列出论文的 3-5 个核心贡献点

3. **技术亮点**：
   - 简述论文中的关键技术或方法

4. **适用场景**：
   - 这项工作适用于哪些实际场景？

请用中文回复，保持专业性和可读性。
"""

# 代码仓库发现 Prompt
CODE_DISCOVERY_PROMPT = """你是一位技术研究员，擅长查找学术论文相关的开源代码实现。

## 论文信息

**标题**: {title}
**作者**: {authors}
**年份**: {year}

## 任务

基于论文信息，请推测可能的代码仓库位置：

1. 检查论文标题，生成可能的 GitHub 搜索关键词
2. 基于作者信息，推测可能的 GitHub 用户名
3. 给出最可能的代码仓库 URL 格式

请以 JSON 格式输出：
```json
{{
  "search_keywords": ["关键词1", "关键词2"],
  "possible_authors": ["github_user1", "github_user2"],
  "suggested_urls": ["https://github.com/...", "..."],
  "notes": "其他说明"
}}
```
"""

# 论文领域分类 Prompt
PAPER_CLASSIFICATION_PROMPT = """请分析以下论文的研究领域和主题。

**标题**: {title}
**摘要**: {abstract}

请识别：
1. 主要研究领域（如：机器学习、网络安全、系统等）
2. 具体研究方向（如：对抗样本、漏洞检测等）
3. 关键技术标签

以 JSON 格式输出：
```json
{{
  "primary_field": "主要领域",
  "secondary_field": "次要领域",
  "research_direction": "具体研究方向",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}
```
"""

# Prompt 集合
RESEARCH_PROMPTS = {
    "paper_summary": PAPER_SUMMARY_PROMPT,
    "code_discovery": CODE_DISCOVERY_PROMPT,
    "paper_classification": PAPER_CLASSIFICATION_PROMPT,
}
