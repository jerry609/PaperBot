# prompts/code_analysis_prompts.py
"""
代码分析相关 Prompt 模板
"""

# 代码仓库质量评估 Prompt
REPO_QUALITY_PROMPT = """你是一位资深的代码审查专家。请评估以下代码仓库的质量。

## 仓库信息

**仓库名称**: {repo_name}
**主要语言**: {language}
**Star 数**: {stars}
**Fork 数**: {forks}
**最后更新**: {last_updated}
**描述**: {description}

## 文件结构

{file_structure}

## README 内容（部分）

{readme_excerpt}

## 任务

请从以下维度评估代码仓库质量：

1. **文档完整性** (0-100)
   - README 是否清晰？
   - 是否有安装和使用说明？
   - 是否有 API 文档？

2. **代码组织** (0-100)
   - 目录结构是否合理？
   - 是否有模块化设计？
   - 命名规范是否良好？

3. **可复现性** (0-100)
   - 是否有依赖管理（requirements.txt, package.json 等）？
   - 是否有示例代码或教程？
   - 环境配置是否简单？

4. **维护状态** (0-100)
   - 最近是否有更新？
   - Issue 处理是否及时？
   - 社区是否活跃？

请以 JSON 格式输出：
```json
{{
  "documentation_score": 85,
  "code_organization_score": 75,
  "reproducibility_score": 80,
  "maintenance_score": 70,
  "overall_score": 78,
  "summary": "整体评价总结",
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "recommendations": ["建议1", "建议2"]
}}
```
"""

# 代码安全性分析 Prompt
SECURITY_ANALYSIS_PROMPT = """你是一位安全研究员。请分析以下代码片段的潜在安全问题。

## 代码

```{language}
{code_snippet}
```

## 任务

请识别可能的安全问题：
1. 常见漏洞类型（如注入、XSS、不安全的反序列化等）
2. 不安全的编码实践
3. 潜在的隐私泄露风险

以 JSON 格式输出分析结果。
"""

# 代码复杂度分析 Prompt
COMPLEXITY_ANALYSIS_PROMPT = """请分析以下代码的复杂度特征。

## 仓库统计

**文件数量**: {file_count}
**代码行数**: {lines_of_code}
**语言分布**: {languages}

## 任务

评估代码复杂度：
1. 整体规模评估
2. 架构复杂度
3. 学习曲线估计

以 JSON 格式输出。
"""

# Prompt 集合
CODE_ANALYSIS_PROMPTS = {
    "repo_quality": REPO_QUALITY_PROMPT,
    "security_analysis": SECURITY_ANALYSIS_PROMPT,
    "complexity_analysis": COMPLEXITY_ANALYSIS_PROMPT,
}
