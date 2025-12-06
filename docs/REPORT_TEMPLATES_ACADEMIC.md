## 学术报告模板与字段契约（MVP）

目标：为学术版 BettaFish / PaperBot 输出的报告提供统一的数据契约，便于 ReportEngine 渲染 IMRaD 风格模板（如 `reports/templates/academic_report.md.j2`）。

---

### 1. 渲染所需上下文（必填/可选）

- `report`（可选字段）：  
  - `title`（可选，默认“学术实验报告”）  
  - `abstract`（可选，摘要文本）  
  - `background`（可选，研究背景）  
  - `research_questions`（可选，list[str]）  
  - `hypothesis`（可选，字符串）  
  - `findings`（可选，结果解读文本）  
  - `discussion`（可选，dict：`main_points`/`limitations`/`related_work`）  
  - `conclusion`、`future_work`（可选）

- `dataset`（建议必填关键字段）：  
  - `name`（必填）  
  - `license`（必填：明确许可/合规性）  
  - `size`（必填：样本条数）  
  - `label_distribution`（必填：可为 dict 或字符串）  
  - `splits`（可选：train/val/test 配比或说明）  
  - `preprocessing`（可选：清洗/脱敏/分词等）

- `experiment`（建议必填关键字段）：  
  - `name`（必填，与 `experiment_name` 对应）  
  - `seed`（可选，默认 42）  
  - `task`（可选，示例：`classification`）  
  - `models`（必填：list[dict]，每个包含 `name`、`type`，可选 `checkpoint`）  
  - `metrics`（可选：渲染时显示的指标名列表）  
  - `runtime`（可选：本次运行耗时）

- `results`（必填）：list[dict]，每个元素应包含：  
  - `model`: 模型名  
  - `metrics`: dict，键为指标名，值为数值或字符串（模板按 `{{ "%.4f"|format(v) }}` 渲染数值）

- `charts`（可选）：dict，用于传递图表占位路径/文本，例如：  
  - `sentiment_distribution`、`timeline`、`extra`

- `generated_at`（必填）：生成时间字符串。

---

### 2. 最小可用示例（与 ExperimentManager 输出对齐）

```json
{
  "report": {
    "title": "Weibo Sentiment Benchmark",
    "abstract": "对 sample_sentiment 数据集进行基线对比……",
    "research_questions": ["RQ1: 简单基线在小数据集的表现如何？"],
    "findings": "keyword_rule 在正例上优于 majority……"
  },
  "dataset": {
    "name": "sample_sentiment",
    "license": "CC-BY-SA 4.0（示例）",
    "size": 5,
    "label_distribution": {"0": 3, "1": 2},
    "splits": "all for eval",
    "preprocessing": "去重、UTF-8 清洗"
  },
  "experiment": {
    "name": "sentiment_benchmark_v1",
    "seed": 42,
    "task": "classification",
    "models": [
      {"name": "majority_baseline", "type": "majority"},
      {"name": "keyword_rule", "type": "rule_keyword"},
      {"name": "random_label", "type": "random"}
    ],
    "metrics": ["accuracy", "f1_macro", "f1_0", "f1_1"]
  },
  "results": [
    {"model": "majority_baseline", "metrics": {"accuracy": 0.60, "f1_macro": 0.43}},
    {"model": "keyword_rule", "metrics": {"accuracy": 0.80, "f1_macro": 0.78}},
    {"model": "random_label", "metrics": {"accuracy": 0.50, "f1_macro": 0.45}}
  ],
  "generated_at": "2025-12-06 12:00:00"
}
```

---

### 3. 渲染指引（示例）

1. 准备上述上下文（可以由 ExperimentManager 运行结果 + 元数据组合得到）。  
2. 选择模板：`reports/templates/academic_report.md.j2`。  
3. 使用 Jinja2 渲染生成 Markdown，再按需转 HTML/PDF。  
4. 若缺少必填字段，建议在渲染前做校验并报错（例如缺少 `dataset.name` / `results`）。

---

### 4. 常见校验建议

- **字段必填**：`dataset.name`、`dataset.license`、`dataset.size`、`experiment.name`、`results`、`generated_at`。  
- **数值格式**：`results[*].metrics` 应为可格式化的数值或字符串；若为空请填 `"-"` 避免渲染异常。  
- **编码与脱敏**：数据描述中注意隐私与许可，示例数据需注明不可用于生产。  
- **时间与时区**：`generated_at` 建议统一 UTC 或明确时区，避免跨机器差异。  

