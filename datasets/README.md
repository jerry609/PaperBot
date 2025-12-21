# Datasets 目录说明

该目录用于存放 PaperBot 学术版所需的静态数据集，帮助实现“数据集优先、可复现”的实验流程。约定的结构如下：

```
datasets/
  raw/         # 原始数据（爬虫导出或外部获取，未经清洗）
  processed/   # 清洗、脱敏并补充标签后的数据，实验直接使用
  metadata/    # 每个数据集的元信息（来源、字段说明、许可等）
```

> ✅ 建议：对外发布或共享实验结果时，请同时提供 `processed/` 中的数据文件与 `metadata/` 描述，方便他人复现。

## 示例数据集

- `processed/sample_sentiment.csv`：一个极简的中文情感分类示例，包含 `text` 与 `label` 两列，可用于验证 ExperimentRunner 的情感模型对比流程是否运行正常。
- `metadata/sample_sentiment.yaml`：配套的元信息文件，说明数据来源、字段含义及许可要求。

## 使用提示

1. **命名约定**：数据集名称统一使用小写 + 下划线，例如 `weibo_covid_2020`。相应的处理后文件应放在 `processed/weibo_covid_2020.csv`，元信息位于 `metadata/weibo_covid_2020.yaml`。
2. **格式要求**：推荐使用 UTF-8 编码的 CSV 或 Parquet 文件。情感/分类任务至少包含 `text` 与 `label` 两列。
3. **版本管理**：如需区分不同版本，可在文件名中追加时间戳或 `v1/v2` 等后缀，并在 metadata 中注明。
4. **合规性**：确保所有数据均符合来源网站/机构的使用许可与隐私要求；如包含内部数据，请在 metadata 中写明访问限制。

## 下一步

- MindSpider / 爬虫导出脚本可将在线数据整理后写入 `raw/`，再通过清洗脚本写入 `processed/`。
- `ExperimentRunner` 默认会在 `processed/` 中查找与配置名匹配的 CSV 文件；如需自定义路径，可在实验配置中显式设置 `dataset_path` 字段。

准备好数据集后，即可通过 `python main.py run-exp --config <yaml>` 运行实验并生成统一的结果文件（或 `python -m paperbot.utils.experiment_runner --config <yaml>`）。

