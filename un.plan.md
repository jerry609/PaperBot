<!-- 276e04e2-3de8-4edc-9683-381bcbb5e086 255e3d0f-388f-477f-b8ce-399f01beef01 -->
# 架构改进 TODO（不改代码版）

## 1) Agent 目录与职责收敛

- 收敛 `agent_module/` 与 `agents/`，保留一套实现与入口；迁移或删除重复 Agent。
- 拆分 `ResearchAgent`：会议抓取/下载节点 vs 单篇摘要+代码发现节点；其他 Agent 保持单一职责。

## 2) 配置集中与模式分离

- 在 `config/settings.py` 增加 `mode`（academic/production）及数据源/模板/LLM 配置分支。
- CLI 子命令仅做覆盖，所有参数统一从 settings/.env/YAML 注入。

## 3) DataSource 抽象落地

- 引入 `BaseDataSource`（local/db/hybrid），在追踪/分析时统一走 DataSource；MindSpider 仅作为离线导出工具。

## 4) 流水线契约与评分对齐

- 在 `ScholarWorkflowCoordinator` 输出统一字段结构，缺省值兜底，兼容 `paper_report` 与 `academic_report` 模板。
- InfluenceCalculator 输出结构对齐模板字段，避免 KeyError。

## 5) 复现与日志

- 在 CLI 入口统一设置 seed、时区、日志等级/路径；ExperimentManager meta 记录 git hash/依赖版本。

## 6) 校验与兜底

- 报告/流水线前增加必填字段校验（paper_id/title/year/abstract/url；dataset text/label）。
- CodeAnalysisAgent 无代码时跳过并写占位字段，LLM 失败写默认文案。

## 7) CLI 与命令整理

- 封装常用命令：download（会议+模式）、track（学者）、run-exp（实验）、render-report（meta+模板）。

### To-dos

- [x] 收敛 `agent_module/` 与 `agents/`，保留一套实现与入口；迁移或删除重复 Agent。
- [x] 拆分 `ResearchAgent`：会议抓取/下载节点 vs 单篇摘要+代码发现节点；其他 Agent 保持单一职责。
- [x] settings 增加 `mode`（academic/production）及数据源/模板/LLM 配置分支。
- [x] 实现 BaseDataSource（local/db/hybrid），在追踪/分析时统一走 DataSource；MindSpider 仅作为离线导出工具。
- [x] 对齐 Workflow 输出与模板字段，缺省值兜底，兼容 `paper_report` 与 `academic_report` 模板；InfluenceCalculator 输出结构对齐。
- [x] 在 CLI 入口统一设置 seed、时区、日志等级/路径；ExperimentManager meta 记录 git hash/依赖版本。
- [x] 报告/流水线前增加必填字段校验；CodeAnalysisAgent 无代码时跳过并写占位字段，LLM 失败写默认文案。
- [x] 整理 download（会议+模式）/track/run-exp/render-report 子命令。

