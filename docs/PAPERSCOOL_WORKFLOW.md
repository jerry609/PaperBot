# Papers.cool Topic Workflow（MVP + V2）

本文档说明当前实现的主题检索工作流：围绕 `ICL压缩 / ICL隐式偏置 / KV Cache加速`，支持单次多主题检索、汇总、日报输出、定时执行与 UI 操作。

## 1. 端到端流程

```text
Input Queries
   -> Query Normalize
   -> Source Injection (papers_cool by default)
   -> Branch Search (arxiv + venue)
   -> Parse & Normalize Records
   -> Merge / Dedup / Score
   -> Query Summary + Global Summary
   -> (Optional) DailyPaper Markdown/JSON
   -> (Optional) Scheduler Cron + Feed Events
```

### 关键步骤

1. **查询规范化**
   - 内置中文映射：
     - `ICL压缩` -> `icl compression`
     - `ICL隐式偏置` -> `icl implicit bias`
     - `KV Cache加速` -> `kv cache acceleration`
   - 同义输入去重，避免重复请求。

2. **单源检索 + 分支策略**
   - Source：`papers_cool`
   - Branch：`arxiv` / `venue`
   - URL：`/{branch}/search?query=...&highlight=1&show=...`

3. **聚合**
   - 先按 URL 去重，后按归一化标题兜底去重。
   - 输出 `matched_keywords / matched_queries / score`。

4. **日报输出（DailyPaper）**
   - 支持 `markdown/json/both`。
   - 支持直接写盘。

5. **调度与信息流**
   - ARQ job 定时生成日报。
   - 将日报高亮项桥接到 feed 的 recommendation 事件。

## 2. 代码位置

- Source 注入层：`src/paperbot/application/workflows/topic_search_sources.py`
- 工作流编排：`src/paperbot/application/workflows/paperscool_topic_search.py`
- DailyPaper 报告：`src/paperbot/application/workflows/dailypaper.py`
- API 路由：`src/paperbot/api/routes/paperscool.py`
- CLI 命令：`src/paperbot/presentation/cli/main.py`
- Scheduler/ARQ：`src/paperbot/infrastructure/queue/arq_worker.py`
- Feed 桥接：`src/paperbot/workflows/feed.py`
- Web 页面：`web/src/app/workflows/page.tsx`

## 3. API 使用

### Topic Search

`POST /api/research/paperscool/search`

```json
{
  "queries": ["ICL压缩", "ICL隐式偏置", "KV Cache加速"],
  "sources": ["papers_cool"],
  "branches": ["arxiv", "venue"],
  "top_k_per_query": 5,
  "show_per_branch": 25
}
```

### DailyPaper

`POST /api/research/paperscool/daily`

```json
{
  "queries": ["ICL压缩", "ICL隐式偏置", "KV Cache加速"],
  "sources": ["papers_cool"],
  "branches": ["arxiv", "venue"],
  "top_k_per_query": 5,
  "show_per_branch": 25,
  "top_n": 10,
  "formats": ["both"],
  "save": true,
  "output_dir": "./reports/dailypaper"
}
```

## 4. CLI 使用

```bash
# 主题检索
python -m paperbot.presentation.cli.main topic-search \
  -q "ICL压缩" -q "ICL隐式偏置" -q "KV Cache加速" \
  --source papers_cool --branch arxiv --branch venue --json

# 生成日报
python -m paperbot.presentation.cli.main daily-paper \
  -q "ICL压缩" -q "ICL隐式偏置" -q "KV Cache加速" \
  --source papers_cool --format both --save --output-dir ./reports/dailypaper
```

## 5. 新源注入（扩展）

新增数据源时，实现 `TopicSearchSource` 协议并注册到 `TopicSearchSourceRegistry`：

```python
class MySource:
    name = "my_source"

    def search(self, *, query: str, branches: Sequence[str], show_per_branch: int):
        return [TopicSearchRecord(...)]

registry.register("my_source", MySource)
```

然后请求中传入：`"sources": ["my_source"]`。

## 6. Scheduler 配置（DailyPaper）

- `PAPERBOT_DAILYPAPER_ENABLED`：是否启用 daily cron（true/false）
- `PAPERBOT_DAILYPAPER_CRON_HOUR` / `PAPERBOT_DAILYPAPER_CRON_MINUTE`
- `PAPERBOT_DAILYPAPER_RUN_AT_STARTUP`
- `PAPERBOT_DAILYPAPER_QUERIES`（逗号分隔）
- `PAPERBOT_DAILYPAPER_SOURCES`（逗号分隔）
- `PAPERBOT_DAILYPAPER_BRANCHES`（逗号分隔）
- `PAPERBOT_DAILYPAPER_TOP_K` / `PAPERBOT_DAILYPAPER_SHOW` / `PAPERBOT_DAILYPAPER_TOP_N`
- `PAPERBOT_DAILYPAPER_TITLE`
- `PAPERBOT_DAILYPAPER_OUTPUT_DIR`

## 7. UI 设计说明

当前采用 **参数化工作流面板**（`/workflows`）而非 n8n/coze 式自由拖拽，原因：

- MVP 目标是“先可用 + 可验证 + 可运维”。
- 当前流程节点固定（Source -> Search -> Rank -> DailyPaper -> Schedule），参数面板已经覆盖主要操作。
- 后续如需拖拽，建议在现有节点模型基础上演进（保留当前 API 合约不变）。
