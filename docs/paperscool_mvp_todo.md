# Papers.cool 单源检索 MVP TODO

目标：先做一个“最小可用版本（MVP）”，支持一次输入多个主题（例如：ICL压缩、ICL隐式偏置、KV Cache加速），在 **papers.cool** 单一源头完成检索，并返回可读的聚合结果。

## 0. 范围与边界

- [ ] 只接入 `https://papers.cool/`（不混入 Semantic Scholar / arXiv API）
- [ ] 默认同时检索 `arxiv` 与 `venue` 两个分支（同一站点内）
- [ ] 先不接 LLM 摘要，使用规则化摘要（可离线、稳定）
- [ ] 先不做数据库持久化，先返回 JSON 结果

## 1. 查询输入与标准化

- [ ] 定义输入结构：`queries: list[str]`
- [ ] 清洗规则：去空、去重、标准化空白
- [ ] 中文术语扩展（MVP内置）
  - `ICL压缩` -> `ICL compression`
  - `ICL隐式偏置` -> `ICL implicit bias`
  - `KV Cache加速` -> `KV cache acceleration`
- [ ] 每个主题输出 `normalized_query` 与 `raw_query`

**验收标准**
- 相同查询（大小写/空白差异）只检索一次
- 输出中可追踪原始词与标准化词

## 2. papers.cool 连接器

- [ ] 构造检索 URL
  - `/arxiv/search?query=...&highlight=1`
  - `/venue/search?query=...&highlight=1`
- [ ] 实现 HTML 解析（卡片 `.paper`）
- [ ] 抽取字段
  - `paper_id`, `title`, `url`, `source_branch`
  - `authors[]`, `subject_or_venue`, `published_at`
  - `summary/snippet`, `keywords[]`, `pdf_stars`, `kimi_stars`
- [ ] 解析失败时容错（跳过脏卡片，不中断整个查询）

**验收标准**
- fixture 测试可稳定解析 arxiv/venue 两类页面
- 至少返回标题、链接、摘要片段

## 3. 聚合与排序

- [ ] 将多个 query 的结果合并
- [ ] 去重策略（优先 URL，再退化到标题归一化）
- [ ] 规则打分（MVP）：
  - query 词命中数量
  - 关键词命中
  - 热度信号（`pdf_stars` + `kimi_stars`）
  - 时间轻微加权（新年份优先）
- [ ] 结果裁剪（每个主题 top-k）

**验收标准**
- 同一论文出现在 arxiv/venue 或多个 query 下只保留一份主记录
- 每条结果给出 `score` 与 `matched_keywords`

## 4. 输出汇总（无 LLM）

- [ ] 定义统一输出 JSON schema
  - `source`, `fetched_at`, `queries[]`, `items[]`, `summary`
- [ ] 每个 query 生成一句摘要
  - 命中数、top论文、主要关键词
- [ ] 全局摘要
  - 总论文数、去重后数量、最相关 top N

**验收标准**
- 不依赖外部模型也能得到可读总结
- 输出可直接喂给后续日报/订阅模块

## 5. 最小接口

- [ ] CLI：新增 `paperbot` 子命令（如 `topic-search`）
  - 输入多个 `--query`
  - 支持 `--top-k` 和 `--json`
- [ ] API：新增最小路由（如 `/api/research/paperscool/search`）
- [ ] 参数校验与错误码

**验收标准**
- 一条命令可跑通三主题检索并得到结构化输出
- API 可被 dashboard 或脚本调用

## 6. 测试与回归

- [ ] parser fixture 测试（arxiv + venue）
- [ ] 聚合去重/排序单测
- [ ] API/CLI smoke test（使用 mock fetch，避免网络依赖）

**验收标准**
- `pytest` 目标用例通过
- 核心逻辑离线可测

## 7. 交付节奏（按功能提交）

- [ ] Commit 1: `docs` - 增加本 TODO
- [ ] Commit 2: `feat` - papers.cool 连接器 + 解析测试
- [ ] Commit 3: `feat` - 聚合/去重/排序 + 测试
- [ ] Commit 4: `feat` - 汇总摘要生成 + 测试
- [ ] Commit 5: `feat` - CLI + API 接口 + smoke test

