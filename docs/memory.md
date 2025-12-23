# Cross-Platform Memory Middleware (PaperBot)

目标：把来自不同大模型平台（ChatGPT / Gemini / Claude / OpenRouter 等）的聊天记录统一解析、提炼为“长期记忆”，并在后续对话里作为上下文注入，避免各平台记忆割裂。

## 核心概念

- **Source（来源）**：一次导入的聊天记录文件（会记录 `sha256`、平台、文件名、解析统计）。
- **Memory Item（记忆条目）**：从对话中提炼出的稳定信息，例如偏好、目标、约束、长期项目背景等。
- **Context（上下文块）**：把若干记忆条目格式化成可直接塞进 system prompt 的片段。

## 已实现能力

- 解析导入：
  - ChatGPT 导出（`conversations.json` 结构，尽量容忍分支/缺字段）
  - Gemini / 各类 API 日志（宽松 JSON：支持 `request.contents` / `response.candidates`、`messages`/`contents`/`history` 等常见结构）
  - 纯文本（`User:`/`Assistant:` 或中文 `我:`/`助手:` 前缀的松散格式）
  - 其它 JSON：尝试 `{"messages": [{"role": "...", "content": "..."}]}` 的通用结构
- 记忆提炼：
  - **默认：启发式规则**（离线可用，不依赖 API）
  - 可选：**LLM 抽取**（需要配置 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 等，失败自动回退规则）
- 检索与注入：
  - 关键词检索（按 query 匹配 content/tags）
  - Chat API 可选启用 `use_memory=true`，按 query 注入最相关的记忆上下文

## API

启动 API：

```bash
python -m uvicorn src.paperbot.api.main:app --reload --port 8000
```

### 1) 导入聊天记录并抽取记忆

`POST /api/memory/ingest`

- Query 参数：
  - `user_id`：记忆命名空间（建议用用户/团队唯一 id）
  - `platform`：平台提示（`chatgpt/gemini/claude/...`）
  - `use_llm`：是否启用 LLM 抽取（默认 `false`）
  - `redact`：是否自动脱敏（邮箱/电话，默认 `true`）
- Body：multipart file `file`

示例：

```bash
curl -F "file=@conversations.json" \
  "http://localhost:8000/api/memory/ingest?user_id=jerry&platform=chatgpt"
```

### 2) 获取可注入的记忆上下文块

`POST /api/memory/context`

```bash
curl -X POST "http://localhost:8000/api/memory/context" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"jerry","query":"我想做一个中间件把多个大模型的记忆合并","limit":8}'
```

返回字段 `context` 可直接拼进任意 LLM 的 system prompt。

### 3) 列出记忆条目

`GET /api/memory/list?user_id=jerry&limit=100`

## Chat API 中启用记忆

`POST /api/chat` body 增加：

- `user_id`：对应记忆命名空间
- `use_memory: true`：启用注入

这样 PaperBot 会先调用本地 memory store 检索，再把上下文附加为额外 system 消息。

## 数据库

默认 SQLite：`sqlite:///data/paperbot.db`（可用 `PAPERBOT_DB_URL` 覆盖）。

新增表：
- `memory_sources`
- `memory_items`

## 安全与隐私

- 默认启用基础脱敏（邮箱/电话）。
- 建议：导入前先手动删除敏感信息，或在接入生产时增加更强的 PII 检测/加密策略。
