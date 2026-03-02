# OpenClaw 源码综述

> **版本**: v2.0 (综述版)
> **分析对象**: [openclaw/openclaw](https://github.com/openclaw/openclaw) `HEAD` (2026-02-27 clone)
> **代码规模**: 3,940 TypeScript 文件 / 726,408 行 / ~70 个子模块
> **技术栈**: TypeScript (核心) + Swift (iOS) + Kotlin (Android)
> **分析方法**: 5 个并行分析 Agent 读取 200+ 源文件, 含行号级引用

---

## 目录

1. [代码结构总览与模块度量](#1-代码结构总览与模块度量)
2. [Memory 模块 — 三层记忆系统](#2-memory-模块--三层记忆系统)
3. [Agents 模块 — Agent 运行时与工具体系](#3-agents-模块--agent-运行时与工具体系)
4. [Gateway 模块 — 常驻网关核心](#4-gateway-模块--常驻网关核心)
5. [Cron 模块 — 定时任务与隔离 Session](#5-cron-模块--定时任务与隔离-session)
6. [Hooks 模块 — 内部事件总线](#6-hooks-模块--内部事件总线)
7. [Channels 模块 — 多渠道插件体系](#7-channels-模块--多渠道插件体系)
8. [Compaction — 上下文压缩与记忆保全](#8-compaction--上下文压缩与记忆保全)
9. [Provider 层 — 多模型路由与缓存](#9-provider-层--多模型路由与缓存)
10. [Config 模块 — 配置体系与热重载](#10-config-模块--配置体系与热重载)
11. [Security 模块 — 安全审计与内容防护](#11-security-模块--安全审计与内容防护)
12. [Secrets 模块 — 凭证管理与加密](#12-secrets-模块--凭证管理与加密)
13. [Auto-Reply 模块 — 消息处理全链路](#13-auto-reply-模块--消息处理全链路)
14. [Infrastructure 模块 — 基础设施服务](#14-infrastructure-模块--基础设施服务)
15. [Browser 模块 — 浏览器自动化](#15-browser-模块--浏览器自动化)
16. [Media & TTS 模块 — 多媒体理解与语音合成](#16-media--tts-模块--多媒体理解与语音合成)
17. [Plugin 系统 — 插件架构与 SDK](#17-plugin-系统--插件架构与-sdk)
18. [ACP 模块 — Agent Communication Protocol](#18-acp-模块--agent-communication-protocol)
19. [CLI / Commands / TUI — 命令行与终端界面](#19-cli--commands--tui--命令行与终端界面)
20. [Sessions & Routing — 会话与路由](#20-sessions--routing--会话与路由)
21. [Daemon 模块 — 守护进程管理](#21-daemon-模块--守护进程管理)
22. [跨模块架构总结](#22-跨模块架构总结)

---

## 1. 代码结构总览与模块度量

### 1.1 目录结构

```
src/
├── memory/            # 82 文件, 10,123 行 — 三层记忆: 索引、检索、embedding、衰减、MMR
├── agents/            # 348 文件, 70,690 行 — Agent 运行时: system prompt、tools、skills、sandbox、compaction
│   ├── tools/         # 25+ 内置工具实现
│   ├── skills/        # Skill 发现/加载/过滤
│   ├── sandbox/       # Docker 沙箱
│   └── pi-extensions/ # Compaction safeguard 等扩展
├── gateway/           # 187 文件, 35,582 行 — WebSocket 网关: 服务端、chat loop、hooks HTTP、cron 集成
│   ├── server-methods/# 200+ WS RPC 方法
│   └── protocol/      # Zod schema 定义
├── cron/              # 36 文件, 6,555 行 — 定时任务: 调度、隔离 session、投递
│   ├── service/       # CronService 核心（ops/jobs/timer/state）
│   └── isolated-agent/# 隔离 cron agent 运行器
├── hooks/             # 23 文件, 3,809 行 — 内部事件总线 + 生命周期 hooks
│   └── bundled/       # 内置 hooks（boot-md, session-memory 等）
├── channels/          # 98 文件, 11,526 行 — 渠道插件体系（27 adapter 接口）
│   └── plugins/       # 插件类型定义 + 加载器
├── config/            # 120 文件, 23,555 行 — Zod-based 配置 schema + 热重载 + 迁移
├── security/          # 19 文件, 6,593 行 — 审计、ACL、内容防护、ReDoS 检测
├── secrets/           # 11 文件, 3,848 行 — 三种 provider (env/file/exec) 凭证管理
├── auto-reply/        # 179 文件, 30,915 行 — 消息处理全链路: 去重、分发、队列、TTS
├── infra/             # 199 文件, 34,601 行 — SSRF 防护、网关锁、心跳调度、成本分析
├── browser/           # 77 文件, 12,973 行 — Playwright/CDP 浏览器自动化
├── media/             # 18 文件, 2,594 行 — 媒体处理 + MIME 检测
├── media-understanding/# 35 文件, 4,016 行 — 音视频理解 (Whisper/Gemini/Vision)
├── tts/               # 2 文件, 1,620 行 — 三 Provider TTS (Edge/ElevenLabs/OpenAI)
├── plugins/           # 36 文件, 7,515 行 — 插件注册表、加载器、生命周期
├── plugin-sdk/        # 25 文件, 2,199 行 — 插件开发 SDK 公共 API
├── acp/               # 30 文件, 5,276 行 — Agent Communication Protocol (IDE 集成)
├── cli/               # 174 文件, 25,004 行 — CLI 入口、懒加载命令、快速路由
├── commands/          # 214 文件, 37,038 行 — 35+ 顶层命令实现
├── tui/               # 28 文件, 5,161 行 — 终端 UI (Ink 风格)
├── web/               # 47 文件, 6,424 行 — WhatsApp Web 渠道集成
├── sessions/          # 7 文件, 487 行 — Session 身份、Provenance、SendPolicy
├── routing/           # 5 文件, 855 行 — 7 级路由绑定、身份链接
├── daemon/            # 28 文件, 3,898 行 — macOS/Linux/Windows 守护进程
├── providers/         # 6 文件, 612 行 — LLM provider 注册 + Copilot token 交换
├── telegram/          # 54 文件, 12,279 行 — Telegram 渠道实现
├── discord/           # 78 文件, 20,763 行 — Discord 渠道实现
├── slack/             # 59 文件, 8,759 行 — Slack 渠道实现
├── signal/            # 28 文件, 3,235 行 — Signal 渠道实现
├── imessage/          # 16 文件, 2,227 行 — iMessage 渠道实现
├── line/              # 43 文件, 5,603 行 — LINE 渠道实现
└── ...                # logging, markdown, pairing, process, shared, terminal, types, utils, wizard
```

### 1.2 模块规模 Top 10

| 模块 | 文件数 | 代码行 | 职责 |
|------|--------|--------|------|
| agents/ | 348 | 70,690 | Agent 运行时 + 工具 |
| commands/ | 214 | 37,038 | CLI 命令 |
| gateway/ | 187 | 35,582 | WebSocket 网关 |
| infra/ | 199 | 34,601 | 基础设施 |
| auto-reply/ | 179 | 30,915 | 消息处理 |
| cli/ | 174 | 25,004 | CLI 框架 |
| config/ | 120 | 23,555 | 配置体系 |
| discord/ | 78 | 20,763 | Discord 渠道 |
| browser/ | 77 | 12,973 | 浏览器自动化 |
| telegram/ | 54 | 12,279 | Telegram 渠道 |

---

## 2. Memory 模块 — 三层记忆系统

**路径**: `src/memory/` (82 文件, 核心 18 文件约 5,766 行)

### 2.1 类层次结构

三级继承分离关注点：

```
MemoryManagerSyncOps        (1,216 行) ← 数据库、文件监听、同步编排
  └── MemoryManagerEmbeddingOps  (807 行) ← embedding 生成、批量、缓存
        └── MemoryIndexManager   (640 行) ← 公共 API: search, readFile, status, close
```

### 2.2 实例管理 — `manager.ts` (640 行)

构造函数为 **private** (L141)。通过静态工厂 `MemoryIndexManager.get()` (L103-139) 创建：

1. 解析 `ResolvedMemorySearchConfig`
2. 计算缓存 key = `agentId + workspaceDir + settings`
3. 命中 `INDEX_CACHE` (模块级 `Map`) 则复用
4. 否则调 `createEmbeddingProvider()` 创建 provider，构造实例并缓存

**关键常量** (L33-37):
```typescript
const SNIPPET_MAX_CHARS = 700;
const VECTOR_TABLE = "chunks_vec";
const FTS_TABLE = "chunks_fts";
const EMBEDDING_CACHE_TABLE = "embedding_cache";
const BATCH_FAILURE_LIMIT = 2;
```

### 2.3 搜索流水线 — `search()` (L207-293)

```
query
  │
  ├─ warmSession() ← 每 session 首次同步
  │
  ├─ if dirty && sync.onSearch → 非阻塞后台同步
  │
  ├─ 解析 minScore, maxResults, candidates (= maxResults × candidateMultiplier, 上限 200)
  │
  ├─── [无 embedding provider] ─→ FTS-only 模式 (L234-267)
  │       extractKeywords() → 多关键词并行搜索 → 去重取最高分
  │
  ├─── [有 provider + hybrid 启用] ─→ Hybrid 模式 (L269-293)
  │       FTS 关键词搜索 ∥ 向量搜索 → mergeHybridResults()
  │
  └─── [有 provider + hybrid 禁用] ─→ Vector-only 模式 (L279-281)
```

### 2.4 SQLite Schema — `memory-schema.ts` (96 行)

| 表 | 类型 | 核心列 |
|----|------|--------|
| `meta` | 常规 | `key TEXT PK`, `value TEXT` |
| `files` | 常规 | `path TEXT PK`, `source TEXT`, `hash TEXT`, `mtime INT`, `size INT` |
| `chunks` | 常规 | `id TEXT PK`, `path`, `source`, `start_line`, `end_line`, `hash`, `model`, `text`, `embedding TEXT (JSON)`, `updated_at` |
| `chunks_fts` | FTS5 虚拟表 | `text` (索引), `id/path/source/model/start_line/end_line` (UNINDEXED) |
| `chunks_vec` | sqlite-vec 虚拟表 | 二进制 float 向量 |
| `embedding_cache` | 常规 | `(provider, model, provider_key, hash)` 复合 PK, `embedding TEXT`, `dims INT` |

**索引**: `idx_chunks_path`, `idx_chunks_source`, `idx_embedding_cache_updated_at`

**Schema 迁移**: `ensureColumn()` (L85-95) 用 `PRAGMA table_info()` + `ALTER TABLE ADD COLUMN` 做增量升级。

### 2.5 Hybrid Search 融合 — `hybrid.ts` (149 行)

**FTS Query 构建** — `buildFtsQuery()` (L33-44):
```typescript
// 输入 "hello world" → 输出 '"hello" AND "world"'
// Unicode-aware: /[\p{L}\p{N}_]+/gu
```

**BM25 Rank→Score** — `bm25RankToScore()` (L46-49):
```typescript
return 1 / (1 + Math.max(0, rank));  // rank 0 = 1.0, rank 9 = 0.1
```

**融合算法** — `mergeHybridResults()` (L51-149):

1. **Union by chunk ID** (L73-119): `Map<chunkId, {vectorScore, textScore}>`, 非 Intersection
2. **加权线性组合** (L121-131):
   ```
   score = vectorWeight × vectorScore + textWeight × textScore
   ```
3. **Temporal Decay** (L133-139): `applyTemporalDecayToHybridResults()`
4. **降序排序** (L140)
5. **MMR 去重** (L143-146): `applyMMRToHybridResults()`

### 2.6 Temporal Decay — `temporal-decay.ts` (167 行)

**衰减函数**:
```typescript
lambda = Math.LN2 / halfLifeDays;        // L17-22
multiplier = Math.exp(-lambda * ageDays); // L24-34
decayedScore = score * multiplier;        // L36-42
```

默认 `halfLifeDays = 30` (默认关闭, `enabled: false`)。

**日期提取优先级** — `extractTimestamp()` (L82-114):
1. 文件路径匹配 `memory/YYYY-MM-DD.md` → 从路径取日期
2. Evergreen 路径 (`MEMORY.md`, `memory/` 下非日期命名文件) → **不衰减** (返回 null)
3. 其他文件 → fallback 到 `mtime`

**Evergreen 判定** — `isEvergreenMemoryPath()` (L71-80): `MEMORY.md` 和 `memory/*.md` (非日期格式) 永不衰减。

### 2.7 MMR 去重 — `mmr.ts` (214 行)

**公式** (L101, Carbonell & Goldstein 1998):
```
MMR = λ × relevance − (1 − λ) × max_similarity_to_selected
```

默认 `lambda = 0.7`（0 = 最大多样性, 1 = 最大相关性）。

**相似度**: Jaccard on token sets — `tokenize()` (L32-35) 提取 `[a-z0-9_]+`; `jaccardSimilarity()` (L41-61) = `|A∩B| / |A∪B|`。

**算法** — `mmrRerank()` (L116-183):
1. 预分词，缓存所有 token sets
2. Score 归一化到 [0,1]
3. 贪心迭代选择: 每轮选 MMR 最高的候选, tie-break 取原始分更高者
4. λ 钳位到 [0, 1]

### 2.8 向量搜索 — `manager-search.ts` (191 行)

**sqlite-vec 路径** (L34-69):
```sql
SELECT c.*, vec_distance_cosine(v.embedding, ?) AS dist
FROM chunks_vec v JOIN chunks c ON v.rowid = c.rowid
WHERE c.model = ? AND c.source IN (?)
ORDER BY dist ASC LIMIT ?
-- score = 1 - dist
```

**内存 fallback** (L71-93): 加载所有 chunks → JSON 解析 embedding → `cosineSimilarity()` → 排序。

**关键词搜索** (L136-191):
```sql
SELECT *, bm25(chunks_fts) AS rank
FROM chunks_fts WHERE chunks_fts MATCH ?
ORDER BY rank ASC LIMIT ?
```

### 2.9 Embedding Provider — `embeddings.ts` (296 行)

**5 个 Provider**:
| Provider | 默认模型 |
|----------|----------|
| **local** (`node-llama-cpp`) | `embeddinggemma-300m-qat-Q8_0` (GGUF) |
| **openai** | `text-embedding-3-small` |
| **gemini** | Gemini embeddings |
| **voyage** | Voyage AI |
| **mistral** | Mistral embeddings |

**"auto" 模式解析** (L174-211):
1. 磁盘上有本地模型 → 优先 local
2. 依次尝试 `["openai", "gemini", "voyage", "mistral"]`
3. API key 缺失 → 跳过; 网络错误 → 直接 fatal
4. 全部无 key → `provider: null` (FTS-only 模式)

**批量重试**: 指数退避 + jitter, base 500ms, max 8000ms, 3 次重试, 连续失败 2 次后永久禁用 batch mode。

### 2.10 同步编排 — `manager-sync-ops.ts` (1,216 行)

**全量重索引触发条件** (L859-868): `force` 标记、无已有元数据、Model/provider 变更、配置的 sources 变更、chunk token/overlap 设置变更、向量可用但无 `vectorDims`。

**原子重索引** — `runSafeReindex()` (L996-1103):
1. 创建临时 DB `{dbPath}.tmp-{uuid}`
2. 从原 DB 种入 embedding cache
3. 在临时 DB 同步所有文件
4. **原子替换** (swap) 临时 DB → 原 DB
5. 失败则从 backup 恢复

**文件监听** (L356-398): `chokidar` 监听 `MEMORY.md`, `memory.md`, `memory/**/*.md`, 加上 `extraPaths`。变更后 debounce → `dirty = true` → 调度同步。

**Session Delta 追踪** (L400-560): 监听 `onSessionTranscriptUpdate`, debounce 5s, 按字节/消息数阈值增量同步, 64KB chunk 读取。

### 2.11 多语言查询扩展 — `query-expansion.ts` (806 行)

支持 **8 种语言** 的停用词和分词: English, Spanish, Portuguese, Arabic, Korean (助词剥离), Japanese (假名+汉字 bigram), Chinese (字 unigram + bigram)。

### 2.12 Backend Fallback — `search-manager.ts` (238 行)

`FallbackMemoryManager`: 首选 QMD backend; 首次 `search()` 失败后 **永久切换** 到内置 `MemoryIndexManager`。

---

## 3. Agents 模块 — Agent 运行时与工具体系

**路径**: `src/agents/` (348 文件, 70,690 行)

### 3.1 System Prompt 装配 — `system-prompt.ts` (704 行)

`buildAgentSystemPrompt()` (L189-664) 在 **每次 Agent turn** 动态装配 system prompt，共 **21 个 section**:

| # | Section | 行号 | 说明 |
|---|---------|------|------|
| 1 | Identity | L419 | `"You are a personal assistant running inside OpenClaw."` |
| 2 | Tooling | L421-453 | 可用工具列表 + 摘要, 按 `toolOrder` 排序 |
| 3 | Tool Call Style | L456-461 | 叙述风格指导 |
| 4 | Safety | L390-396 | 反权力追求、遵守 stop/pause、不操纵 |
| 5 | CLI Quick Ref | L464-471 | OpenClaw 子命令 |
| 6 | Skills | L473 | `buildSkillsSection()` — 按需读取 |
| 7 | Memory Recall | L474 | `memory_search`/`memory_get` 指引 |
| 8 | Self-Update | L476-486 | 仅 full mode |
| 9 | Model Aliases | L489-498 | 仅 full mode |
| 10 | Workspace | L502-506 | 工作目录 + sandbox 指引 |
| 11 | Docs | L507 | 本地文档路径 |
| 12 | Sandbox | L508-552 | 容器 workdir, browser bridge |
| 13 | User Identity | L553 | Owner numbers |
| 14 | Time | L554-556 | 仅时区 (利于缓存) |
| 15 | Bootstrap Files | L557-558 | SOUL.md, IDENTITY.md, AGENTS.md |
| 16 | Messaging | L561-568 | Session 路由 |
| 17 | Reasoning | L601-603 | `<think>...</think>` tags |
| 18 | Project Context | L605-625 | 遍历 contextFiles[] |
| 19 | Silent Replies | L628-643 | `SILENT_REPLY_TOKEN` 规则 |
| 20 | Heartbeats | L646-656 | `HEARTBEAT_OK` ack |
| 21 | Runtime Footer | L658-664 | Agent ID, host, OS, model, channel |

**Prompt Mode**: `"full"` (主 agent) / `"minimal"` (子 agent, 跳过 memory/docs/self-update) / `"none"` (仅 identity)。

### 3.2 Agent 生命周期 — `pi-embedded-runner/run.ts`

**入口**: `runEmbeddedPiAgent()` (L192)

1. 解析 session lane + global lane
2. Auth Profile 解析: 按优先级轮转 API key, 支持 cooldown/failure 追踪
3. Context Window Guard: 硬最小 16,000 tokens, 警告阈值 32,000 tokens
4. **重试循环**: 基础 24 次, 每个 auth profile 加 8 次, 最小 32, 最大 160 次
5. **错误分类与恢复**:
   - `isAuthAssistantError` → 轮转到下一个 auth profile
   - `isBillingAssistantError` → 格式化计费错误
   - `isLikelyContextOverflowError` → 触发 compaction
   - `isRateLimitAssistantError` → 退避
   - `isFailoverAssistantError` → fallback 到备选模型

**Active Run 追踪** — `runs.ts`:
```typescript
const ACTIVE_EMBEDDED_RUNS = new Map<string, EmbeddedPiQueueHandle>();
// queueMessage, isStreaming, isCompacting, abort
```

### 3.3 工具体系 — `tool-catalog.ts` (326 行) + 工具实现

**25+ 核心工具** 按 section 分组:

| Section | 工具 |
|---------|------|
| **fs** | `read`, `write`, `edit`, `apply_patch` |
| **runtime** | `exec`, `process` |
| **web** | `web_search`, `web_fetch` |
| **memory** | `memory_search`, `memory_get` |
| **sessions** | `sessions_list`, `sessions_history`, `sessions_send`, `sessions_spawn` |
| **ui** | `browser`, `canvas` |
| **messaging** | `message` |
| **automation** | `cron`, `gateway` |
| **agents** | `subagents` |
| **media** | `image`, `tts` |

#### 3.3.1 Exec Tool — `bash-tools.exec.ts`

- **Shell-bleed 检测** (L55-149): 解析命令字符串，检测 Python/JS 文件包含裸 shell 变量语法 (`$HOME`)，引导使用 `os.environ.get()` 或 `process.env[]`
- **三种执行宿主**: Local (直接 shell), Gateway (代理), Node (远程节点)
- **审批工作流**: `bash-tools.exec-approval-request.ts` 管理工具调用审批

#### 3.3.2 Web Search — `tools/web-search.ts`

**5 个搜索 Provider**: Brave, Perplexity, Grok, Gemini, Kimi

每个 provider 有独立实现路径，搜索结果经 `wrapWebContent()` 安全包装防注入。缓存层: 15 分钟 TTL。

#### 3.3.3 Web Fetch — `tools/web-fetch.ts`

```typescript
DEFAULT_FETCH_MAX_CHARS = 50_000
DEFAULT_FETCH_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_FETCH_MAX_REDIRECTS = 3
```

支持 Firecrawl 集成 (可选高级抓取)。所有请求经 SSRF 防护 (`fetchWithWebToolsNetworkGuard`)。

#### 3.3.4 Memory Tools — `tools/memory-tool.ts`

- `createMemorySearchTool()`: 语义搜索 `MEMORY.md` + `memory/*.md`，结果附 `path#L{start}-L{end}` 引用，受 `maxInjectedChars` 预算限制
- `createMemoryGetTool()`: 安全片段读取器，按行号范围读取

#### 3.3.5 Browser Tool — `tools/browser-tool.ts`

支持操作: `browserStart`, `browserStop`, `browserSnapshot`, `browserTabs`, `browserAct`, `browserNavigate`, `browserScreenshotAction`, `browserPdfSave`。可路由到远程 Node。所有浏览器提取内容经 `wrapExternalContent()` 防注入。

#### 3.3.6 Message Tool — `tools/message-tool.ts`

跨渠道富消息投递: Discord (按钮/菜单/embeds), Telegram (内联按钮/引用), 通用 (send/reply/thread-reply/broadcast)。

### 3.4 Subagent 系统 — `subagent-spawn.ts` (551 行)

**Spawn 流程** (L166-550):

1. **深度守卫**: `callerDepth >= maxSpawnDepth` → 拒绝
2. **并发守卫**: `maxChildrenPerAgent` (默认 5)
3. **Agent 授权**: 检查 `subagents.allowAgents` 列表 (支持 `*`)
4. **Session 创建**: `agent:${targetAgentId}:subagent:${crypto.randomUUID()}`
5. **Gateway 调用**: 3 次 `sessions.patch` (spawnDepth, model, thinking)
6. **Thread 绑定**: `ensureThreadBindingForSubagentSpawn()`
7. **System Prompt**: `buildSubagentSystemPrompt()` 注入请求者上下文
8. **执行**: `callGateway({ method: "agent" })`, AGENT_LANE_SUBAGENT lane
9. **注册**: `registerSubagentRun()` 记录到注册表
10. **生命周期 Hook**: 触发 `subagent_spawned`

**Spawn 模式**: `"run"` (一次性, 自动公告结果) / `"session"` (持久线程, 支持跟进)

### 3.5 Skills 系统 — `skills/workspace.ts` (760 行)

**6 层发现来源** (优先级从低到高): extraDirs < Bundled < Managed < Personal < Project < Workspace

**限制**:
```typescript
maxCandidatesPerRoot: 300
maxSkillsLoadedPerSource: 200
maxSkillsInPrompt: 150
maxSkillsPromptChars: 30_000
maxSkillFileBytes: 256_000
```

**Prompt 生成**: 二分搜索截断到 `maxSkillsPromptChars`。

### 3.6 Tool Policy — `tool-policy.ts`

- **Owner-only 强制**: 非 owner 调用受限工具抛 "Tool restricted to owner senders."
- **Allow/Deny 策略**: `ToolPolicyLike` 支持 `allow[]`/`deny[]` + 组展开 (`group:plugins`)
- **Sandbox 工具策略**: 沙箱内允许 fs/runtime/memory/sessions, 禁止 browser/canvas/cron/gateway

### 3.7 Context Window Guard — `context-window-guard.ts`

```typescript
CONTEXT_WINDOW_HARD_MIN_TOKENS = 16_000
CONTEXT_WINDOW_WARN_BELOW_TOKENS = 32_000
```

解析链: config override → model reported → `agents.defaults.contextTokens` cap → fallback。

---

## 4. Gateway 模块 — 常驻网关核心

**路径**: `src/gateway/` (187 文件, 35,582 行)

### 4.1 启动序列 — `server.impl.ts` (935 行)

`startGatewayServer(port=18789, opts)` 的完整 **21 步启动序列**:

| 步骤 | 行号 | 操作 |
|------|------|------|
| 1 | L212-233 | Config 校验, 遗留格式迁移 |
| 2 | L248-260 | Plugin 自动启用 (env vars) |
| 3 | L262-371 | Secrets 激活 (锁 + 级联) |
| 4 | L347-365 | Auth bootstrap (生成 gateway token) |
| 5 | L376-379 | SIGUSR1 重启策略 |
| 6 | L380 | Subagent registry init |
| 7 | L383-401 | 加载 plugins, 合并 channel + base methods |
| 8 | L403-428 | Runtime config 解析 |
| 9 | L494-518 | 创建 runtime state |
| 10 | L520-538 | Node registry (IoT/mobile) |
| 11 | L540-545 | Cron service 构建 |
| 12 | L547-553 | Channel manager 启动 |
| 13 | L555-570 | Discovery (mDNS/Bonjour, Tailscale) |
| 14 | L572-595 | Skills remote registry (debounced 30s) |
| 15 | L620-633 | Agent event handler 订阅 |
| 16 | L641-646 | Heartbeat runner |
| 17 | L658 | Cron start |
| 18 | L662-673 | Delivery recovery (crash 恢复) |
| 19 | L696-768 | 挂载所有 WS handlers |
| 20 | L814-822 | 触发 `gateway_start` hook |
| 21 | L824-884 | Config reloader (watch openclaw.json) |

### 4.2 WS RPC 方法注册 — `server-methods.ts`

```typescript
export const coreGatewayHandlers = {
  ...connectHandlers,      // 连接握手
  ...chatHandlers,         // chat send/abort/history/inject
  ...sessionsHandlers,     // session CRUD
  ...agentHandlers,        // agent run/wait
  ...nodeHandlers,         // node pairing/invoke
  ...channelsHandlers,     // channel status/logout
  ...configHandlers,       // config get/set/apply/patch
  ...cronHandlers,         // cron CRUD
  ...modelsHandlers,       // model catalog
  ...skillsHandlers,       // skills listing
  ...browserHandlers,      // browser operations
  ...sendHandlers,         // outbound delivery
  ...usageHandlers,        // usage stats
  ...ttsHandlers,          // text-to-speech
  // ...还有 health, wizard, push, talk, web, system, update, device, logs 等
};
```

**请求分发** (L97-149):
1. `authorizeGatewayMethod()` 检查 role + scopes
2. 控制面写方法 (`config.apply`, `config.patch`, `update.run`) 限流: 3 次/60s
3. Handler 查找: extra handlers 优先, 再查 core handlers

### 4.3 关键 RPC 方法

#### `chat.send`
- **输入清洗**: NFC 归一化, 去 null 字节, 移除禁止控制字符
- **历史清洗**: 截断文本 12,000 字符, 去 thinking 签名, 去 base64 图片
- **分发**: 通过 `dispatchInboundMessage()` 进入 auto-reply 系统

#### `sessions.*`
- **sessions.list**: 从 store 列出, 支持 derived titles + last message preview
- **sessions.patch**: 15+ 可 patch 字段 (model, thinkingLevel, verboseLevel, execHost, sendPolicy)
- **sessions.reset**: 归档 transcript, 重置状态, 解绑 thread
- **sessions.compact**: 触发 context compaction

#### `agent.*`
- **agent** (主方法): 接收消息, 解析 session, 通过 auto-reply 分发。支持 `/new`, `/reset`
- **agent.wait**: 阻塞等待 agent run 完成

### 4.4 协议定义 — `protocol/schema/frames.ts`

**三种帧类型** (discriminated union on `type`):
```typescript
RequestFrame:  { type: "req",   id, method, params? }
ResponseFrame: { type: "res",   id, ok, payload?, error? }
EventFrame:    { type: "event", event, payload?, seq?, stateVersion? }
```

**连接握手** — `ConnectParams`: protocol version 协商, client identification, 设备签名, auth credentials, capabilities。

**Hello 响应** — `HelloOk`: 协商后 protocol version, server version, feature advertisement, full state snapshot, canvas host URL, auth result, policy limits。

### 4.5 Auth 系统 — `auth.ts` (488 行)

**Auth 模式**: `"none" | "token" | "password" | "trusted-proxy"`

**`authorizeGatewayConnect()`** (L364-469) 多策略认证:
1. **Trusted-proxy**: 验证远程地址 + 可信代理头部
2. **Tailscale**: whois 查找 + header 三方验证
3. **Token/Password**: 常时间比较 (`safeEqualSecret()`)
4. **Rate limiting**: 按 IP 追踪失败次数

### 4.6 Chat 事件处理 — `server-chat.ts` (501 行)

- **ChatRunRegistry**: 队列式 run 追踪 (per-session queue)
- **Delta 节流**: 150ms 最小间隔
- **Heartbeat 隐藏**: 根据 `resolveHeartbeatVisibility()` 配置压制心跳输出
- **安全**: strip heartbeat tokens, 压制 silent replies

---

## 5. Cron 模块 — 定时任务与隔离 Session

**路径**: `src/cron/` (36 文件, 6,555 行)

### 5.1 调度类型 — `types.ts` (144 行)

```typescript
type CronSchedule =
  | { kind: "at", at: string }           // 一次性绝对时间
  | { kind: "every", ms: number }         // 间隔
  | { kind: "cron", expression: string }  // cron 表达式 + 时区 + stagger
```

**Session 目标**: `"main"` (注入主 session) | `"isolated"` (一次性隔离 session)
**载荷**: `systemEvent` (文本注入) | `agentTurn` (完整 agent turn)

### 5.2 两阶段执行 — `ops.ts` (458 行)

`run()` (L336-451):
1. **加锁阶段**: 设置 `runningAtMs`, 持久化, 释放锁
2. **无锁执行**: 在锁外运行, 保持 service 对读请求的响应能力
3. **重新加锁**: 应用结果, 处理 `deleteAfterRun`, 重算 nextRunAtMs

**防雷群**: SHA-256(jobId) % stagger window — 确定性散列到不同时间偏移。

**安全阈值**:
- `MAX_SCHEDULE_ERRORS = 3` → 自动 disable
- `STUCK_RUN_MS = 2 * 60 * 60 * 1000` (2 小时卡死检测)
- `MAX_TIMER_DELAY_MS = 60_000`, `MIN_REFIRE_GAP_MS = 2_000`

### 5.3 隔离 Agent — `isolated-agent/run.ts`

`runCronIsolatedAgentTurn()` (L90):
1. 解析 agent config, workspace, model catalog
2. `resolveCronSession()` — 评估新鲜度, stale 则新建
3. `resolveDeliveryTarget()` — 解析投递目标
4. 运行 agent (CLI or embedded Pi) + model fallback
5. `dispatchCronDelivery()` — 投递结果

### 5.4 投递路由 — `delivery-dispatch.ts` (437 行)

- **direct**: 结构化内容 → `deliverOutboundPayloads()`
- **announce**: 文本 → 主 session system-message 注入 → `runSubagentAnnounceFlow()`

---

## 6. Hooks 模块 — 内部事件总线

**路径**: `src/hooks/` (23 文件, 3,809 行)

### 6.1 事件类型

```typescript
type InternalHookEventType = "command" | "session" | "agent" | "gateway" | "message"
```

| 事件 | Action | 用途 |
|------|--------|------|
| `agent:bootstrap` | `"bootstrap"` | 修改注入文件 |
| `gateway:startup` | `"startup"` | 启动时逻辑 |
| `command:new` | `"new"` | 新会话事件 |
| `command:reset` | `"reset"` | 会话重置 |
| `message:received` | `"received"` | 消息入站 |
| `message:sent` | `"sent"` | 消息出站 |

### 6.2 两级分发 — `internal-hooks.ts` (285 行)

`triggerInternalHook(event)` (L192): **同时** 触发通用类型 (如 `"command"`) 和具体 action (如 `"command:new"`)。错误被 catch + log, 不阻止后续。

### 6.3 发现与加载 — `loader.ts` + `workspace.ts`

**来源优先级**: `extra < bundled < managed < workspace`

**发现流程**: 扫描 `HOOK.md` → 定位 handler → 解析 frontmatter → 资格检查 (OS/binary/env) → **路径边界校验** → 动态 import → 注册

### 6.4 内置 Hooks

| Hook | 事件 | 功能 |
|------|------|------|
| **boot-md** | `gateway:startup` | 启动时执行 `BOOT.md` checklist |
| **session-memory** | `command:new/reset` | 保存记忆到 `memory/<date>-<slug>.md` (LLM 生成 slug) |
| **bootstrap-extra-files** | `agent:bootstrap` | 加载额外 bootstrap 文件 |
| **command-logger** | `command` | JSON 日志到 `logs/commands.log` |

---

## 7. Channels 模块 — 多渠道插件体系

**路径**: `src/channels/` (98 文件, 11,526 行)

### 7.1 渠道注册 — `registry.ts` (190 行)

`CHAT_CHANNEL_ORDER`: `telegram, whatsapp, discord, irc, googlechat, slack, signal, imessage`

> **注**: 代码库中 **未找到** Feishu/Lark 原生实现。

### 7.2 Plugin 接口 — `types.plugin.ts`

`ChannelPlugin` 定义了 **27 个可选 adapter 槽位**:

| Adapter | 职责 |
|---------|------|
| `config` (必需) | 账号列表/解析 |
| `setup` | 账号验证 |
| `pairing` | 设备配对 |
| `security` | DM 策略, 安全警告 |
| `groups` | 群组 mention/intro |
| `outbound` | 消息发送 (`direct/gateway/hybrid`) |
| `status` | 探测/审计/快照 |
| `gateway` | 启停账号, QR 登录 |
| `streaming` | 流式回复合并 |
| `threading` | 线程/回复模式 |
| `directory` | peer/group 列表 |
| `actions` | send/edit/unsend/react |
| `heartbeat` | 就绪检查 |
| ...等 | 共 27 个 |

### 7.3 Dock 系统 — `dock.ts` (636 行)

**ChannelDock**: 轻量级元数据/行为层, 与重量级 plugin 实现分离。`buildDockFromPlugin()` 从 plugin 提取 dock 兼容字段。

### 7.4 渠道能力对比

| 特性 | Telegram | Discord | WhatsApp | Slack | IRC | Signal |
|------|----------|---------|----------|-------|-----|--------|
| Chat Types | direct, group, channel, thread | direct, channel, thread | direct, group | direct, channel, thread | direct, group | direct, group |
| Reactions | - | yes | yes | yes | - | yes |
| Commands | yes | yes | - | yes | - | - |
| Threads | - | yes | - | yes | - | - |
| Block Streaming | yes | - | - | - | yes | - |
| Text Limit | 4000 | 2000 | 4000 | 4000 | 350 | 4000 |

### 7.5 Discord 实现示例

**Dock** (`dock.ts` L321-363): capabilities 含 `polls, reactions, media, nativeCommands, threads`, textChunkLimit=2000。

**Outbound** (`plugins/outbound/discord.ts`): Webhook 优先 → 降级到直接发送。支持 persona (username + avatar)。

**Agent Tools** (`agents/tools/discord-actions*.ts`, ~50KB): 5 个文件覆盖 guild 管理、消息操作、审核、存在状态。

### 7.6 Telegram 实现示例

**Dock** (`dock.ts` L234-277): textChunkLimit=4000, blockStreaming=true。

**Threading 细节**: 只使用 `MessageThreadId` (topic ID), 不使用 `ReplyToId` (避免 DM 中的 `invalid message_thread_id` 错误)。

**HTML 格式化**: `markdownToTelegramHtmlChunks()`, textMode="html"。

---

## 8. Compaction — 上下文压缩与记忆保全

### 8.1 核心算法 — `agents/compaction.ts` (454 行)

| 函数 | 算法 |
|------|------|
| `estimateMessagesTokens()` | 逐消息估算, `stripToolResultDetails()` 安全过滤 |
| `splitMessagesByTokenShare()` | 按 token 预算等分 N 份 |
| `chunkMessagesByMaxTokens()` | `SAFETY_MARGIN = 1.2` (20% buffer) |
| `computeAdaptiveChunkRatio()` | 大消息 >10% window → 从 `BASE=0.4` 降至 `MIN=0.15` |
| `summarizeWithFallback()` | 完整摘要 → 排除超大消息 → 尺寸描述 fallback |
| `summarizeInStages()` | 分段 → 独立摘要 → merge prompt 合并 |
| `pruneHistoryForContextShare()` | `maxHistoryShare=0.5`, 最旧优先丢, 修复 tool_use/result 配对 |

### 8.2 Safeguard — `pi-extensions/compaction-safeguard.ts` (399 行)

Hook 到 `session_before_compact`:
1. 无真实对话 → **取消**
2. 收集文件操作 + tool 失败 (最多 8 条, 240 字截断)
3. 自适应 chunk ratio + `summarizeInStages()`
4. 追加 tool failure section + AGENTS.md "Red Lines"
5. **任何失败 → `{ cancel: true }`** (不冒损坏风险)

### 8.3 Pre-Compaction Memory Flush — `auto-reply/reply/memory-flush.ts` (145 行)

**触发条件**: `totalTokens > contextWindow - reserve - softThreshold` (默认 `softThreshold = 4000`)

**Flush Prompt**: "Store durable memories now (use `memory/YYYY-MM-DD.md`)... If nothing to store, reply with `NO_REPLY`."

### 8.4 Post-Compaction Context — `post-compaction-context.ts` (118 行)

从 `AGENTS.md` 提取 "Session Startup" + "Red Lines" sections (max 3000 字符), 注入系统消息: "Session was just compacted. Execute your Session Startup sequence now."

### 8.5 Compaction Timeout 安全 — `compaction-timeout.ts` (55 行)

超时 → fallback 到 **pre-compaction snapshot** (不丢数据)。

---

## 9. Provider 层 — 多模型路由与缓存

### 9.1 Model 解析 — 五级 fallback

1. `modelRegistry.find(provider, modelId)` — 内置注册表
2. `cfg.models.providers` — 用户自定义
3. `resolveForwardCompatModel()` — 版本兼容
4. OpenRouter passthrough — 任何 model ID 直通
5. Config provider fallback

### 9.2 StreamFn 装饰链 — `extra-params.ts` (741 行)

`applyExtraParamsToAgent()` 链式组合 10+ wrapper:

| Wrapper | 功能 |
|---------|------|
| Cache retention | `"short"` (5min Anthropic 直连) / `"long"` / `"none"` |
| Anthropic betas | `context1m` (Claude Opus 4/Sonnet 4 1M context) |
| OpenRouter cache_control | system message 注入 `cache_control: { type: "ephemeral" }` |
| Google thinking | sanitize `thinkingBudget`, map to `thinkingLevel` |
| Z.AI | `tool_stream=true` |
| 等 | SiliconFlow, Bedrock, OpenAI Responses... |

### 9.3 GitHub Copilot Token 交换 — `providers/github-copilot-token.ts` (138 行)

1. 检查本地缓存 `~/.openclaw/credentials/github-copilot.token.json`
2. 有效 (>5min) 则复用
3. 否则: `https://api.github.com/copilot_internal/v2/token`
4. 从 token 提取 `proxy-ep` → `api.*` 作为 base URL
5. 缓存到磁盘

---

## 10. Config 模块 — 配置体系与热重载

**路径**: `src/config/` (120 文件, 23,555 行)

### 10.1 Zod Schema — `zod-schema.ts`

`OpenClawSchema` (L131-813): ~40 个顶层 section 的 `z.object().strict()`:

`meta`, `env`, `wizard`, `diagnostics`, `logging`, `update`, `browser`, `ui`, `secrets`, `auth`, `acp`, `models`, `nodeHost`, `agents`, `tools`, `bindings`, `broadcast`, `audio`, `media`, `messages`, `commands`, `approvals`, `session`, `cron`, `hooks`, `web`, `channels`, `discovery`, `canvasHost`, `talk`, `gateway`, `memory`, `skills`, `plugins`

使用 `.superRefine()` (L783-813) 交叉验证 broadcast agent 引用。

### 10.2 Config I/O 管线

```
文件读取 → JSON5 解析 → $include 解析 (递归, 深度限制 10) →
${ENV_VAR} 替换 → 遗留格式迁移 → Zod 验证 → 默认值应用 → 返回类型化 OpenClawConfig
```

### 10.3 Include 系统 — `includes.ts` (347 行)

- `MAX_INCLUDE_DEPTH = 10`
- `MAX_INCLUDE_FILE_BYTES = 2MB`
- 循环引用检测 (visited set)
- 路径遍历阻止
- `deepMerge()`: 数组拼接, 对象递归合并

### 10.4 默认值 — `defaults.ts` (537 行)

**Model Aliases**:
```typescript
opus → anthropic/claude-opus-4-6
gpt → openai/gpt-5.2
```

**Compaction 默认模式**: `"safeguard"`

### 10.5 遗留迁移 — 三部分迁移系统 (~1,220 行)

- Part 1 (571 行): 10 次迁移 — bindings/providers/routing 重构
- Part 2 (427 行): agent model-config-v2, routing-agents-v2
- Part 3 (222 行): memorySearch, tools.bash→exec, identity→agents.list

### 10.6 JSON Merge Patch — `merge-patch.ts` (97 行)

RFC 7396 风格, 扩展支持 id-based 数组合并 (元素有 `id` 字段时 merge 而非替换)。

---

## 11. Security 模块 — 安全审计与内容防护

**路径**: `src/security/` (19 文件, 6,593 行)

### 11.1 审计引擎 — `audit.ts` (1,021 行)

`runSecurityAudit()` 编排 **20+ 审计收集器**:

**同步收集器** (`audit-extra.sync.ts`): 仅配置分析
**异步收集器** (`audit-extra.async.ts`): I/O-bound 检查

Gateway 配置检查 15+ 项: Bind 地址、Auth 方法、CORS、Tailscale funnel、Rate limiting、TLS。

### 11.2 外部内容防护 — `external-content.ts` (326 行)

`wrapExternalContent()` (L219-245):
1. 生成随机 boundary ID
2. 清洗伪造 boundary 标记
3. 折叠 Unicode 同形字 (全角 ASCII, 角括号变体)
4. 检测可疑模式 (prompt injection 指标)
5. 包裹 `[BEGIN_EXTERNAL_{id}]...[END_EXTERNAL_{id}]`

**12 个可疑模式** (L17-30): `system:`, `<|endoftext|>`, `[INST]`, `@assistant` 等。

### 11.3 ReDoS 防护 — `safe-regex.ts` (152 行)

`hasNestedRepetition()`: 栈式解析器检测嵌套量词。`compileSafeRegex()`: 缓存编译 (max 256 条)。

### 11.4 DM 策略 — `dm-policy-shared.ts` (303 行)

`resolveDmGroupAccessDecision()`: 状态机解析 group/DM 策略: open → pairing → allowlist → disabled。

8 个 reason code: `ALLOWED_DEFAULT`, `BLOCKED_BY_POLICY`, `BLOCKED_PAIRING_REQUIRED` 等。

### 11.5 Skill 代码扫描 — `skill-scanner.ts` (427 行)

**静态分析规则**:
- `dangerous-exec`: child_process
- `dynamic-code-execution`: eval/new Function
- `crypto-mining`: stratum/coinhive/xmrig
- `potential-exfiltration`: readFile + fetch
- `obfuscated-code`: 长 hex/base64 序列
- `env-harvesting`: process.env + fetch

### 11.6 自动修复 — `fix.ts` (478 行)

`fixSecurityFootguns()`: 启用 `redactSensitive`, 设置限制性 group 策略, 修复文件权限 (Unix: `chmod 700/600`, Windows: `icacls`)。

---

## 12. Secrets 模块 — 凭证管理与加密

**路径**: `src/secrets/` (11 文件, 3,848 行)

### 12.1 三种 Provider

| Provider | 实现 | 安全措施 |
|----------|------|----------|
| **env** | `process.env[id]` | - |
| **file** | `fs.readFile()` | `assertSecurePath()` 验证权限/所有权/symlink |
| **exec** | 外部二进制 + JSON 协议 | stdin/stdout, `protocolVersion: 1` |

### 12.2 核心解析 — `resolve.ts` (715 行)

`resolveSecretRefValues()` (L604-676):
1. 按 provider type 分组所有 `SecretRef`
2. env refs → 读 `process.env`
3. file refs → `assertSecurePath()` → `fs.readFile()`
4. exec refs → spawn binary, JSON stdin `{protocolVersion:1, ids}` → JSON stdout `{values}`
5. 并发限制

`assertSecurePath()` (L99-179): 检查所有权、权限 (非 world/group 可读)、symlink 目标。

### 12.3 原子快照 — `runtime.ts` (427 行)

`prepareSecretsRuntimeSnapshot()` → 收集赋值 → 解析所有 refs → 返回快照
`activateSecretsRuntimeSnapshot()` → 原子写入 config + auth-store

### 12.4 审计 — `audit.ts` (756 行)

扫描: config 文件明文、auth-profiles 明文、.env 暴露、legacy auth.json、ref 可解析性、shadowing 检测。

审计码: `PLAINTEXT_FOUND`, `REF_UNRESOLVED`, `REF_SHADOWED`, `LEGACY_RESIDUE`。

---

## 13. Auto-Reply 模块 — 消息处理全链路

**路径**: `src/auto-reply/` (179 文件, 30,915 行)

### 13.1 架构总览

```
Inbound Layer: inbound-debounce.ts → 聚合快速消息
       ↓
Dispatch Layer: dispatch-from-config.ts → 去重, hooks, abort, send policy, ACP, 跨渠道路由, TTS
       ↓
Reply Generation: get-reply.ts → 解析 model/skills/directives/commands → runPreparedReply
       ↓
Delivery Layer: reply-dispatcher.ts → 序列化投递, reservation-based idle 检测, 人类延迟
       ↓
Queue Layer: queue/ → 6 种队列模式, drop policy, debounce, 跨渠道 drain
       ↓
Memory Layer: memory-flush.ts + post-compaction-context.ts → 保全 agent 身份
```

### 13.2 核心类型 — `types.ts` (86 行)

- `GetReplyOptions`: 主选项包, 含 15+ 回调 hooks (onAgentRunStart, onPartialReply, onToolResult, onModelSelected...)
- `ReplyPayload`: 通用回复单元: text, mediaUrl, replyToId, channelData, isReasoning
- `TypingPolicy`: `"auto" | "user_message" | "system_event" | "heartbeat"`

### 13.3 Dispatch Pipeline — `dispatch-from-config.ts` (581 行)

`dispatchReplyFromConfig()` 完整流水线:
1. 检查重复消息 (L155)
2. 触发 plugin hooks (`message_received`) (L182-243)
3. **跨渠道路由** (L252-260): 如果来源渠道与当前 surface 不同, 路由回原渠道
4. Fast abort `/stop` (L300-334)
5. Send policy 检查 (L338-358)
6. ACP 分发尝试 (L361-379)
7. Block reply + TTS 累积 (L384)
8. 调用 `getReplyFromConfig()` (L406-468)
9. 迭代 reply payloads, 应用 TTS, 路由投递 (L470-511)

### 13.4 Reply Dispatcher — `reply-dispatcher.ts` (242 行)

**序列化投递队列**, reservation-based idle 检测:
- pending counter 初始为 1 (reservation)
- `markComplete()` 通过 `Promise.resolve().then()` 延迟释放, 让 in-flight enqueue 完成
- 人类延迟: `800ms ~ 2500ms` 随机间隔 (block replies)

### 13.5 Queue 系统 — `queue/`

**6 种模式**: `steer | followup | collect | steer-backlog | interrupt | queue`

**Drop Policy**: `old | new | summarize`

**默认**: debounce 1000ms, cap 20, drop=summarize

### 13.6 命令注册 — `commands-registry.ts`

缓存式命令注册表, 支持多 text alias + 参数解析模式 + 作用域 (text/native/both)。

### 13.7 Thinking 级别 — `thinking.ts` (228 行)

```typescript
ThinkLevel: "off" | "minimal" | "low" | "medium" | "high" | "xhigh"
VerboseLevel: "off" | "on" | "full"
ElevatedLevel: "off" | "on" | "ask" | "full"
```

`XHIGH_MODEL_REFS`: `openai/gpt-5.2`, `openai-codex/gpt-5.3-codex` 等。

---

## 14. Infrastructure 模块 — 基础设施服务

**路径**: `src/infra/` (199 文件, 34,601 行)

### 14.1 SSRF 防护 — `net/ssrf.ts` (364 行)

**两阶段验证**:
1. Phase 1 (pre-DNS): 检查字面主机/IP
2. Phase 2 (post-DNS): 检查解析后地址

**DNS Pinning**: 创建 pinned lookup 函数, 返回预解析地址, round-robin 轮转。防 TOCTOU 攻击。

**阻止列表**: `localhost`, `localhost.localdomain`, `metadata.google.internal` + `.localhost`, `.local`, `.internal` 后缀。

### 14.2 Fetch Guard — `net/fetch-guard.ts` (219 行)

`fetchWithSsrFGuard()`: SSRF 安全 HTTP fetch + 手动重定向跟随:
1. 验证 URL scheme (http/https)
2. 解析 pinned hostname
3. 创建 dispatcher (环境代理或 pinned)
4. `redirect: "manual"` → 跨域重定向去敏感 header (authorization, cookie)
5. 重定向循环检测 (visited Set)

### 14.3 网关锁 — `gateway-lock.ts` (294 行)

文件式单例锁, 防多实例:
1. 排他文件创建 (`wx` flag)
2. 多策略存活检测: port probe → `kill(0)` → Linux `/proc/pid/stat` startTime → `/proc/pid/cmdline`
3. 默认超时 5s, 轮询 100ms, 过期 30s

### 14.4 System Events — `system-events.ts` (119 行)

Session 级临时事件队列 (max 20), 连续去重, 不持久化。事件前缀到下次 agent prompt, 然后 drain。

### 14.5 成本分析 — `session-cost-usage.ts` (1,017 行)

流式 JSONL 分析引擎:
- `readJsonlRecords()`: async generator 逐行解析
- `loadCostUsageSummary()`: 跨 session 聚合 + 日桶
- `loadSessionCostSummary()`: 单 session 详细: message 统计, tool 使用, model 分布, 延迟百分位 (p50/p90/p95/p99)

### 14.6 心跳调度 — `heartbeat-runner.ts` (1,213 行)

**多 Agent 心跳调度器**:
- 可配置 per-agent intervals
- HEARTBEAT.md 文件门控 (无内容则跳过)
- 24h 窗口重复抑制
- Transcript 修剪 (HEARTBEAT_OK 时)
- 活跃时间段限制
- 多渠道投递 (channel plugin 就绪检查)

### 14.7 重启哨兵 — `restart-sentinel.ts` (147 行)

文件式跨进程消息传递: 重启进程写哨兵 → 新进程消费 → 报告状态。投递上下文跨重启保留。

---

## 15. Browser 模块 — 浏览器自动化

**路径**: `src/browser/` (77 文件, 12,973 行)

### 15.1 架构

```
CDP Discovery (chrome.ts, cdp.ts)
  → 连接运行中的 Chrome 实例
    ↓
Session Management (pw-session.ts)
  → Playwright Browser 连接池, WeakMap per-page 状态
    ↓
Tool Implementation (pw-tools-core.*.ts)
  → navigate, click, type, snapshot, screenshot...
    ↓
HTTP Server (server.ts, routes/)
  → 作为 REST 端点暴露
    ↓
Client (client.ts)
  → 类型化 HTTP client
```

### 15.2 Session 管理 — `pw-session.ts` (500+ 行)

- `ConnectedBrowser`: Browser + CDP WS URL + disconnect callback
- `PageState` (WeakMap per-page): console messages, errors, network requests, snapshot ref cache
- `getPageForTargetId()`: 通过 CDP 连接, 按 targetId 查找 page, 缓存连接
- `refLocator()`: 将 snapshot ref ("e1") 解析为 Playwright Locator

### 15.3 交互工具 — `pw-tools-core.interactions.ts`

统一模式: resolve page → ensure state → resolve element → apply action → return result

- `navigateViaPlaywright()`: SSRF guard → `page.goto()` → 重定向后二次检查
- `clickViaPlaywright()`: 支持 button/modifiers/double-click
- `typeViaPlaywright()`: 支持 slow (pressSequentially) / fast (fill)
- `fillFormViaPlaywright()`: 批量填充 (textbox/checkbox/radio/combobox/slider)
- `evaluateViaPlaywright()`: 任意 JS 执行
- `armDialogViaPlaywright()`: 预注册 dialog handler

### 15.4 Snapshot-Ref-Action 循环

AI agent 的浏览器交互范式:
1. 拍 accessibility snapshot → 获得元素 refs (e1, e2...)
2. 用 ref 调用交互工具 (click, type...)
3. 重新 snapshot 观察结果

### 15.5 CDP-via-Playwright 混合

大多操作用 Playwright 高层 API; locale/timezone/device emulation fallback 到 CDP session commands。

---

## 16. Media & TTS 模块 — 多媒体理解与语音合成

### 16.1 Media Understanding — `media-understanding/runner.ts` (806 行)

**策略模式 + 回退链**: 多个 backend 按优先级尝试, 首次成功胜出。

**Auto-discovery**: 检测 API key + 本地二进制:
1. 当前 agent 的 active model (如有 vision 能力)
2. 本地 CLI: `whisper-cli`, `whisper`, `sherpa-onnx-offline`
3. Gemini CLI
4. API key 检测 (OPENAI_API_KEY 等)

**Provider**: OpenAI, Google, Anthropic + 本地 Whisper/Sherpa

### 16.2 TTS — `tts/tts.ts` (948 行)

**三 Provider**: Edge / ElevenLabs / OpenAI

**Directive 系统**: LLM 可在输出中嵌入 `[[tts:...]]` 控制:
- `[[tts:provider=openai]]` — 切换 provider
- `[[tts:voice=coral]]` — 切换声音
- `[[tts:stability=0.7 speed=1.2]]` — 调节参数
- `[[tts:text]]...[[/tts:text]]` — 自定义 TTS 文本

**自动模式**: `off | always | inbound | tagged`

**长文本处理**: 超过 maxLength → LLM 自动摘要 → TTS → 临时文件 → 5 分钟后清理

**OpenAI 14 声音**: alloy, ash, ballad, cedar, coral, echo, fable, juniper, marin, onyx, nova, sage, shimmer, verse

---

## 17. Plugin 系统 — 插件架构与 SDK

**路径**: `src/plugins/` (36 文件, 7,515 行) + `src/plugin-sdk/` (25 文件, 2,199 行)

### 17.1 Plugin 类型 — `types.ts` (764 行)

**24 个生命周期 Hook**: `before_model_resolve`, `before_prompt_build`, `before_agent_start`, `llm_input`, `llm_output`, `agent_end`, `before_compaction`, `after_compaction`, `before_reset`, `message_received`, `message_sending`, `message_sent`, `before_tool_call`, `after_tool_call`, `tool_result_persist`, `before_message_write`, `session_start`, `session_end`, `subagent_spawning`, `subagent_delivery_target`, `subagent_spawned`, `subagent_ended`, `gateway_start`, `gateway_stop`

**Plugin API** (`OpenClawPluginApi`): `registerTool`, `registerHook`, `registerHttpHandler`, `registerHttpRoute`, `registerChannel`, `registerGatewayMethod`, `registerCli`, `registerService`, `registerProvider`, `registerCommand`

### 17.2 加载管线 — `loader.ts` (726 行)

```
发现 (扫描 workspace/global/config-paths)
  → Manifest 解析
    → 边界文件检查 (openBoundaryFileSync, 防 symlink 穿越)
      → Jiti 动态加载 (TypeScript JIT 编译)
        → Config 校验 (JSON Schema)
          → register(api) 调用
            → 注册到 PluginRegistry
```

**安全**: 边界文件检查、provenance 追踪 (未安装来源的插件告警)、配置 JSON Schema 校验。

### 17.3 Plugin SDK — `plugin-sdk/index.ts` (597 行)

"Kitchen sink" 风格公共 API, 暴露:
- Channel adapters (Discord/Telegram/Slack/等)
- ACP runtime
- 安全工具 (SSRF, allowlist, DM policy)
- 媒体工具 (MIME, store, web media)
- 基础设施 (file locks, webhook, dedup, JSON store)

---

## 18. ACP 模块 — Agent Communication Protocol

**路径**: `src/acp/` (30 文件, 5,276 行)

### 18.1 概述

ACP = **Agent Client Protocol** — 开放协议, 用于 IDE/编辑器与 AI Agent 集成 (类似 LSP 之于语言服务器)。

### 18.2 架构: 协议桥接

```
ACP Client (IDE 扩展) ←→ stdin/stdout ndjson ←→ AcpGatewayAgent ←→ WebSocket ←→ Gateway
```

### 18.3 Translator — `translator.ts` (499 行)

`AcpGatewayAgent` 实现 `Agent` 接口:

- **initialize()** (L122): 返回能力: loadSession, image prompts, embedded context
- **newSession()** (L145): 限流 → UUID → 解析 session key → 创建
- **prompt()** (L252): 提取 text + image → 2MB 限制 (CWE-400) → `chat.send` → Promise 等待完成
- **cancel()** (L311): abort controller → `chat.abort`
- **handleDeltaEvent()** (L435): 增量文本流 (delta = fullText.slice(sentSoFar))
- **handleAgentEvent()** (L331): tool call 事件 → `inferToolKind()` 映射 (read/edit/execute/fetch)

### 18.4 Session Store — `session.ts` (191 行)

内存 LRU store: max 5000 sessions, 24h idle TTL, active run 保护 (不会被淘汰)。

### 18.5 安全

- `extractAttachmentsFromPrompt()`: 仅接受 base64 image
- `escapeInlineControlChars()`: 转义 NUL/CR/LF/TAB/Unicode 行分隔符 — 防日志注入
- Prompt 大小限制: 2MB (CWE-400)
- Session 创建限流

---

## 19. CLI / Commands / TUI — 命令行与终端界面

### 19.1 CLI 架构 — `cli/` (174 文件, 25,004 行)

**分层启动**:
1. `entry.ts`: 进程级 — respawn (加 `--disable-warning`), profile 解析, 环境归一化
2. `cli/run-main.ts`: **快速路由** (9 个只读命令绕过 Commander) + Commander program 构建
3. `cli/program/`: Commander.js 框架 + **懒加载命令注册**

**懒加载模式**: 命令注册为轻量级 placeholder (`allowUnknownOption(true)`), 首次调用时动态 import 真正的模块, 然后重新解析 argv。显著降低启动时间。

### 19.2 Commands — `commands/` (214 文件, 37,038 行)

**10 个核心命令组**:
1. `setup` — 初始化 config/workspace
2. `onboard` — 交互式引导
3. `configure` — 凭证/渠道设置
4. `config` — 非交互式 get/set
5. `doctor` — 20+ 步诊断修复
6. `message` — 消息发送/轮询
7. `memory` — 搜索/重索引
8. `agent` — Agent turn 执行 (828 行主函数)
9. `status/health/sessions` — 渠道健康/会话列表
10. `browser` — Chromium 管理

**24 个子命令组**: `acp`, `gateway`, `daemon`, `logs`, `system`, `models`, `approvals`, `nodes`, `sandbox`, `tui`, `cron`, `dns`, `docs`, `hooks`, `plugins`, `channels`, `security`, `secrets`, `skills`, `update`, `completion`...

#### agent 命令 — `commands/agent.ts` (828 行)

核心执行入口:
1. 验证消息体和目标
2. 加载 config, 解析 agent ID
3. 解析 model + thinking/verbose levels
4. 解析 session (支持 ACP session)
5. 构建 skills snapshot
6. `runWithModelFallback()` → `runAgentAttempt()` (CLI provider 或 embedded Pi)
7. 生命周期事件 + session store 更新 + 结果投递

#### doctor 命令 — `commands/doctor.ts` (327 行)

20+ 步全面诊断: CLI 更新, config 迁移, auth token, legacy state, sandbox, 安全, hooks, systemd linger, gateway 健康, 记忆探测...

### 19.3 TUI — `tui/` (28 文件, 5,161 行)

**Component-Tree + Event-Driven** 架构, 基于 `@mariozechner/pi-tui`:

```
Root Container
├── Header (Text)
├── ChatLog (Container) — 消息流, max 180 组件, LRU 裁剪
├── StatusBar (Container)
├── Footer (Text)
└── Editor (CustomEditor) — 多行编辑器 + 自动补全
```

**18 个 Slash 命令**: `/help`, `/status`, `/agent`, `/model`, `/think`, `/verbose`, `/reasoning`, `/elevated`, `/new`, `/reset`, `/usage`, `/settings`, `/exit`...

**快捷键**: Escape (abort), Ctrl+C (clear/warn/exit), Ctrl+L (model 选择), Ctrl+G (agent 选择), Ctrl+P (session 选择), Ctrl+T (toggle thinking)

**Gateway Chat Client** (`tui/gateway-chat.ts`): WebSocket RPC 封装, 支持 `chat.send/abort/history`, `sessions.list/patch/reset`, `models.list`。

**主题**: 20 色调色板, 语法高亮 (cli-highlight), Markdown 渲染, 超链接 (OSC 8)。

---

## 20. Sessions & Routing — 会话与路由

### 20.1 Sessions — `sessions/` (7 文件, 487 行)

**Input Provenance**: `"external_user" | "inter_session" | "internal_system"` — 追踪每条消息来源, 支持多 agent 场景的信任分级。

**Session Key 结构**: `agent:<agentId>:<channel>:<chatType>:<peerId>`

**辅助函数**:
- `parseAgentSessionKey()`: 解析 `agent:id:rest`
- `deriveSessionChatType()`: 提取 `direct/group/channel`
- `isCronRunSessionKey()`: 匹配 `cron:<name>:run:<id>`
- `isSubagentSessionKey()`: 检查 `subagent:` 前缀
- `getSubagentDepth()`: 计数 `:subagent:` 层级

**Send Policy** — `send-policy.ts` (124 行): 多层规则匹配 (channel, chatType, keyPrefix), 首个 deny 即拒, 有 allow 则放行, 默认 allow。

### 20.2 Routing — `routing/` (5 文件, 855 行)

**7 级绑定解析** — `resolve-route.ts` (444 行):

| 优先级 | Tier | 匹配条件 |
|--------|------|----------|
| 1 | `binding.peer` | 精确 peer ID |
| 2 | `binding.peer.parent` | 线程父 peer 继承 |
| 3 | `binding.guild+roles` | Guild ID + Discord 角色 |
| 4 | `binding.guild` | Guild ID |
| 5 | `binding.team` | Team ID (Slack/Teams) |
| 6 | `binding.account` | 特定 account (非通配) |
| 7 | `binding.channel` | 通配 account (`*`) |

首次匹配的 tier 胜出; 无匹配则 fallback 到 default agent。

**DM 范围**: 控制 session 粒度:
- `"main"`: 所有 DM 折叠到 `agent:<agentId>:main`
- `"per-peer"`: `agent:<agentId>:direct:<peerId>`
- `"per-channel-peer"`: `agent:<agentId>:<channel>:direct:<peerId>`
- `"per-account-channel-peer"`: 全隔离

**身份链接** (`identityLinks`): 跨平台身份折叠 → 统一 session。

**缓存**: `WeakMap<OpenClawConfig, EvaluatedBindingsCache>`, max 2000 条目。

---

## 21. Daemon 模块 — 守护进程管理

**路径**: `src/daemon/` (28 文件, 3,898 行)

### 21.1 平台策略 — `service.ts` (115 行)

```typescript
resolveGatewayService() → platform 分发:
  darwin → LaunchAgent service
  linux → systemd user service
  win32 → Scheduled Task service
```

### 21.2 macOS LaunchAgent — `launchd.ts` (497 行)

`installLaunchAgent()`: plist 写入 → legacy 卸载 → bootstrap → kickstart

`restartLaunchAgent()`: bootout → PID 等待 (10s, 200ms poll) → bootstrap → kickstart

### 21.3 Linux systemd — `systemd.ts` (431 行)

`installSystemdService()`: unit 文件 → backup → write → legacy 清理 → daemon-reload → enable → restart

所有操作通过 `systemctl --user` (非 root)。

### 21.4 命名 — `constants.ts` (114 行)

Profile-aware 命名 + 3 代遗留兼容 (openclaw, clawdbot, moltbot):
```
GATEWAY_LAUNCH_AGENT_LABEL = "ai.openclaw.gateway"
GATEWAY_SYSTEMD_SERVICE_NAME = "openclaw-gateway"
GATEWAY_WINDOWS_TASK_NAME = "OpenClaw Gateway"
```

### 21.5 诊断 — `inspect.ts` (433 行)

跨平台扫描额外/遗留 gateway 实例: macOS (plist), Linux (systemd units), Windows (schtasks)。

---

## 22. 跨模块架构总结

### 22.1 System Prompt 装配全链路

```
[Gateway] resolveSessionAgentIds()
    │
    ▼
[Agents] resolveBootstrapFilesForRun()
    │  ← agent:bootstrap hook 可修改 files
    ▼
[Agents] loadWorkspaceSkillEntries() → buildWorkspaceSkillsPrompt()
    │
    ▼
[Agents] buildSystemPromptParams() → resolveModel, timezone, runtime
    │
    ▼
[Agents] buildAgentSystemPrompt()
    │  ← 21 个 section 动态拼装
    ▼
    System Prompt (注入 LLM)
```

### 22.2 消息处理全链路

```
[Channel Plugin] 接收消息
    │
    ├─ inbound-debounce (聚合快速消息)
    │
    ▼
[Auto-Reply] dispatchReplyFromConfig()
    ├─ 去重检查
    ├─ plugin hooks (message_received)
    ├─ 跨渠道路由检查
    ├─ /stop fast abort
    ├─ send policy 检查
    ├─ ACP 分发尝试
    │
    ▼
[Auto-Reply] getReplyFromConfig()
    ├─ 解析 model, skills, directives
    ├─ media understanding + link understanding
    ├─ command/status/skill 内联处理
    │
    ▼
[Agents] runEmbeddedPiAgent()
    ├─ auth profile 轮转
    ├─ context window guard
    ├─ 重试循环 (32-160 次)
    │
    ▼
[Reply Dispatcher] 序列化投递
    ├─ reservation-based idle 检测
    ├─ 人类延迟 (800-2500ms)
    ├─ TTS 应用
    │
    ▼
[Channel Plugin] 发送消息
```

### 22.3 记忆 Lifecycle

```
[Memory Flush]  ← softThresholdTokens 前触发
    │  agent 写 memory/YYYY-MM-DD.md
    ▼
[Compaction]    ← 83.5% context window 触发
    │  safeguard → summarizeInStages()
    │  任何失败 → cancel (保留原文)
    ▼
[Post-Compaction] ← 注入 "Session Startup" 指令
    │
    ▼
[Session Memory Hook] ← /new 或 /reset 时
    │  LLM 生成 slug → memory/<date>-<slug>.md
    ▼
[Memory Search] ← Agent turn 中按需调用
    │  FTS5 ∥ sqlite-vec → hybrid merge → decay → MMR → Top-K
```

### 22.4 Cron Job 执行全链路

```
[Timer fires]
    │
    ▼
[ops.run()] ← 加锁: 设 runningAtMs, 持久化
    │
    ▼ (释放锁)
[executeJobCoreWithTimeout()]
    │
    ├── systemEvent → enqueueSystemEvent (注入主 session)
    │
    └── agentTurn → runCronIsolatedAgentTurn()
            │  resolveCronSession() → 新建或复用
            │  resolveDeliveryTarget()
            │  runWithModelFallback() → CLI or embedded Pi
            │  dispatchCronDelivery() → direct or announce
            ▼
[Timer re-arms] ← 重新加锁: 应用结果, recompute, arm
```

### 22.5 关键设计模式

| 模式 | 应用位置 | 说明 |
|------|----------|------|
| **三级类继承** | Memory Manager | SyncOps → EmbeddingOps → IndexManager 分离关注点 |
| **原子替换** | Memory reindex | 临时 DB → 原子 swap, 失败从 backup 恢复 |
| **两阶段执行** | Cron ops.run() | 加锁预约 → 无锁执行 → 加锁应用 |
| **StreamFn 装饰链** | Provider extra-params | 10+ 可组合 wrapper |
| **两级事件分发** | Hooks | 通用 type + 具体 type:action 同时触发 |
| **防御性 cancel** | Compaction safeguard | 任何失败 cancel 而非冒损坏风险 |
| **Pre-compaction snapshot** | Compaction timeout | 超时恢复到前快照 |
| **确定性散列 stagger** | Cron jobs | SHA-256(jobId) % window 防雷群 |
| **Union 非 Intersection** | Hybrid search | 任一通道命中即保留 |
| **Evergreen 文件豁免** | Temporal decay | MEMORY.md 永不衰减 |
| **懒加载命令** | CLI | Commander placeholder → 动态 import → reparse |
| **快速路由** | CLI | 9 个只读命令绕过完整 Commander 解析 |
| **Reservation-based idle** | Reply dispatcher | pending counter=1 → markComplete → settle |
| **Snapshot-Ref-Action** | Browser | accessibility snapshot → element ref → interact |
| **协议桥接** | ACP | IDE ←→ ndjson ←→ AcpGatewayAgent ←→ WS ←→ Gateway |
| **Discovery-Manifest-Load-Register** | Plugins | 目录扫描 → manifest → 边界检查 → Jiti 加载 → register() |
| **7 级绑定解析** | Routing | peer > parent > guild+roles > guild > team > account > channel > default |
| **文件式单例锁** | Gateway lock | exclusive `wx` create + 多策略存活检测 |
| **跨进程哨兵** | Restart sentinel | 写 JSON → 消费 → 报告 (投递上下文保留) |
| **三 Provider 凭证模型** | Secrets | env / file / exec, 原子快照激活 |
| **Findings-Based 审计** | Security | 20+ 收集器 → 标准化 Finding 对象 → 自动修复 |

---

## 23. PaperBot 集成可行性评估 (2026-03-02 补充)

> 完整方案见 [AGENTIC_RESEARCH_EVOLUTION.md](./AGENTIC_RESEARCH_EVOLUTION.md)。

### 23.1 集成方式: Hybrid Plugin + Bridge

PaperBot (Python/FastAPI) 通过 TypeScript 薄层插件接入 OpenClaw 生态, 保留独立性的同时利用 OpenClaw 基础设施。

```
OpenClaw Gateway ←→ paperbot-openclaw plugin (TS shim, ~300 行)
                          │ HTTP
                    PaperBot FastAPI (Python, 不改核心)
```

### 23.2 各模块集成评分

| 模块 | 适配度 | 说明 |
|------|--------|------|
| Plugin/Skill | 9/10 | 24 lifecycle hooks, registerTool API 完善, 仅需 TS/Python 桥接 |
| Memory | 7/10 | 用户记忆用 OpenClaw, 领域记忆 (paper/code) 保留 PaperBot |
| Cron | 9/10 | 大幅升级 PaperBot 的 asyncio 朴素调度器 |
| Channels | 10/10 | 从 console-only 升级到 8 渠道 (Telegram/Discord/Slack/...) |
| ACP | 7/10 | 适合交互式查询, 不适合批量 pipeline |
| Subagents | 8/10 | PaperBot 8 种 Agent 自然映射, 隔离 session 解决上下文污染 |
| **综合** | **8/10** | 推荐实施 |

### 23.3 关键集成点

**Plugin 注册 6 个工具**: `paper_search`, `paper_analyze`, `paper_track`, `gen_code`, `review_paper`, `research_context`

**4 个 Cron Job**: `paper-monitor` (每日), `weekly-digest` (每周), `conference-deadlines` (每日), `citation-monitor` (每小时)

**记忆分层**: OpenClaw 管 Layer 0-1 (用户 profile + session), PaperBot 管 Layer 2-3 (query-relevant + paper-scoped)

详见 [AGENTIC_RESEARCH_EVOLUTION.md §3](./AGENTIC_RESEARCH_EVOLUTION.md#3-openclaw-集成架构)。
