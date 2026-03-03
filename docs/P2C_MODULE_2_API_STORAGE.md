# P2C Module 2: API & Storage — 后端接口与持久化

- 日期：2026-02-22
- 状态：Draft
- 负责范围：HTTP 接口、持久化层、外接 Provider Bridge
- 上游依赖：Module 1（Core Engine）产出的 `ReproContextPack`
- 下游消费方：Module 3（Frontend）

---

## 1. 模块职责

1. 提供 HTTP API 供前端调用（生成、查询、创建会话）；
2. 持久化 `ReproContextPack` 及阶段中间结果；
3. 管理 SSE 进度推送（利用现有 `streaming.py` 能力）；
4. 可选双写到外接 context provider（OneContext 等）；
5. 不负责提取逻辑（由 Module 1 Core Engine 执行）。

---

## 2. 与现有代码的关系

| 现有组件 | 路径 | 复用方式 |
|---|---|---|
| `APIRouter` 注册 | `src/paperbot/api/main.py` | 新增 router 注册 |
| SSE 工具 | `src/paperbot/api/streaming.py` | 复用 `sse_response()` |
| `SqlAlchemyResearchStore` | `src/paperbot/infrastructure/stores/research_store.py` | 扩展或新增表 |
| `SqlAlchemyMemoryStore` | `src/paperbot/infrastructure/stores/memory_store.py` | 读取用户记忆 |
| Alembic 迁移 | `alembic/` | 新增迁移脚本 |
| `ContextEngine` | `src/paperbot/context_engine/engine.py` | 注入用户上下文 |

---

## 3. API 端点设计

### 3.1 生成上下文包

```
POST /api/research/repro/context/generate
```

**请求体：**

```json
{
  "user_id": "default",
  "project_id": "proj_001",
  "paper_id": "paper_xxx",
  "track_id": 30,
  "depth": "standard"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `paper_id` | string | 是 | 论文 ID（S2 / arXiv / DOI） |
| `user_id` | string | 否 | 默认 "default" |
| `project_id` | string | 否 | 关联项目 ID |
| `track_id` | int | 否 | 关联 track |
| `depth` | enum | 否 | `"fast"` / `"standard"` / `"deep"`，默认 `"standard"` |

**响应体（SSE 流）：**

SSE 模式下推送阶段进度事件，最终推送完成事件：

```
event: stage_progress
data: {"stage": "blueprint_extract", "progress": 0.33, "message": "Extracting blueprint..."}

event: stage_progress
data: {"stage": "environment_infer", "progress": 0.50, "message": "Inferring environment..."}

...

event: completed
data: {
  "context_pack_id": "ctxp_a1b2c3",
  "status": "completed",
  "summary": "ResNet-50 复现包，含 6 步路线图，置信度 0.81",
  "confidence": {"overall": 0.81, "blueprint": 0.84, "environment": 0.78, "metrics": 0.73},
  "warnings": [],
  "next_action": "create_repro_session"
}

event: error  (仅在失败时)
data: {"error": "Blueprint extraction failed after retry", "partial_pack_id": "ctxp_a1b2c3"}
```

**非 SSE 回退（同步模式）：**

请求头 `Accept: application/json` 时返回同步 JSON 响应，结构同 `completed` 事件的 data。

### 3.2 获取上下文包详情

```
GET /api/research/repro/context/{context_pack_id}
```

**响应体：** 完整的 `ReproContextPack` JSON 序列化。

```json
{
  "context_pack_id": "ctxp_a1b2c3",
  "version": "v1",
  "created_at": "2026-02-22T10:00:00Z",
  "paper": {
    "paper_id": "...",
    "title": "...",
    "year": 2026,
    "authors": ["..."],
    "identifiers": {"doi": "...", "arxiv": "...", "s2": "..."}
  },
  "objective": "复现论文的核心方法并验证主要指标",
  "literature_digest": {
    "problem_definition": "...",
    "core_innovation": "...",
    "relation_to_user": "...",
    "key_references": ["..."]
  },
  "blueprint": { "architecture_type": "transformer", "module_hierarchy": {}, "..." : "..." },
  "environment": { "python_version": "3.10", "framework": "pytorch", "..." : "..." },
  "implementation_spec": { "optimizer": "adamw", "learning_rate": 1e-4, "..." : "..." },
  "task_roadmap": [
    {"id": "T1", "title": "数据预处理", "acceptance_criteria": ["..."], "depends_on": []}
  ],
  "success_criteria": [
    {"metric_name": "Top-1", "target_value": ">= 93.0", "source": "Table 2"}
  ],
  "evidence_links": [ {"type": "paper_span", "ref": "...", "supports": ["..."], "confidence": 0.9} ],
  "confidence": {"overall": 0.81, "literature": 0.75, "blueprint": 0.84, "environment": 0.78, "spec": 0.80, "roadmap": 0.82, "metrics": 0.73},
  "warnings": []
}
```

### 3.3 列出上下文包

```
GET /api/research/repro/context?user_id=default&paper_id=xxx&limit=20&offset=0
```

**查询参数：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `user_id` | string | 按用户过滤 |
| `paper_id` | string | 按论文过滤 |
| `project_id` | string | 按项目过滤 |
| `limit` | int | 分页大小，默认 20 |
| `offset` | int | 分页偏移 |

**响应体：**

```json
{
  "items": [
    {
      "context_pack_id": "ctxp_a1b2c3",
      "paper_title": "...",
      "created_at": "...",
      "confidence_overall": 0.81,
      "status": "completed",
      "warning_count": 0
    }
  ],
  "total": 5
}
```

### 3.4 由上下文包创建复现会话

```
POST /api/research/repro/context/{context_pack_id}/session
```

**请求体：**

```json
{
  "executor_preference": "auto",
  "override_env": null,
  "override_roadmap": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `executor_preference` | enum | `"auto"` / `"claude_code"` / `"codex"` / `"local"` |
| `override_env` | object | 可选，覆盖环境配置 |
| `override_roadmap` | array | 可选，用户编辑后的路线图 |

**响应体：**

```json
{
  "session_id": "sess_xxx",
  "runbook_id": "rb_xxx",
  "initial_steps": [
    {"step_id": "S1", "title": "Setup environment", "command": "...", "status": "pending"}
  ],
  "initial_prompt": "Based on the reproduction context pack, implement..."
}
```

> 此接口将 `ReproContextPack` 转换为现有 Studio runbook 格式，复用 `src/paperbot/api/routes/runbook.py` 的 runbook 创建能力。

### 3.5 删除上下文包

```
DELETE /api/research/repro/context/{context_pack_id}
```

软删除（标记 `deleted_at`），不物理删除。

---

## 4. 存储设计

### 4.1 新增表

```sql
-- 主表：上下文包
CREATE TABLE repro_context_pack (
    id              TEXT PRIMARY KEY,          -- "ctxp_{uuid}"
    user_id         TEXT NOT NULL DEFAULT 'default',
    project_id      TEXT,
    paper_id        TEXT NOT NULL,
    paper_title     TEXT,
    version         TEXT NOT NULL DEFAULT 'v1',
    depth           TEXT NOT NULL DEFAULT 'standard',
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/running/completed/failed
    objective       TEXT,
    pack_json       TEXT NOT NULL,             -- 完整 ReproContextPack JSON
    confidence_overall REAL DEFAULT 0.0,
    warning_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at      TIMESTAMP,

    -- 查询索引
    -- CREATE INDEX ix_rcp_user_paper ON repro_context_pack(user_id, paper_id, created_at DESC);
    -- CREATE INDEX ix_rcp_project ON repro_context_pack(project_id, created_at DESC);
);

-- 阶段中间结果（调试用，可选）
CREATE TABLE repro_context_stage_result (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    context_pack_id TEXT NOT NULL REFERENCES repro_context_pack(id),
    stage_name      TEXT NOT NULL,             -- "literature_distill", "blueprint_extract", ...
    status          TEXT NOT NULL,             -- "completed" / "failed" / "skipped"
    result_json     TEXT,                      -- 阶段输出 JSON
    confidence      REAL DEFAULT 0.0,
    duration_ms     INTEGER DEFAULT 0,
    error_message   TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 证据链接（可选，用于审计）
CREATE TABLE repro_context_evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    context_pack_id TEXT NOT NULL REFERENCES repro_context_pack(id),
    evidence_type   TEXT NOT NULL,             -- "paper_span" / "table" / "figure" / ...
    ref             TEXT NOT NULL,
    supports        TEXT NOT NULL,             -- JSON array of field names
    confidence      REAL DEFAULT 0.0
);

-- 用户反馈（上线后收集）
CREATE TABLE repro_context_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    context_pack_id TEXT NOT NULL REFERENCES repro_context_pack(id),
    user_id         TEXT NOT NULL,
    rating          INTEGER,                   -- 1-5
    comment         TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 SQLAlchemy Model

```python
# 放在 src/paperbot/infrastructure/stores/models.py 或独立文件

class ReproContextPackModel(Base):
    __tablename__ = "repro_context_pack"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, default="default")
    project_id = Column(String, nullable=True)
    paper_id = Column(String, nullable=False)
    paper_title = Column(String, nullable=True)
    version = Column(String, nullable=False, default="v1")
    depth = Column(String, nullable=False, default="standard")
    status = Column(String, nullable=False, default="pending")
    objective = Column(Text, nullable=True)
    pack_json = Column(Text, nullable=False)
    confidence_overall = Column(Float, default=0.0)
    warning_count = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime, nullable=True)
```

### 4.3 Repository 接口

遵循现有 Repository Pattern（`application/ports/` 定义接口，`infrastructure/stores/` 实现）。

```python
# Port 定义
class ReproContextPackPort(ABC):
    @abstractmethod
    async def save(self, pack: ReproContextPack, user_id: str, depth: str) -> str: ...

    @abstractmethod
    async def get(self, pack_id: str) -> Optional[ReproContextPack]: ...

    @abstractmethod
    async def list_by_user(self, user_id: str, paper_id: Optional[str], limit: int, offset: int) -> Tuple[List[ReproContextPackSummary], int]: ...

    @abstractmethod
    async def soft_delete(self, pack_id: str) -> bool: ...

    @abstractmethod
    async def save_stage_result(self, pack_id: str, stage_name: str, result: dict, confidence: float, duration_ms: int) -> None: ...
```

---

## 5. Provider Bridge（外接同步）

### 5.1 职责

- 主写入：本地 SQLAlchemy store（必须成功）
- 可选双写：OneContext 或其他外接 provider（失败不阻断）
- 通过 feature flag 控制

### 5.2 接口

```python
class ContextProviderBridge:
    """本地存储 + 可选外接 provider 双写。"""

    def __init__(
        self,
        local_store: ReproContextPackPort,
        external_providers: List[ExternalContextProvider] = [],
    ):
        ...

    async def save(self, pack: ReproContextPack, user_id: str, depth: str) -> str:
        """
        1. 写入 local_store（必须成功）
        2. 异步双写到 external_providers（失败仅 log warning）
        """
        ...

    async def get(self, pack_id: str) -> Optional[ReproContextPack]:
        """仅从 local_store 读取。"""
        ...


class ExternalContextProvider(ABC):
    """外接 context provider 抽象。"""

    @abstractmethod
    async def sync(self, pack: ReproContextPack) -> bool: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


class OneContextProvider(ExternalContextProvider):
    """OneContext 实现，behind feature flag。"""

    def __init__(self, api_url: str, api_key: str):
        ...

    async def sync(self, pack: ReproContextPack) -> bool:
        """将 pack 转换为 OneContext 格式并上传。"""
        ...
```

### 5.3 Feature Flag

```python
# config/settings.py 或 .env
PAPERBOT_P2C_ONECONTEXT_ENABLED=false
PAPERBOT_P2C_ONECONTEXT_API_URL=
PAPERBOT_P2C_ONECONTEXT_API_KEY=
```

---

## 6. 路由注册

```python
# src/paperbot/api/routes/repro_context.py

from fastapi import APIRouter
router = APIRouter(prefix="/api/research/repro/context", tags=["P2C"])

@router.post("/generate")
async def generate_context_pack(request: GenerateContextPackRequest): ...

@router.get("/{context_pack_id}")
async def get_context_pack(context_pack_id: str): ...

@router.get("/")
async def list_context_packs(user_id: str = "default", ...): ...

@router.post("/{context_pack_id}/session")
async def create_repro_session(context_pack_id: str, request: CreateSessionRequest): ...

@router.delete("/{context_pack_id}")
async def delete_context_pack(context_pack_id: str): ...
```

在 `src/paperbot/api/main.py` 中注册：

```python
from paperbot.api.routes.repro_context import router as repro_context_router
app.include_router(repro_context_router)
```

---

## 7. SSE 进度推送实现

复用现有 `src/paperbot/api/streaming.py` 中的 SSE 能力。

```python
# generate endpoint 内部
async def generate_context_pack(request: GenerateContextPackRequest):
    async def event_generator():
        def on_stage_complete(stage_name: str, progress: float):
            # 通过 asyncio.Queue 传递事件
            queue.put_nowait({
                "event": "stage_progress",
                "data": {"stage": stage_name, "progress": progress}
            })

        # 启动 Core Engine
        pack = await orchestrator.run(
            normalized_input,
            depth=request.depth,
            on_stage_complete=on_stage_complete,
        )

        # 持久化
        pack_id = await bridge.save(pack, request.user_id, request.depth)

        queue.put_nowait({
            "event": "completed",
            "data": {"context_pack_id": pack_id, "confidence": pack.confidence}
        })

    return sse_response(event_generator())
```

---

## 8. 代码组织

```
src/paperbot/
    api/routes/
        repro_context.py                   # 新增：P2C API 端点
    application/ports/
        repro_context_port.py              # 新增：Repository 接口
    infrastructure/
        stores/
            repro_context_store.py         # 新增：SQLAlchemy 实现
        connectors/
            onecontext_connector.py        # 新增：OneContext provider（feature flag）
    p2c/
        provider_bridge.py                 # 新增：双写桥接
```

Alembic 迁移：

```bash
alembic revision --autogenerate -m "add repro_context_pack tables"
alembic upgrade head
```

---

## 9. 与 Module 1 / Module 3 的接口契约

### 调用 Module 1（Core Engine）

```python
# 在 generate endpoint 中
from paperbot.p2c.input_normalizer import InputNormalizer
from paperbot.p2c.orchestrator import ExtractionOrchestrator

normalizer = InputNormalizer(research_store, memory_store, context_engine)
normalized = await normalizer.normalize(request)

orchestrator = ExtractionOrchestrator(...)
pack = await orchestrator.run(normalized, depth=request.depth, on_stage_complete=callback)
```

### 供 Module 3（Frontend）消费

前端通过以下 HTTP 接口消费数据：

| 前端操作 | 调用的 API | 返回数据 |
|---|---|---|
| 点击 "Generate Reproduction Session" | `POST /generate`（SSE） | 阶段进度 + 完成结果 |
| 查看上下文包详情 | `GET /{pack_id}` | 完整 `ReproContextPack` JSON |
| 列出历史包 | `GET /` | 摘要列表 |
| 创建执行会话 | `POST /{pack_id}/session` | session_id + runbook steps |

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| 外接 provider 不可用 | local-first，双写异步，失败仅 log |
| `pack_json` 存储过大 | 单独表存阶段结果，主表 pack_json 为最终输出 |
| SSE 连接中断 | 支持 `GET /{pack_id}` 轮询回退 |
| 并发生成同一论文 | 幂等检查：同一 `(user_id, paper_id, depth)` 10 分钟内复用 |
| 迁移兼容性 | `pack_json` 为 JSON text，schema 变更向后兼容（新增字段带默认值） |
