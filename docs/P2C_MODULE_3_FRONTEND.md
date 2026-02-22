# P2C Module 3: Frontend Integration — Research/Studio UI 对接

- 日期：2026-02-22
- 状态：Draft
- 负责范围：Research 页面入口、Studio 页面注入、前端状态管理
- 上游依赖：Module 2（API & Storage）提供的 HTTP 接口
- 技术栈：Next.js 16 + React 19 + Tailwind CSS

---

## 1. 模块职责

1. 在 Research 页面的论文卡片上新增 "Generate Reproduction Session" 入口；
2. 展示生成进度（SSE 实时 / 轮询回退）；
3. 生成完成后跳转 Studio 并注入 context pack 数据；
4. 在 Studio 中可视化 context pack（Blueprint、Roadmap、Metrics）；
5. 支持用户编辑后再执行（human-in-the-loop）。

---

## 2. 与现有代码的关系

| 现有组件 | 路径 | 复用方式 |
|---|---|---|
| Studio Store | `web/src/lib/store/studio-store.ts` | 扩展 state，注入 context pack |
| RunbookPanel | `web/src/components/studio/RunbookPanel.tsx` | 复用 runbook 渲染能力 |
| FilesPanel | `web/src/components/studio/FilesPanel.tsx` | 展示生成的文件结构 |
| Research 页面 | `web/src/app/research/` 或相关路由 | 增加入口按钮 |
| API 调用约定 | `web/src/app/api/` | 新增 proxy route |

---

## 3. 消费的 API 接口

全部由 Module 2 提供，前端通过 Next.js API Route 代理调用。

| 操作 | 方法 | 端点 | 响应格式 |
|---|---|---|---|
| 生成上下文包 | POST | `/api/research/repro/context/generate` | SSE stream |
| 获取包详情 | GET | `/api/research/repro/context/{pack_id}` | JSON |
| 列出历史包 | GET | `/api/research/repro/context?user_id=...` | JSON |
| 创建执行会话 | POST | `/api/research/repro/context/{pack_id}/session` | JSON |
| 删除包 | DELETE | `/api/research/repro/context/{pack_id}` | JSON |

---

## 4. 数据类型定义（TypeScript）

```typescript
// web/src/types/p2c.ts

export interface ReproContextPack {
  context_pack_id: string;
  version: string;
  created_at: string;

  paper: PaperIdentity;
  objective: string;

  literature_digest: LiteratureDigest | null;
  blueprint: Blueprint | null;
  environment: EnvironmentSpec | null;
  implementation_spec: ImplementationSpec | null;
  task_roadmap: TaskCheckpoint[];
  success_criteria: SuccessCriterion[];

  evidence_links: EvidenceLink[];
  confidence: ConfidenceScores;
  warnings: string[];
}

export interface PaperIdentity {
  paper_id: string;
  title: string;
  year: number;
  authors: string[];
  identifiers: Record<string, string>;
}

export interface LiteratureDigest {
  problem_definition: string;
  core_innovation: string;
  relation_to_user: string;
  key_references: string[];
}

export interface Blueprint {
  architecture_type: string;
  module_hierarchy: Record<string, string[]>;
  data_flow: [string, string][];
  core_algorithms: AlgorithmSpec[];
  loss_functions: string[];
  optimization_strategy: string;
  key_hyperparameters: Record<string, unknown>;
  input_output_spec: Record<string, unknown>;
  paper_title: string;
  paper_year: number;
  framework_hints: string[];
  domain: string;
}

export interface AlgorithmSpec {
  name: string;
  pseudocode: string;
  complexity: string;
  inputs: string[];
  outputs: string[];
}

export interface EnvironmentSpec {
  python_version: string;
  pytorch_version: string | null;
  tensorflow_version: string | null;
  cuda_version: string | null;
  base_image: string;
  pip_requirements: string[];
}

export interface ImplementationSpec {
  model_type: string;
  optimizer: string;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  extra_params: Record<string, unknown>;
}

export interface TaskCheckpoint {
  id: string;
  title: string;
  description: string;
  acceptance_criteria: string[];
  depends_on: string[];
  estimated_difficulty: "low" | "medium" | "high";
}

export interface SuccessCriterion {
  metric_name: string;
  target_value: string;
  dataset_split: string;
  aggregation: string;
  source: string;
  tolerance: string | null;
}

export interface EvidenceLink {
  type: "paper_span" | "table" | "figure" | "code_snippet" | "metadata";
  ref: string;
  supports: string[];
  confidence: number;
}

export interface ConfidenceScores {
  overall: number;
  literature: number;
  blueprint: number;
  environment: number;
  spec: number;
  roadmap: number;
  metrics: number;
}

// SSE 事件类型
export interface StageProgressEvent {
  stage: string;
  progress: number;
  message: string;
}

export interface GenerateCompletedEvent {
  context_pack_id: string;
  status: "completed";
  summary: string;
  confidence: ConfidenceScores;
  warnings: string[];
  next_action: "create_repro_session";
}

export interface GenerateErrorEvent {
  error: string;
  partial_pack_id?: string;
}

// 列表项摘要
export interface ContextPackSummary {
  context_pack_id: string;
  paper_title: string;
  created_at: string;
  confidence_overall: number;
  status: string;
  warning_count: number;
}
```

---

## 5. 前端状态管理

### 5.1 扩展 Studio Store

在 `web/src/lib/store/studio-store.ts` 中扩展：

```typescript
// 新增 state 字段
interface StudioState {
  // ... 现有字段 ...

  // P2C 相关
  contextPack: ReproContextPack | null;
  contextPackLoading: boolean;
  contextPackError: string | null;
  generationProgress: StageProgressEvent[];
}

// 新增 actions
interface StudioActions {
  // ... 现有 actions ...

  setContextPack: (pack: ReproContextPack | null) => void;
  setContextPackLoading: (loading: boolean) => void;
  appendGenerationProgress: (event: StageProgressEvent) => void;
  clearGenerationProgress: () => void;
  injectContextPackToRunbook: () => void;  // pack → runbook steps 转换
}
```

### 5.2 独立 Hook: `useContextPackGeneration`

```typescript
// web/src/hooks/useContextPackGeneration.ts

export function useContextPackGeneration() {
  const [status, setStatus] = useState<"idle" | "generating" | "completed" | "error">("idle");
  const [progress, setProgress] = useState<StageProgressEvent[]>([]);
  const [result, setResult] = useState<GenerateCompletedEvent | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(async (params: {
    paperId: string;
    userId?: string;
    projectId?: string;
    trackId?: number;
    depth?: "fast" | "standard" | "deep";
  }) => {
    setStatus("generating");
    setProgress([]);
    setError(null);

    try {
      const response = await fetch("/api/research/repro/context/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
        body: JSON.stringify({
          paper_id: params.paperId,
          user_id: params.userId ?? "default",
          project_id: params.projectId,
          track_id: params.trackId,
          depth: params.depth ?? "standard",
        }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      // SSE 解析循环
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        // 解析 SSE event/data 行
        for (const event of parseSSEEvents(text)) {
          if (event.type === "stage_progress") {
            setProgress(prev => [...prev, event.data as StageProgressEvent]);
          } else if (event.type === "completed") {
            setResult(event.data as GenerateCompletedEvent);
            setStatus("completed");
          } else if (event.type === "error") {
            setError((event.data as GenerateErrorEvent).error);
            setStatus("error");
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  }, []);

  return { status, progress, result, error, generate };
}
```

---

## 6. UI 组件设计

### 6.1 Research 页面入口按钮

**位置**：论文卡片（discovery、collections、saved papers）的操作区域。

```
┌─────────────────────────────────────────┐
│  📄 Attention Is All You Need (2017)    │
│  Vaswani et al.                         │
│  ─────────────────────────────────────  │
│  Abstract: We propose a new simple...   │
│                                         │
│  [View Details]  [Save]                 │
│  [🔬 Generate Reproduction Session]  ← 新增按钮
└─────────────────────────────────────────┘
```

**组件**：`GenerateReproButton`

```typescript
// web/src/components/research/GenerateReproButton.tsx

interface Props {
  paperId: string;
  paperTitle: string;
  disabled?: boolean;
}

export function GenerateReproButton({ paperId, paperTitle, disabled }: Props) {
  const { status, progress, result, error, generate } = useContextPackGeneration();
  const router = useRouter();

  const handleClick = async () => {
    await generate({ paperId });
  };

  useEffect(() => {
    if (result) {
      // 生成完成，跳转 Studio
      router.push(`/studio?context_pack_id=${result.context_pack_id}`);
    }
  }, [result, router]);

  return (
    <div>
      <button onClick={handleClick} disabled={disabled || status === "generating"}>
        {status === "generating" ? "Generating..." : "Generate Reproduction Session"}
      </button>

      {/* 进度指示器 */}
      {status === "generating" && (
        <GenerationProgressBar stages={progress} />
      )}

      {/* 错误提示 */}
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
}
```

### 6.2 生成进度条

```
┌──────────────────────────────────────────────────────┐
│  Generating Reproduction Context...                  │
│                                                      │
│  ✅ Literature Distill                               │
│  ✅ Blueprint Extract                                │
│  🔄 Environment Infer ━━━━━━━━━━░░░░░ 50%           │
│  ⬜ Spec & Hyperparams                               │
│  ⬜ Task Roadmap                                     │
│  ⬜ Success Criteria                                 │
│                                                      │
│  Overall: ━━━━━━━━━━━━━░░░░░░░░░░ 33%               │
└──────────────────────────────────────────────────────┘
```

**组件**：`GenerationProgressBar`

```typescript
// web/src/components/research/GenerationProgressBar.tsx

const STAGE_LABELS: Record<string, string> = {
  literature_distill: "Literature Distill",
  blueprint_extract: "Blueprint Extract",
  environment_infer: "Environment Infer",
  spec_extract: "Spec & Hyperparams",
  task_roadmap: "Task Roadmap",
  success_criteria: "Success Criteria",
};

const ALL_STAGES = Object.keys(STAGE_LABELS);

interface Props {
  stages: StageProgressEvent[];
}

export function GenerationProgressBar({ stages }: Props) {
  const completedStages = new Set(
    stages.filter(s => s.progress >= 1.0).map(s => s.stage)
  );
  const currentStage = stages.length > 0 ? stages[stages.length - 1] : null;
  const overallProgress = stages.length > 0
    ? stages[stages.length - 1].progress
    : 0;

  return (
    <div className="mt-3 p-3 bg-gray-50 rounded-lg text-sm space-y-1">
      <p className="font-medium">Generating Reproduction Context...</p>
      {ALL_STAGES.map(stage => (
        <div key={stage} className="flex items-center gap-2">
          <span>
            {completedStages.has(stage) ? "✅" :
             currentStage?.stage === stage ? "🔄" : "⬜"}
          </span>
          <span className={completedStages.has(stage) ? "text-green-700" : ""}>
            {STAGE_LABELS[stage]}
          </span>
        </div>
      ))}
      {/* Overall progress bar */}
      <div className="mt-2 w-full bg-gray-200 rounded h-2">
        <div
          className="bg-blue-500 rounded h-2 transition-all"
          style={{ width: `${overallProgress * 100}%` }}
        />
      </div>
    </div>
  );
}
```

### 6.3 Studio 页面：ContextPackPanel

生成完成跳转 Studio 后，展示 context pack 的结构化数据。

```
┌─ Context Pack: Attention Is All You Need ────────────┐
│                                                      │
│  Confidence: ████████░░ 81%                          │
│                                                      │
│  ┌─ Objective ─────────────────────────────────────┐ │
│  │ 复现论文的核心 Transformer 架构并验证翻译指标     │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  [Blueprint]  [Environment]  [Roadmap]  [Metrics]    │ ← Tab 切换
│                                                      │
│  ┌─ Blueprint ─────────────────────────────────────┐ │
│  │ Architecture: transformer                       │ │
│  │ Modules:                                        │ │
│  │   model → [encoder, decoder, attention]         │ │
│  │ Core Algorithms:                                │ │
│  │   - Multi-Head Attention (O(n²d))               │ │
│  │   - Positional Encoding                         │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ⚠️ Warnings:                                        │
│  │ CUDA version inferred from paper year, verify   │ │
│                                                      │
│  [Edit & Customize]     [Create Repro Session →]     │
└──────────────────────────────────────────────────────┘
```

**组件**：`ContextPackPanel`

```typescript
// web/src/components/studio/ContextPackPanel.tsx

interface Props {
  pack: ReproContextPack;
  onCreateSession: () => void;
  onEdit: () => void;
}

export function ContextPackPanel({ pack, onCreateSession, onEdit }: Props) {
  const [activeTab, setActiveTab] = useState<
    "blueprint" | "environment" | "roadmap" | "metrics"
  >("blueprint");

  return (
    <div className="border rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-semibold text-lg">{pack.paper.title}</h3>
          <p className="text-sm text-gray-500">{pack.paper.year} · {pack.paper.authors.join(", ")}</p>
        </div>
        <ConfidenceBadge score={pack.confidence.overall} />
      </div>

      {/* Objective */}
      <div className="bg-blue-50 p-3 rounded text-sm">{pack.objective}</div>

      {/* Tabs */}
      <div className="flex gap-2 border-b">
        {(["blueprint", "environment", "roadmap", "metrics"] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={activeTab === tab ? "border-b-2 border-blue-500 pb-1 font-medium" : "pb-1"}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "blueprint" && pack.blueprint && <BlueprintView blueprint={pack.blueprint} />}
      {activeTab === "environment" && pack.environment && <EnvironmentView env={pack.environment} />}
      {activeTab === "roadmap" && <RoadmapView checkpoints={pack.task_roadmap} />}
      {activeTab === "metrics" && <MetricsView criteria={pack.success_criteria} />}

      {/* Warnings */}
      {pack.warnings.length > 0 && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 text-sm">
          <p className="font-medium">Warnings:</p>
          <ul className="list-disc list-inside">
            {pack.warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-3 pt-2">
        <button onClick={onEdit} className="btn-secondary">Edit & Customize</button>
        <button onClick={onCreateSession} className="btn-primary">Create Repro Session →</button>
      </div>
    </div>
  );
}
```

### 6.4 子组件

#### `ConfidenceBadge`

```typescript
function ConfidenceBadge({ score }: { score: number }) {
  const color = score >= 0.8 ? "green" : score >= 0.6 ? "yellow" : "red";
  return (
    <span className={`px-2 py-1 rounded text-sm font-medium bg-${color}-100 text-${color}-800`}>
      {Math.round(score * 100)}% confidence
    </span>
  );
}
```

#### `RoadmapView`

```
T1: 数据预处理         ✅ acceptance: [可加载训练集]
 │
 ▼
T2: 模型实现           acceptance: [forward 正确]
 │
 ▼
T3: 训练循环           acceptance: [loss 下降]
 │
 ▼
T4: 评估与指标验证     acceptance: [Top-1 >= 93.0]
```

```typescript
function RoadmapView({ checkpoints }: { checkpoints: TaskCheckpoint[] }) {
  return (
    <div className="space-y-3">
      {checkpoints.map((cp, i) => (
        <div key={cp.id} className="flex gap-3">
          {/* 连接线 */}
          <div className="flex flex-col items-center">
            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-sm font-medium">
              {cp.id}
            </div>
            {i < checkpoints.length - 1 && <div className="w-0.5 h-full bg-gray-300" />}
          </div>

          {/* 内容 */}
          <div className="flex-1 pb-4">
            <p className="font-medium">{cp.title}</p>
            {cp.description && <p className="text-sm text-gray-600">{cp.description}</p>}
            <div className="text-xs text-gray-500 mt-1">
              <span className={`inline-block px-1.5 py-0.5 rounded ${
                cp.estimated_difficulty === "high" ? "bg-red-100" :
                cp.estimated_difficulty === "medium" ? "bg-yellow-100" : "bg-green-100"
              }`}>
                {cp.estimated_difficulty}
              </span>
              {cp.acceptance_criteria.length > 0 && (
                <span className="ml-2">
                  Acceptance: {cp.acceptance_criteria.join(", ")}
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
```

#### `MetricsView`

```typescript
function MetricsView({ criteria }: { criteria: SuccessCriterion[] }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-gray-500 border-b">
          <th className="pb-2">Metric</th>
          <th className="pb-2">Target</th>
          <th className="pb-2">Split</th>
          <th className="pb-2">Source</th>
        </tr>
      </thead>
      <tbody>
        {criteria.map((c, i) => (
          <tr key={i} className="border-b last:border-0">
            <td className="py-2 font-medium">{c.metric_name}</td>
            <td className="py-2 font-mono">{c.target_value}{c.tolerance ? ` (${c.tolerance})` : ""}</td>
            <td className="py-2">{c.dataset_split}</td>
            <td className="py-2 text-gray-500">{c.source}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

---

## 7. 页面路由与数据流

### 7.1 完整用户流程

```
Research Page                    Studio Page
┌──────────────┐                ┌──────────────────────────────┐
│ 论文卡片      │                │                              │
│              │  ① click       │  ContextPackPanel            │
│ [Generate    │ ──────────→    │  ┌────────────────────────┐  │
│  Repro       │  ② SSE 进度    │  │ Blueprint / Env /      │  │
│  Session]    │ ←─────────→    │  │ Roadmap / Metrics      │  │
│              │  ③ 完成后跳转  │  └────────────────────────┘  │
└──────────────┘  ──────────→   │                              │
                                │  ④ [Create Repro Session]    │
                                │     ↓                        │
                                │  RunbookPanel (现有)          │
                                │  ┌────────────────────────┐  │
                                │  │ Step 1: Setup env      │  │
                                │  │ Step 2: Implement      │  │
                                │  │ Step 3: Verify         │  │
                                │  └────────────────────────┘  │
                                └──────────────────────────────┘
```

### 7.2 Studio 页面加载逻辑

```typescript
// web/src/app/studio/page.tsx (或对应路由)

export default function StudioPage() {
  const searchParams = useSearchParams();
  const contextPackId = searchParams.get("context_pack_id");
  const { contextPack, setContextPack } = useStudioStore();

  useEffect(() => {
    if (contextPackId && !contextPack) {
      // 从 API 加载 context pack
      fetch(`/api/research/repro/context/${contextPackId}`)
        .then(res => res.json())
        .then(setContextPack);
    }
  }, [contextPackId, contextPack, setContextPack]);

  return (
    <div>
      {contextPack && (
        <ContextPackPanel
          pack={contextPack}
          onCreateSession={handleCreateSession}
          onEdit={handleEdit}
        />
      )}
      {/* 现有 Studio 内容 */}
    </div>
  );
}
```

### 7.3 创建会话后注入 Runbook

```typescript
async function handleCreateSession() {
  const res = await fetch(
    `/api/research/repro/context/${contextPack.context_pack_id}/session`,
    { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ executor_preference: "auto" }) }
  );
  const session = await res.json();

  // 注入到现有 Studio runbook
  studioStore.setRunbookId(session.runbook_id);
  studioStore.setRunbookSteps(session.initial_steps);
}
```

---

## 8. Next.js API Route（代理层）

前端需要一个代理 route 将请求转发到 Python 后端，复用现有 proxy 模式。

```
web/src/app/api/research/repro/context/
    route.ts                 # GET (list) + POST (generate)
    [packId]/
        route.ts             # GET (detail) + DELETE
        session/
            route.ts         # POST (create session)
```

```typescript
// web/src/app/api/research/repro/context/route.ts

const BACKEND_URL = process.env.PAPERBOT_API_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const backendRes = await fetch(`${BACKEND_URL}/api/research/repro/context/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
    body: JSON.stringify(body),
  });

  // 透传 SSE stream
  return new Response(backendRes.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.toString();
  const backendRes = await fetch(`${BACKEND_URL}/api/research/repro/context?${query}`);
  const data = await backendRes.json();
  return NextResponse.json(data);
}
```

---

## 9. 代码组织

```
web/src/
    types/
        p2c.ts                                    # 新增：TypeScript 类型定义
    hooks/
        useContextPackGeneration.ts               # 新增：SSE 生成 hook
    components/
        research/
            GenerateReproButton.tsx                # 新增：入口按钮
            GenerationProgressBar.tsx              # 新增：进度条
        studio/
            ContextPackPanel.tsx                   # 新增：Pack 可视化面板
            BlueprintView.tsx                      # 新增：Blueprint tab
            EnvironmentView.tsx                    # 新增：Environment tab
            RoadmapView.tsx                        # 新增：Roadmap tab
            MetricsView.tsx                        # 新增：Metrics tab
            ConfidenceBadge.tsx                    # 新增：置信度徽章
    app/
        api/research/repro/context/
            route.ts                              # 新增：API proxy
            [packId]/
                route.ts                          # 新增：API proxy
                session/
                    route.ts                      # 新增：API proxy
    lib/store/
        studio-store.ts                           # 修改：扩展 P2C state
```

---

## 10. 与 Module 2 的协调事项

| 协调点 | 说明 |
|---|---|
| SSE 事件格式 | 前端假设 `event: stage_progress` / `event: completed` / `event: error`，需与后端对齐 |
| `ReproContextPack` JSON 字段 | 前端 TypeScript 类型需与后端 Python dataclass 的 JSON 序列化一致 |
| API 路径 | 统一使用 `/api/research/repro/context/*` 前缀 |
| 错误响应格式 | 统一为 `{ "error": "message", "detail": "..." }` |
| 分页参数 | `limit` + `offset`，返回 `{ items: [...], total: N }` |

---

## 11. 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| SSE 连接断开 | `useContextPackGeneration` 内置重连逻辑 + 轮询 fallback |
| 大 pack JSON 导致渲染卡顿 | Tab 懒加载，仅渲染当前 tab 内容 |
| 用户编辑 roadmap 后与后端不同步 | 编辑后的 override 通过 `create_session` 请求提交 |
| 前后端类型漂移 | 考虑后续引入 OpenAPI spec 自动生成 TypeScript 类型 |
