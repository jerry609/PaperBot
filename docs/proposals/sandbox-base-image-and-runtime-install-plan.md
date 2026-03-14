# 沙箱依赖管理实施计划：自定义基础镜像 + 运行时自安装

## 概述

借鉴 Manus 的沙箱三道防线策略，为 PaperBot 实现前两道防线：

1. **第一道防线：自定义 E2B 基础镜像** — 预装论文复现常用的 ML/科学计算包，消除运行时安装延迟
2. **第二道防线：Agent 运行时自安装** — 当基础镜像缺少特定包时，Agent 可在沙箱内自行 `pip install`

> 第三道防线（API 预授权代理）暂不实施。PaperBot 的核心场景是代码复现，不需要天气/金融等外部数据 API。未来如需对接 Semantic Scholar、HuggingFace Hub 等学术 API，可按此模式扩展。

---

## 第一道防线：自定义 E2B 基础镜像

### 1.1 目标

将 `E2BExecutor` 从通用 `"Python3"` 模板切换到预装 ML 栈的自定义模板 `"paperbot-repro"`。沙箱启动即拥有 PyTorch、Transformers、科学计算全家桶，无需运行时安装。

### 1.2 E2B 自定义模板机制

E2B 支持通过 Dockerfile 构建自定义沙箱模板：

```bash
# 安装 E2B CLI
npm install -g @e2b/cli

# 登录
e2b auth login

# 初始化模板项目
mkdir e2b-template && cd e2b-template
e2b template init
```

构建后的模板会被快照（snapshot），每次创建沙箱时从快照恢复，启动时间 ~150ms，不会重复安装。

### 1.3 基础镜像包清单

根据论文复现场景，分层选择预装包：

#### 核心 ML 框架

| 包 | 用途 | 大小影响 |
|---|------|---------|
| `torch` (CPU) | PyTorch 推理/训练 | ~800MB（CPU-only 精简） |
| `transformers` | HuggingFace 模型加载 | ~50MB |
| `datasets` | HuggingFace 数据集 | ~30MB |
| `tokenizers` | 分词器 | ~10MB |
| `accelerate` | 分布式/混合精度训练 | ~5MB |
| `safetensors` | 安全模型序列化 | ~2MB |

#### 科学计算

| 包 | 用途 |
|---|------|
| `numpy`, `scipy` | 矩阵运算、科学计算 |
| `pandas` | 数据处理 |
| `matplotlib`, `seaborn` | 可视化 |
| `scikit-learn` | 传统 ML |
| `pillow` | 图像处理 |

#### 开发工具

| 包 | 用途 |
|---|------|
| `pytest` | 单元测试 |
| `black`, `pylint`, `flake8` | 代码格式/质量 |
| `jupyter`, `ipykernel` | Notebook 执行 |
| `pyyaml`, `toml` | 配置文件解析 |
| `requests`, `httpx` | HTTP 客户端 |
| `tqdm` | 进度条 |

#### 系统工具（apt）

| 包 | 用途 |
|---|------|
| `git` | 版本控制 |
| `curl`, `wget` | 下载 |
| `build-essential` | C 编译（部分包依赖） |
| `ffmpeg` | 多媒体处理（语音/视频论文） |
| `libgl1` | OpenCV 依赖 |

### 1.4 Dockerfile 设计

```dockerfile
# e2b-template/e2b.Dockerfile
FROM e2b/code-interpreter:latest

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python ML 核心（CPU-only PyTorch，控制镜像大小）
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    transformers datasets tokenizers accelerate safetensors \
    numpy scipy pandas matplotlib seaborn scikit-learn pillow \
    opencv-python-headless

# 开发工具
RUN pip install --no-cache-dir \
    pytest black pylint flake8 \
    jupyter ipykernel \
    pyyaml toml requests httpx tqdm

# Node.js 工具（如需前端复现）
RUN npm install -g typescript ts-node

# 工作目录
WORKDIR /home/user
```

> **镜像大小权衡**：CPU-only PyTorch (~800MB) vs GPU PyTorch (~2.5GB)。默认选 CPU-only 以缩短构建和快照恢复时间。如论文需要 GPU 推理，Agent 可运行时安装 GPU 版本（第二道防线），或后续提供 `paperbot-repro-gpu` 模板。

### 1.5 模板配置文件

```toml
# e2b-template/e2b.toml
template_id = "paperbot-repro"
dockerfile = "e2b.Dockerfile"
template_name = "PaperBot Reproduction Environment"

[resources]
# 默认 2 vCPU / 512MB RAM，可按需调整
cpu_count = 2
memory_mb = 512
```

### 1.6 构建与发布流程

```bash
cd e2b-template

# 构建模板（约 5-10 分钟，取决于包下载速度）
e2b template build --name "paperbot-repro"

# 构建成功后会返回模板 ID，类似：
# ✅ Template paperbot-repro built successfully
# Template ID: paperbot-repro

# 验证模板
e2b sandbox create --template "paperbot-repro"
# 在沙箱中验证预装包
# python3 -c "import torch; print(torch.__version__)"
# python3 -c "import transformers; print(transformers.__version__)"
```

### 1.7 代码改动

#### 1.7.1 `src/paperbot/repro/e2b_executor.py`

```python
# 修改前
TEMPLATE_PYTHON = "Python3"

# 修改后
TEMPLATE_PYTHON = os.getenv("E2B_TEMPLATE", "paperbot-repro")
```

一行改动。环境变量 `E2B_TEMPLATE` 允许覆盖（测试时可回退到 `"Python3"`）。

#### 1.7.2 `config/settings.py` / `.env`

```bash
# .env 新增
E2B_TEMPLATE=paperbot-repro       # 自定义模板名（默认值即为此）
```

#### 1.7.3 `env.example` 更新

```bash
# E2B sandbox
E2B_API_KEY=e2b_...
E2B_TEMPLATE=paperbot-repro       # Custom template with ML stack pre-installed
```

---

## 第二道防线：Agent 运行时自安装

### 2.1 目标

当 Agent 遇到基础镜像未预装的包时，能自主通过 `run_command` 执行 `pip install` 安装。需要解决三个问题：

1. Agent 知道自己可以安装包（系统提示词）
2. Agent 知道哪些包已预装（环境感知）
3. 安装行为安全可控（超时、白名单可选）

### 2.2 系统提示词改造

#### 当前提示词（`codex_dispatcher.py:378-392`）

```python
def _tool_system_prompt(self) -> str:
    return (
        "You are an expert coding agent working in a workspace with tools.\n\n"
        "Workflow:\n"
        "1. Use list_files/read_file first to understand existing code.\n"
        "2. Use write_file to implement changes.\n"
        "3. Use run_command only when available to verify behavior.\n"
        "4. Update progress with update_subtask.\n"
        "5. Call task_done with a short summary when complete.\n\n"
        "Rules:\n"
        "- Make minimal, correct changes.\n"
        "- Inspect tool outputs before taking the next action.\n"
        "- Avoid repeated identical tool calls.\n"
        "- If a command or step fails, diagnose and fix before continuing."
    )
```

#### 改造后提示词

```python
def _tool_system_prompt(self) -> str:
    base = (
        "You are an expert coding agent working in a workspace with tools.\n\n"
        "Workflow:\n"
        "1. Use list_files/read_file first to understand existing code.\n"
        "2. Use write_file to implement changes.\n"
        "3. Use run_command only when available to verify behavior.\n"
        "4. Update progress with update_subtask.\n"
        "5. Call task_done with a short summary when complete.\n\n"
        "Rules:\n"
        "- Make minimal, correct changes.\n"
        "- Inspect tool outputs before taking the next action.\n"
        "- Avoid repeated identical tool calls.\n"
        "- If a command or step fails, diagnose and fix before continuing.\n"
    )

    # 沙箱可用时追加环境说明
    if self._sandbox_available():
        base += (
            "\nEnvironment:\n"
            "- You are running inside a sandboxed Ubuntu VM.\n"
            "- Pre-installed packages: torch, transformers, datasets, numpy, scipy, "
            "pandas, matplotlib, scikit-learn, pillow, opencv, pytest, black.\n"
            "- If you need a package NOT listed above, install it with:\n"
            "  run_command({\"command\": \"pip install <package-name> -q\"})\n"
            "- Always use -q (quiet) flag to minimize output.\n"
            "- Do NOT reinstall pre-installed packages.\n"
            "- For apt packages: run_command({\"command\": \"sudo apt-get install -y <pkg>\"})\n"
        )

    return base
```

#### 关键设计决策

- **显式列出预装包**：避免 Agent 浪费迭代次数重复安装已有包
- **`-q` 静默标志**：减少安装日志对上下文窗口的污染
- **仅在沙箱可用时注入**：无沙箱时不提示安装能力（Fail-Closed 原则一致）

### 2.3 安装超时保护

当前 `_tool_run_command` 超时为 120 秒（`worker_tools.py:261`）。`pip install` 大型包可能超过此限。

#### 改造方案

```python
# worker_tools.py

# 新增常量
MAX_INSTALL_TIMEOUT_SEC = 300   # pip install 最长 5 分钟
MAX_COMMAND_TIMEOUT_SEC = 120   # 普通命令 2 分钟

async def _tool_run_command(self, args: Dict[str, Any]) -> str:
    enable_run = os.getenv("CODEX_ENABLE_RUN_COMMAND", "false").lower() == "true"
    if not enable_run:
        return "Error: run_command is disabled (set CODEX_ENABLE_RUN_COMMAND=true to enable)."
    if self.sandbox is None or not self.sandbox.available():
        return "Error: run_command requires an available sandbox executor."

    command = str(args.get("command", "")).strip()
    if not command:
        return "Error: command is required."

    # pip/apt install 命令给予更长超时
    is_install = _is_install_command(command)
    timeout = MAX_INSTALL_TIMEOUT_SEC if is_install else MAX_COMMAND_TIMEOUT_SEC

    result = await asyncio.to_thread(
        self.sandbox.run,
        workdir=self.workspace,
        commands=[command],
        timeout_sec=timeout,
    )
    body = result.logs or ""
    if result.error:
        body = f"{body}\n[error] {result.error}".strip()
    body = self._truncate(body, MAX_COMMAND_OUTPUT_CHARS)
    return f"exit_code: {result.exit_code}\n{body}".strip()


def _is_install_command(command: str) -> bool:
    """检测是否为包安装命令，给予更长超时。"""
    cmd = command.strip().lower()
    return any(
        cmd.startswith(prefix)
        for prefix in ("pip install", "pip3 install", "apt install", "apt-get install",
                        "sudo apt install", "sudo apt-get install", "npm install",
                        "conda install")
    )
```

### 2.4 安装日志压缩

`pip install` 输出冗长（依赖解析、下载进度等），会快速消耗 LLM 上下文。

#### 改造方案

在 `_tool_run_command` 返回结果前，对安装命令做日志压缩：

```python
def _compress_install_output(self, output: str, command: str) -> str:
    """压缩 pip/npm install 输出，只保留关键信息。"""
    if not _is_install_command(command):
        return output

    lines = output.splitlines()
    kept: list[str] = []
    for line in lines:
        lower = line.lower().strip()
        # 保留：成功/失败/版本/错误信息
        if any(kw in lower for kw in (
            "successfully installed", "already satisfied",
            "error", "failed", "not found", "installed",
            "collecting", "warning",
        )):
            kept.append(line)

    if not kept:
        return "(install completed, no notable output)"

    return "\n".join(kept[-20:])  # 最多保留最后 20 行关键信息
```

在 `_tool_run_command` 返回前调用：

```python
    body = self._compress_install_output(body, command) if is_install else body
    body = self._truncate(body, MAX_COMMAND_OUTPUT_CHARS)
```

### 2.5 可选：包安装白名单（安全加固）

> 此功能为可选项。默认不启用，保持 Manus 的开放式安装体验。如需收紧安全边界可开启。

```python
# 环境变量控制
CODEX_PIP_ALLOWLIST=torch,transformers,numpy,...   # 空=不限制
```

```python
def _check_pip_allowlist(self, command: str) -> Optional[str]:
    """如配置了白名单，检查 pip install 的包名。"""
    allowlist_raw = os.getenv("CODEX_PIP_ALLOWLIST", "").strip()
    if not allowlist_raw:
        return None  # 无白名单，放行

    if not command.strip().lower().startswith(("pip install", "pip3 install")):
        return None

    allowed = {p.strip().lower() for p in allowlist_raw.split(",") if p.strip()}
    # 提取包名（忽略 flags 如 -q, --no-cache-dir）
    parts = command.split()
    packages = [p for p in parts[2:] if not p.startswith("-")]

    blocked = [p for p in packages if p.lower() not in allowed]
    if blocked:
        return f"Error: packages not in allowlist: {', '.join(blocked)}"
    return None
```

---

## 文件清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `e2b-template/e2b.Dockerfile` | 自定义沙箱基础镜像 |
| **新建** | `e2b-template/e2b.toml` | E2B 模板配置 |
| **修改** | `src/paperbot/repro/e2b_executor.py` | 模板名改为可配置（1 行） |
| **修改** | `src/paperbot/infrastructure/swarm/codex_dispatcher.py` | 系统提示词注入环境说明 |
| **修改** | `src/paperbot/infrastructure/swarm/worker_tools.py` | 安装超时、日志压缩、可选白名单 |
| **修改** | `env.example` | 新增 `E2B_TEMPLATE` 说明 |
| **新建** | `tests/unit/test_install_command_detection.py` | 测试安装命令检测 |
| **新建** | `tests/unit/test_install_output_compression.py` | 测试日志压缩 |

---

## 实施步骤

### Step 1：构建自定义 E2B 模板（~30 分钟）

```bash
# 前提：已有 E2B 账号和 API Key
npm install -g @e2b/cli
e2b auth login

mkdir -p e2b-template
# 创建 e2b.Dockerfile 和 e2b.toml（内容见上文 1.4, 1.5）
cd e2b-template
e2b template build --name "paperbot-repro"
```

**验证**：
```bash
# 创建临时沙箱，验证包可用
e2b sandbox create --template "paperbot-repro"
python3 -c "import torch, transformers, numpy, pandas, sklearn; print('All OK')"
pytest --version
```

### Step 2：修改 `e2b_executor.py` 模板引用（~5 分钟）

将 `TEMPLATE_PYTHON = "Python3"` 改为 `os.getenv("E2B_TEMPLATE", "paperbot-repro")`。

### Step 3：改造系统提示词（~15 分钟）

修改 `codex_dispatcher.py` 的 `_tool_system_prompt()`，在沙箱可用时注入预装包列表和安装指引。添加 `_sandbox_available()` 辅助方法。

### Step 4：添加安装超时和日志压缩（~20 分钟）

修改 `worker_tools.py`：
- 添加 `_is_install_command()` 函数
- 修改 `_tool_run_command()` 的超时逻辑
- 添加 `_compress_install_output()` 方法

### Step 5：更新配置文件（~5 分钟）

更新 `env.example`，添加 `E2B_TEMPLATE` 条目。

### Step 6：编写测试（~20 分钟）

- `test_install_command_detection.py`：验证各种安装命令的识别
- `test_install_output_compression.py`：验证日志压缩保留关键信息

### Step 7：端到端验证（~15 分钟）

```bash
# 启动后端
E2B_API_KEY=e2b_... \
E2B_TEMPLATE=paperbot-repro \
CODEX_ENABLE_RUN_COMMAND=true \
python -m uvicorn src.paperbot.api.main:app --reload --port 8000

# 通过 Agent Board 创建 session，观察：
# 1. 沙箱启动即有 torch/transformers（无安装延迟）
# 2. Agent 遇到未知包时自动 pip install
# 3. 安装日志被压缩，不爆上下文
```

---

## 可观测性指标

| 指标 | 采集方式 | 目标 |
|------|---------|------|
| 沙箱启动时间 | `E2BExecutor` 计时 | < 5s（自定义模板 vs 通用模板） |
| `pip install` 命中率 | ToolExecutor 日志统计 | 预装命中 > 80%（即 < 20% 需运行时安装） |
| 安装成功率 | `run_command` 返回码 | > 95% |
| 安装超时率 | `run_command` 超时计数 | < 5% |
| 上下文利用率 | 安装输出 token 数 / 总输出 token 数 | 压缩后 < 5% |

---

## 未来扩展（暂不实施）

### GPU 模板

```bash
# paperbot-repro-gpu 模板，预装 CUDA PyTorch
# 适用于需要 GPU 推理的论文（大模型、图像生成等）
e2b template build --name "paperbot-repro-gpu"
```

Agent Board 可根据论文类型自动选择模板：
```python
template = "paperbot-repro-gpu" if paper.requires_gpu else "paperbot-repro"
```

### 第三道防线：API 预授权代理

如需在沙箱中提供学术 API 访问（Semantic Scholar、HuggingFace Hub、arXiv），可参考 Manus 的 `data_api` 模式：
- 在沙箱 `/opt/paperbot/api_client.py` 中提供预授权客户端
- 通过 PaperBot 后端代理转发请求（服务端管理 API Key）
- Agent 直接 `from api_client import PaperBotAPI` 使用

### 模板版本管理

```bash
# 定期更新基础镜像（新版 PyTorch 等）
e2b template build --name "paperbot-repro" --tag "v2"
```

通过 `E2B_TEMPLATE=paperbot-repro:v2` 控制版本切换。
