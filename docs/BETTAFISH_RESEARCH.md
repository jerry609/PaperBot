# BettaFish 项目调研报告

## 📋 项目概况

**项目名称**：BettaFish（微舆）  
**项目链接**：https://github.com/666ghj/BettaFish  
**核心定位**：多智能体舆情分析系统  
**技术栈**：纯 Python，从0实现，不依赖任何框架  
**开源许可**：GPL-2.0  
**项目状态**：活跃开发中（35+ 贡献者，4个Release）

### 项目简介

BettaFish 是一个创新型的多智能体舆情分析系统，通过 Agent 协作机制，全自动分析 30+ 主流社交媒体与数百万条大众评论。用户只需像聊天一样提出分析需求，系统即可生成专业的舆情分析报告。

---

## 🏗️ 核心架构分析

### 1. 多 Agent 协作架构

BettaFish 采用了**五大专业 Agent** 的协作模式：

```
┌─────────────────────────────────────────────┐
│            ForumEngine（协作层）             │
│        Agent "论坛" 协作机制 + 主持人         │
└─────────────────────────────────────────────┘
                     ↓
    ┌────────────────┴────────────────┐
    ↓                ↓                ↓
┌─────────┐    ┌──────────┐    ┌─────────┐
│ Query   │    │  Media   │    │ Insight │
│ Agent   │    │  Agent   │    │ Agent   │
│         │    │          │    │         │
│ 新闻搜索 │    │ 多模态   │    │ 数据库  │
└─────────┘    └──────────┘    └─────────┘
                     ↓
              ┌──────────┐
              │  Report  │
              │  Agent   │
              │          │
              │ 报告生成  │
              └──────────┘
```

**关键特性**：
- **Query Agent**：国内外新闻广度搜索
- **Media Agent**：多模态内容分析（视频、图片）
- **Insight Agent**：私有数据库深度挖掘
- **Report Agent**：智能报告生成（多轮迭代）
- **Forum Agent**：协作主持人，通过"论坛"机制进行链式思维碰撞与辩论

### 2. 模块化分层设计

每个 Agent 都采用统一的分层结构：

```
Agent/
├── agent.py              # Agent 主逻辑
├── llms/                 # LLM 接口封装
│   └── base.py          # 统一的 OpenAI 兼容客户端
├── nodes/                # 处理节点
│   ├── base_node.py     # 基础节点类
│   ├── search_node.py   # 搜索节点
│   ├── formatting_node.py
│   └── summary_node.py
├── tools/                # 工具集（可定制）
├── utils/                # 工具函数
├── state/                # 状态管理
│   └── state.py
└── prompts/              # 提示词模板
    └── prompts.py
```

**优势**：
- 清晰的职责分离
- 易于扩展和替换
- 统一的接口标准

### 3. 数据流架构

```
用户输入 → QueryEngine（搜索新闻）
         ↓
         MediaEngine（提取多模态内容）
         ↓
         InsightEngine（挖掘私有数据）
         ↓
         ForumEngine（Agent 协作辩论）
         ↓
         ReportEngine（生成最终报告）
```

---

## 🎯 可借鉴的核心特性

### 1. ⭐ Agent "论坛" 协作机制（**强烈推荐**）

**原理**：
- 为不同 Agent 赋予独特的工具集与思维模式
- 引入**辩论主持人模型**（`ForumEngine/monitor.py`）
- 通过"论坛"机制进行链式思维碰撞与辩论
- 避免单一模型的思维局限与交流导致的同质化

**适用场景（PaperBot）**：
```
ResearchAgent（论文分析）
    ↓
CodeAnalysisAgent（代码分析）
    ↓
QualityAgent（质量评估）
    ↓
ForumCoordinator（主持人）
    ↓
DocumentationAgent（文档生成）
```

**实现要点**：
- 创建 `ForumCoordinator` 作为主持人
- 定义 Agent 间的通信协议（`utils/forum_reader.py`）
- 实现日志监控和论坛管理（`ForumEngine/monitor.py`）

### 2. ⭐ 统一的模块化 Agent 框架

**核心组件**：

```python
# base_agent.py 统一接口
class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config
        self.llm = self._init_llm()      # LLM 客户端
        self.nodes = self._init_nodes()  # 处理节点
        self.tools = self._init_tools()  # 工具集
        self.state = {}                  # 状态管理
    
    @abstractmethod
    async def process(self, input_data):
        """主处理流程"""
        pass
```

**节点化处理**（Nodes）：

```python
# nodes/base_node.py
class BaseNode:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    async def execute(self, state):
        """执行节点逻辑"""
        pass
```

**适用于 PaperBot**：
- 将现有的 `BaseAgent` 扩展为完整的节点化架构
- 每个 Agent 拥有独立的 `nodes/`、`tools/`、`prompts/`
- 统一的状态管理机制

### 3. ⭐ 中间件集成机制

BettaFish 不仅依赖 LLM，还融合了多种中间件：

**情感分析中间件**：
```python
# tools/sentiment_analyzer.py
class SentimentAnalyzer:
    """微调 BERT/GPT-2 模型进行情感分析"""
    def analyze(self, text):
        # 调用微调模型
        pass
```

**关键词优化中间件**：
```python
# tools/keyword_optimizer.py
class KeywordOptimizer:
    """使用 Qwen 优化关键词"""
    def optimize(self, keywords):
        pass
```

**适用于 PaperBot**：
- 集成代码质量分析模型（如 CodeBERT）
- 添加漏洞检测模型
- 使用微调模型进行论文分类

### 4. ⭐ ReportEngine 的 IR（中间表示）架构

**核心设计**：

BettaFish 的 `ReportEngine` 采用了**文档中间表示（Document IR）**架构：

```
流程：LLM 生成 → JSON Schema 校验 → IR 装订 → 多格式渲染
```

**关键模块**：

```python
# ir/schema.py - 定义标准块类型
BLOCK_TYPES = {
    'heading', 'paragraph', 'list', 
    'table', 'chart', 'code', 'quote'
}

# ir/validator.py - 校验章节 JSON 结构
def validate_chapter(chapter_json):
    """确保 LLM 输出符合标准"""
    pass

# core/stitcher.py - 装订器
def stitch_document(chapters):
    """将多个章节装订为完整文档"""
    pass

# renderers/ - 多格式渲染器
html_renderer.py   # IR → HTML
pdf_renderer.py    # HTML → PDF
```

**多轮生成流程**：

```python
# nodes/template_selection_node.py
# 1. 模板选择
templates = select_templates(user_query)

# nodes/document_layout_node.py  
# 2. 文档布局设计
layout = design_layout(templates)

# nodes/word_budget_node.py
# 3. 篇幅规划
budget = plan_word_budget(layout)

# nodes/chapter_generation_node.py
# 4. 章节级 JSON 生成 + 校验
for section in layout:
    chapter_json = generate_chapter(section, budget)
    validate_chapter(chapter_json)  # 校验
    chapters.append(chapter_json)

# core/stitcher.py
# 5. 装订为完整文档
document_ir = stitch_document(chapters)

# renderers/
# 6. 渲染为 HTML/PDF
html = render_html(document_ir)
pdf = render_pdf(html)
```

**适用于 PaperBot**：
- 为论文分析报告定义 JSON Schema
- 实现多轮报告生成（概述 → 代码分析 → 质量评估 → 总结）
- 支持导出为 Markdown/HTML/PDF

### 5. ⭐ 工具集可定制化设计

每个 Agent 的 `tools/` 目录都是可插拔的：

```python
# QueryEngine/tools/
├── baidu_search.py      # 百度搜索
├── bing_search.py       # Bing 搜索
├── google_search.py     # Google 搜索
└── serper_search.py     # Serper API

# 使用时动态加载
class QueryAgent:
    def _init_tools(self):
        enabled_tools = self.config.get('enabled_tools', [])
        return {
            name: import_tool(name) 
            for name in enabled_tools
        }
```

**适用于 PaperBot**：
- 为 `ResearchAgent` 添加多种论文搜索工具（ACM、IEEE、arXiv）
- 为 `CodeAnalysisAgent` 添加多种代码分析工具（SonarQube、Pylint）
- 通过配置文件动态启用/禁用工具

### 6. ⭐ Prompt 模板化管理

BettaFish 将所有 Prompt 集中管理：

```python
# prompts/prompts.py
TEMPLATE_SELECTION_PROMPT = """
你是一个专业的报告模板选择专家...
任务：{task}
可用模板：{templates}
请选择最适合的模板...
"""

CHAPTER_GENERATION_PROMPT = """
你是一个专业的{section_type}章节撰写专家...
主题：{topic}
字数预算：{word_budget}
输出格式：严格的 JSON Schema
...
"""
```

**适用于 PaperBot**：
- 为不同分析任务设计专门的 Prompt
- 集中管理和版本控制 Prompt
- 支持 A/B 测试不同的 Prompt 策略

### 7. Streamlit 单 Agent 独立应用

BettaFish 为每个 Agent 提供了独立的 Streamlit 应用：

```python
# SingleEngineApp/query_engine_streamlit_app.py
def main():
    st.title("Query Engine - 新闻搜索分析")
    
    query = st.text_input("输入搜索关键词")
    if st.button("开始分析"):
        with st.spinner("正在搜索..."):
            result = query_agent.process(query)
        st.json(result)
```

**适用于 PaperBot**：
- 为每个 Agent 创建独立的调试界面
- 方便开发者单独测试和调优
- 降低开发和调试复杂度

### 8. 统一的配置管理

BettaFish 使用 `.env` 文件统一管理配置：

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM 配置
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    OPENAI_MODEL: str
    
    # 数据库配置
    DATABASE_URL: str
    
    # 爬虫配置
    CRAWLER_THREADS: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**适用于 PaperBot**：
- 使用 Pydantic 进行类型安全的配置管理
- 环境变量优先级：`.env` < 配置文件 < 命令行参数
- 支持多环境配置（dev/staging/prod）

---

## 🔄 与 PaperBot 的对比分析

| 维度 | BettaFish | PaperBot（现状） | 可借鉴点 |
|------|-----------|-----------------|---------|
| **Agent 架构** | 5个专业 Agent + 论坛协作 | 4个基础 Agent | ✅ 论坛协作机制 |
| **模块化程度** | 极高（nodes/tools/prompts） | 中等 | ✅ 节点化架构 |
| **中间件集成** | 微调模型 + 统计模型 | 仅 LLM | ✅ 集成专业模型 |
| **报告生成** | IR + 多轮生成 + 多格式导出 | - | ✅ IR 架构 + PDF 导出 |
| **工具集** | 可插拔 | 耦合度较高 | ✅ 工具插件化 |
| **配置管理** | 统一 .env | YAML + 代码混合 | ✅ Pydantic 统一管理 |
| **独立调试** | Streamlit 单 Agent 应用 | - | ✅ 独立调试界面 |
| **数据处理** | 爬虫 + 数据库 + 情感分析 | 论文下载 | ⚠️ 数据管道设计 |

---

## 📊 迁移优先级建议

### 🔴 高优先级（立即实施）

1. **Agent 论坛协作机制**
   - 创建 `ForumCoordinator` 作为主持人
   - 实现 Agent 间通信协议
   - 引入辩论机制提升分析质量
   
2. **节点化 Agent 架构**
   - 重构 `BaseAgent` 为节点化设计
   - 统一 `nodes/`、`tools/`、`prompts/` 结构
   - 实现状态管理机制

3. **Prompt 模板化管理**
   - 创建 `prompts/` 目录集中管理
   - 为不同任务设计专门的 Prompt
   - 版本控制和 A/B 测试

### 🟡 中优先级（短期规划）

4. **Report IR 架构**
   - 定义论文分析报告的 JSON Schema
   - 实现多轮报告生成流程
   - 支持 PDF/HTML 导出

5. **工具集插件化**
   - 重构为可插拔的工具架构
   - 支持通过配置动态加载工具
   - 为每个 Agent 设计专属工具集

6. **统一配置管理**
   - 使用 Pydantic 替代 YAML
   - 统一 `.env` 文件管理
   - 支持多环境配置

### 🟢 低优先级（长期规划）

7. **中间件集成**
   - 集成 CodeBERT 进行代码质量分析
   - 添加漏洞检测模型
   - 微调模型进行论文分类

8. **Streamlit 调试界面**
   - 为每个 Agent 创建独立应用
   - 方便单独测试和调优

---

## 🛠️ 具体实施方案

### 方案 1：重构 Agent 为节点化架构

**目标文件结构**：

```
agents/
├── research_agent/
│   ├── agent.py              # 主逻辑
│   ├── nodes/
│   │   ├── paper_search_node.py
│   │   ├── metadata_extract_node.py
│   │   └── github_extract_node.py
│   ├── tools/
│   │   ├── acm_search.py
│   │   ├── ieee_search.py
│   │   └── arxiv_search.py
│   ├── prompts/
│   │   └── prompts.py
│   └── state/
│       └── state.py
├── code_analysis_agent/
│   ├── agent.py
│   ├── nodes/
│   │   ├── clone_node.py
│   │   ├── static_analysis_node.py
│   │   └── security_scan_node.py
│   ├── tools/
│   │   ├── github_clone.py
│   │   ├── sonarqube.py
│   │   └── bandit.py
│   └── ...
└── forum_coordinator/
    ├── coordinator.py        # 主持人逻辑
    ├── monitor.py            # 日志监控
    └── communication.py      # Agent 通信协议
```

**代码示例**：

```python
# agents/research_agent/nodes/paper_search_node.py
class PaperSearchNode(BaseNode):
    async def execute(self, state):
        query = state['query']
        conference = state['conference']
        
        # 使用工具搜索论文
        tool = self.tools[f'{conference}_search']
        papers = await tool.search(query)
        
        # 更新状态
        state['papers'] = papers
        return state
```

### 方案 2：实现 Forum Coordinator

**核心逻辑**：

```python
# agents/forum_coordinator/coordinator.py
class ForumCoordinator:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.moderator = ModeratorLLM()  # 主持人 LLM
        self.forum_log = []
    
    async def coordinate(self, task):
        """协调多个 Agent 完成任务"""
        
        # 1. 任务分解
        subtasks = await self.moderator.decompose(task)
        
        # 2. Agent 执行
        results = {}
        for subtask in subtasks:
            agent = self._select_agent(subtask)
            result = await agent.process(subtask)
            results[subtask.id] = result
            self.forum_log.append({
                'agent': agent.name,
                'subtask': subtask,
                'result': result
            })
        
        # 3. 论坛辩论
        debate_result = await self._forum_debate(results)
        
        # 4. 生成最终结论
        conclusion = await self.moderator.synthesize(debate_result)
        
        return conclusion
    
    async def _forum_debate(self, results):
        """Agent 论坛辩论"""
        rounds = 3
        for round in range(rounds):
            # 每个 Agent 对其他 Agent 的结果提出质疑
            critiques = {}
            for agent_name, result in results.items():
                other_agents = [a for a in self.agents if a.name != agent_name]
                for critic in other_agents:
                    critique = await critic.critique(result)
                    critiques[f'{critic.name}_on_{agent_name}'] = critique
            
            # 主持人总结
            summary = await self.moderator.summarize_debate(critiques)
            
            # Agent 根据批评修正结果
            for agent in self.agents:
                if agent.name in results:
                    results[agent.name] = await agent.revise(
                        results[agent.name], 
                        critiques
                    )
        
        return results
```

### 方案 3：Report IR 架构

**JSON Schema 定义**：

```json
{
  "document": {
    "meta": {
      "title": "论文代码分析报告",
      "author": "PaperBot",
      "timestamp": "2024-12-01T15:00:00Z"
    },
    "sections": [
      {
        "type": "summary",
        "title": "执行摘要",
        "blocks": [
          {
            "type": "paragraph",
            "content": "本报告分析了..."
          },
          {
            "type": "list",
            "items": ["关键发现1", "关键发现2"]
          }
        ]
      },
      {
        "type": "code_analysis",
        "title": "代码质量分析",
        "blocks": [
          {
            "type": "table",
            "headers": ["指标", "数值", "评级"],
            "rows": [...]
          },
          {
            "type": "chart",
            "chart_type": "bar",
            "data": {...}
          }
        ]
      }
    ]
  }
}
```

**渲染器实现**：

```python
# renderers/markdown_renderer.py
class MarkdownRenderer:
    def render(self, document_ir):
        md = f"# {document_ir['meta']['title']}\n\n"
        
        for section in document_ir['sections']:
            md += f"## {section['title']}\n\n"
            
            for block in section['blocks']:
                if block['type'] == 'paragraph':
                    md += f"{block['content']}\n\n"
                elif block['type'] == 'list':
                    for item in block['items']:
                        md += f"- {item}\n"
                    md += "\n"
                elif block['type'] == 'table':
                    md += self._render_table(block)
                elif block['type'] == 'chart':
                    md += self._render_chart(block)
        
        return md
```

---

## 🎓 学习价值

### 1. 系统设计理念

BettaFish 的设计哲学值得深入学习：

- **模块化优先**：每个组件都可以独立开发、测试、替换
- **接口标准化**：统一的 Agent/Node/Tool 接口
- **关注点分离**：LLM、中间件、工具各司其职
- **可扩展性**：轻松添加新的 Agent/Node/Tool

### 2. 工程最佳实践

- **类型安全**：使用 Pydantic 进行配置管理
- **错误处理**：统一的重试机制（`utils/retry_helper.py`）
- **日志管理**：完善的日志系统
- **测试覆盖**：单元测试 + 集成测试

### 3. 多 Agent 协作模式

- **论坛机制**：通过辩论提升决策质量
- **主持人模型**：协调 Agent 间的交互
- **状态共享**：跨 Agent 的状态管理

---

## 🚀 下一步行动建议

### 第一阶段：基础架构重构（1-2周）

1. **重构 BaseAgent**
   - 引入 `nodes/` 架构
   - 统一 `tools/`、`prompts/` 目录结构
   - 实现状态管理机制

2. **创建 Prompt 库**
   - 提取现有 Prompt 到独立文件
   - 为不同任务设计专门的 Prompt

3. **统一配置管理**
   - 迁移到 Pydantic + `.env`
   - 支持多环境配置

### 第二阶段：Agent 协作机制（2-3周）

1. **实现 ForumCoordinator**
   - 创建主持人 Agent
   - 实现 Agent 间通信协议
   - 引入辩论机制

2. **为现有 Agent 添加节点**
   - ResearchAgent → PaperSearchNode + MetadataExtractNode
   - CodeAnalysisAgent → CloneNode + StaticAnalysisNode

### 第三阶段：报告生成增强（2-3周）

1. **设计 Report IR**
   - 定义 JSON Schema
   - 实现校验器

2. **实现多轮报告生成**
   - 模板选择 → 布局设计 → 章节生成

3. **添加渲染器**
   - Markdown/HTML/PDF 导出

### 第四阶段：工具集扩展（持续）

1. **论文搜索工具**
   - ACM、IEEE、arXiv 等多源搜索

2. **代码分析工具**
   - SonarQube、Bandit、Pylint 等集成

3. **中间件集成**
   - CodeBERT、漏洞检测模型等

---

## 📚 参考资源

- **BettaFish 官方文档**：https://github.com/666ghj/BettaFish
- **Deep Search Agent Demo**：https://github.com/666ghj/DeepSearchAgent-Demo（初学者友好）
- **L站讨论**：https://linux.do/t/topic/1009280

---

## 💡 总结

BettaFish 项目为 PaperBot 提供了极具价值的参考：

1. **最具价值**：Agent 论坛协作机制 + 节点化架构
2. **易于实施**：Prompt 模板化管理 + 统一配置管理
3. **长期收益**：Report IR 架构 + 工具集插件化
4. **工程质量**：模块化设计 + 完善的测试覆盖

建议从**基础架构重构**开始，逐步引入**论坛协作机制**和**报告生成增强**，最终打造一个更加智能、模块化、可扩展的学术论文分析系统。
