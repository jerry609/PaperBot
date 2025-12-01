# PaperBot 学者追踪系统使用指南

## 📚 功能概述

学者追踪系统允许您：
- 订阅关注的学者
- 自动检测学者的新发表论文
- 计算论文的影响力评分 (PIS)
- 生成结构化的分析报告

---

## 🚀 快速开始

### 1. 配置订阅学者

编辑 `config/scholar_subscriptions.yaml`：

```yaml
subscriptions:
  scholars:
    - name: "Dawn Song"
      semantic_scholar_id: "1741101"  # 在 Semantic Scholar 网站上获取
      keywords:
        - "AI Security"
        - "Blockchain"
    
    - name: "Nicolas Papernot"
      semantic_scholar_id: "2810933"
      keywords:
        - "Machine Learning Security"

  settings:
    check_interval: "weekly"
    papers_per_scholar: 20
    min_influence_score: 50
```

> 💡 **如何获取 Semantic Scholar ID**: 
> 访问 https://www.semanticscholar.org/，搜索学者，在 URL 中找到 ID

### 2. 运行追踪

```bash
# 查看追踪状态摘要
python main.py track --summary

# 追踪所有订阅学者
python main.py track

# 仅追踪特定学者
python main.py track --scholar-id 1741101

# 强制重新检测（清除缓存）
python main.py track --force

# 仅检测，不生成报告（Dry Run）
python main.py track --dry-run
```

### 3. 查看报告

报告默认保存在 `output/reports/{学者名}/` 目录：

```
output/reports/
├── Dawn_Song/
│   ├── 2024-12-01_abc123.md
│   └── 2024-12-01_def456.md
└── Nicolas_Papernot/
    └── 2024-12-01_xyz789.md
```

---

## 📊 影响力评分 (PIS)

PaperBot 使用综合评分公式评估论文影响力：

$$Score = 0.6 \times I_a + 0.4 \times I_e$$

### 学术影响力 ($I_a$)

| 指标 | 权重 | 说明 |
|------|------|------|
| 引用数 | 60% | 根据引用数量映射到 0-100 分 |
| 发表渠道 | 40% | 顶会(tier1) +100, 优秀会议(tier2) +60 |

### 工程影响力 ($I_e$)

| 指标 | 权重 | 说明 |
|------|------|------|
| 代码可用性 | 30% | 有公开代码 +100 |
| GitHub Stars | 40% | 根据 Star 数映射到 0-100 分 |
| 可复现性 | 30% | 基于文档、更新频率等评估 |

### 推荐级别

| 分数范围 | 推荐级别 |
|----------|----------|
| 80-100 | 🌟🌟🌟 强烈推荐深入阅读 |
| 60-79 | 🌟🌟 建议关注 |
| 40-59 | 🌟 可选阅读 |
| 0-39 | ⚪ 低优先级 |

---

## ⚙️ 高级配置

### 顶会列表

编辑 `config/top_venues.yaml` 自定义顶会列表：

```yaml
security:
  tier1:
    - "CCS"
    - "S&P"
    - "USENIX Security"
    - "NDSS"
  tier2:
    - "ACSAC"
    - "RAID"
    - "ESORICS"
```

### API 配置

```yaml
settings:
  api:
    semantic_scholar:
      api_key: null  # 可选，有 API Key 可获得更高请求限制
      timeout: 30
      request_interval: 1.0  # 请求间隔（秒）
    
    github:
      token: null  # 可选，用于获取私有仓库信息
```

---

## 📂 目录结构

```
PaperBot/
├── config/
│   ├── scholar_subscriptions.yaml  # 学者订阅配置
│   └── top_venues.yaml             # 顶会列表
├── scholar_tracking/               # 学者追踪子系统
│   ├── agents/                     # 追踪 Agents
│   ├── models/                     # 数据模型
│   └── services/                   # 服务层
├── influence/                      # 影响力计算模块
├── reports/                        # 报告生成模块
├── prompts/                        # Prompt 模板
├── cache/scholar_papers/           # 论文缓存
└── output/reports/                 # 生成的报告
```

---

## 🔧 故障排查

### 常见问题

**Q: API 请求失败，提示 Rate Limit**
- A: Semantic Scholar 有请求限制，增大 `request_interval` 或申请 API Key

**Q: 未发现任何新论文**
- A: 首次运行会将所有论文标记为"已处理"，之后才会检测新论文
- A: 使用 `--force` 参数清除缓存重新检测

**Q: 报告生成失败**
- A: 检查是否安装了 Jinja2：`pip install jinja2`
- A: 系统会自动使用备用模板，不影响基本功能

---

## 📝 定时任务

使用 cron 设置每周自动追踪：

```bash
# 每周一早上 8 点运行
0 8 * * 1 cd /path/to/PaperBot && python main.py track >> logs/tracking.log 2>&1
```

---

## 🔗 相关资源

- [Semantic Scholar API 文档](https://api.semanticscholar.org/api-docs/)
- [MVP 设计文档](MVP_DESIGN.md)
- [BettaFish 调研报告](BETTAFISH_RESEARCH.md)
