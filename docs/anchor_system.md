# 锚点作者系统

> 信息源锚点模型：理论建模、实施设计与实施计划。

---

## 第一部分：理论模型 — 形式化建模与系统设计

> 从噪声信息流中发现高质量锚点，建模信息源之间的权威性传播关系，并个性化标定锚点价值。

### 1. 本质问题

这是一个**异构信息网络中的权威性发现与传播问题**（Authority Discovery in Heterogeneous Information Networks）。

它结合了三个经典问题：

1. **信号检测**（Signal Detection）— 从噪声中识别高质量信号
2. **权威性传播**（Authority Propagation）— PageRank/HITS 的核心思想：权威性不是孤立的属性，而是通过关系网络传播的
3. **锚点标定**（Anchor Calibration）— 锚点不是绝对的，是相对于观察者（用户研究方向）和时间的

#### 1.1 信号检测：从噪声中找锚点

每天面对的信息流本质是一个**低信噪比信道**。arXiv 每天 ~500 篇 CS 论文，绝大多数是噪声（对特定研究方向而言）。锚点就是这个信道中的**高信噪比节点** — 它们不只是自身有价值，而且它们的存在能帮你**校准其他信号的价值**。

**一个好的锚点的本质特征是：它能减少你评估其他信息时的不确定性。**

例：当你知道 Dawn Song 在做 AI Safety，她的新论文就是一个锚点 — 不只因为这篇论文好，而是因为它帮你快速判断：
- 这个方向是活跃的
- 这些合作者值得关注
- 这些 venue 是相关的
- 这些被引论文可能是基础工作

锚点的信息论定义：**锚点是观测到后能最大程度降低你对信息空间不确定性的节点**。

```
H(信息空间 | 观测到锚点) << H(信息空间)

其中 H 是信息熵。锚点的质量 ∝ 互信息 I(锚点; 信息空间)
```

#### 1.2 权威传播：锚点之间的关系

这就是 PageRank 的核心洞察：**权威性不是孤立属性，而是通过关系网络传播的**。

```
锚点学者 ──发表于──→ 锚点 venue
    │                    │
    └──引用──→ 锚点论文 ──被引用──→ 新论文（被锚点网络"背书"）
    │
    └──合作──→ 新学者（通过合作关系获得"锚点传播"）
```

**关键：锚点不是单个实体的属性，是整个网络中相对位置的函数。**

#### 1.3 个性化标定：锚点是相对的

同一个学者，对研究 NLP 的人和研究 Systems 的人是完全不同的锚点。锚点评分实际上是一个**四元函数**：

```
anchor_score(source, domain, time, observer) → [0, 1]
```

- **source**: 信息源实体（学者/venue/网站/repo）
- **domain**: 领域上下文（安全/ML/系统/SE）
- **time**: 时间窗口（最近 6 个月 vs 历史全量）
- **observer**: 用户的研究方向和偏好

### 2. 形式化建模

#### 2.1 异构信息网络定义

定义异构信息网络 $G = (V, E, \phi, \psi)$，其中：

- $V$ 是节点集合
- $E \subseteq V \times V$ 是边集合
- $\phi: V \rightarrow T_V$ 是节点类型映射函数
- $\psi: E \rightarrow T_E$ 是边类型映射函数
- $|T_V| + |T_E| > 2$（异构性条件）

**节点类型** $T_V$：

```
T_V = {Scholar, Paper, Venue, Website, Topic, Repo}
```

| 节点类型 | 属性集 | 内在质量信号 |
|---------|--------|------------|
| **Scholar** | id, name, h_index, citation_count, paper_count, fields, affiliations | h-index, 总引用, 论文产出率 |
| **Paper** | id, title, year, citations, venue, judge_scores, abstract | 引用数, Judge 综合分, venue tier |
| **Venue** | name, domain, tier, acceptance_rate, impact_factor | 领域排名, 接收率, 影响因子 |
| **Website** | url, type, freshness, coverage | 覆盖率, 更新频率, 数据质量 |
| **Topic** | keyword, field, trending_score | 论文量增速, 引用集中度 |
| **Repo** | url, stars, forks, language, last_commit, contributors | stars, 活跃度, 贡献者数 |

**边类型** $T_E$：

```
T_E = {authors, published_at, cites, coauthors, belongs_to, listed_on, has_repo, researches}
```

| 边类型 | 源节点 → 目标节点 | 权威传播含义 | 权重来源 |
|--------|------------------|------------|---------|
| **authors** | Scholar → Paper | 学者为论文背书 | 作者排序位置 |
| **published_at** | Paper → Venue | Venue 为论文背书（接收=认可） | 接收年份 |
| **cites** | Paper → Paper | 被引论文获得引用方的传播 | 引用上下文（正/负/中性） |
| **coauthors** | Scholar ↔ Scholar | 合作者之间的信任传递 | 合作频次, 合作年限 |
| **belongs_to** | Paper → Topic | 论文质量反哺主题热度 | 主题匹配置信度 |
| **listed_on** | Paper → Website | 数据源的覆盖质量 | 上线时间 |
| **has_repo** | Paper → Repo | 代码实现增强论文可信度 | 代码与论文匹配度 |
| **researches** | Scholar → Topic | 学者定义研究方向的权威性 | 该方向的论文数占比 |

**元路径（Meta-path）**：

```
Scholar Authority Paths:
  P1: Scholar ──authors──→ Paper ──published_at──→ Venue
  P2: Scholar ──authors──→ Paper ──cites──→ Paper ──authors──→ Scholar
  P3: Scholar ──coauthors──→ Scholar ──authors──→ Paper

Venue Authority Paths:
  P4: Venue ←──published_at── Paper ──cites──→ Paper ──published_at──→ Venue

Topic Authority Paths:
  P5: Topic ←──belongs_to── Paper ──authors──→ Scholar ──researches──→ Topic

Emerging Source Detection:
  P6: Scholar ──coauthors──→ Scholar(anchor) ──researches──→ Topic
```

#### 2.2 锚点评分公式

对一个 source 节点 $s$，其锚点评分是四个分量的组合：

$$
\text{AnchorScore}(s) = \alpha \cdot Q(s) + \beta \cdot N(s) + \gamma \cdot T(s) + \delta \cdot R(s, o)
$$

其中 $\alpha + \beta + \gamma + \delta = 1$，建议初始权重：

| 分量 | 权重 | 说明 |
|------|------|------|
| $\alpha$ (内在质量) | 0.30 | 基础门槛，但不应主导 |
| $\beta$ (网络位置) | 0.35 | 最重要的信号 — 网络效应 |
| $\gamma$ (时间动态) | 0.15 | 区分活跃 vs 历史锚点 |
| $\delta$ (观察者相关) | 0.20 | 个性化校准 |

**Q(s) — 内在质量**

Scholar 内在质量：

```
Q_scholar(s) = normalize(
    w_h · log(1 + h_index) +
    w_c · log(1 + citation_count) +
    w_p · min(paper_count / 50, 1.0) +
    w_v · avg_venue_tier
)
```

Paper 内在质量：

```
Q_paper(s) = normalize(
    w_cite · citation_score(citations) +
    w_venue · venue_tier_score(venue) +
    w_judge · judge_overall / 5.0 +
    w_code · has_code_score
)
```

**N(s) — 网络位置（异构 PageRank）**

$$
N(v) = \frac{1 - d}{|V|} + d \sum_{u \in \text{in}(v)} \frac{w_{\psi(u,v)} \cdot N(u)}{Z(u)}
$$

边类型传播权重：

```python
EDGE_PROPAGATION_WEIGHTS = {
    "cites":        0.30,   # 引用传播最强
    "coauthors":    0.25,   # 合作关系
    "published_at": 0.20,   # venue 背书
    "has_repo":     0.15,   # 代码关联
    "belongs_to":   0.10,   # 主题归属
}
```

**T(s) — 时间动态**

```
T_scholar(s) = w_rec · recency(s) + w_vel · velocity(s) + w_trend · trend(s)

recency(s) = exp(-λ · months_since_last_paper)    λ = 0.693 / 12
velocity(s) = min(papers_last_12_months / 5, 1.0)
```

**R(s, o) — 观察者相关性**

```
R(s, o) = cosine_similarity(embed(source_profile), embed(observer_profile))
```

#### 2.3 锚点层级判定

| 层级 | AnchorScore 区间 | 含义 | 系统行为 |
|------|-----------------|------|---------|
| **核心锚点** (Core Anchor) | >= 0.8 | 领域奠基者/顶会常客 | 主动追踪，新论文自动推送，搜索结果置顶 |
| **活跃锚点** (Active Anchor) | 0.5 ~ 0.8 | 当前产出高、引用增速快 | 纳入 DailyPaper 优先排序 |
| **潜力锚点** (Emerging Anchor) | 0.3 ~ 0.5 | 新兴学者/新趋势 | 标记关注，定期复查 |
| **普通源** (Background) | < 0.3 | 背景噪声 | 仅在搜索命中时展示 |

#### 2.4 潜力锚点检测

特征定义：内在质量不高，但动态信号异常强。

Scholar 潜力信号：

| 信号 | 检测方法 | 示例 |
|------|---------|------|
| **顶会突破** | 首次在 tier1 venue 发表 | 博士生的第一篇 NeurIPS |
| **锚点合作** | 首次与核心锚点合作 | 新学者与 Dawn Song 合著 |
| **引用爆发** | 近 6 个月引用增速 > 同年龄段 2σ | 一篇论文突然被广泛引用 |
| **跨领域迁移** | 原本在 A 领域，开始在 B 领域发表 | Systems 学者开始做 ML Security |
| **代码影响力** | 关联 repo stars 快速增长 | 论文 repo 一月内 1000+ stars |

### 3. 与经典算法的关系

- **vs. PageRank**: N(s) 分量是 PageRank 在异构图上的扩展，增加边类型权重和 Q(s) 先验初始化
- **vs. HITS**: 可更好区分"综述型学者"（hub 分高）和"原创型学者"（authority 分高），建议 Phase 2 考虑
- **vs. TrustRank**: `scholar_subscriptions.yaml` 和 `top_venues.yaml` 是天然的种子锚点
- **vs. Metapath2Vec / HAN**: 对 PaperBot 规模，显式 PageRank 比深度图模型更实用

### 4. 与 PaperBot 现有基础的对接

#### 4.1 已有基础设施

| 已有组件 | 对应锚点模型分量 | 当前限制 |
|---------|-----------------|---------|
| `InfluenceCalculator` (PIS) | Q(s) — Paper 内在质量 | 仅评估 Paper，未扩展到 Scholar/Venue |
| `DynamicPISCalculator` | T(s) — 时间动态 | 仅估算引用速度，无真实引用历史 |
| `top_venues.yaml` (tier1/tier2) | Q(s) — Venue 内在质量 | 静态配置，无自动发现新 venue |
| Scholar domain model | Q(s) — Scholar 内在质量 | h-index 无领域归一化 |
| Scholar Network API (coauthor graph) | N(s) — 网络位置输入 | 已有 coauthor 数据，但未计算传播分数 |
| Judge 5 维评分 | Q(s) — Paper 细粒度质量 | 独立评分，未反哺 source 权威 |
| `TrackRouter` (keyword matching) | R(s,o) — 观察者相关性 | 基于 keyword overlap，非语义嵌入 |
| `EmbeddingProvider` (OpenAI) | R(s,o) — 语义计算 | 已有基础设施，但未用于 source 评分 |

#### 4.2 关键缺口

1. **无 Source 统一注册** — Scholar/Venue/Repo 是独立的，没有统一的 `Source` 抽象
2. **无网络级评分** — 所有评分都是实体级的，没有权威传播
3. **无 Source 间关系建模** — coauthor 数据已有但未用于评分
4. **Judge 评分单向流** — Judge 评完论文后分数不反哺到学者/venue 的权威
5. **搜索排序忽略 source** — `_score_record()` 不考虑论文作者/venue 的锚点地位

#### 4.3 反哺回路

```
Source Authority Layer
    │
    ├──→ DailyPaper: global_top 排序时加入 anchor_boost
    ├──→ Topic Search: _score_record() 中加入 source_authority_factor
    ├──→ Judge: 评分上下文中注入 anchor_level 信息
    ├──→ Scholar Tracking: 自动发现潜力锚点，建议订阅
    ├──→ Trending: trending_score 中加入 source_authority 权重
    └──→ 推荐系统: 基于用户锚点偏好推荐新论文
```

---

## 第二部分：实施设计

> 从理论模型到可运行代码的落地方案。

### 5. 数据层：authors + paper_authors

#### 5.1 新表设计

```sql
-- 作者实体表
CREATE TABLE authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name_normalized TEXT NOT NULL,
    display_name TEXT NOT NULL,
    semantic_scholar_id TEXT,
    affiliation TEXT,
    h_index INTEGER,
    citation_count INTEGER DEFAULT 0,
    paper_count INTEGER DEFAULT 0,
    resolved_at DATETIME,
    created_at DATETIME NOT NULL,
    UNIQUE(name_normalized)
);

-- 论文-作者多对多关联
CREATE TABLE paper_authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    UNIQUE(paper_id, author_id)
);
```

#### 5.2 Name Normalization

```python
def normalize_author_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'\b[a-z]\.\s*', '', name)
    if ',' in name:
        parts = [p.strip() for p in name.split(',', 1)]
        name = f"{parts[1]} {parts[0]}"
    return name.strip()
```

### 6. 双轨评分：防过拟合

```
anchor_score = α × subjective_score + (1 - α) × objective_score
```

α 随用户成熟度动态调整（α 范围 [0.1, 0.7]，客观分永远至少占 30%）：

- 新用户 (saves < 10): α ≈ 0.1 → 客观质量主导
- 成熟用户 (saves > 30) + 高校准: α ≈ 0.5 → 主观客观各半

**Objective Score 信号**：judge_score_avg (0.35) + citation_velocity (0.25) + venue_tier_avg (0.20) + h_index_norm (0.20)

**Subjective Score 信号**：save_count (0.40) + recency (0.25) + track_spread (0.20) + first_author_ratio (0.15)

**锚点分层**：Core Anchor (>= 0.7) / Rising (0.4 ~ 0.7) / Background (< 0.4)

### 7. Search Hit Overlay

搜索结果中标注锚点作者命中，提供 anchor_hits summary + by_author 结构。

前端展示：
- PaperCard 作者名旁显示锚点 badge + track 计数
- 搜索结果顶部统计栏显示命中率

### 8. 合作网络 + 机构关系

从 `paper_authors` 构建 co-author 共现图，按 `authors.affiliation` 分组做机构聚类。

### 9. 探索机制

- **Community Picks**: 纯客观质量推荐，不依赖用户行为
- **Blind Spot Detection**: 检测用户哪些 track 缺少锚点覆盖
- **Serendipity Injection**: 搜索结果中混入 10-15% 非锚点但客观质量高的论文

### 10. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/research/scholars/discover` | 锚点发现 |
| `GET` | `/research/scholars/community` | Community Picks |
| `GET` | `/research/scholars/network` | 合作网络图 |
| `GET` | `/research/scholars/blind-spots` | 盲区检测 |
| `POST` | `/research/scholars/{author_id}/resolve` | 触发 S2 API 解析 |

### 11. 系统架构

```
                    ┌─────────────────────────────────┐
                    │     Source Authority Layer        │
                    │                                   │
                    │  1. Source Registry                │
                    │  2. Relation Graph                 │
                    │  3. Authority Propagation          │
                    │  4. Anchor Classifier              │
                    │  5. Observer Projection             │
                    └─────────────────┬─────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
        DailyPaper              Scholar Tracking        推荐系统
```

---

## 第三部分：实施计划 (TODO)

### P0 — 最小闭环：建表 → 回填 → 发现 → 展示

**数据层**
- [ ] `models.py`: 新增 `AuthorModel` + `PaperAuthorModel`
- [ ] Alembic 迁移脚本，创建 authors + paper_authors 表
- [ ] `paper_store.py`: `normalize_author_name()` + `get_or_create_author()` + `link_paper_authors()`
- [ ] `paper_store.py`: 修改 `upsert_paper()` 同步写入 `paper_authors`
- [ ] 回填脚本: `scripts/backfill_paper_authors.py`

**评分服务**
- [ ] `anchor_service.py`: objective_score / subjective_score / compute_calibration / compute_alpha / anchor_score / discover_anchors / community_picks

**API 端点**
- [ ] `GET /research/scholars/discover`

**前端**
- [ ] 替换 mock 数据为真实锚点列表
- [ ] AuthorCard 组件 + Community Picks + Blind Spots

### P1 — 搜索增强：Search Hit Overlay

- [ ] `compute_anchor_hits()` — 搜索结果与用户锚点交叉匹配
- [ ] 搜索端点附加 `anchor_hits`
- [ ] PaperCard 作者名旁显示锚点 badge
- [ ] AnchorCoverageBanner 组件

### P2 — 合作网络 + S2 解析

- [ ] `build_coauthor_network()` — nodes + edges + clusters
- [ ] `resolve_author_profile()` — S2 API 解析补全
- [ ] `GET /research/scholars/network`
- [ ] `POST /research/scholars/{author_id}/resolve`
- [ ] 前端 Network tab

### P3 — 智能推荐 + Context Engine 集成

- [ ] `AnchorBoostStep` — 锚点作者论文 judge score boost
- [ ] `detect_blind_spots()` + `serendipity_injection()`
- [ ] Context Engine 搜索排序加入 anchor_boost

### P4 — 高级功能（远期）

- [ ] Author Momentum: 锚点分数随时间变化追踪
- [ ] Citation Chain Discovery: 锚点作者频繁引用的作者 → 潜在新锚点
- [ ] Track-Author Matrix / Institutional Heatmap / Anchor Alert / Diversity Warning

---

## 参考文献

### 算法基础

- Page, L., et al. (1999). The PageRank Citation Ranking.
- Kleinberg, J. M. (1999). Authoritative Sources in a Hyperlinked Environment. JACM.
- Gyöngyi, Z., et al. (2004). Combating Web Spam with TrustRank. VLDB.
- Sun, Y., et al. (2011). PathSim: Meta Path-Based Top-K Similarity Search. VLDB.
- Dong, Y., et al. (2017). metapath2vec: Scalable Representation Learning. KDD.
- Wang, X., et al. (2019). Heterogeneous Graph Attention Network. WWW.

### 学术影响力度量

- Hirsch, J. E. (2005). An index to quantify scientific research output. PNAS.
- Radicchi, F., et al. (2008). Universality of citation distributions. PNAS.
- Wang, D., et al. (2013). Quantifying Long-Term Scientific Impact. Science.

### PaperBot 现有实现

- `src/paperbot/domain/influence/calculator.py` — PIS 评分计算器
- `src/paperbot/domain/influence/analyzers/dynamic_pis.py` — 引用速度分析
- `src/paperbot/domain/influence/weights.py` — 评分权重配置
- `src/paperbot/domain/scholar.py` — Scholar 领域模型
- `config/top_venues.yaml` — Venue tier 配置
- `config/scholar_subscriptions.yaml` — 种子锚点配置
