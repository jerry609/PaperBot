# Search Architecture Upgrade Plan

> **Status**: Ready for Implementation
> **Date**: 2026-02-13
> **Goal**: Enhance paper search with RRF fusion, Anchor Author integration, and lightweight LTR

---

## 1. Current State Analysis

### What's Already Working

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CURRENT SEARCH ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Query: "transformer"                                              │
│  Active Track: "NLP" (keywords: ["attention", "bert", "llm"])           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Query Enhancement                                        │   │
│  │ ─────────────────────────                                        │   │
│  │ • Acronym expansion: "rag" → "retrieval augmented generation"    │   │
│  │ • Track keywords merged: query + track.keywords                  │   │
│  │ • Result: "transformer attention bert llm"                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: Multi-Source Search (Parallel)                          │   │
│  │ ──────────────────────────────────────                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │Semantic  │ │  arXiv   │ │ OpenAlex │ │ Papers   │            │   │
│  │  │Scholar   │ │ (lexical)│ │(semantic)│ │  Cool    │            │   │
│  │  │(semantic)│ │          │ │          │ │          │            │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘            │   │
│  │       └────────────┴────────────┴────────────┘                  │   │
│  │                         │                                        │   │
│  │              Simple Title-Hash Deduplication                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 3: Paper Scoring                                           │   │
│  │ ────────────────────                                            │   │
│  │ score = 0.55 × keyword_match(paper, track)                      │   │
│  │       + 0.30 × log(citations)                                   │   │
│  │       + 0.15 × recency                                          │   │
│  │       + boost(saved: +0.25, liked: +0.15)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: Diversification & Return                                │   │
│  │ ───────────────────────────────                                 │   │
│  │ • Limit per author/venue/field                                  │   │
│  │ • Stage-aware exploration ratio                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Existing Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| Query expansion (acronyms) | `context_engine/engine.py` | ✅ Working |
| Track keyword merging | `context_engine/engine.py` | ✅ Working |
| Multi-source search | `PaperSearchService` | ✅ Working |
| Paper scoring with track | `_paper_score()` | ✅ Working |
| Saved/liked boosts | `build_context_pack()` | ✅ Working |
| Stage-aware weights | `_stage_defaults()` | ✅ Working |
| Feedback collection | `PaperFeedbackModel` | ✅ Working |
| Anchor Authors discovery | `AnchorService` | ✅ Working (just fixed) |
| Author extraction | `AuthorStore` | ✅ Working (just fixed) |
| Personalized/Global toggle | `SearchBox.tsx` | ⚠️ UI only, not connected |

### What's Missing

| Feature | Impact | Effort |
|---------|--------|--------|
| RRF cross-source fusion | High | Medium |
| Anchor Author boost in search | High | Low |
| Connect Personalized/Global toggle | Medium | Low |
| Click tracking for LTR | High | Medium |
| Learned ranking weights | High | High |

---

## 2. Upgraded Architecture

### Overview Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      UPGRADED SEARCH ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Query + Track + Mode (Personalized/Global)                         │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: QUERY ENHANCEMENT                              [Exists] │   │
│  │ ─────────────────────────────                                    │   │
│  │ • Acronym expansion (RAG, LLM, etc.)                             │   │
│  │ • Track keywords appended to query                               │   │
│  │ • [NEW] Query rewriting for arXiv (synonym expansion)            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: MULTI-SOURCE RETRIEVAL                         [Exists] │   │
│  │ ──────────────────────────────                                   │   │
│  │                                                                  │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │   │  Semantic   │  │   arXiv     │  │  OpenAlex   │  ...         │   │
│  │   │  Scholar    │  │             │  │             │              │   │
│  │   │  (semantic) │  │  (lexical)  │  │  (semantic) │              │   │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │   │
│  │          │                │                │                     │   │
│  │          │    Each returns: papers + ranks (1, 2, 3, ...)        │   │
│  │          └────────────────┼────────────────┘                     │   │
│  │                           │                                      │   │
│  └───────────────────────────┼──────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: RRF FUSION                                        [NEW] │   │
│  │ ──────────────────                                               │   │
│  │                                                                  │   │
│  │  For each paper:                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │                                                         │    │   │
│  │  │  rrf_score = Σ  (w_source / (k + rank_source))          │    │   │
│  │  │              source                                     │    │   │
│  │  │                                                         │    │   │
│  │  │  where:                                                 │    │   │
│  │  │    k = 60 (default RRF constant)                        │    │   │
│  │  │    w_s2 = 1.0, w_arxiv = 0.8, w_openalex = 0.9          │    │   │
│  │  │    (weights can be learned via LTR)                     │    │   │
│  │  │                                                         │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  │  • Deduplication: merge same paper from multiple sources         │   │
│  │  • Combine RRF scores when paper appears in multiple sources     │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: RELEVANCE SCORING                              [Exists] │   │
│  │ ─────────────────────────                                        │   │
│  │                                                                  │   │
│  │  relevance_score = w_kw  × keyword_match(paper, track)           │   │
│  │                  + w_cit × log(citations)                        │   │
│  │                  + w_rec × recency                               │   │
│  │                                                                  │   │
│  │  Weights vary by research stage:                                 │   │
│  │    survey:   w_kw=0.45, w_cit=0.15, w_rec=0.40                   │   │
│  │    writing:  w_kw=0.60, w_cit=0.30, w_rec=0.10                   │   │
│  │    rebuttal: w_kw=0.50, w_cit=0.40, w_rec=0.10                   │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: PERSONALIZATION                                   [NEW] │   │
│  │ ───────────────────────                                          │   │
│  │                                                                  │   │
│  │  if mode == "personalized":                                      │   │
│  │    ┌─────────────────────────────────────────────────────────┐  │   │
│  │    │ A. Feedback Boosts (existing)                           │  │   │
│  │    │    saved papers:  +0.25                                 │  │   │
│  │    │    liked papers:  +0.15                                 │  │   │
│  │    │                                                         │  │   │
│  │    │ B. Anchor Author Boost (NEW)                            │  │   │
│  │    │    if paper.author in followed_anchors:                 │  │   │
│  │    │      boost += 0.20 × anchor_score                       │  │   │
│  │    │                                                         │  │   │
│  │    │ C. Similar-to-Saved Boost (NEW, optional)               │  │   │
│  │    │    if similar_to_saved_papers(paper):                   │  │   │
│  │    │      boost += 0.10 × similarity                         │  │   │
│  │    └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  if mode == "global":                                            │   │
│  │    skip all personalization boosts                               │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: FINAL SCORING & RANKING                                 │   │
│  │ ───────────────────────────────                                  │   │
│  │                                                                  │   │
│  │  final_score = α × rrf_score                                     │   │
│  │              + β × relevance_score                               │   │
│  │              + γ × personalization_boost                         │   │
│  │                                                                  │   │
│  │  Default weights: α=0.35, β=0.50, γ=0.15                         │   │
│  │  (Can be learned via LTR)                                        │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 7: DIVERSIFICATION                                [Exists] │   │
│  │ ───────────────────────                                          │   │
│  │ • Limit per first-author (max 2)                                 │   │
│  │ • Limit per venue (max 3)                                        │   │
│  │ • Limit per field (max 4)                                        │   │
│  │ • Exploration ratio based on research stage                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│                      Return Ranked Results                               │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 8: FEEDBACK COLLECTION (for LTR)                     [NEW] │   │
│  │ ─────────────────────────────────                                │   │
│  │ • Track which result user clicked                                │   │
│  │ • Record: (query, clicked_paper, position, skipped_papers)       │   │
│  │ • Used for learning RRF weights and personalization factors      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Phases

### Phase 1: RRF Fusion (Priority: High)

**Goal**: Replace simple deduplication with proper RRF score fusion.

**Files to modify**:
- `src/paperbot/application/services/paper_search_service.py`

**Changes**:
```python
# Current: Simple deduplication
seen_hashes: set[str] = set()
for p in all_papers:
    if p.title_hash in seen_hashes:
        continue  # Just skip duplicates
    seen_hashes.add(p.title_hash)
    unique.append(p)

# New: RRF fusion
from collections import defaultdict

def compute_rrf_scores(
    results_by_source: Dict[str, List[PaperCandidate]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute RRF scores for papers across sources."""
    weights = weights or {
        "semantic_scholar": 1.0,
        "arxiv": 0.8,
        "openalex": 0.9,
        "papers_cool": 0.7,
        "hf_daily": 0.6,
    }

    scores: Dict[str, float] = defaultdict(float)
    paper_by_hash: Dict[str, PaperCandidate] = {}

    for source, papers in results_by_source.items():
        w = weights.get(source, 0.5)
        for rank, paper in enumerate(papers, start=1):
            h = paper.title_hash
            scores[h] += w / (k + rank)
            if h not in paper_by_hash:
                paper_by_hash[h] = paper

    return scores, paper_by_hash
```

**Estimated effort**: 2-3 hours

---

### Phase 2: Connect Personalized/Global Toggle (Priority: High)

**Goal**: Make the UI toggle actually affect search behavior.

**Files to modify**:
- `web/src/components/research/ResearchPageNew.tsx` (pass mode to API)
- `src/paperbot/api/routes/research.py` (accept mode parameter)
- `src/paperbot/context_engine/engine.py` (apply/skip personalization)

**Frontend change**:
```typescript
// In handleSearch()
const body = {
  user_id: userId,
  query,
  paper_limit: 10,
  memory_limit: 8,
  sources: searchSources,
  personalized: anchorPersonalized,  // ADD THIS
  // ...
}
```

**Backend change**:
```python
# In ContextEngineConfig
@dataclass
class ContextEngineConfig:
    # ... existing fields
    personalized: bool = True  # NEW

# In build_context_pack()
if self.config.personalized:
    # Apply boosts from saved/liked papers
    for pid in saved_ids:
        boosts[pid] = boosts.get(pid, 0.0) + 0.25
    for pid in liked_ids:
        boosts[pid] = boosts.get(pid, 0.0) + 0.15
    # Apply anchor author boosts (NEW)
    # ...
else:
    boosts = {}  # No personalization
```

**Estimated effort**: 1-2 hours

---

### Phase 3: Anchor Author Boost (Priority: High)

**Goal**: Boost papers from followed Anchor Authors.

**Files to modify**:
- `src/paperbot/context_engine/engine.py`
- `src/paperbot/application/services/anchor_service.py` (add helper method)

**Implementation**:
```python
# In AnchorService, add method:
def get_followed_author_ids(
    self, *, user_id: str, track_id: int
) -> Dict[int, float]:
    """Return {author_id: anchor_score} for followed authors."""
    with self._provider.session() as session:
        rows = session.execute(
            select(UserAnchorActionModel, UserAnchorScoreModel)
            .join(UserAnchorScoreModel, ...)
            .where(UserAnchorActionModel.action == "follow")
            .where(UserAnchorActionModel.user_id == user_id)
            .where(UserAnchorActionModel.track_id == track_id)
        ).all()
        return {row.author_id: row.personalized_anchor_score for row in rows}

# In ContextEngine.build_context_pack():
if self.config.personalized and routed_track:
    anchor_service = _get_anchor_service()
    followed_authors = anchor_service.get_followed_author_ids(
        user_id=user_id, track_id=int(routed_track["id"])
    )

    # Get author IDs for each paper
    for paper in papers:
        paper_author_ids = get_paper_author_ids(paper["paper_id"])
        for author_id in paper_author_ids:
            if author_id in followed_authors:
                anchor_score = followed_authors[author_id]
                boosts[paper["paper_id"]] += 0.20 * anchor_score
```

**Estimated effort**: 2-3 hours

---

### Phase 4: Click Tracking for LTR (Priority: Medium)

**Goal**: Record which search results users click for future learning.

**New table**:
```python
class SearchClickModel(Base):
    __tablename__ = "search_clicks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    query: Mapped[str] = mapped_column(Text)
    query_hash: Mapped[str] = mapped_column(String(64), index=True)
    clicked_paper_id: Mapped[int] = mapped_column(Integer, index=True)
    click_position: Mapped[int] = mapped_column(Integer)  # 1-based rank
    results_shown: Mapped[int] = mapped_column(Integer)   # Total results
    sources_json: Mapped[str] = mapped_column(Text)       # Which sources
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
```

**Frontend tracking**:
```typescript
// When user clicks a paper in search results
async function trackClick(paperId: number, position: number) {
  await fetch("/api/research/search/click", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      track_id: activeTrackId,
      query: lastQuery,
      paper_id: paperId,
      position: position,
      results_count: searchResults.length,
    }),
  })
}
```

**Estimated effort**: 3-4 hours

---

### Phase 5: Lightweight LTR (Priority: Low, Future)

**Goal**: Learn optimal RRF weights and personalization factors from click data.

**Approach**:
1. Collect click data for 2-4 weeks
2. Compute click-through rate (CTR) by position and source
3. Adjust source weights based on which sources produce more clicks
4. Optionally: Train a simple linear model on features

**Simple weight learning**:
```python
def learn_source_weights(clicks: List[SearchClick]) -> Dict[str, float]:
    """Learn source weights from click data."""
    source_clicks = defaultdict(int)
    source_impressions = defaultdict(int)

    for click in clicks:
        sources = json.loads(click.sources_json)
        clicked_source = get_paper_source(click.clicked_paper_id)

        for source in sources:
            source_impressions[source] += 1
        source_clicks[clicked_source] += 1

    # Compute CTR-based weights
    weights = {}
    for source in source_impressions:
        ctr = source_clicks[source] / max(1, source_impressions[source])
        weights[source] = 0.5 + ctr  # Base weight + CTR bonus

    return weights
```

**Estimated effort**: 1-2 weeks (including data collection)

---

## 4. Summary Table

| Phase | Task | Files | Effort | Impact |
|-------|------|-------|--------|--------|
| 1 | RRF Fusion | `paper_search_service.py` | 2-3h | High |
| 2 | Connect Toggle | `ResearchPageNew.tsx`, `engine.py` | 1-2h | Medium |
| 3 | Anchor Boost | `engine.py`, `anchor_service.py` | 2-3h | High |
| 4 | Click Tracking | New model, API, frontend | 3-4h | Medium |
| 5 | LTR Learning | New service | 1-2 weeks | High |

**Recommended order**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

---

## 5. API Changes

### Modified Endpoints

**POST /api/research/context**
```json
{
  "user_id": "default",
  "query": "transformer attention",
  "track_id": 1,
  "paper_limit": 10,
  "sources": ["semantic_scholar", "arxiv", "openalex"],
  "personalized": true  // NEW: enables/disables personalization
}
```

### New Endpoints

**POST /api/research/search/click** (Phase 4)
```json
{
  "user_id": "default",
  "track_id": 1,
  "query": "transformer",
  "paper_id": 123,
  "position": 3,
  "results_count": 10
}
```

**GET /api/research/search/weights** (Phase 5, admin)
```json
{
  "source_weights": {
    "semantic_scholar": 1.0,
    "arxiv": 0.85,
    "openalex": 0.92
  },
  "last_updated": "2026-02-13T12:00:00Z",
  "training_clicks": 1523
}
```

---

## 6. Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Search result CTR | Unknown | +15% | Click tracking |
| Save rate from search | Unknown | +10% | Existing feedback |
| Empty result rate | Unknown | -20% | API logs |
| P95 latency | ~800ms | <1200ms | API monitoring |

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| RRF adds latency | Compute in parallel, cache source weights |
| Anchor boost creates filter bubble | Global mode provides unbiased alternative |
| LTR overfits to few users | Minimum click threshold before learning |
| External API failures | Graceful degradation, return partial results |

---

## 8. References

- [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Elasticsearch Hybrid Search](https://www.elastic.co/docs/solutions/search/hybrid-search)
- [Learning to Rank Overview](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/)
- Current codebase: `src/paperbot/context_engine/engine.py`, `src/paperbot/application/services/paper_search_service.py`
