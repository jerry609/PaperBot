# P0: Memory Scope and Acceptance Criteria - Technical Design Document

> **Status**: Draft
> **Author**: Claude Code
> **Date**: 2025-12-27
> **Estimated Effort**: 1-2 days (17h)

---

## 0. Architecture Context: Where P0 Fits

P0 spans **three layers** of the PaperBot 5-layer architecture, focusing on the **Memory subsystem's foundational definitions and quality metrics**.

### 0.1 PaperBot 5-Layer Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PaperBot Standard Architecture                           │
│         (Offline Ingestion → Storage → Online Retrieval → Generation → Feedback)│
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 1 · Ingestion (Async)                                                    │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────────────┐    │
│  │ Sources  │──▶│ Parse/Norm   │──▶│Chunk/Embed │──▶│ EventLog (evidence)  │    │
│  │ arXiv    │   │ Connectors   │   │ (optional) │   │ SqlAlchemyEventLog   │    │
│  │ Reddit   │   │              │   │            │   │                      │    │
│  └──────────┘   └──────────────┘   └────────────┘   └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 2 · Storage                                                              │
│  ┌────────────────────────────────────────┐   ┌─────────────────────────────┐   │
│  │  SQL 主库 (SQLite/Postgres)            │   │  Vector Index (optional)   │   │
│  │  ┌─────────────────────────────────┐   │   │  Qdrant / Milvus           │   │
│  │  │ ╔═══════════════════════════╗   │   │   └─────────────────────────────┘   │
│  │  │ ║  memory_items  ◀──────────╠───┼───┼─── P0: Type Boundaries        │   │
│  │  │ ║  memory_audit_log         ║   │   │        Namespace/Isolation     │   │
│  │  │ ╚═══════════════════════════╝   │   │                                │   │
│  │  │   research_tracks / tasks       │   │                                │   │
│  │  │   paper_feedback                │   │                                │   │
│  │  └─────────────────────────────────┘   │                                │   │
│  └────────────────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 3 · Retrieval (Online)                                                   │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────────────────────────────┐    │
│  │ContextEng │──▶│ TrackRouter │──▶│ ╔════════════════════════════════╗   │    │
│  │ine        │   │             │   │ ║  MemoryStore                   ║   │    │
│  │           │   │ keyword+    │   │ ║  search_memories()  ◀──────────╠───┼─── P0
│  │           │   │ mem+task+   │   │ ║  (scope-aware retrieval)       ║   │    │
│  │           │   │ embedding   │   │ ╚════════════════════════════════╝   │    │
│  └────────────┘   └─────────────┘   └──────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────────────────────┐   │
│  │ Paper Searcher │──▶│ Rank/Filter/   │──▶│ Replay write                   │   │
│  │ S2 / offline   │   │ Dedupe         │   │ context_run + impressions      │   │
│  └────────────────┘   └────────────────┘   └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 4 · Generation                                                           │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────────────────────┐   │
│  │ PromptComposer │──▶│ LLM + Tools    │──▶│ Output Parser                  │   │
│  │ budget+format  │   │ GPT/Claude     │   │ structured + citations         │   │
│  └────────────────┘   └────────────────┘   └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 5 · Feedback / Governance / Replay                                       │
│  ┌─────────────────────────────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │ ╔═════════════════════════════╗ │   │ Paper Feedback │   │ Evals/Replay   │  │
│  │ ║  Memory Inbox (governance)  ║ │   │ like/save/     │   │ Summary        │  │
│  │ ║  suggest → pending          ║ │   │ dislike        │   │                │  │
│  │ ║  approve/reject/move ◀──────╠─┼───┼────────────────┼───┼─── P0: Metrics │  │
│  │ ╚═════════════════════════════╝ │   │                │   │    Evaluation  │  │
│  └─────────────────────────────────┘   └────────────────┘   └────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘

Legend:
  ╔═══╗  P0 Focus Area (Memory Scope & Acceptance)
  ╚═══╝
  ───▶   Data Flow
  ◀────  P0 Touch Point
```

### 0.2 P0 Components Mapped to Architecture Layers

| Layer | Component | P0 Deliverable |
|-------|-----------|----------------|
| **Layer 2: Storage** | `memory_items` table | Memory type boundary definitions (taxonomy) |
| | Namespace fields | `user_id` / `workspace_id` / `scope_type:scope_id` isolation rules |
| **Layer 3: Retrieval** | `MemoryStore.search_memories()` | Scope-aware retrieval logic validation |
| | Query resolution | Isolation rule implementation verification |
| **Layer 5: Feedback** | Memory Inbox | Governance workflow (pending → approved) |
| | `memory_eval_metrics` (new) | Acceptance criteria measurement |
| | Audit log | Deletion compliance verification |

### 0.3 P0 Focus: Cross-Cutting Concerns

```
                    ┌─────────────────────────────────────────┐
                    │           P0: Scope & Acceptance        │
                    └─────────────────────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  Type Boundaries    │   │  Namespace/Isolation│   │  Acceptance Metrics │
│                     │   │                     │   │                     │
│  - User Memory      │   │  - user_id (L1)     │   │  - Extraction       │
│    profile/pref/    │   │  - workspace_id(L2) │   │    Precision ≥85%   │
│    goal/constraint  │   │  - scope_type:id(L3)│   │  - False Positive   │
│  - Episodic Memory  │   │  - provider (meta)  │   │    Rate ≤5%         │
│    fact/decision/   │   │                     │   │  - Retrieval Hit    │
│    hypothesis       │   │  Query Resolution:  │   │    Rate ≥80%        │
│  - Workspace Memory │   │  scope > workspace  │   │  - Injection        │
│    keyword_set/note │   │  > user (fallback)  │   │    Pollution ≤2%    │
│                     │   │                     │   │  - Deletion         │
│  NOT Memory:        │   │                     │   │    Compliance 100%  │
│  - Context cache    │   │                     │   │                     │
│  - Code index       │   │                     │   │                     │
│  - Search results   │   │                     │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │        Implementation Artifacts         │
                    │                                         │
                    │  docs/memory_types.md                   │
                    │  docs/memory_isolation.md               │
                    │  src/paperbot/memory/eval/collector.py  │
                    │  evals/memory/test_*.py                 │
                    └─────────────────────────────────────────┘
```

### 0.4 Data Flow with P0 Touch Points

```
                                 INGESTION
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  POST /api/memory/ingest                                │  │
     │  │  file + user_id + platform + workspace_id               │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  Parsers (ChatGPT/Gemini/Plaintext)                     │  │
     │  │  → NormalizedMessage[]                                  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  Extractor (heuristic + optional LLM)                   │  │
     │  │  → MemoryCandidate[]                                    │  │
     │  │  ╔═══════════════════════════════════════════════════╗  │  │
     │  │  ║ P0: kind must be in defined taxonomy              ║  │  │
     │  │  ║ P0: confidence scoring for precision metrics      ║  │  │
     │  │  ╚═══════════════════════════════════════════════════╝  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     └───────────────────────────────┼───────────────────────────────┘
                                     │
                                 STORAGE
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  MemoryStore.add_memories()                             │  │
     │  │  ╔═══════════════════════════════════════════════════╗  │  │
     │  │  ║ P0: scope-aware dedup (content_hash includes scope)║  │  │
     │  │  ║ P0: status = pending if confidence < 0.6          ║  │  │
     │  │  ║ P0: namespace isolation (user_id + workspace_id)  ║  │  │
     │  │  ╚═══════════════════════════════════════════════════╝  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  memory_items table                                     │  │
     │  │  memory_audit_log table                                 │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                                                               │
     └───────────────────────────────────────────────────────────────┘
                                     │
                                RETRIEVAL
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  POST /api/memory/context                               │  │
     │  │  query + user_id + workspace_id + scope                 │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  MemoryStore.search_memories()                          │  │
     │  │  ╔═══════════════════════════════════════════════════╗  │  │
     │  │  ║ P0: only approved, non-deleted, non-expired       ║  │  │
     │  │  ║ P0: scope isolation enforced                      ║  │  │
     │  │  ║ P0: hit rate metrics collected                    ║  │  │
     │  │  ╚═══════════════════════════════════════════════════╝  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  ContextPack (formatted for prompt injection)           │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                                                               │
     └───────────────────────────────────────────────────────────────┘
                                     │
                               GOVERNANCE
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  Memory Inbox: /api/research/memory/inbox               │  │
     │  │  ╔═══════════════════════════════════════════════════╗  │  │
     │  │  ║ P0: pending → approved/rejected workflow          ║  │  │
     │  │  ║ P0: bulk_moderate, bulk_move, clear               ║  │  │
     │  │  ╚═══════════════════════════════════════════════════╝  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  DELETE /api/memory/items/{id}                          │  │
     │  │  ╔═══════════════════════════════════════════════════╗  │  │
     │  │  ║ P0: soft_delete / hard_delete                     ║  │  │
     │  │  ║ P0: deletion compliance = 100% (never retrieved)  ║  │  │
     │  │  ╚═══════════════════════════════════════════════════╝  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                                                               │
     └───────────────────────────────────────────────────────────────┘
                                     │
                               EVALUATION
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  GET /api/memory/metrics (NEW in P0)                    │  │
     │  │  ╔═══════════════════════════════════════════════════╗  │  │
     │  │  ║ P0: MetricCollector aggregates 5 metrics          ║  │  │
     │  │  ║     - extraction_precision                        ║  │  │
     │  │  ║     - false_positive_rate                         ║  │  │
     │  │  ║     - retrieval_hit_rate                          ║  │  │
     │  │  ║     - injection_pollution_rate                    ║  │  │
     │  │  ║     - deletion_compliance                         ║  │  │
     │  │  ╚═══════════════════════════════════════════════════╝  │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                               │                               │
     │                               ▼                               │
     │  ┌─────────────────────────────────────────────────────────┐  │
     │  │  memory_eval_metrics table (NEW in P0)                  │  │
     │  │  evals/memory/test_*.py (NEW in P0)                     │  │
     │  └─────────────────────────────────────────────────────────┘  │
     │                                                               │
     └───────────────────────────────────────────────────────────────┘
```

---

## 1. Executive Summary

**Objective**: Define clear boundaries for memory types, establish namespace/isolation strategies, and implement measurable acceptance criteria for the cross-platform memory middleware.

**Current State**: The infrastructure (database schema, CRUD APIs, basic extraction/search) is largely complete. P0 focuses on **formalizing definitions** and **implementing evaluation metrics** rather than building new features.

**Scope**: This document covers the three main deliverables from `docs/memory_todo.md` P0 section:
1. Memory type boundary definitions
2. Namespace and isolation strategy
3. Acceptance criteria (5 metrics)

---

## 2. Technical Solution Design

### 2.1 Memory Type Boundaries

Based on the existing `MemoryKind` enum and the TODO requirements, the following formal taxonomy is proposed:

#### 2.1.1 Memory Type Taxonomy

| Category | Kind | Definition | Persistence | Example |
|----------|------|------------|-------------|---------|
| **User Memory** | `profile` | Identity facts (name, role, affiliation) | Permanent until edited | "My name is Jerry" |
| | `preference` | Style/format preferences | Semi-permanent | "I prefer concise answers" |
| | `goal` | Long-term objectives | Session-spanning | "I'm researching LLM memory" |
| | `constraint` | Hard rules/requirements | Permanent | "Never use Chinese in responses" |
| | `project` | Project context/background | Project-scoped | "PaperBot is a research tool" |
| **Episodic Memory** | `fact` | Session-derived facts | Decaying | "User mentioned deadline is Friday" |
| | `decision` | Key decisions made | Project-scoped | "We chose SQLite over Postgres" |
| | `hypothesis` | Tentative assumptions | Low-priority | "User may be a graduate student" |
| | `todo` | Action items | Time-bound | "Need to implement FTS5" |
| **Workspace/Project** | `keyword_set` | Domain keywords | Project-scoped | "focus areas: RAG, memory, agents" |
| | `note` | Free-form annotations | Project-scoped | "Important: check GDPR compliance" |

#### 2.1.2 Exclusions (NOT Memory)

The following are explicitly **NOT** part of the memory system and should be handled elsewhere:

| Item | Reason | Proper Location |
|------|--------|-----------------|
| **Context Caching** | Latency optimization, not long-term storage | LLM provider layer |
| **Code Index** | Repository structure, symbols, dependencies | `context_engine/` or future `code_index` subsystem |
| **Search Results** | Ephemeral retrieval results | Not persisted |
| **Embeddings** | Vector representations | Separate `memory_embeddings` table (P1) |
| **Conversation History** | Raw message logs | `memory_sources` table (input, not output) |

### 2.2 Namespace and Isolation Strategy

#### 2.2.1 Isolation Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                    user_id (required)               │
│  ┌───────────────────────────────────────────────┐  │
│  │            workspace_id (optional)            │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │   scope_type:scope_id (track/project)  │  │  │
│  │  │  ┌─────────────────────────────────────┐│  │  │
│  │  │  │   provider (metadata only)         ││  │  │
│  │  │  └─────────────────────────────────────┘│  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

#### 2.2.2 Isolation Rules

| Level | Field | Required | Behavior |
|-------|-------|----------|----------|
| **L1** | `user_id` | Yes | Primary boundary. Cross-user access is forbidden. |
| **L2** | `workspace_id` | No | Team/project isolation. Memories with `workspace_id=X` are only visible when querying with the same `workspace_id`. |
| **L3** | `scope_type:scope_id` | No | Track-level isolation for research contexts. |
| **Metadata** | `provider` | No | Provenance tracking only (ChatGPT/Gemini/Claude/...). Not used for query filtering. |

#### 2.2.3 Scope Type Values

| scope_type | Description | Visibility |
|------------|-------------|------------|
| `global` | User-wide memories | Visible across all tracks/workspaces for that user |
| `track` | Research direction specific | Only visible when that track is active |
| `project` | Project-specific context | Only visible within project scope |
| `paper` | Paper-specific notes | Only visible when analyzing that paper |

#### 2.2.4 Query Resolution Logic

```python
def resolve_visible_memories(user_id, workspace_id=None, scope_type=None, scope_id=None):
    """
    Resolution order (most specific to least):
    1. Exact scope match (scope_type + scope_id)
    2. Global scope within workspace
    3. Global scope for user (if no workspace specified)
    """
    base_filter = (user_id == user_id)

    if workspace_id:
        base_filter &= (workspace_id == workspace_id)

    if scope_type and scope_id:
        # Include both specific scope AND global
        scope_filter = (
            (scope_type == scope_type AND scope_id == scope_id) |
            (scope_type == 'global')
        )
    else:
        scope_filter = (scope_type == 'global')

    return base_filter & scope_filter
```

### 2.3 Acceptance Criteria Framework

#### 2.3.1 Metric Definitions

| Metric | Definition | Formula | Target |
|--------|------------|---------|--------|
| **Extraction Precision** | Fraction of extracted items that are correct | `correct_items / total_extracted_items` | ≥ 85% |
| **False Positive Rate (Dirty Memory Rate)** | Fraction of approved items that are incorrect or harmful | `incorrect_approved / total_approved` | ≤ 5% |
| **Retrieval Hit Rate** | Fraction of relevant memories retrieved when needed | `retrieved_relevant / total_relevant` | ≥ 80% |
| **Injection Pollution Rate** | Fraction of responses negatively affected by wrong memory | `polluted_responses / total_responses_with_memory` | ≤ 2% |
| **Deletion Compliance** | Deleted items must never be retrieved or injected | `deleted_items_retrieved == 0` | 100% |

#### 2.3.2 Measurement Methods

| Metric | Data Source | Frequency | Automation Level |
|--------|-------------|-----------|------------------|
| Extraction Precision | Human sampling (50 items/week) | Weekly | Manual with script support |
| False Positive Rate | User corrections + manual audit | Weekly | Semi-automated |
| Retrieval Hit Rate | Synthetic test queries + user feedback | Daily (CI) + Weekly (user) | Automated + Manual |
| Injection Pollution Rate | A/B testing + user reports | Monthly | Manual |
| Deletion Compliance | Automated regression test | Per-commit (CI) | Fully automated |

#### 2.3.3 Metric Storage Schema

```sql
CREATE TABLE memory_eval_metrics (
    id INTEGER PRIMARY KEY,
    metric_name TEXT NOT NULL,        -- 'extraction_precision', 'retrieval_hit_rate', etc.
    metric_value REAL NOT NULL,       -- 0.0 to 1.0
    sample_size INTEGER,              -- Number of items evaluated
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluator_id TEXT,                -- 'ci', 'human:<user_id>', 'automated'
    detail_json TEXT,                 -- Additional context (item IDs, query info, etc.)

    INDEX idx_metric_name_time (metric_name, evaluated_at)
);
```

---

## 3. Implementation Principles

### 3.1 Core Design Principles

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Explicit over Implicit** | All memory types must have clear written definitions | Document in `docs/memory_types.md` + code docstrings |
| **Fail-Safe Defaults** | Low confidence or high PII risk → require review | `confidence < 0.6` → `status='pending'` |
| **Auditability First** | Every mutation must be logged | All changes create `memory_audit_log` entries |
| **Isolation by Default** | Scope to narrowest applicable boundary | Default `scope_type='global'`, require explicit promotion |
| **Separation of Concerns** | Each layer has single responsibility | Extraction → Storage → Retrieval → Injection |

### 3.2 Fail-Safe Status Transitions

```
                    ┌──────────────┐
                    │   Extracted  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ confidence ≥ 0.6│      │ confidence < 0.6│
    │ pii_risk < 2    │      │ OR pii_risk ≥ 2 │
    └────────┬────────┘      └────────┬────────┘
             │                        │
             ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐
    │    approved     │      │     pending     │
    │  (auto-active)  │      │ (needs review)  │
    └─────────────────┘      └────────┬────────┘
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                        ▼                           ▼
               ┌─────────────────┐        ┌─────────────────┐
               │    approved     │        │    rejected     │
               │ (human review)  │        │  (discarded)    │
               └─────────────────┘        └─────────────────┘
```

### 3.3 Component Responsibilities

| Component | Responsibility | Should NOT Do |
|-----------|----------------|---------------|
| **Extractor** | Parse input, produce candidates with confidence | Apply business rules, write to DB |
| **Store** | Dedup, apply status rules, persist, audit | Parse input, format output |
| **Retriever** | Find relevant approved items | Return pending/deleted items |
| **Injector** | Format for prompt, respect token budget | Store or modify memories |

---

## 4. Technology Selection Rationale

### 4.1 Metrics Storage: SQLite

| Criterion | SQLite | InfluxDB | Prometheus |
|-----------|--------|----------|------------|
| Consistency with stack | ✅ Same DB | ❌ New dependency | ❌ New infra |
| Volume fit | ✅ Hundreds/day | Overkill | Overkill |
| Backup simplicity | ✅ Single file | ❌ Complex | ❌ Complex |
| Query flexibility | ✅ Full SQL | ⚠️ InfluxQL | ⚠️ PromQL |

**Decision**: SQLite (same as main DB)

### 4.2 Precision Evaluation: Human-in-the-Loop

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Human sampling | Ground truth quality | Labor-intensive | ✅ Selected |
| LLM-as-judge | Automated | Circular dependency | ❌ Rejected |
| Rule-based | Fast | Cannot judge semantic correctness | ❌ Rejected |

**Decision**: Manual sampling (50 items/week) with script support. Build annotated dataset for future automation.

### 4.3 Retrieval Evaluation: Synthetic Test Queries

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Synthetic fixtures | Reproducible, CI-friendly | May miss edge cases | ✅ Selected |
| Production sampling | Real-world coverage | Privacy concerns, non-deterministic | ⚠️ Future |
| User feedback | Direct signal | Sparse, delayed | ⚠️ Supplementary |

**Decision**: Pre-defined query-answer pairs in `evals/memory/fixtures/`. Supplement with user feedback in P1.

---

## 5. Best Practices and References

### 5.1 Industry References

| Source | Key Insight | Application |
|--------|-------------|-------------|
| **Generative Agents** (Stanford, 2023) | Memory streams with reflection layers | Validates episodic vs. long-term distinction |
| **MemGPT** (Berkeley, 2023) | Hierarchical memory with OS-like management | Validates scope-based isolation design |
| **ChatGPT Memory** (OpenAI, 2024) | User-controllable explicit memory | Justifies `pending/approved` workflow |
| **Claude Projects** (Anthropic, 2024) | Project-scoped context isolation | Validates workspace_id pattern |
| **GDPR Article 17** | Right to erasure | Mandates 100% deletion compliance |

### 5.2 Academic Papers

| Paper | Relevance |
|-------|-----------|
| Lewis et al., "RAG for Knowledge-Intensive NLP Tasks" (2020) | Retrieval + generation pattern for injection |
| Thakur et al., "BEIR: Zero-shot IR Evaluation" (2021) | Retrieval metric methodology |
| Park et al., "Generative Agents" (2023) | Memory stream architecture |
| Packer et al., "MemGPT" (2023) | Tiered memory management |

### 5.3 Internal Documents

| Document | Content |
|----------|---------|
| `docs/memory_survey.md` | Survey of memory architectures across products |
| `docs/memory_ui_controls_matrix.md` | UI patterns from ChatGPT, Claude, Gemini, etc. |
| `docs/memory.md` | Current API documentation |
| `src/paperbot/memory/schema.py` | Existing type definitions |

---

## 6. Risks and Mitigations

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Metrics collection slows API | Medium | Medium | Async logging; batch writes; sampling |
| Human labeling bottleneck | High | Low | Start small (50/week); automate obvious cases |
| Deletion compliance gaps | Low | Critical | DB constraint; nightly audit; hard delete for GDPR |
| Scope confusion in queries | Medium | Medium | Clear error messages; API validation; docs |

### 6.2 Process Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Metric definitions drift | Medium | Medium | Document in code; review in PRs |
| Evaluation dataset staleness | High | Low | Quarterly refresh; add edge cases from production |
| Unrealistic targets | Medium | Low | Start with industry benchmarks; adjust after first month |

### 6.3 Compliance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PII in metrics | Low | High | Never log raw content; only IDs and scores |
| Audit log tampering | Low | Critical | Append-only design; consider signed entries |

---

## 7. Workload Estimation

### 7.1 Task Breakdown

| Task | Effort | Dependencies |
|------|--------|--------------|
| **Documentation** | | |
| Create `docs/memory_types.md` with taxonomy | 2h | None |
| Create `docs/memory_isolation.md` with rules | 1h | Taxonomy |
| Update `schema.py` docstrings | 1h | Taxonomy |
| **Evaluation Infrastructure** | | |
| Add `memory_eval_metrics` table | 1h | None |
| Implement `MemoryMetricCollector` class | 3h | Table |
| Hook collector into API endpoints | 2h | Collector |
| Add `/api/memory/metrics` endpoint | 1h | Collector |
| **Test Harness** | | |
| Create extraction precision test | 2h | Collector |
| Create retrieval hit rate test | 2h | Collector |
| Create deletion compliance test | 1h | Collector |
| Add CI job for regression | 1h | Tests |

### 7.2 Summary

| Category | Hours |
|----------|-------|
| Documentation | 4h |
| Evaluation Infrastructure | 7h |
| Test Harness | 6h |
| **Total** | **17h (~2 days)** |

### 7.3 Suggested Timeline

```
Day 1 (Morning):   Documentation (4h)
                   - memory_types.md
                   - memory_isolation.md
                   - schema.py docstrings

Day 1 (Afternoon): Evaluation Infrastructure (4h)
                   - memory_eval_metrics table
                   - MetricCollector class (partial)

Day 2 (Morning):   Evaluation Infrastructure + Tests (5h)
                   - MetricCollector completion
                   - API integration
                   - Test harness

Day 2 (Afternoon): CI Integration + Review (4h)
                   - CI job setup
                   - Code review
                   - Update memory_todo.md
```

---

## 8. Deliverables Checklist

Upon completion of P0, the following artifacts will exist:

### 8.1 Documentation
- [ ] `docs/memory_types.md` - Formal taxonomy with definitions
- [ ] `docs/memory_isolation.md` - Namespace and isolation rules
- [ ] `src/paperbot/memory/schema.py` - Updated docstrings

### 8.2 Code
- [ ] `src/paperbot/memory/eval/__init__.py` - Evaluation module
- [ ] `src/paperbot/memory/eval/collector.py` - MetricCollector implementation
- [ ] `src/paperbot/infrastructure/stores/models.py` - `MemoryEvalMetricModel` added
- [ ] `src/paperbot/api/routes/memory.py` - `/api/memory/metrics` endpoint

### 8.3 Tests
- [ ] `evals/memory/test_extraction_precision.py`
- [ ] `evals/memory/test_retrieval_hit_rate.py`
- [ ] `evals/memory/test_deletion_compliance.py`
- [ ] `evals/memory/fixtures/` - Synthetic test data

### 8.4 CI/CD
- [ ] `.github/workflows/ci.yml` - Memory eval job added

### 8.5 Tracking
- [ ] `docs/memory_todo.md` - P0 items marked complete

---

## 9. Open Questions

The following questions require user input before implementation:

1. **Extraction language support**: Current heuristics are Chinese-focused. Should P0 include English patterns, or defer to later phase?

2. **Human labeling workflow**: Should we build a simple web UI for labeling, or use external tools (Label Studio, spreadsheet)?

3. **Metrics retention period**: How long should evaluation metrics be stored? (30 days / 90 days / indefinitely)

4. **Deletion compliance scope**: Should 100% target apply to soft-delete only, or require hard-delete for GDPR?

---

## Appendix A: Existing Implementation Summary

### A.1 Current Schema (Already Implemented)

**`memory_items` table fields**:
- Core: `id`, `user_id`, `workspace_id`, `scope_type`, `scope_id`
- Content: `kind`, `content`, `content_hash`, `confidence`
- Status: `status` (pending/approved/rejected/superseded), `supersedes_id`
- Lifecycle: `expires_at`, `last_used_at`, `use_count`
- Security: `pii_risk`, `deleted_at`, `deleted_reason`
- Audit: `created_at`, `updated_at`, `source_id`

**`memory_audit_log` table fields**:
- `id`, `ts`, `actor_id`, `user_id`, `workspace_id`
- `action` (create/update/approve/reject/delete/hard_delete/use)
- `item_id`, `source_id`, `detail_json`

### A.2 Current API Endpoints

| Endpoint | Method | Status |
|----------|--------|--------|
| `/api/memory/ingest` | POST | ✅ Implemented |
| `/api/memory/list` | GET | ✅ Implemented |
| `/api/memory/context` | POST | ✅ Implemented |
| `/api/memory/items/{id}` | PATCH | ✅ Implemented |
| `/api/memory/items/{id}` | DELETE | ✅ Implemented |
| `/api/memory/metrics` | GET | ❌ To be added |

### A.3 Current Parsers

| Platform | Format | Status |
|----------|--------|--------|
| ChatGPT | conversations.json | ✅ Working |
| Gemini | API logs (loose JSON) | ✅ Working |
| Generic | {"messages": [...]} | ✅ Working |
| Plain text | User:/Assistant: prefix | ✅ Working |
| Claude | Export format | ❌ Not implemented |
| Perplexity | Unknown | ❌ Not implemented |


  Phase 1: Documentation

  | File                     | Status  |
  |--------------------------|---------|
  | docs/memory_types.md     | Created |
  | docs/memory_isolation.md | Created |

  Phase 2: Evaluation Infrastructure

  | File                                         | Status                                         |
  |----------------------------------------------|------------------------------------------------|
  | src/paperbot/infrastructure/stores/models.py | Modified (added MemoryEvalMetricModel)         |
  | src/paperbot/memory/eval/__init__.py         | Created                                        |
  | src/paperbot/memory/eval/collector.py        | Created                                        |
  | src/paperbot/api/routes/memory.py            | Modified (added /api/memory/metrics endpoints) |

  Phase 3: Test Harness

  | File                                         | Status  |
  |----------------------------------------------|---------|
  | evals/memory/__init__.py                     | Created |
  | evals/memory/fixtures/sample_memories.json   | Created |
  | evals/memory/fixtures/retrieval_queries.json | Created |
  | evals/memory/test_deletion_compliance.py     | Created |
  | evals/memory/test_retrieval_hit_rate.py      | Created |

  Phase 4: CI Integration

  | File                     | Status                           |
  |--------------------------|----------------------------------|
  | .github/workflows/ci.yml | Modified (added memory P0 tests) |