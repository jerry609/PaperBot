# Memory Namespace and Isolation Strategy

This document describes the isolation boundaries and query resolution rules implemented in the memory system.

> **Reference Implementation:** `src/paperbot/infrastructure/stores/memory_store.py`

## Isolation Hierarchy

Memory items are isolated using a three-level namespace hierarchy, as defined in `memory_todo.md`:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Level 1: user_id (required)                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Level 2: workspace_id (optional)             │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │        Level 3: scope_type:scope_id (optional)      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Level 1: user_id (Required)

The primary isolation boundary. Cross-user access is forbidden.

**Implementation:** Every query in `memory_store.py` starts with:
```python
stmt = select(MemoryItemModel).where(MemoryItemModel.user_id == user_id)
```

## Level 2: workspace_id (Optional)

Team or project-level isolation within a user's memories.

**Implementation:** `memory_store.py` line 293-294:
```python
if workspace_id is not None:
    stmt = stmt.where(MemoryItemModel.workspace_id == workspace_id)
```

**Behavior:**
- If `workspace_id` is set on a memory, it is only visible when querying with the same `workspace_id`
- Memories with `workspace_id=NULL` are user-global

## Level 3: scope_type:scope_id (Optional)

Fine-grained isolation for research tracks, projects, or papers.

**Implementation:** `memory_store.py` lines 296-301:
```python
if scope_type is not None:
    if scope_type == "global":
        stmt = stmt.where(or_(MemoryItemModel.scope_type == scope_type, MemoryItemModel.scope_type.is_(None)))
    else:
        stmt = stmt.where(MemoryItemModel.scope_type == scope_type)
if scope_id is not None:
    stmt = stmt.where(MemoryItemModel.scope_id == scope_id)
```

**Scope Types:**

| scope_type | scope_id | Usage |
|------------|----------|-------|
| `global` | NULL | User-wide memories, visible across all tracks |
| `track` | track_id | Research direction specific |
| `project` | project_id | Project-specific context |
| `paper` | paper_id | Paper-specific notes |

## Content Hash Deduplication

To prevent duplicates while allowing the same content in different scopes, the content hash includes scope information.

**Implementation:** `memory_store.py` lines 168-171:
```python
# Dedup within a user's scope boundary (prevents cross-track pollution/dedup collisions).
content_hash = _sha256_text(
    f"{effective_scope_type}:{effective_scope_id or ''}:{m.kind}:{content}"
)
```

## Status Filtering

Queries only return approved, non-deleted, non-expired items.

**Implementation:** `memory_store.py` lines 303-305:
```python
stmt = stmt.where(MemoryItemModel.deleted_at.is_(None))
stmt = stmt.where(MemoryItemModel.status == "approved")
stmt = stmt.where(or_(MemoryItemModel.expires_at.is_(None), MemoryItemModel.expires_at > now))
```

## Auto-Status Assignment

Status is automatically set based on confidence and PII risk.

**Implementation:** `memory_store.py` lines 174-178:
```python
effective_status = "approved" if float(m.confidence) >= 0.60 else "pending"
pii_risk = _estimate_pii_risk(content)
if pii_risk >= 2 and effective_status == "approved":
    effective_status = "pending"
```

**Rules:**
- `confidence >= 0.60` → `approved`
- `confidence < 0.60` → `pending`
- `pii_risk >= 2` → override to `pending` (requires human review)

## Provider as Metadata

The source platform (ChatGPT/Gemini/Claude) is stored in `memory_sources.platform`, not in `memory_items`.

**Rationale:** From `memory_survey.md` section 6.1:
> 建议拆成三张逻辑表：
> - `memory_sources`：导入批次（平台、文件名、sha256、统计信息）
> - `memory_items`：稳定条目

Provider is for provenance tracking, not query filtering.

## Audit Trail

All mutations are logged in `memory_audit_log`.

**Implementation:** `memory_store.py` lines 205-220:
```python
session.add(
    MemoryAuditLogModel(
        ts=now,
        actor_id=actor_id,
        user_id=user_id,
        workspace_id=workspace_id,
        action="create",
        item_id=row.id,
        ...
    )
)
```

## References

- Requirements: `docs/memory_todo.md` P0 section
- Survey: `docs/memory_survey.md` section 6
- Implementation: `src/paperbot/infrastructure/stores/memory_store.py`
