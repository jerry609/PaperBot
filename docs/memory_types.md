# Memory Type Taxonomy

This document defines the formal taxonomy for memory types in PaperBot's cross-platform memory middleware.

## Overview

Memory items are categorized into three main categories based on their purpose and lifecycle:

1. **User Memory** - Stable, cross-session information about the user
2. **Episodic Memory** - Session-derived facts with potential decay
3. **Workspace/Project Memory** - Scope-specific context

## Memory Type Definitions

### User Memory

Long-term, stable information about the user that persists across sessions.

| Kind | Definition | Persistence | Confidence Threshold | Examples |
|------|------------|-------------|---------------------|----------|
| `profile` | Identity facts: name, role, affiliation, background | Permanent until edited | 0.85 | "My name is Jerry", "I'm a PhD student at MIT" |
| `preference` | Style, format, or interaction preferences | Semi-permanent | 0.72 | "I prefer concise answers", "Use Chinese for responses" |
| `goal` | Long-term objectives or research directions | Session-spanning | 0.68 | "I'm researching LLM memory systems", "Working on a paper about RAG" |
| `constraint` | Hard rules or requirements that must be followed | Permanent | 0.62 | "Never include code examples", "Always cite sources" |
| `project` | Project-level context and background | Project-scoped | 0.70 | "PaperBot is a research workflow tool", "The codebase uses FastAPI" |

### Episodic Memory

Information derived from specific sessions or interactions, subject to decay or update.

| Kind | Definition | Persistence | Confidence Threshold | Examples |
|------|------------|-------------|---------------------|----------|
| `fact` | Specific facts mentioned during interaction | Decaying (use_count tracked) | 0.65 | "User's deadline is Friday", "Paper was submitted to NeurIPS" |
| `decision` | Key decisions made during sessions | Project-scoped | 0.70 | "We chose SQLite over Postgres", "Using Docker for sandboxing" |
| `hypothesis` | Tentative assumptions or inferences | Low-priority, may expire | 0.50 | "User may be a graduate student", "Likely interested in NLP" |
| `todo` | Action items or pending tasks | Time-bound (expires_at) | 0.58 | "Need to implement FTS5", "Review the memory module" |

### Workspace/Project Memory

Scope-specific information tied to a particular workspace or research track.

| Kind | Definition | Persistence | Confidence Threshold | Examples |
|------|------------|-------------|---------------------|----------|
| `keyword_set` | Domain-specific keywords or focus areas | Project-scoped | 0.75 | "Focus areas: RAG, memory, multi-agent" |
| `note` | Free-form annotations or observations | Project-scoped | 0.60 | "Important: check GDPR compliance", "Consider using vector DB" |

## What is NOT Memory

The following are explicitly **excluded** from the memory system:

| Item | Reason | Where It Belongs |
|------|--------|------------------|
| **Context Caching** | Performance optimization, not long-term storage | LLM provider layer (prompt caching) |
| **Code Index** | Repository structure, symbols, dependencies | `context_engine/` or `code_index` subsystem |
| **Search Results** | Ephemeral retrieval results | Not persisted; used immediately |
| **Embeddings** | Vector representations of content | Separate `memory_embeddings` table (future) |
| **Raw Conversation History** | Unprocessed message logs | `memory_sources` table (input, not output) |
| **Temporary Variables** | Session-only working memory | In-memory only, not persisted |

## Status Workflow

Memory items follow a status workflow based on confidence and risk:

```
Extracted (MemoryCandidate)
         │
         ├── confidence >= 0.60 AND pii_risk < 2
         │         │
         │         ▼
         │   ┌─────────────┐
         │   │  approved   │ ──── Can be retrieved and injected
         │   └─────────────┘
         │
         └── confidence < 0.60 OR pii_risk >= 2
                   │
                   ▼
             ┌─────────────┐
             │   pending   │ ──── Requires human review
             └─────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
   ┌─────────────┐    ┌─────────────┐
   │  approved   │    │  rejected   │
   └─────────────┘    └─────────────┘
```

## Confidence Score Guidelines

When implementing extractors, use these confidence score ranges:

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.85 - 1.00 | High confidence, explicit statement | Auto-approve |
| 0.70 - 0.84 | Good confidence, clear inference | Auto-approve |
| 0.60 - 0.69 | Moderate confidence | Auto-approve (borderline) |
| 0.50 - 0.59 | Low confidence | Pending (needs review) |
| 0.00 - 0.49 | Very low confidence | Pending or reject |

## PII Risk Levels

| Level | Description | Behavior |
|-------|-------------|----------|
| 0 | No PII detected | Normal processing |
| 1 | Possible PII (redacted patterns found) | Normal, but flagged |
| 2 | High PII risk (raw PII detected) | Auto-pending, requires review |

## Usage in Code

```python
from paperbot.memory.schema import MemoryKind, MemoryCandidate

# Valid memory kinds
valid_kinds = [
    # User Memory
    "profile", "preference", "goal", "constraint", "project",
    # Episodic Memory
    "fact", "decision", "hypothesis", "todo",
    # Workspace Memory
    "keyword_set", "note"
]

# Creating a memory candidate
candidate = MemoryCandidate(
    kind="preference",
    content="User prefers concise answers",
    confidence=0.75,
    tags=["style", "output"],
    evidence="From conversation: 'Please keep responses brief'",
    scope_type="global",
    scope_id=None,
    status="approved"  # Will be set by store based on confidence
)
```

## References

- Design Document: `docs/P0_memory_scope_acceptance_design.md`
- Schema Definition: `src/paperbot/memory/schema.py`
- Store Implementation: `src/paperbot/infrastructure/stores/memory_store.py`
