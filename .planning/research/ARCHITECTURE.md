# Architecture Patterns: v2.0 PostgreSQL Migration & Async Data Layer

**Domain:** Database migration + async refactoring for existing PaperBot app
**Researched:** 2026-03-14
**Milestone:** v2.0 PostgreSQL Migration & Data Layer Refactoring

---

## Current Architecture Snapshot

### SessionProvider and the Engine-Per-Instance Problem

Every store, service, and event log creates its own `SessionProvider(db_url)`, which in turn
calls `create_engine()`. With 17+ store classes plus services and the event log, the process
holds 20+ distinct connection pools at runtime. On SQLite this is tolerable (file-based). On
PostgreSQL, each `create_engine()` call opens a separate `asyncpg` connection pool, wasting
connections and preventing any cross-pool transaction semantics.

```
# Current pattern (17+ instances of this):
class PaperStore:
    def __init__(self, db_url=None):
        self._provider = SessionProvider(db_url)  # creates engine + pool

class SqlAlchemyEventLog:
    def __init__(self, db_url=None):
        self._provider = SessionProvider(db_url)  # another engine + pool
```

### Session Context Manager Usage

All stores use `with self._provider.session() as session:` — a synchronous context manager
returning a sync `Session`. The `sessionmaker` returns the session; stores call
`session.commit()` / `session.add()` directly. There is no async session anywhere today.

### FTS5 Virtual Tables (SQLite-Only)

Two stores create SQLite FTS5 virtual tables at startup via raw DDL:

- `SqlAlchemyMemoryStore._ensure_fts5()` creates `memory_items_fts` + 3 triggers
- `DocumentIndexStore._ensure_fts5()` creates `document_chunks_fts` + triggers

These are outside Alembic metadata. The `_search_fts5()` method explicitly checks
`if not db_url.startswith("sqlite"): return None` — it degrades silently on PostgreSQL.

### sqlite-vec Embedding Storage

`MemoryItemModel` stores embeddings as `LargeBinary` bytes packed as `struct.pack("...f", *vec)`.
On SQLite, `SqlAlchemyMemoryStore._ensure_vec_table()` creates a `vec_items` virtual table.
On PostgreSQL, there is no equivalent; vector search falls back to keyword-only (FTS5 path
returns None, vec path returns empty). This is the biggest functional gap in the migration.

### Alembic: Already Dual-DB Aware

`alembic/env.py` already detects PG URLs and applies `prepare_threshold: 0` for PgBouncer
compatibility. `ensure_tables()` on `SessionProvider` skips table creation for PostgreSQL —
it relies on Alembic exclusively. This is correct architecture already.

### MCP Tool Pattern: anyio.to_thread.run_sync()

All MCP tools that call sync stores wrap with `anyio.to_thread.run_sync(lambda: ...)`. This
is the current async/sync boundary. It is correct and safe for the interim period, but adds
thread overhead. Once stores become async, this bridge can be removed.

### FastAPI Routes: Sync Store Calls in Async Handlers

Route handlers are `async def` but call stores synchronously. Example from `runs.py`:
```python
async def list_runs(request: Request):
    return {"runs": event_log.list_runs(limit=limit)}  # sync call in async handler
```
This blocks the event loop. On SQLite with typical loads it is hidden. On PostgreSQL under
concurrent load it will degrade. Converting stores to async eliminates this.

### ARQ Worker: Sync Event Log in Async Jobs

ARQ job functions are `async def` but use `SqlAlchemyEventLog.append()` which is synchronous.
The worker module holds a module-level `_EVENT_LOG` singleton. This is a concurrency hazard
if tasks run concurrently (ARQ parallelism > 1) because the same sync session factory is
used across tasks. With async stores, each ARQ task should get its own `AsyncSession`.

### DI Container: Synchronous Factory Registry

`Container.register(interface, factory, singleton=True)` stores callable factories. There is
no concept of async factories or async initialization. The container must gain support for
async-initialized singletons (specifically: the shared async engine).

---

## Integration Architecture for v2.0

### Core Principle: Single Shared Async Engine

Replace the N-engine-per-store pattern with one shared `AsyncEngine` created at process
startup and injected via the DI container. All stores receive an `async_sessionmaker` from
this shared engine.

```
# v2.0 target: one engine, many session factories sharing the pool
AsyncEngine (created once at startup)
  |
  +-- async_sessionmaker (one factory)
        |
        +-- PaperStore (receives factory)
        +-- ResearchStore (receives factory)
        +-- MemoryStore (receives factory)
        +-- SqlAlchemyEventLog (receives factory)
        +-- (all 17+ stores)
```

### New Component: AsyncSessionProvider

`AsyncSessionProvider` replaces `SessionProvider`. It accepts an `async_sessionmaker` rather
than creating its own engine. The store's `__init__` no longer calls `create_engine()`.

```python
# infrastructure/stores/async_db.py (NEW)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine

def create_async_db_engine(db_url: str | None = None) -> AsyncEngine:
    url = _coerce_to_async_url(db_url or get_db_url())
    connect_args = {}
    if "postgresql" in url:
        connect_args = {"prepare_threshold": 0}  # PgBouncer compat
    return create_async_engine(
        url,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args=connect_args,
    )

def create_async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, autoflush=False, expire_on_commit=False)

class AsyncSessionProvider:
    """Thin wrapper: accepts an injected async_sessionmaker. Does NOT create an engine."""
    def __init__(self, factory: async_sessionmaker[AsyncSession]):
        self._factory = factory

    def session(self) -> AsyncSession:
        return self._factory()
```

Key difference from `SessionProvider`: `expire_on_commit=False` is mandatory. In async
contexts, accessing expired attributes after commit raises `MissingGreenlet`. Setting
`expire_on_commit=False` means attribute access post-commit is safe.

### URL Coercion Helper

asyncpg requires `postgresql+asyncpg://` scheme. The helper converts existing env var URLs:

```python
def _coerce_to_async_url(url: str) -> str:
    """Convert postgresql:// or postgres:// to postgresql+asyncpg://"""
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return url.replace("://", "+asyncpg://", 1)
    if url.startswith("sqlite:"):
        return url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    return url
```

The `PAPERBOT_DB_URL` env var does not need to change format. The coercion happens
transparently in `create_async_db_engine()`.

### Modified Component: DI Container

Add an `AsyncEngine` registration slot to `bootstrap_dependencies`. The engine is created
once and registered as a singleton. All stores resolve it.

```python
# core/di/bootstrap.py additions
async def bootstrap_async_db(container: Container, db_url: str | None = None) -> None:
    """Call once at app startup (inside async startup event)."""
    from paperbot.infrastructure.stores.async_db import (
        create_async_db_engine, create_async_session_factory
    )
    engine = create_async_db_engine(db_url)
    factory = create_async_session_factory(engine)
    container.register(AsyncEngine, lambda: engine, singleton=True)
    container.register(async_sessionmaker, lambda: factory, singleton=True)
```

FastAPI startup hook wires this:
```python
@app.on_event("startup")
async def _startup_db():
    await bootstrap_async_db(Container.instance())
```

### Modified Pattern: Store Constructor

Stores change from creating their own `SessionProvider` to receiving an injected factory:

```python
# Before
class PaperStore:
    def __init__(self, db_url=None):
        self.db_url = db_url or get_db_url()
        self._provider = SessionProvider(self.db_url)

# After
class PaperStore:
    def __init__(self, factory: async_sessionmaker | None = None):
        resolved = factory or Container.instance().resolve(async_sessionmaker)
        self._provider = AsyncSessionProvider(resolved)
```

### Modified Pattern: Store Methods

All store methods become `async def` using `async with` session context:

```python
# Before
def get_paper(self, paper_id: int) -> PaperModel | None:
    with self._provider.session() as session:
        return session.get(PaperModel, paper_id)

# After
async def get_paper(self, paper_id: int) -> PaperModel | None:
    async with self._provider.session() as session:
        result = await session.get(PaperModel, paper_id)
        return result
```

For relationship access, use `selectinload` / `joinedload` eagerly. Lazy loading raises
`MissingGreenlet` in async context:

```python
# Relationships must be eagerly loaded
from sqlalchemy.orm import selectinload

async def get_paper_with_authors(self, paper_id: int):
    async with self._provider.session() as session:
        stmt = (
            select(PaperModel)
            .where(PaperModel.id == paper_id)
            .options(selectinload(PaperModel.author_links))
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
```

### Modified Pattern: FastAPI Route Handlers

After stores become async, route handlers call `await store.method()` directly. The
`anyio.to_thread.run_sync()` wrapper in MCP tools is also removed:

```python
# Before (MCP tools)
result = await anyio.to_thread.run_sync(lambda: store.add_memories(...))

# After (MCP tools, stores async)
result = await store.add_memories(...)
```

Route handlers already use `async def`. After the store conversion they simply `await`:

```python
# FastAPI route handler - no change to signature
@router.get("/runs")
async def list_runs(request: Request):
    return {"runs": await event_log.list_runs(limit=limit)}  # now truly async
```

### Modified Pattern: ARQ Worker

ARQ job functions are already `async def`. The module-level `_EVENT_LOG` singleton is
replaced with a per-process `DatabaseConnectionManager` (started in ARQ's `startup` hook):

```python
# infrastructure/queue/arq_worker.py changes
from contextvars import ContextVar

_db_session_context: ContextVar[str | None] = ContextVar("arq_session_ctx", default=None)
_db_manager: DatabaseConnectionManager | None = None

async def startup(ctx) -> None:
    global _db_manager
    _db_manager = DatabaseConnectionManager(get_db_url())
    await _db_manager.connect()
    ctx["db_manager"] = _db_manager

async def shutdown(ctx) -> None:
    if _db_manager:
        await _db_manager.disconnect()

async def on_job_start(ctx, cid=None) -> None:
    _db_session_context.set(ctx.get("job_id", ""))

# Each task receives a fresh AsyncSession via scoped session
async def cron_track_subscriptions(ctx) -> dict:
    async with _db_manager.get_session() as session:
        elog = AsyncSqlAlchemyEventLog(session)
        # ... rest of job
```

This ensures each ARQ task has its own `AsyncSession` (scoped by `job_id` ContextVar),
preventing session sharing across concurrent tasks.

### Modified Component: Alembic env.py

Alembic's `run_migrations_online()` must use an async-aware runner for asyncpg. The standard
pattern for async Alembic:

```python
# alembic/env.py additions for async support
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

def run_migrations_online_async() -> None:
    url = _get_db_url()
    if not (url.startswith("postgresql") or url.startswith("postgres")):
        # SQLite still uses sync path during dev/test
        run_migrations_online_sync()
        return

    async_url = _coerce_to_async_url(url)
    connectable = create_async_engine(async_url, poolclass=pool.NullPool)

    async def _run():
        async with connectable.connect() as connection:
            await connection.run_sync(
                lambda sync_conn: context.configure(
                    connection=sync_conn,
                    target_metadata=target_metadata,
                    compare_type=True,
                    render_as_batch=False,  # PG supports native ALTER
                )
            )
            async with connection.begin():
                await connection.run_sync(context.run_migrations)

    asyncio.run(_run())
```

SQLite batch migrations remain on the sync path. PostgreSQL uses native ALTER TABLE, so
`render_as_batch=False` is correct.

---

## PostgreSQL-Native Feature Integration

### FTS5 → tsvector

The two FTS5 tables (`memory_items_fts`, `document_chunks_fts`) are replaced by PostgreSQL
tsvector columns and GIN indexes. This is a pure Alembic migration — no store code change
beyond swapping the SQL query.

```sql
-- Migration: add tsvector column to memory_items
ALTER TABLE memory_items ADD COLUMN content_tsv tsvector;
UPDATE memory_items SET content_tsv = to_tsvector('english', coalesce(content, ''));
CREATE INDEX idx_memory_items_content_tsv ON memory_items USING GIN (content_tsv);

-- Auto-update trigger
CREATE TRIGGER memory_items_tsv_update
BEFORE INSERT OR UPDATE ON memory_items
FOR EACH ROW EXECUTE FUNCTION
  tsvector_update_trigger(content_tsv, 'pg_catalog.english', content);
```

The `_search_fts5()` method becomes `_search_tsvector()` with a dialect check:

```python
def _search_tsvector(self, tokens: list[str], **scope):
    """PostgreSQL tsvector FTS. Returns None on SQLite (use keyword fallback)."""
    if self._is_sqlite:
        return None
    query = " & ".join(tokens)
    stmt = (
        select(MemoryItemModel)
        .where(MemoryItemModel.content_tsv.match(query))
        .order_by(func.ts_rank(MemoryItemModel.content_tsv, func.plainto_tsquery(query)).desc())
        .limit(limit)
    )
```

### sqlite-vec → pgvector

Replace `LargeBinary` embedding storage with `pgvector`'s `VECTOR(1536)` column type:

```python
# models.py: swap LargeBinary for pgvector
from pgvector.sqlalchemy import Vector

class MemoryItemModel(Base):
    # Before: embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(1536), nullable=True)
```

Alembic migration: drop the `LargeBinary` column, add `VECTOR(1536)`, create HNSW index:

```sql
ALTER TABLE memory_items DROP COLUMN embedding;
ALTER TABLE memory_items ADD COLUMN embedding vector(1536);
CREATE INDEX idx_memory_items_embedding ON memory_items USING hnsw (embedding vector_cosine_ops);
```

The `pgvector` Python package (`pip install pgvector`) provides the `Vector` type for SQLAlchemy.
This is MEDIUM confidence — pgvector is well-established but requires the PostgreSQL extension
to be enabled in the server (`CREATE EXTENSION IF NOT EXISTS vector`). Docker image and migration
must handle this.

### JSON Text Columns → JSONB

All `*_json` columns (e.g., `authors_json`, `keywords_json`, `payload_json`) currently store
Python-serialized strings with manual `json.loads()` / `json.dumps()` helpers. On PostgreSQL
these can become native `JSONB`:

```python
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON

# Use JSON (generic) in model; let dialect map to JSONB on PG, TEXT on SQLite
class PaperModel(Base):
    keywords: Mapped[dict | list] = mapped_column(JSON, default=list)
```

Using `sqlalchemy.JSON` (not `postgresql.JSONB`) keeps models dialect-neutral. SQLAlchemy
maps `JSON` to `jsonb` on PostgreSQL and `TEXT` with serialization on SQLite. All the
manual `get_keywords()` / `set_keywords()` helpers become unnecessary once models use `JSON`.

**Caution:** Migrating existing `*_json TEXT` columns to `JSONB` requires a data migration.
Alembic can use `ALTER COLUMN ... TYPE jsonb USING column::jsonb` but only if existing data
is valid JSON. Rows with empty strings or malformed JSON must be cleaned first.

---

## Data Model Refactoring Plan

### Normalization Targets

| Current Pattern | Problem | PostgreSQL Target |
|----------------|---------|-------------------|
| `authors_json TEXT` (JSON string in every `papers` row) | No FK to `authors` table; denormalized | `paper_authors` join table + `authors` table (already exists, link properly) |
| `keywords_json TEXT` | No indexing, full string match only | `JSONB` column with `@>` containment queries |
| `sources_json TEXT` | Ad-hoc string; no enum validation | `JSONB` + `CHECK (sources_json @> '[]')` |
| `metadata_json TEXT` (on 20+ models) | Catch-all dump; poor queryability | Keep as `JSONB`; add specific columns for frequently queried fields |
| `status` strings (no constraint) | Any string accepted | `VARCHAR(32)` + `CHECK (status IN (...))` |
| Nullable `created_at` (many models) | Inconsistent audit trail | `NOT NULL DEFAULT now()` |

### Models with PG-Native Upgrade Opportunity

| Model | Current | v2.0 |
|-------|---------|-------|
| `MemoryItemModel` | `embedding: LargeBinary`, `content: Text`, FTS via virtual table | `embedding: VECTOR(1536)`, `content_tsv: TSVECTOR`, tsvector trigger |
| `PaperModel` | `keywords_json: Text`, `authors_json: Text` | `keywords: JSONB`, `authors: JSONB` |
| `AgentEventModel` | `payload_json: Text`, `metrics_json: Text`, `tags_json: Text` | `payload: JSONB`, `metrics: JSONB`, `tags: JSONB` |
| `AgentRunModel` | `metadata_json: Text` | `metadata: JSONB` |
| `ResearchTrackModel` | `keywords_json: Text`, `venues_json: Text`, `methods_json: Text` | `keywords: JSONB`, `venues: JSONB`, `methods: JSONB` |

### Constraint Hardening

Add to PostgreSQL-specific Alembic migrations:
- `CHECK (status IN ('pending', 'running', 'completed', 'failed'))` on status columns
- `CHECK (confidence BETWEEN 0.0 AND 1.0)` on `MemoryItemModel.confidence`
- `NOT NULL DEFAULT NOW()` on all `created_at` columns that are currently nullable
- `CHECK (pii_risk IN (0, 1, 2))` on `MemoryItemModel.pii_risk`

---

## What Is Preserved vs. Must Change

### Preserved (Zero Changes)

| Component | Why Preserved |
|-----------|--------------|
| `Base(DeclarativeBase)` | No change; add JSONB/VECTOR types incrementally |
| All model `__tablename__` values | Schema names do not change |
| `alembic/versions/` directory | Existing migrations remain valid history |
| `Container` class interface | `register()` / `resolve()` API unchanged |
| `AgentEventEnvelope` schema | Event envelope structure unchanged |
| All port interfaces (`EventLogPort`, `RegistryPort`, etc.) | Contracts preserved; implementations change internally |
| MCP server registration pattern | `register(mcp)` pattern unchanged |
| ARQ `WorkerSettings.functions` | Function names unchanged; internals refactored |
| FastAPI route signatures | `async def` already; `await` additions only |

### Must Change

| Component | Change Required | Risk |
|-----------|-----------------|------|
| `sqlalchemy_db.py` — `SessionProvider` | Add `AsyncSessionProvider`; keep `SessionProvider` for backward compat during transition | Low |
| All 17+ store `__init__` | Accept injected `async_sessionmaker` instead of creating engine | Medium (mechanical but many files) |
| All store methods | Convert `def` to `async def`, `with` to `async with` | High (pervasive change) |
| `sqlalchemy_event_log.py` | Convert `append()`, `stream()`, `list_runs()`, `list_events()` to async | Medium |
| `bootstrap.py` | Add `bootstrap_async_db()` async factory | Low |
| `arq_worker.py` | Add `startup/shutdown/on_job_start` hooks; per-task session management | Medium |
| `alembic/env.py` | Add async migration path for PostgreSQL | Low |
| All MCP tools using `anyio.to_thread` | Remove wrapper after stores go async | Low (cleanup) |
| `memory_store.py` `_ensure_fts5()` / `_ensure_vec_table()` | Replace with tsvector + pgvector; keep SQLite fallback in `_search_*` methods | High |
| `document_index_store.py` `_ensure_fts5()` | Replace with tsvector | Medium |
| JSON helper methods (`get_keywords`, `set_keywords`, etc.) | Remove after JSON column type switch; direct attribute access | Medium |
| `models.py` JSON columns (`*_json: Text`) | Rename + change type to `JSON`/`JSONB` per model | High (requires data migration) |
| `models.py` embedding column | Change `LargeBinary` to `Vector(1536)` | High (data migration + pgvector extension) |

---

## Backward Compatibility Strategy

### Phase Approach: Sync-First, Then Async

Do NOT attempt a big-bang sync-to-async conversion. The risk of breaking 40+ test files and
all CI gates is too high. Use a two-phase approach:

**Phase A — PostgreSQL + Schema (sync stays):**
- Set up PostgreSQL + Docker dev environment
- Add `asyncpg` + `aiosqlite` to dependencies
- Create new Alembic migrations for PG-native columns (JSONB, tsvector, pgvector)
- Run all existing tests against PostgreSQL — sync stores still work on PG
- Fix any PostgreSQL-incompatible DDL (FTS5 virtual tables, sqlite-vec)
- Deliver: PG works with existing sync stores

**Phase B — Async Data Layer:**
- Add `AsyncSessionProvider` to `sqlalchemy_db.py` alongside `SessionProvider`
- Convert stores one domain at a time (memory, papers, research, event log, etc.)
- For each converted store: update tests to use `pytest-anyio` / `asyncio` fixtures
- Update MCP tools to drop `anyio.to_thread.run_sync()` wrapper
- Update FastAPI routes to `await` store calls
- Update ARQ worker with lifecycle hooks
- Deliver: full async data layer

**Phase C — Model Refactoring:**
- Convert `*_json TEXT` columns to `JSON`/`JSONB` with data migration scripts
- Add constraint checks
- Remove JSON helper methods from models; use direct attribute access
- Clean up dead code

### Keeping SQLite Dev Support

During Phase A and B, SQLite continues to work for local `pytest`. The `AsyncSessionProvider`
with `aiosqlite` makes this possible. Only Phase C features (tsvector, pgvector, JSONB
operators) are PostgreSQL-only. Tests that exercise FTS or vector search can be marked
`@pytest.mark.skipif(is_sqlite, reason="PG-only")`.

---

## Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `async_db.py` (NEW) | Create and own the single shared `AsyncEngine`; provide `AsyncSessionProvider` | DI container (receives engine), all stores (receive factory) |
| `AsyncSessionProvider` (NEW) | Thin wrapper: yields `AsyncSession` from injected factory | Store methods (`async with`) |
| `SessionProvider` (KEEP) | Sync wrapper for test fixtures and migration scripts | Alembic env, unit tests |
| `bootstrap_async_db()` (NEW) | One-time startup: create engine, register in DI | FastAPI `startup` event, ARQ `startup` hook |
| Each store (MODIFIED) | Same domain logic, now with `async def` methods | `AsyncSessionProvider`, SQLAlchemy ORM |
| `SqlAlchemyEventLog` (MODIFIED) | Async `append()` + `list_runs()` | ARQ worker, FastAPI startup, CompositeEventLog |
| `alembic/env.py` (MODIFIED) | Dual path: async PG migrations, sync SQLite migrations | Alembic CLI |
| MCP tools (MODIFIED) | Remove `anyio.to_thread`; directly `await` store methods | Async stores |

---

## Data Flow Changes

### Before (sync everywhere)

```
FastAPI async handler
  |
  v (blocking call — blocks event loop)
Store.sync_method()
  |
  v
SessionProvider.session() — sync context manager
  |
  v
SQLAlchemy sync Session
  |
  v
psycopg2 / sqlite3 driver (blocking I/O)
```

### After (async throughout)

```
FastAPI async handler
  |
  v (non-blocking await)
await Store.async_method()
  |
  v
AsyncSessionProvider.session() — async context manager
  |
  v
SQLAlchemy AsyncSession
  |
  v
asyncpg / aiosqlite driver (non-blocking I/O)
```

### MCP Tools Before/After

```
# Before
async def _save_to_memory_impl(...):
    store = _get_store()
    result = await anyio.to_thread.run_sync(
        lambda: store.add_memories(user_id, [candidate])
    )

# After
async def _save_to_memory_impl(...):
    store = _get_store()
    result = await store.add_memories(user_id, [candidate])
```

---

## Scalability Considerations

| Concern | Phase A (PG, sync stores) | Phase B (PG, async stores) | Phase C (full refactor) |
|---------|--------------------------|---------------------------|------------------------|
| Concurrent API requests | Event loop blocks on sync DB calls | Non-blocking; connection pool shared | Same as Phase B |
| Connection pool exhaustion | 20+ independent pools | Single pool, configurable size | Same as Phase B |
| FTS search | Sync tsvector queries (still blocks) | Async tsvector queries | Same as Phase B |
| Vector search | Sync pgvector queries | Async pgvector queries | Same as Phase B |
| ARQ job concurrency | Per-task sync sessions (risk of contention) | Per-task async sessions (safe) | Same as Phase B |

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Converting All Stores in One PR
**What goes wrong:** 17+ stores, all tests fail simultaneously, CI blocked for days.
**Prevention:** Convert one domain group at a time. Each group has its own PR + test pass.
**Domain groups:** (1) event log, (2) memory store, (3) paper store + research store, (4) remaining 13 stores.

### Anti-Pattern 2: Lazy-Loading Relationships in Async Context
**What goes wrong:** `session.get(Model, id)` succeeds; `model.relationship_attr` raises
`MissingGreenlet` after session closes.
**Prevention:** Add `selectinload()` / `joinedload()` to every query that accesses relationships.
Set `expire_on_commit=False` on the session factory (already noted above).

### Anti-Pattern 3: Running Alembic Autogenerate on Mixed Schema
**What goes wrong:** Alembic sees FTS5 virtual tables in SQLite metadata as "extra tables" and
generates `DROP TABLE memory_items_fts` migrations that break SQLite.
**Prevention:** FTS5 tables are created outside `Base.metadata`; Alembic autogenerate does not
see them. Do not change this. PostgreSQL tsvector columns go in regular models and ARE seen by
autogenerate — which is correct.

### Anti-Pattern 4: Using `create_all()` on PostgreSQL
**What goes wrong:** `metadata.create_all(engine)` on PostgreSQL bypasses Alembic; migration
history becomes inconsistent.
**Prevention:** `ensure_tables()` on `SessionProvider` already skips PostgreSQL (`startswith("sqlite")`).
Keep this guard. Never call `create_all()` on a PostgreSQL URL.

### Anti-Pattern 5: Sharing AsyncSession Across Concurrent ARQ Tasks
**What goes wrong:** `AsyncSession` is not thread-safe or task-safe. Multiple concurrent ARQ
tasks using the same session cause data corruption or connection errors.
**Prevention:** Use `async_scoped_session` with a `ContextVar` scoped to the ARQ job ID, as
documented by the ARQ + SQLAlchemy pattern. One session per task, always.

### Anti-Pattern 6: Migrating JSON Columns Without Data Cleanup
**What goes wrong:** `ALTER COLUMN keywords_json TYPE jsonb USING keywords_json::jsonb` fails
if any row contains `""` (empty string) or malformed JSON.
**Prevention:** Run a cleanup query before the type migration:
`UPDATE papers SET keywords_json = '[]' WHERE keywords_json = '' OR keywords_json IS NULL`.
Do this in the Alembic `upgrade()` before the `ALTER COLUMN`.

---

## Build Order (Dependency-Driven)

1. **Docker + PostgreSQL dev environment** — Nothing works without a PG target.
   - Blocks: all subsequent phases

2. **Alembic dual-path env.py + async deps** — `asyncpg`, `aiosqlite`, `pgvector` in
   `pyproject.toml`; Alembic async runner for PG.
   - Depends on: Docker PG
   - Blocks: all migrations

3. **Schema migrations (PostgreSQL-compatible models)** — Convert FTS5 → tsvector, sqlite-vec
   → pgvector column, JSON text → JSONB columns. Write new Alembic migrations (0020+).
   - Depends on: Alembic async env
   - Blocks: PG-native feature usage

4. **AsyncSessionProvider + bootstrap_async_db** — New `async_db.py`, DI registration.
   - Depends on: nothing (new file)
   - Blocks: async store conversion

5. **Data migration scripts** — pgloader or custom Python to move SQLite → PostgreSQL data.
   - Depends on: schema migrations
   - Blocks: production cutover

6. **Store-by-store async conversion** — Four domain groups, one at a time. Start with
   `SqlAlchemyEventLog` (smallest, most impactful for ARQ) then memory, then papers/research,
   then remaining stores.
   - Depends on: AsyncSessionProvider
   - Blocks: MCP tool cleanup, route cleanup

7. **ARQ worker async lifecycle** — `startup/shutdown/on_job_start` hooks; per-task sessions.
   - Depends on: async event log (step 6, group 1)
   - Blocks: safe concurrent ARQ execution

8. **MCP tool cleanup** — Remove `anyio.to_thread.run_sync()` wrappers.
   - Depends on: all stores async (step 6 complete)
   - Blocks: nothing (cleanup)

9. **Model refactoring** — Remove JSON helper methods; add constraints; normalize authors.
   - Depends on: JSONB migrations (step 3)
   - Blocks: nothing (cleanup + hardening)

---

## Sources

- Codebase inspection: `src/paperbot/infrastructure/stores/sqlalchemy_db.py` (SessionProvider)
- Codebase inspection: `src/paperbot/infrastructure/stores/models.py` (46 models, LargeBinary embedding, JSON text columns)
- Codebase inspection: `src/paperbot/infrastructure/stores/memory_store.py` (FTS5 + sqlite-vec patterns)
- Codebase inspection: `src/paperbot/infrastructure/stores/document_index_store.py` (FTS5 pattern)
- Codebase inspection: `src/paperbot/infrastructure/event_log/sqlalchemy_event_log.py` (sync event log)
- Codebase inspection: `src/paperbot/infrastructure/queue/arq_worker.py` (module-level singleton, async jobs)
- Codebase inspection: `src/paperbot/mcp/tools/save_to_memory.py` (anyio.to_thread pattern)
- Codebase inspection: `alembic/env.py` (dual-DB detection already present)
- Codebase inspection: `pyproject.toml` (`psycopg[binary]>=3.2.0` already a dependency)
- [SQLAlchemy 2.0 Asyncio Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) — HIGH confidence
- [ARQ + SQLAlchemy Done Right](https://wazaari.dev/blog/arq-sqlalchemy-done-right) — MEDIUM confidence (async_scoped_session + ContextVar pattern)
- [FastAPI SQLAlchemy 2.0 Modern Async Patterns](https://dev-faizan.medium.com/fastapi-sqlalchemy-2-0-modern-async-database-patterns-7879d39b6843) — MEDIUM confidence
- [Alembic Batch Migrations for SQLite](https://alembic.sqlalchemy.org/en/latest/batch.html) — HIGH confidence
- [pgvector GitHub](https://github.com/pgvector/pgvector) — HIGH confidence
- Project context: `.planning/PROJECT.md` (v2.0 milestone definition)
