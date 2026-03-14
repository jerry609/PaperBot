# Feature Landscape

**Domain:** PostgreSQL migration + async data layer + systematic model refactoring (brownfield)
**Researched:** 2026-03-14
**Confidence:** HIGH — patterns are well-established; specific complexity estimates are from codebase analysis

---

## Scope Note

This file covers **v2.0: PostgreSQL Migration & Data Layer Refactoring** only. The existing file
covered v1.1 Agent Orchestration Dashboard. This milestone inherits a specific brownfield baseline:

- 46 SQLAlchemy 2.0 `Mapped`/`mapped_column` models in a single `models.py` (1 500+ lines)
- Sync `SessionProvider` + `session()` pattern across 17 stores
- FTS5 virtual tables + sqlite-vec virtual table in `memory_store.py` and `document_index_store.py`
- 92 JSON-serialized `Text` columns (hand-rolled `_json` suffix + `json.dumps/loads` helpers)
- 32 Alembic migrations (SQLite chain)
- `psycopg[binary]>=3.2.0` already in `pyproject.toml` — sync driver present, async driver absent
- `create_async_engine` / `asyncpg` / `psycopg[async]` — none present anywhere in `src/`

---

## Table Stakes

Features that must exist for the milestone to be considered complete. Missing any of these means
the migration is not production-ready.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **PostgreSQL engine + async session factory** | AsyncSession + asyncpg replaces sync SessionProvider. Without this, no other feature in the milestone is possible. | MEDIUM | Replace `create_engine` / `sessionmaker` in `sqlalchemy_db.py` with `create_async_engine` / `async_sessionmaker`. New `AsyncSessionProvider` class. Keep sync path for Alembic env.py (sync is required for `run_migrations_online`). |
| **Alembic env.py async-aware config** | Alembic must be able to apply migrations against PostgreSQL. Common trap: using sync Alembic env against async engine breaks in production. | MEDIUM | Standard pattern: add `run_async_migrations()` function in `env.py` using `AsyncEngine.begin()`. Sync fallback remains for SQLite CI tests. Alembic 1.13+ supports this natively. |
| **Store-by-store async conversion (17 stores)** | Every store that calls `self._provider.session()` currently blocks the event loop when used in FastAPI async routes. 17 stores × ~10 methods each = ~170 method conversions. | HIGH | Each `def method` becomes `async def method` with `await session.execute()`, `await session.commit()`, `await session.refresh()`. Most critical path: `paper_store`, `memory_store`, `research_store`, `document_index_store`. ARQ worker stores need ARQ-specific session lifecycle (on_job_start/after_job_end hooks), not FastAPI DI. |
| **Eager loading for all relationships** | Async SQLAlchemy silently breaks lazy loading — attribute access on an unloaded relationship raises `MissingGreenlet` in async context. All ORM relationships currently use default `lazy="select"` (sync). | HIGH | Audit every `relationship()` declaration in models.py. Most relationships are append-only audit trails (one-to-many) → use `lazy="write_only"`. For read paths: add `selectinload()` to queries that access related collections. `joinedload` for simple many-to-one FKs. |
| **Text → JSONB column migration** | 92 `Text` columns storing hand-serialized JSON. PostgreSQL can store and query these natively as JSONB, which is both faster and queryable. Without this, the migration is superficial. | MEDIUM | Replace `Text` + `json.dumps/loads` helpers with `sqlalchemy.dialects.postgresql.JSONB`. Cross-DB compatibility: use `TypeDecorator` with `with_variant(JSONB(), "postgresql")` for columns that must still work in SQLite test env. Alembic migration: `op.alter_column(..., type_=JSONB, postgresql_using="col::jsonb")`. 92 columns across 46 models — batch by model group. |
| **FTS5 → PostgreSQL tsvector (memory + documents)** | `memory_store._search_fts5()` and `document_index_store._search_fts5()` create SQLite-only virtual tables. These break on PostgreSQL silently (they fall back to no FTS). | HIGH | Two replacement strategies: (A) **Generated tsvector column** — `ALTER TABLE memory_items ADD COLUMN fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED` + GIN index. Query with `@@` operator via `func.to_tsvector(...).bool_op("@@")(func.plainto_tsquery(...))`. (B) **Application-side tsquery** — call `to_tsvector`/`plainto_tsquery` in SQLAlchemy Core expressions. Strategy A is preferred: index is maintained by PG automatically, no trigger management. Remove existing FTS5 virtual table creation + 3 insert/update/delete triggers per table. |
| **sqlite-vec → pgvector (memory embeddings)** | `memory_store._ensure_vec()` creates `vec_items USING vec0(...)` virtual table. This is SQLite-only. `MemoryItemModel.embedding` stores raw `LargeBinary` blobs. | MEDIUM | Enable `pgvector` extension via Alembic: `op.execute("CREATE EXTENSION IF NOT EXISTS vector;")`. Replace `LargeBinary` with `pgvector.sqlalchemy.Vector(N_DIM)`. Replace `vec_items` virtual table with native column on `memory_items`. Replace `vec0 MATCH` query with pgvector cosine distance operator `<=>`. Register `vector` type in Alembic `ischema_names` to silence `alembic check` warnings. |
| **Docker Compose for local PostgreSQL dev** | Developers need a PG instance without manual setup. This is the baseline dev environment assumption for all migration work. | LOW | `docker-compose.yml` with `postgres:16-alpine`, named volume, health check. `.env` update: `PAPERBOT_DB_URL=postgresql+asyncpg://paperbot:paperbot@localhost:5432/paperbot`. |
| **Data migration tooling (SQLite → PG)** | Existing users have SQLite databases that contain real data. Without a migration path, the version upgrade is a breaking change with data loss. | HIGH | Two-phase approach: (1) `alembic upgrade head` against fresh PG to create schema; (2) data export script using pgloader or custom Python script to transfer rows. Key risks: FTS5 virtual tables cannot be exported by pgloader (skip them; they rebuild from source data). JSONB cast: pgloader handles `TEXT → JSONB` automatically if JSON is valid. Vector blobs: custom Python script to re-read `LargeBinary` bytes, decode as `float32` array, insert as pgvector. |
| **Systematic model refactoring** | 46 models accumulated organically. Normalization, constraint correctness, and redundancy removal are required before PG adoption or the schema debt compounds. | HIGH | Four categories of work: (a) Add missing `NOT NULL` constraints (many nullable columns are never actually null); (b) Extract repeated JSON payload patterns into proper FK relationships where query frequency justifies it; (c) Add missing `UniqueConstraint` declarations that are currently enforced only in application code; (d) Normalize `String(64)` IDs to `String(36)` UUID columns where appropriate. Do not over-normalize: embedded `_json` arrays that are write-once and never filtered should stay as JSONB. |
| **CI parity: PostgreSQL in test matrix** | Tests currently run on SQLite in-process (`:memory:` or `tmp_path`). Some behaviors diverge (JSONB operator support, tsvector syntax, pgvector operators). Without a CI PostgreSQL target, regressions will reach production. | MEDIUM | Add `pytest` fixture for PostgreSQL test database (use `pytest-asyncio` + `asyncpg` test URL). Keep existing SQLite fixtures for fast unit tests. Add a `@pytest.mark.postgres` marker for integration tests that require PG features. GitHub Actions matrix: add a `postgres:16` service container. |

---

## Differentiators

Features that go beyond a minimum viable migration and meaningfully improve the system.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Hybrid search: pgvector + tsvector** | Current `_hybrid_merge()` in `memory_store.py` combines FTS5 (BM25) + sqlite-vec cosine similarity. PostgreSQL enables a proper hybrid search: `ts_rank` for text relevance + `<=>` cosine distance, merged by RRF (Reciprocal Rank Fusion). This is the production-quality RAG pattern. | MEDIUM | `ts_rank(fts, plainto_tsquery(...))` + `embedding <=> :query_vec` in a single CTE with RRF merge. Replaces the Python-side `_hybrid_merge()` function with a server-side SQL query. Fewer round-trips, better ranking. |
| **GIN indexes on JSONB payload columns** | Once `_json` columns become JSONB, frequently-filtered payloads (e.g., `AgentEventModel.tags_json`, `MemoryItemModel.evidence_json`) can have GIN indexes for sub-document queries. Currently unindexable. | LOW | Per-column decision: only add GIN index if the column is actually queried with `@>`, `?`, or `?|` operators. Start with `agent_events.tags_json` and `memory_items.evidence_json` based on current query patterns. |
| **Connection pooling configuration** | `asyncpg` + `create_async_engine` support `pool_size`, `max_overflow`, `pool_timeout`. Current sync SQLite has no meaningful pooling. Proper pooling is critical for FastAPI concurrency. PgBouncer-compatible: `prepare_threshold=0` already in `sqlalchemy_db.py` (a forward-looking comment). | LOW | Configure: `pool_size=10`, `max_overflow=20`, `pool_timeout=30`, `pool_recycle=1800`. Document PgBouncer connection string format. Parameterize via env vars. |
| **ARRAY columns for list-of-strings payloads** | Some JSONB columns store flat string arrays (e.g., `keywords_json`, `venues_json`, `methods_json`, `topics_json`). PostgreSQL `ARRAY(Text)` is queryable with `ANY()`, supports GIN indexing with `gin__int_ops`, and avoids JSONB parsing overhead for flat lists. | LOW | Evaluate case-by-case. Arrays that need `ANY(:keyword) = ANY(column)` queries benefit. Arrays that are read-only aggregations (author lists, venue history) can stay JSONB. Do not convert everything. |
| **Alembic branch for PG-only features** | Current Alembic chain has 32 SQLite-era migrations. PG migration can be a new branch head rather than a continuation, allowing clean separation between SQLite legacy and PG-native schema. | LOW | `alembic revision --autogenerate -m "pg_initial_schema" --head base` creates a fresh branch. Stamp PG databases at this revision. SQLite tests continue on the old chain. Use `alembic merge heads` only if cross-DB support is truly needed long-term. |
| **Async ARQ worker with asyncpg sessions** | Current ARQ worker (`WorkerSettings`) uses sync stores which block its async event loop. Proper async ARQ + asyncpg integration uses `on_job_start`/`after_job_end` hooks with `AsyncSession` context vars. | MEDIUM | Pattern: ARQ `ctx["db"]` key holds `AsyncSession` created at job start, closed at job end. Each job function receives `ctx` and reads `ctx["db"]`. Avoids connection leaks across job boundaries. This is documented in the ARQ + SQLAlchemy community pattern (wazaari.dev). |

---

## Anti-Features

Features that seem valuable for this migration but should be explicitly avoided.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full ORM re-architecture during migration** | Tempting to redesign relationships, add polymorphic inheritance, or switch to SQLModel while migrating. Scope explosion: each design change requires migration, store rewrite, test update, and integration validation. A data migration is already high-risk without adding schema redesign. | Migrate schema and async layer first. Model refactoring is a separate, bounded task within the milestone. Decouple: async conversion → JSONB/tsvector/pgvector → normalization. Never all three simultaneously on the same model. |
| **`run_sync()` as the async migration strategy** | `run_sync()` lets sync store methods work inside `AsyncSession` without full conversion. Appealing as a shortcut. In practice it serializes all DB work through a greenlet, provides no real concurrency benefit, obscures errors, and is explicitly documented as "partial upgrade" not a destination. | Convert stores properly to `async def`. For the small number of sync-only callers (Alembic env.py, tests), use a separate sync engine instance. |
| **SQLite + PostgreSQL dual-target parity** | Maintaining identical behavior on both databases requires `with_variant()` on every JSONB column, conditional FTS code paths, no pgvector columns, and no PG-specific operators. This is the current state and is the problem being solved. | Accept SQLite for fast unit tests only (no FTS, no vectors, no JSONB operators). PostgreSQL for integration tests and production. The test matrix has both, but SQLite tests cannot be expected to cover PG-native features. |
| **Zero-downtime dual-write migration** | Running writes to both SQLite and PostgreSQL simultaneously during transition sounds safe but requires application-level dual-write logic, consistency checks, and a cutover procedure. For PaperBot's current scale (single-server, non-SLA), this complexity is not warranted. | Simple cutover: export SQLite data → apply Alembic on PG → migrate data via pgloader/script → update `PAPERBOT_DB_URL` → restart. Maintenance window acceptable. |
| **pgloader for FTS5 virtual tables** | pgloader handles most SQLite → PG data migration automatically, but it cannot export FTS5 virtual tables (`memory_items_fts`, `document_chunks_fts`) or sqlite-vec virtual tables (`vec_items`). Attempting to pgload these tables will fail or produce garbage. | Skip virtual tables in pgloader. Regenerate FTS data: tsvector generated columns auto-populate on first `UPDATE` or can be bulk-populated via `UPDATE memory_items SET updated_at = updated_at`. For embeddings: re-run the embedding pipeline on existing content after migration. |
| **Big-bang model normalization** | Normalizing all 46 models in a single Alembic revision is the highest-risk operation in the milestone. One constraint violation in production data stops the entire migration. | Normalize incrementally: one model group per Alembic revision. Test each revision against a copy of production data before applying. Use `ALTER TABLE ... ADD CONSTRAINT IF NOT EXISTS` to be idempotent. |

---

## Feature Dependencies

```
Docker Compose (PostgreSQL local dev)
  |
  +-> Alembic env.py async config
  |     |
  |     +-> PostgreSQL engine + AsyncSessionProvider
  |           |
  |           +-> Store-by-store async conversion (17 stores)
  |           |     |
  |           |     +-> Async ARQ worker integration
  |           |     |
  |           |     +-> CI PostgreSQL test matrix
  |           |
  |           +-> Eager loading audit (all relationships)
  |           |
  |           +-> Text -> JSONB column migration
  |           |     |
  |           |     +-> GIN indexes on queryable JSONB columns
  |           |     |
  |           |     +-> ARRAY columns for flat string lists (optional)
  |           |
  |           +-> FTS5 -> tsvector (memory + documents)
  |           |     |
  |           |     +-> Hybrid pgvector + tsvector search
  |           |
  |           +-> sqlite-vec -> pgvector (memory embeddings)
  |           |     |
  |           |     +-> Hybrid pgvector + tsvector search
  |           |
  |           +-> Systematic model normalization
  |
  +-> Data migration tooling (SQLite -> PG)
```

### Dependency Notes

- **AsyncSessionProvider requires Docker Compose:** PG must be running locally before any async engine code can be tested.
- **Store conversion requires eager loading audit:** Converting a store to async without fixing its lazy-loaded relationships will produce `MissingGreenlet` errors at runtime, not at conversion time. These must be done together per-store, not sequentially across the full codebase.
- **JSONB migration requires Alembic PG target:** The `postgresql_using` cast expression in `op.alter_column` is PostgreSQL-only. Migration scripts must be run against PG, not SQLite.
- **pgvector requires FTS5 → tsvector:** The hybrid search feature uses both. Neither can be delivered alone if hybrid search is the goal.
- **Data migration tooling is independent:** pgloader/script migration of existing SQLite data can run after schema is in place. It is not on the critical path for new installations.
- **Model normalization is last:** Schema constraints should be added after data is migrated. Adding `NOT NULL` constraints to a column with nulls in production data will fail. Normalization runs against real data, so data migration must precede it.

---

## MVP Definition

### Ship First (Milestone Core)

The minimum needed to make PaperBot run on PostgreSQL with async stores.

- [ ] Docker Compose PG setup — required for any local development
- [ ] Alembic env.py async config — required to create PG schema
- [ ] `AsyncSessionProvider` + `create_async_engine` — replaces sync engine
- [ ] `paper_store` async conversion — highest-traffic store, most API routes depend on it
- [ ] `memory_store` async conversion + FTS5 → tsvector + sqlite-vec → pgvector — memory system is a first-class feature
- [ ] `document_index_store` async conversion + FTS5 → tsvector — document search depends on it
- [ ] `research_store` async conversion — research tracks are core to paper workflows
- [ ] Remaining 13 stores converted — stores that only write/read without FTS or vector search; low risk
- [ ] Text → JSONB for all 92 columns — prerequisite for any JSONB indexing or querying
- [ ] Eager loading audit — required per-store as part of async conversion

### Add After MVP Validated

Once the app runs cleanly on PG in development:

- [ ] Hybrid pgvector + tsvector search — improves retrieval quality, but BM25-only is functional
- [ ] GIN indexes on JSONB columns — performance optimization, not correctness
- [ ] Async ARQ worker integration — ARQ currently works with sync stores wrapped in thread pool; proper async integration is an improvement
- [ ] Data migration tooling — needed only when upgrading existing SQLite installations
- [ ] CI PostgreSQL service container — add after PG codebase is stable

### Defer to Post-v2.0

- [ ] Systematic model normalization — correctness improvement, not functionality blocker
- [ ] ARRAY columns for flat string lists — micro-optimization, schema change risk
- [ ] Alembic branch strategy — architectural decision with no runtime impact
- [ ] Connection pool tuning — production concern, not development milestone

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Docker Compose PG | HIGH (unblocks all dev) | LOW | P1 |
| Alembic async env.py | HIGH (unblocks schema) | LOW | P1 |
| AsyncSessionProvider | HIGH (core architecture) | LOW | P1 |
| paper_store async | HIGH (most-used store) | MEDIUM | P1 |
| memory_store async + FTS5/vec | HIGH (search is core) | HIGH | P1 |
| research_store async | HIGH (tracks are core) | MEDIUM | P1 |
| document_index_store async + FTS5 | MEDIUM | MEDIUM | P1 |
| Remaining 13 stores async | HIGH (completeness) | HIGH (volume) | P1 |
| Text → JSONB | HIGH (semantic correctness) | MEDIUM | P1 |
| Eager loading audit | HIGH (correctness) | HIGH | P1 |
| Hybrid pgvector + tsvector search | MEDIUM (quality boost) | MEDIUM | P2 |
| Async ARQ worker | MEDIUM (worker efficiency) | MEDIUM | P2 |
| GIN indexes on JSONB | MEDIUM (query performance) | LOW | P2 |
| CI PostgreSQL matrix | HIGH (regression safety) | LOW | P2 |
| Data migration tooling | HIGH (for existing users) | MEDIUM | P2 |
| Model normalization | MEDIUM (schema hygiene) | HIGH | P3 |
| ARRAY columns | LOW (micro-optimization) | LOW | P3 |
| Connection pool tuning | MEDIUM (production ops) | LOW | P3 |

**Priority key:** P1 = milestone is incomplete without it; P2 = adds significant value, ship after P1 stable; P3 = polish/optimization.

---

## Complexity Drivers

These are the aspects that make this migration harder than average:

| Driver | Impact | Mitigation |
|--------|--------|------------|
| 17 stores × ~10 async method conversions | ~170 method rewrites | Prioritize by traffic; use store-by-store Alembic revisions to isolate risk |
| Lazy loading is pervasive — default `lazy="select"` on all relationships | Runtime errors discovered only at test time, not at conversion time | Add `lazy="raise"` temporarily to all relationships after conversion; run full test suite to surface N+1 violations |
| FTS5 + sqlite-vec virtual tables do not export | Data migration cannot use pgloader for these tables | Skip in pgloader; regenerate from source data after PG import |
| 92 JSON Text columns need Alembic cast migrations | Each column needs `postgresql_using` cast; invalid JSON will cause migration failure | Pre-validate all JSON columns before migration: `SELECT id FROM table WHERE col IS NOT NULL AND col != '{}' AND (col::jsonb IS NULL)` — this will fail on bad JSON, surfacing rows to fix first |
| ARQ worker and FastAPI share the same store classes | ARQ does not have FastAPI DI; session lifecycle is different | Use ARQ lifecycle hooks (`on_job_start`/`after_job_end`) to manage `AsyncSession` in worker context; do not use FastAPI `Depends` patterns in worker code |
| `_ensure_fts5` and `_ensure_vec` are called on `__init__` of stores | Bootstrap code that runs at startup must detect DB type and skip SQLite-only setup | Add DB dialect check: `if session.bind.dialect.name == "postgresql"` before creating PG-specific structures |

---

## Sources

- [SQLAlchemy 2.0 Async I/O Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) — AsyncSession, async_sessionmaker, selectinload, run_sync
- [SQLAlchemy: The Async-ening](https://matt.sh/sqlalchemy-the-async-ening) — practical lazy loading pitfalls in async conversion
- [FastAPI + SQLAlchemy 2.0 Modern Async Patterns](https://dev-faizan.medium.com/fastapi-sqlalchemy-2-0-modern-async-database-patterns-7879d39b6843) — session lifecycle, expire_on_commit
- [ARQ + SQLAlchemy Done Right](https://wazaari.dev/blog/arq-sqlalchemy-done-right) — ARQ lifecycle hooks for async session management
- [Alembic Batch Migrations (SQLite + PG)](https://alembic.sqlalchemy.org/en/latest/batch.html) — cross-database migration portability
- [pgvector Python Library](https://github.com/pgvector/pgvector-python) — SQLAlchemy Vector type, Alembic integration, ischema_names fix
- [SQLAlchemy PostgreSQL Dialect — JSONB](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html) — JSONB type, GIN index, with_variant, MutableDict
- [Alembic JSONB Column Migration Discussion](https://github.com/sqlalchemy/alembic/discussions/984) — Text → JSONB alter_column with postgresql_using cast
- [PostgreSQL tsvector FTS with SQLAlchemy](https://amitosh.medium.com/full-text-search-fts-with-postgresql-and-sqlalchemy-edc436330a0c) — generated tsvector column, GIN index, ts_rank
- [pgloader SQLite Reference](https://pgloader.readthedocs.io/en/latest/ref/sqlite.html) — data migration tool, type conversion, FK constraint handling
- [How to Migrate from SQLite to PostgreSQL](https://render.com/articles/how-to-migrate-from-sqlite-to-postgresql) — boolean, datetime, JSON type differences
- [Mixing Async/Sync in FastAPI](https://github.com/fastapi/fastapi/discussions/12995) — run_in_threadpool vs full async conversion
- [Advanced SQLAlchemy 2.0 selectinload Strategies 2025](https://www.johal.in/advanced-sqlalchemy-2-0-selectinload-and-withparent-strategies-2025/) — selectinload pitfalls (composite PKs, recursive relations, fan-outs)

---
*Feature research for: PostgreSQL migration + async data layer + model refactoring (PaperBot v2.0)*
*Researched: 2026-03-14*
