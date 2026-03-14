# Project Research Summary

**Project:** PaperBot v2.0 — PostgreSQL Migration & Async Data Layer
**Domain:** Brownfield database migration — SQLite to PostgreSQL, sync to async SQLAlchemy
**Researched:** 2026-03-14
**Confidence:** HIGH

## Executive Summary

PaperBot v2.0 is a brownfield database migration, not a greenfield build. The project inherits 46 SQLAlchemy 2.0 models, 17 stores all using a sync `SessionProvider`, 84 `Text` columns hand-serializing JSON, two SQLite-only FTS5 virtual table subsystems, and a sqlite-vec embedding layer — none of which function correctly on PostgreSQL without explicit replacement. The recommended approach is a three-layer migration executed in strict sequence: (1) establish PostgreSQL infrastructure and schema compatibility while keeping sync stores, (2) convert all stores to async SQLAlchemy with a single shared asyncpg engine, and (3) clean up model schema and remove dead code. Each layer is independently deliverable and verifiable, which is the core risk-mitigation strategy for a ~170 method conversion across 17 stores.

The most dangerous failure mode is attempting any two layers simultaneously. The lazy-loading pitfall (`MissingGreenlet`) is pervasive and silent — it surfaces only at runtime, not at conversion time, and can affect every store that accesses ORM relationships after session close. The mitigation is to add `lazy="raise"` to all 30+ model relationships before any store conversion begins, so violations are caught during development. The secondary risk is the Text→JSONB migration: PostgreSQL requires an explicit `USING column::jsonb` cast that Alembic autogenerate never emits, and any row with malformed JSON halts the migration mid-table. Every type-change migration must be hand-authored and tested against a seeded database.

The test infrastructure is the single most critical enabler for this milestone. The existing SQLite in-memory test suite cannot validate PostgreSQL behavior — type coercion differs, LIKE case sensitivity differs, and FTS and vector search have no SQLite equivalent. A `testcontainers[postgres]` pytest fixture must be established and integrated into CI before the first store conversion ships, otherwise the CI green signal is meaningless. This is a non-negotiable prerequisite for Phase 3 work.

---

## Key Findings

### Recommended Stack

The existing stack already has `SQLAlchemy>=2.0.0`, `alembic>=1.13.0`, and `psycopg[binary]>=3.2.0`. Only three new packages are required: `asyncpg>=0.31.0` (async PostgreSQL driver — ~5x faster than psycopg3 in async benchmarks), `sqlalchemy[asyncio]>=2.0.0` (re-install with extra to pull in `greenlet`, mandatory in SQLAlchemy 2.1+), and `pgvector>=0.4.2` (typed `Vector(N)` column for SQLAlchemy). The local dev environment uses the `pgvector/pgvector:pg17` Docker image, which bundles the pgvector extension and eliminates a manual `CREATE EXTENSION` step. PostgreSQL 17.3+ is required — versions 17.0–17.2 have a symbol linking bug with pgvector.

The data migration tool for existing SQLite users is `pgloader` (a system-level tool, not a Python dependency). It handles type coercion and FK ordering but cannot export FTS5 or sqlite-vec virtual tables — those must be regenerated from source data after migration. The existing `psycopg[binary]` driver stays; it is used by Alembic for synchronous DDL migrations and must not be replaced.

**Core technologies:**
- `asyncpg>=0.31.0`: async PostgreSQL driver — fastest async PG driver, de facto standard for SQLAlchemy async PG; no libpq dependency
- `sqlalchemy[asyncio]>=2.0.0`: unlocks `create_async_engine`, `AsyncSession`, `async_sessionmaker` — `[asyncio]` extra is mandatory in SQLAlchemy 2.1+
- `pgvector>=0.4.2`: typed `Vector(N)` column with HNSW/IVFFlat index support — replaces `LargeBinary` blob approach for embeddings
- `pgvector/pgvector:pg17` Docker image: local PG with pgvector bundled — eliminates manual extension setup, requires PG 17.3+
- `pgloader` (system tool, one-time): SQLite to PostgreSQL data migration — handles type coercion and FK ordering, not a `requirements.txt` entry

**What NOT to add:** `aiosqlite` (tests use sync SQLite; adding async SQLite complexity has zero value), `databases` (superseded by SQLAlchemy 2.0 async), `tortoise-orm` (would require rewriting 46 models), `psycopg2` (psycopg3 already installed, never install both).

### Expected Features

The milestone has a clear three-tier priority structure based on feature dependencies. All P1 features are correctness blockers — the app cannot run on PostgreSQL without them. P2 features add meaningful capability once P1 is stable. P3 is polish and post-launch optimization.

**Must have (P1 — milestone incomplete without these):**
- Docker Compose PostgreSQL setup — required for all local development; blocks everything else
- Alembic dual-path env.py (async PG + sync SQLite) — required to apply PostgreSQL schema
- `AsyncSessionProvider` + `create_async_engine` with single shared pool — replaces 20+ independent sync engine instances
- All 17 stores converted to `async def` methods with `async with` session context — eliminates event-loop blocking
- Eager loading audit: `lazy="raise"` on all relationships, `selectinload` on query paths — prevents silent `MissingGreenlet`
- Text to JSONB for all 84 JSON columns — semantic correctness and prerequisite for any JSONB indexing
- FTS5 to tsvector for `memory_store` and `document_index_store` — FTS5 is SQLite-only; silent fallback on PG means no search
- sqlite-vec to pgvector for `MemoryItemModel.embedding` — vector search is currently a no-op on PostgreSQL

**Should have (P2 — ship after P1 validated):**
- Hybrid pgvector + tsvector search with Reciprocal Rank Fusion — upgrades BM25-only to production-quality RAG
- Async ARQ worker with per-job session lifecycle — prevents concurrent job session corruption
- GIN indexes on queryable JSONB columns (`agent_events.tags`, `memory_items.evidence`) — query performance
- CI PostgreSQL service container via `testcontainers[postgres]` — regression safety
- Data migration tooling (pgloader + custom script for embeddings) — required for existing user upgrades

**Defer to post-v2.0:**
- Systematic model normalization: removing all 84 `_json` helper methods, adding CHECK constraints, normalizing authors to FK table
- ARRAY columns for flat string lists — micro-optimization with schema change risk
- Connection pool tuning and PgBouncer documentation
- Alembic migration squash / clean single-head baseline

### Architecture Approach

The architectural pivot is from N-engines-per-store to a single shared `AsyncEngine` owned by the DI container. Today every store calls `SessionProvider(db_url)` in `__init__`, creating an independent connection pool — 20+ separate pools at runtime on PostgreSQL, wasting connections and preventing cross-pool transaction semantics. The v2.0 pattern is a `bootstrap_async_db()` function called once at FastAPI startup that creates one `AsyncEngine`, wraps it in an `async_sessionmaker`, registers it as a DI singleton, and injects the factory into every store constructor. Stores no longer own engines.

The build sequence is dependency-driven: (A) PostgreSQL + Schema while sync stores stay in place — proves PG compatibility before any async risk; (B) Async Data Layer conversion in four domain groups, one group per iteration; (C) Model refactoring to remove dead code and add constraints. The sync-first strategy means the existing test suite remains valid throughout Phase A, providing a safety net before the higher-risk Phase B work begins.

**Major components:**
1. `async_db.py` (new) — owns `AsyncEngine` creation, `AsyncSessionProvider` wrapper, URL coercion helper (`postgresql://` to `postgresql+asyncpg://`); injected into DI at startup
2. `bootstrap_async_db()` (new in `core/di/bootstrap.py`) — FastAPI startup hook that wires engine into `Container.instance()`; mirrored in ARQ `startup` hook
3. All 17 stores (modified) — receive injected `async_sessionmaker`, all methods become `async def` with `async with` session context and `selectinload` for relationship access
4. `alembic/env.py` (modified) — dual path: async PG via `connection.run_sync(context.run_migrations)`, sync SQLite path unchanged; `include_object` filter excludes tsvector GIN indexes from autogenerate; `Vector` registered in `ischema_names`
5. `arq_worker.py` (modified) — `startup` creates engine and factory only, `on_job_start`/`on_job_complete` hooks create and close per-job `AsyncSession` scoped by `ContextVar`
6. MCP tools (modified) — all 16 `anyio.to_thread.run_sync()` wrappers removed and replaced with direct `await store.method()` calls, one tool per store as each store is converted

### Critical Pitfalls

1. **MissingGreenlet on lazy-loaded relationships** — All 30+ relationships in `models.py` use `lazy="select"` (SQLAlchemy default). In async context, accessing any unloaded relationship attribute after session close raises `sqlalchemy.exc.MissingGreenlet`. This error is invisible on SQLite sync tests and only surfaces at runtime on async PostgreSQL. Prevention: add `lazy="raise"` to every relationship in `models.py` as the very first step, before any store conversion begins. Set `expire_on_commit=False` on the `async_sessionmaker`. Add explicit `selectinload()` or `joinedload()` to every query that accesses related collections.

2. **Text to JSONB migration fails without explicit CAST** — PostgreSQL will not implicitly cast `text` to `jsonb`. Alembic autogenerate never emits the required `USING column::jsonb` clause. Any row with malformed JSON (empty string, `NULL`, invalid JSON) stops the migration mid-run with a `DatatypeMismatch` error, leaving PostgreSQL in a partially migrated state. Prevention: hand-author every `_json TEXT → JSONB` migration using `op.execute("ALTER TABLE ... ALTER COLUMN ... TYPE jsonb USING col::jsonb")`; run a pre-migration cleanup query to fix empty strings; test against a seeded database — never just an empty schema.

3. **FTS5 sqlite_master queries crash PostgreSQL immediately** — `memory_store.py` and `document_index_store.py` contain 20+ queries against `sqlite_master` and `CREATE VIRTUAL TABLE ... USING fts5(...)` DDL, both called from store `__init__`. These raise `ProgrammingError: relation "sqlite_master" does not exist` on first use with any PostgreSQL URL. Prevention: wrap all SQLite-specific bootstrap code in `is_sqlite` guards as the first act of Phase 1 work, before any other PG integration.

4. **anyio.to_thread.run_sync left in place after async store conversion** — Once a store method becomes `async def`, passing it to `anyio.to_thread.run_sync()` returns the coroutine object rather than executing it. No error is raised; the MCP tool silently returns an empty list or `None`. Prevention: update each MCP tool to `await store.method()` directly, immediately after its corresponding store is converted — not as a final cleanup sweep.

5. **ARQ worker shared AsyncSession across concurrent jobs** — After async conversion, if the worker uses a single `AsyncSession` across concurrent ARQ jobs, one job's `commit()` or `rollback()` affects another job's uncommitted work. Prevention: `startup` hook creates engine and factory only (no session); `on_job_start` creates `ctx["db_session"]` per job; `on_job_complete` closes it. Use `async_scoped_session` with a `ContextVar` scoped to `ctx["job_id"]`.

---

## Implications for Roadmap

Based on research, the build order is dependency-driven and risk-stratified. The critical constraint is that each layer must be verified before the next begins. Attempting Phase 1 and Phase 3 simultaneously is the single highest-risk anti-pattern identified across all research files.

### Phase 1: PostgreSQL Infrastructure and Schema Compatibility

**Rationale:** Nothing works without a running PostgreSQL target and a schema that does not crash on connection. This phase proves PG compatibility with zero async risk — sync stores remain in place, the existing test suite stays valid. All subsequent phases depend on this layer being stable.

**Delivers:**
- Docker Compose with `pgvector/pgvector:pg17`, health check, named volume, `.env` update to `postgresql+asyncpg://` URL
- `pyproject.toml` additions: `asyncpg>=0.31.0`, `sqlalchemy[asyncio]>=2.0.0`, `pgvector>=0.4.2`
- Alembic dual-path `env.py`: async runner for PG URLs, sync path unchanged for SQLite, `include_object` filter for tsvector GIN indexes, `Vector` registered in `ischema_names`
- Alembic migrations 0028+: `CREATE EXTENSION vector`, tsvector columns + GIN indexes + update triggers on `memory_items` and `document_chunks`, pgvector `Vector(1536)` column replacing `LargeBinary` on `memory_items`, JSONB type changes with `USING` casts on all 84 `_json` columns
- `is_sqlite` guards wrapping `_ensure_fts5`, `_ensure_vec_table`, all `sqlite_master` queries
- Existing sync stores running against PostgreSQL (functionally correct, not yet async)

**Avoids:** FTS5 `sqlite_master` crash (#5), Text→JSONB missing CAST (#4), pgvector not registered in env.py (#13), tsvector autogenerate loop (#6), Alembic branch conflicts (#9)

### Phase 2: Test Infrastructure (testcontainers PostgreSQL)

**Rationale:** This phase is a hard prerequisite for all store conversions. The existing SQLite in-memory fixtures cannot validate PostgreSQL-specific behavior: JSONB operators, tsvector queries, pgvector distance operators, LIKE case sensitivity, and datetime type handling all differ. Shipping a converted store without a PG test target means CI green is meaningless.

**Delivers:**
- `testcontainers[postgres]` and `pytest-asyncio` added to `requirements-ci.txt`
- Session-scoped `pg_container` pytest fixture providing a real PostgreSQL database
- `@pytest.mark.postgres` marker for store integration tests
- SQLite sync fixtures retained for pure domain-logic unit tests (no stores)
- Baseline store integration tests running against PostgreSQL, confirming the fixture works before any async conversion begins

**Avoids:** SQLite in-memory tests invalid after AsyncSession migration (#12)

### Phase 3: Async Data Layer — Store Conversion in Domain Groups

**Rationale:** Four domain groups, one group per iteration, each with its own PR and test pass. Converting all 17 stores in a single PR is the highest-risk mistake identified in research. Starting with `SqlAlchemyEventLog` forces the async infrastructure pattern to be proven on the smallest, most tightly-coupled component before the larger stores are touched.

**Delivers:**
- `async_db.py` (new): `AsyncSessionProvider`, `create_async_db_engine`, `create_async_session_factory`, URL coercion helper, `statement_cache_size=0` in `connect_args`
- `bootstrap_async_db()` in `core/di/bootstrap.py`, wired to FastAPI `startup` event
- `lazy="raise"` added to ALL relationships in `models.py` (must be done first, before any store conversion)
- Group 1: `SqlAlchemyEventLog` — async `append()`, `list_runs()`, `list_events()`, `stream()`
- Group 2: `memory_store` — async methods + `_search_tsvector()` replacing `_search_fts5()` + pgvector `<=>` replacing `_search_vec()` + `anyio.to_thread.run_sync` wrappers removed from memory MCP tools
- Group 3: `paper_store` + `research_store` — async methods, all `.like()` calls audited and converted to `.ilike()`, `anyio.to_thread.run_sync` wrappers removed
- Group 4: remaining 13 stores — mechanical async conversion (no FTS or vector complexity), `anyio.to_thread.run_sync` wrappers removed from all remaining MCP tools

**Avoids:** MissingGreenlet (#1), anyio.to_thread.run_sync silent failure (#8), LIKE case sensitivity (#3), asyncpg prepared statement errors (#11)

### Phase 4: Async ARQ Worker

**Rationale:** ARQ requires a distinct session lifecycle from FastAPI — no dependency injection, per-job session scoping via `ContextVar`. This phase is architecturally separate from store conversion because mixing ARQ session lifecycle patterns with FastAPI `Depends` patterns is a documented failure mode. It depends on `SqlAlchemyEventLog` (Group 1 of Phase 3) being complete.

**Delivers:**
- `WorkerSettings` with `startup`, `shutdown`, `on_job_start`, `on_job_complete` hooks
- Per-job `AsyncSession` scoped to `ctx["job_id"]` via `ContextVar`
- Module-level `_EVENT_LOG` singleton replaced with context-scoped session access
- `startup` creates engine + factory only — no session is created at worker startup

**Avoids:** ARQ worker shared session corruption (#7)

### Phase 5: Hybrid Search and Performance Enhancements

**Rationale:** Once the async foundation is stable and tested, the production-quality search features that PostgreSQL enables can be added. These are improvements over a working baseline, not correctness blockers — they depend on both tsvector and pgvector being in place from Phase 1 and the async `memory_store` from Phase 3.

**Delivers:**
- Hybrid pgvector + tsvector search with Reciprocal Rank Fusion (RRF) in `memory_store._hybrid_search()` — replaces Python-side `_hybrid_merge()` with a single server-side SQL CTE
- GIN indexes on queryable JSONB columns: `agent_events.tags`, `memory_items.evidence`
- HNSW index on `memory_items.embedding` (replaces default sequential scan)
- Connection pool parameters (`pool_size`, `max_overflow`, `pool_recycle`) parameterized via env vars

**Addresses:** Hybrid pgvector + tsvector search, GIN indexes, connection pool configuration

### Phase 6: Data Migration Tooling and Model Refactoring

**Rationale:** Model normalization must run against real data already on PostgreSQL, so data migration precedes constraint additions. Adding `NOT NULL` constraints to a column with nulls in migrated data will fail. This phase is post-v2.0 in scope but must be planned as part of the milestone to prevent schema debt from compounding further.

**Delivers:**
- SQLite `PRAGMA foreign_keys = ON` + `PRAGMA integrity_check` as pre-migration gate
- pgloader migration command with FK violation report; custom Python script for re-encoding `LargeBinary` float bytes to pgvector arrays (pgloader cannot handle these)
- Alembic migrations for model normalization: `is_active` Integer to Boolean with all 5 call sites in `research_store.py` updated simultaneously, CHECK constraints on `status`/`confidence`/`pii_risk` columns, `NOT NULL DEFAULT NOW()` on all nullable `created_at` columns
- JSON helper methods removed (`get_keywords`, `set_keywords`, etc.); direct attribute access on JSON/JSONB columns

**Avoids:** Data migration FK violations (#10), is_active integer/boolean type change (#2), big-bang normalization anti-pattern

### Phase Ordering Rationale

- Phase 1 before Phase 3: Alembic migrations and PostgreSQL-native schema (tsvector, pgvector, JSONB) must exist and be verified before async stores can be meaningfully tested against PG-specific features.
- Phase 2 before Phase 3: The testcontainers CI fixture must be established and confirmed working before any store ships as async — otherwise the CI green signal is unreliable for the work being done.
- Phase 3 Group 1 before Phase 4: ARQ worker depends on `SqlAlchemyEventLog` being async; convert the event log first.
- Phase 5 after Phase 3: Hybrid search requires both tsvector (Phase 1 schema + Phase 3 `memory_store` query) and pgvector (same) to be fully operational.
- Phase 6 last: Constraint additions on columns that previously accepted nulls will fail against migrated data that contains nulls. Data must be on PostgreSQL first.

### Research Flags

Phases likely needing deeper research or per-phase planning work:

- **Phase 3, Group 2 (memory_store):** The most complex single store — FTS5 replacement, sqlite-vec replacement, hybrid search paths, and the most MCP tool connections. The specific relationship loading patterns and tsvector query shapes warrant a dedicated mini-plan before the group ships.
- **Phase 6 (Data migration):** The actual FK violation profile of existing SQLite production databases is unknown. Phase 6 planning should not be finalized until a representative database dump has been analyzed with `PRAGMA integrity_check` to quantify the remediation scope.

Phases with standard patterns (skip deeper research):

- **Phase 1:** Docker Compose PostgreSQL + Alembic async env.py are extensively documented with exact code patterns in ARCHITECTURE.md. Standard patterns apply directly.
- **Phase 2:** testcontainers Python pytest fixture is a solved, well-documented pattern with official guides.
- **Phase 4:** The ARQ + AsyncSession per-job lifecycle pattern is documented in ARCHITECTURE.md and can be implemented directly from that spec.
- **Phase 5:** Hybrid pgvector + tsvector RRF is an established production RAG pattern; implementation follows directly from FEATURES.md and ARCHITECTURE.md code samples.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All new packages verified on PyPI; version requirements confirmed against official changelogs and codebase inspection of `pyproject.toml` and `requirements.txt` |
| Features | HIGH | Feature set and priorities derived from direct codebase analysis — 46 models, 17 stores, 84 JSON columns, 28 migrations, 16 MCP tools counted directly, not estimated |
| Architecture | HIGH | Patterns grounded in official SQLAlchemy 2.0 async docs; phase approach confirmed against established brownfield migration guides; all integration points verified by reading source files |
| Pitfalls | HIGH | Pitfalls verified against codebase with specific file locations and line numbers confirmed; backed by official SQLAlchemy/Alembic/asyncpg sources and confirmed upstream issue tracker tickets (Alembic #1390, #1324) |

**Overall confidence:** HIGH

### Gaps to Address

- **SQLAlchemy 2.1 stability:** The `>=2.0.0` pin may resolve to 2.1.x (currently at beta as of 2026-03-14), which changed `greenlet` handling and dropped Python 3.9. Validate actual resolved version in Phase 1 against the CI matrix (3.10, 3.11, 3.12). Consider pinning `>=2.0.0,<2.1.0` until 2.1 stable is released.
- **Production FK violation profile:** The actual number of orphaned rows in existing SQLite deployments is unknown. Phase 6 planning must include a pre-migration audit step before commitments are made on remediation scope.
- **pgloader in CI:** pgloader is a system package not installable via pip. Phase 6 planning must confirm whether the GitHub Actions runner has pgloader available or whether a Docker-based pgloader or custom Python script alternative is needed.
- **aiosqlite for async test fixtures:** ARCHITECTURE.md's `_coerce_to_async_url` includes a SQLite to `aiosqlite` coercion path. Confirm during Phase 2 whether `aiosqlite` is needed for any async test fixture path or whether testcontainers fully replaces it.

---

## Sources

### Primary (HIGH confidence)

- `src/paperbot/infrastructure/stores/sqlalchemy_db.py` — confirmed `SessionProvider` sync pattern, `prepare_threshold=0`, `future=True`
- `src/paperbot/infrastructure/stores/models.py` — confirmed 46 models, `LargeBinary` embedding, 84 `Text` JSON columns, `lazy="select"` default on all relationships, `is_active: Mapped[int]`
- `src/paperbot/infrastructure/stores/memory_store.py` — confirmed `sqlite_master` queries (20+), `_ensure_fts5`, `_ensure_vec_table`, `sqlite_vec.load`
- `src/paperbot/infrastructure/stores/research_store.py` — confirmed `is_active == 1` / `== 0` in 5 locations, `func.lower().like()` search pattern
- `src/paperbot/infrastructure/stores/paper_store.py` — confirmed `.ilike()` in 4 locations
- `alembic/versions/` — confirmed 28 migration files, existing branch merge at `4c71b28a2f67`
- `src/paperbot/mcp/tools/` — confirmed 16 uses of `anyio.to_thread.run_sync` wrapping sync store calls
- `pyproject.toml` — confirmed `psycopg[binary]>=3.2.0`, `SQLAlchemy>=2.0.0`, `alembic>=1.13.0`, CI matrix 3.10/3.11/3.12
- [SQLAlchemy 2.0 Asyncio Extension docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) — `AsyncSession`, `async_sessionmaker`, `selectinload`, `MissingGreenlet` behavior
- [SQLAlchemy 2.1.0b1 release blog](https://www.sqlalchemy.org/blog/2026/01/21/sqlalchemy-2.1.0b1-released/) — `greenlet` no longer auto-installed in 2.1
- [asyncpg PyPI 0.31.0](https://pypi.org/project/asyncpg/) — version and Python version matrix confirmed
- [pgvector-python PyPI 0.4.2](https://pypi.org/project/pgvector/) — version confirmed
- [pgloader SQLite reference](https://pgloader.readthedocs.io/en/latest/ref/sqlite.html) — FK ordering, type casting, violation handling
- [asyncpg FAQ](https://magicstack.github.io/asyncpg/current/faq.html) — prepared statement conflicts with PgBouncer

### Secondary (MEDIUM confidence)

- [ARQ + SQLAlchemy Done Right](https://wazaari.dev/blog/arq-sqlalchemy-done-right) — per-job session lifecycle with `on_job_start`/`on_job_complete` hooks
- [FastAPI SQLAlchemy 2.0 Modern Async Patterns](https://dev-faizan.medium.com/fastapi-sqlalchemy-2-0-modern-async-database-patterns-7879d39b6843) — session lifecycle, `expire_on_commit`
- [Alembic tsvector + JSONB migration patterns](https://berkkaraal.com/blog/2024/09/19/setup-fastapi-project-with-async-sqlalchemy-2-alembic-postgresql-and-docker/)
- [pgvector-python SQLAlchemy integration](https://deepwiki.com/pgvector/pgvector-python/3.1-sqlalchemy-integration) — `register_vector_async`, `ischema_names` pattern
- [Alembic issue #1390](https://github.com/sqlalchemy/alembic/issues/1390) — tsvector GIN index autogenerate false positive loop (confirmed upstream bug)
- [Alembic issue #1324](https://github.com/sqlalchemy/alembic/discussions/1324) — pgvector `ischema_names` fix
- [Alembic issue #697](https://github.com/sqlalchemy/alembic/issues/697) — Text to JSON migration data loss risk
- [psycopg3 vs asyncpg comparison (2026)](https://fernandoarteaga.dev/blog/psycopg-vs-asyncpg/) — performance benchmark rationale for asyncpg choice

---

*Research completed: 2026-03-14*
*Ready for roadmap: yes*
