# Pitfalls Research

**Domain:** PostgreSQL migration + async data layer + model refactoring (v2.0)
**Researched:** 2026-03-14
**Confidence:** HIGH — grounded in codebase inspection + verified SQLAlchemy/Alembic/asyncpg official sources

> This file covers pitfalls specific to the v2.0 milestone: SQLite → PostgreSQL migration,
> sync Session → AsyncSession conversion across 17+ stores, FTS5 → tsvector,
> sqlite-vec → pgvector, JSON Text → JSONB, and Alembic migration tooling.
> It does NOT cover the v1.1 agent orchestration pitfalls (see PITFALLS.md history).

---

## Critical Pitfalls

Mistakes that cause rewrites, data loss, or silent behavioral regressions.

---

### Pitfall 1: MissingGreenletError on Lazy-Loaded Relationships

**What goes wrong:**
After converting stores to `AsyncSession`, any access to a SQLAlchemy relationship attribute that was not explicitly loaded in the original query raises `sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called; can't call await_only() here`. This includes accessing `.events`, `.logs`, `.metrics`, `.runbook_steps`, `.artifacts` on `AgentRunModel`, or `.memories` on `MemorySourceModel`. The error is not raised in tests that use SQLite in-memory with sync sessions — it only appears in production-style async contexts.

**Why it happens:**
SQLAlchemy's default relationship loading strategy is lazy — it issues a synchronous SELECT when the attribute is first accessed. In an async context there is no greenlet in scope to proxy this synchronous I/O, so the ORM raises `MissingGreenlet` instead of silently blocking. The existing `models.py` has 30+ relationships all using the default `lazy="select"` strategy. None are annotated with `lazy="selectin"` or `lazy="raise"`.

**How to avoid:**
- Add `lazy="raise"` to ALL relationships in `models.py` immediately. This converts silent runtime errors into loud errors that surface during development.
- For each store query that needs a relationship, add explicit `.options(selectinload(...))` to the `select()` statement.
- Use `expire_on_commit=False` in the `async_sessionmaker` factory. Without this, attributes accessed after a `commit()` will trigger an implicit reload — which also raises `MissingGreenlet`.
- Never serialize a SQLAlchemy model object to a response dict outside of an `AsyncSession` scope without pre-loading all needed attributes.

**Warning signs:**
- `MissingGreenlet` in logs pointing to a model `.attribute` access.
- Tests pass with SQLite but requests fail in production.
- Pydantic serialization of response models triggers the error.

**Phase to address:** Model schema phase (before any async session work begins). Add `lazy="raise"` to all relationships as the first step of the conversion so violations surface immediately.

---

### Pitfall 2: is_active Stored as Integer, Compared as Boolean After Column Type Change

**What goes wrong:**
`ResearchTrackModel.is_active` is declared as `Mapped[int]` and queried with `ResearchTrackModel.is_active == 1` and `.values(is_active=0)` in `research_store.py` (lines 204, 261, 285, 326, 350). If the model refactoring phase changes this column to `Mapped[bool]` / `Boolean`, all 5 call sites must be updated to `True`/`False` simultaneously. Any site that is missed silently sends `1` to a `BOOLEAN` column in PostgreSQL — which PostgreSQL accepts — but reads back as `True`, not `1`. Code paths that did `bool(int(t.is_active or 0))` (research_store.py:1890) work, but code paths that compared `result == 1` break.

**Why it happens:**
SQLite stores `Boolean` as `0`/`1` integers and Python code learned to treat the column as an integer. PostgreSQL has a native `BOOLEAN` type that returns Python `True`/`False`, not `1`/`0`. The mismatch is invisible in SQLite and explodes in PostgreSQL.

**How to avoid:**
- During model refactoring, grep for ALL `== 0`, `== 1`, `=0`, `=1` assignments to `is_active` before changing the column type. Change all 5 sites in `research_store.py` at the same time as the model change.
- Treat `Boolean` columns and `Integer` flags as separate migration concerns — do not change types incrementally on different days.
- After schema change, add an integration test that reads the `is_active` field back and asserts `isinstance(result, bool)`.

**Warning signs:**
- Any store file using `== 0` or `== 1` comparisons on columns declared as `Boolean`.
- `bool(int(...))` wrapper calls indicate the column was not originally `bool`.

**Phase to address:** Model refactoring phase. Audit all `Integer`-as-boolean columns (also `pii_risk`, `priority` where used as flags) before declaring them `Boolean`.

---

### Pitfall 3: LIKE is Case-Insensitive in SQLite, Case-Sensitive in PostgreSQL

**What goes wrong:**
`paper_store.py` uses `ilike(pattern)` in 4 places (lines 668–690) and `func.lower(...).like(...)` in `research_store.py` (lines 964–968). The `ilike` calls are safe because `ilike` is portable via SQLAlchemy. The `func.lower(...).like(...)` pattern in `research_store.py` is also safe if the input is already `.lower()`. However, any remaining `.like(...)` without `.lower()` or `ilike` wrapping — whether in the stores or in raw SQL strings — will silently return fewer results after moving to PostgreSQL.

**Why it happens:**
SQLite's `LIKE` is case-insensitive for ASCII by default. PostgreSQL's `LIKE` is case-sensitive. Developers test search on SQLite in dev/CI where "python" matches "Python", then the same query on PostgreSQL returns 0 results.

**How to avoid:**
- Audit all `.like(...)` calls across all stores. Any `.like(...)` that is NOT preceded by `func.lower(column)` and `func.lower(value)` must be changed to `.ilike(...)`.
- The existing `ilike` usage in `paper_store.py` is already correct — do not change it.
- The FTS search replacement (tsvector) is inherently case-insensitive via PostgreSQL's text search dictionaries — no action needed there.

**Warning signs:**
- Full-text search returns fewer results on PostgreSQL than SQLite for the same query with mixed-case input.
- A search for "ArXiv" that finds papers on SQLite returns 0 on PostgreSQL.

**Phase to address:** Store migration phase. Add a text search regression test with mixed-case inputs before and after migration.

---

### Pitfall 4: Text → JSONB Column Migration Fails Without Explicit CAST

**What goes wrong:**
PaperBot has 84 `Text` columns storing JSON (`_json` suffix). The model refactoring plan converts these to `JSONB`. Alembic's `autogenerate` cannot automatically migrate `TEXT` data to `JSONB`. Running `alembic upgrade head` on an existing PostgreSQL database with data will fail with:
```
psycopg2.errors.DatatypeMismatch: column "payload_json" is of type jsonb
but expression is of type text.
HINT: You might need to add an explicit cast.
```
Data loss risk: if the migration drops and recreates the column instead of altering it, all JSON data is lost.

**Why it happens:**
PostgreSQL will not implicitly cast `text` to `jsonb`. The `ALTER COLUMN ... TYPE jsonb` command requires an explicit `USING column::jsonb` clause. Alembic's autogenerate does not add this clause automatically.

**How to avoid:**
- Write ALL `Text` → `JSONB` column migrations manually (not autogenerated). Use the pattern:
  ```python
  op.execute("ALTER TABLE agent_events ALTER COLUMN payload_json TYPE jsonb USING payload_json::jsonb")
  ```
- Test every migration in a staging PostgreSQL database with real row data, not just with an empty schema.
- For any row where the JSON text is malformed, `::jsonb` cast will fail. Add a pre-migration validation step: `SELECT id FROM table WHERE payload_json IS NOT NULL AND payload_json::text !~ '^[{\\[]'`.
- Never use `autogenerate` for type change migrations — always review and write by hand.

**Warning signs:**
- Alembic generates `sa.Column('payload_json', postgresql.JSONB(...))` in autogenerate output without a `USING` clause in `op.alter_column`.
- CI migration tests pass on an empty schema but fail on a database with rows.

**Phase to address:** Alembic migration authoring phase. The golden rule: every `Text → JSONB` alter must be hand-authored with `USING` clause and tested on a seeded database.

---

### Pitfall 5: FTS5 Virtual Table and sqlite_master Queries Break on PostgreSQL Immediately

**What goes wrong:**
`memory_store.py` and `document_index_store.py` contain 20+ direct calls to `sqlite_master`:
```python
text("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')")
text("SELECT name FROM sqlite_master WHERE type='trigger'")
text("PRAGMA table_info(memory_items)")
```
These queries will raise `ProgrammingError: relation "sqlite_master" does not exist` the first time any code path that calls `_ensure_fts5()` or `_ensure_vec_table()` runs against PostgreSQL. There are also raw FTS5 queries like `CREATE VIRTUAL TABLE ... USING fts5(...)` that have no PostgreSQL equivalent.

**Why it happens:**
The FTS5 and sqlite-vec tables are created lazily at runtime by the store constructors. They are deeply interleaved with the main store logic, not isolated in migrations. Moving to PostgreSQL requires replacing them entirely — `tsvector` with a GIN index for FTS, and `pgvector`'s `vector` type for ANN search.

**How to avoid:**
- Before writing a single async-migration line, wrap all SQLite-specific code paths in an `is_sqlite` guard:
  ```python
  if str(self._engine.url).startswith("sqlite"):
      self._ensure_fts5(conn)
  ```
  This prevents crash-on-PostgreSQL during the transition period where both backends may be in use.
- Create a `MemorySearchPort` interface that has `search_fts(...)` and `search_vec(...)` methods. Provide a `SqliteMemorySearch` implementation (existing code) and a `PostgresMemorySearch` implementation (tsvector + pgvector). Swap via the DI container based on DB URL.
- The tsvector replacement is a separate migration file: add a `tsvector` column, create a GIN index, add an update trigger. This migration can ONLY run against PostgreSQL — gate it in `alembic/env.py` with `if "postgresql" in db_url`.

**Warning signs:**
- Any test that passes a PostgreSQL URL to a store constructor crashes with `sqlite_master does not exist`.
- The `_ensure_fts5` method is called from `__init__` with no database-type guard.

**Phase to address:** Store interface design phase (before PostgreSQL integration). FTS abstraction behind a port is non-negotiable.

---

### Pitfall 6: Alembic Autogenerate Loops Infinitely on tsvector GIN Indexes

**What goes wrong:**
Once tsvector columns and GIN indexes are added to the PostgreSQL schema, every subsequent `alembic revision --autogenerate` detects the GIN index as "changed" and generates a drop/recreate pair. This produces a stream of identical empty migrations and makes `alembic check` always report "schema out of date" even when nothing has changed.

**Why it happens:**
Alembic's autogenerate cannot correctly fingerprint `to_tsvector()`-based expression indexes. It sees the index definition as different on every comparison cycle, even if nothing changed. This is a confirmed upstream bug in Alembic 1.13+ (GitHub issue #1390).

**How to avoid:**
- After creating the initial tsvector GIN index migration, exclude it from autogenerate using `include_object` in `alembic/env.py`:
  ```python
  def include_object(object, name, type_, reflected, compare_to):
      if type_ == "index" and name and "tsvector" in (name or ""):
          return False
      return True
  ```
- Alternatively, mark the index creation as `manual` (do not autogenerate it at all) and manage it through named migration files.
- Register the `vector` type from pgvector in `env.py` to prevent similar false-positive autogenerate on vector columns:
  ```python
  from pgvector.sqlalchemy import Vector
  connection.dialect.ischema_names["vector"] = Vector
  ```

**Warning signs:**
- Every `alembic revision --autogenerate` produces a non-empty file for the same indexes.
- `alembic check` reports detected changes but running the migration makes no visible difference.

**Phase to address:** Alembic tooling setup phase. Add autogenerate exclusions BEFORE writing any tsvector or pgvector migrations.

---

### Pitfall 7: ARQ Worker Session Not Scoped to Individual Jobs

**What goes wrong:**
After converting stores to `AsyncSession`, the ARQ worker (`arq_worker.py`) uses a shared session across jobs. Two concurrent ARQ jobs both write to `agent_events` or `agent_runs` using the same session. One job's `commit()` commits the other job's uncommitted changes. Or worse, one job's `rollback()` rolls back both jobs' work.

**Why it happens:**
Unlike FastAPI (which scopes sessions per-request via `Depends`), ARQ has no dependency injection. The naive migration wraps the `WorkerSettings` startup in a single `async with async_session_factory() as session: ...` that lives for the entire worker lifetime. All jobs share it.

**How to avoid:**
- Use ARQ's `on_job_start` and `on_job_complete` lifecycle hooks to create and destroy a session per job.
- Store the per-job session in the ARQ context dict (`ctx["db_session"]`) keyed to `ctx["job_id"]`.
- The `startup` hook creates the engine and session factory only — not a session.
- `on_job_start` creates `ctx["db_session"] = async_session_factory()`.
- `on_job_complete` calls `await ctx["db_session"].close()`.

**Warning signs:**
- Jobs that succeed in isolation fail intermittently under concurrent load.
- `IntegrityError` from concurrent jobs writing to the same rows.
- Checking ARQ worker logs: a single session ID appears across multiple job log entries.

**Phase to address:** ARQ worker migration phase. The ARQ session lifecycle must be explicitly designed before any store method is converted to async.

---

### Pitfall 8: anyio.to_thread.run_sync Wrappers Left in Place After AsyncSession Migration

**What goes wrong:**
Currently, 16 MCP tool functions call `anyio.to_thread.run_sync(sync_store_method, ...)` to bridge async MCP handlers to sync SQLAlchemy stores. After converting stores to `AsyncSession`, these bridges become unnecessary and harmful: the store method is now a coroutine, not a callable, so `anyio.to_thread.run_sync(coroutine_method)` silently returns a coroutine object instead of awaiting it. No error is raised; the tool returns empty data.

**Why it happens:**
`anyio.to_thread.run_sync` accepts any callable and runs it in a thread. Passing a coroutine-returning method (e.g., `async def search(...)`) returns the coroutine object itself to `run_sync`, which wraps it and returns a future that resolves to the coroutine object — not its result. This is a silent failure.

**How to avoid:**
- Convert MCP tools to `await store.method(...)` directly after converting each store.
- Do NOT leave `anyio.to_thread.run_sync` wrappers in place "as a safety net" — they will silently break.
- Write an integration test for each MCP tool that asserts the return value is populated data, not a coroutine object or empty list.

**Warning signs:**
- MCP tools return empty lists or `None` after store conversion.
- No `MissingGreenlet` errors (the coroutine was never awaited — the error is silence, not crash).
- Adding `print(result)` shows `<coroutine object ...>`.

**Phase to address:** MCP tool update phase. Each tool must be updated immediately after its corresponding store is converted — not as a final sweep.

---

### Pitfall 9: Alembic Migration Squashing or Merge Breaks PostgreSQL-Specific Branches

**What goes wrong:**
The existing `alembic/versions/` directory has 28 migration files with a known branch conflict (`4c71b28a2f67_merge_structured_card_and_anchor_author_.py`). Adding new PostgreSQL-specific migrations creates additional branches. Running `alembic upgrade head` on a fresh PostgreSQL database with all branches active hits conflicts and either runs migrations in wrong order or fails outright.

**Why it happens:**
Alembic's dependency graph for heads gets confused when multiple "head" revisions exist simultaneously. The existing merge migration handles SQLite-era branches. Adding PostgreSQL-gated migrations (e.g., `CREATE EXTENSION vector`) that cannot run on SQLite creates a new branching problem.

**How to avoid:**
- Before v2.0 migration work starts, squash all existing migrations into a single "v1.x baseline" migration. Test this baseline on both SQLite and a fresh PostgreSQL schema.
- Create a clean single-head starting point for v2.0 work.
- For PostgreSQL-only migrations (tsvector, pgvector), use a dialect check in the migration body, NOT separate branch files:
  ```python
  def upgrade():
      bind = op.get_bind()
      if bind.dialect.name == "postgresql":
          op.execute("CREATE EXTENSION IF NOT EXISTS vector")
  ```

**Warning signs:**
- `alembic heads` shows more than 1 head.
- `alembic upgrade head` on a fresh database takes an unexpected path or skips migrations.

**Phase to address:** Pre-migration setup phase. Squash first, then add PostgreSQL migrations.

---

### Pitfall 10: Data Migration of Existing SQLite Database Loses Rows Due to FK Violations

**What goes wrong:**
The PaperBot SQLite database has 46 tables with foreign keys that SQLite historically did not enforce. When importing the SQLite data into PostgreSQL (where FK constraints are always enforced), rows with dangling FK references fail to insert. For example: `paper_feedback` rows referencing `papers.id` values that were cleaned up in SQLite but not deleted from `paper_feedback` due to missing CASCADE. The import fails mid-table, leaving PostgreSQL in a partially migrated state.

**Why it happens:**
SQLite's foreign key enforcement is opt-in (`PRAGMA foreign_keys = ON`). Most SQLite deployments run without it. PaperBot's models define `ondelete="CASCADE"` on some relationships but not all (e.g., `PaperFeedbackModel.paper_ref_id` has no explicit `ondelete`). Years of data in SQLite may contain orphaned rows that have never caused visible errors.

**How to avoid:**
- Before migration, validate FK integrity on SQLite with `PRAGMA foreign_keys = ON` and a `PRAGMA integrity_check`. Log all violations.
- Use `pgloader` for the actual data transfer — it handles FK ordering and can generate a violation report.
- Alternatively, use a custom Python migration script that inserts parent tables before child tables and collects FK violations to a separate log for manual triage.
- After migration, run `SELECT * FROM information_schema.table_constraints WHERE constraint_type = 'FOREIGN KEY'` and spot-check a sample of FK relationships.

**Warning signs:**
- `pgloader` output shows "condition not verified" rows counted separately from "rows copied".
- `INSERT` failures during migration referencing `foreign key constraint`.
- Row counts differ between SQLite export and PostgreSQL import.

**Phase to address:** Data migration tooling phase. Build validation-first, migrate-second.

---

### Pitfall 11: Prepared Statement Errors with asyncpg and Connection Poolers

**What goes wrong:**
If PostgreSQL is deployed behind PgBouncer (or similar) in transaction pooling mode, asyncpg's automatic use of prepared statements causes intermittent errors: `prepared statement "__asyncpg_stmt_XX__" does not exist` or `already exists`. The existing `sqlalchemy_db.py` already disables prepared statements for `psycopg` connections via `prepare_threshold=0`, but this does NOT apply to asyncpg.

**Why it happens:**
asyncpg prepares statements at the session level by default. PgBouncer in transaction mode does not guarantee the same backend connection across transactions. When a prepared statement exists on backend connection A, and the next query arrives on backend connection B, the statement does not exist on B.

**How to avoid:**
- Add `statement_cache_size=0` to the asyncpg `connect_args` in `create_async_engine`:
  ```python
  create_async_engine(url, connect_args={"statement_cache_size": 0})
  ```
  Note: this cannot be set in the connection URL string — it must be a Python kwarg.
- The existing `prepare_threshold: 0` in `sqlalchemy_db.py` covers the sync `psycopg2` path; mirror it for asyncpg explicitly.
- For local Docker development without PgBouncer, this is a non-issue — but production deployments with connection poolers will hit this.

**Warning signs:**
- Works in Docker dev, fails in production (or CI using a pooler).
- Intermittent errors on high-concurrency endpoints like `POST /api/analyze` or `GET /api/track`.
- Error message references `asyncpg_stmt`.

**Phase to address:** AsyncSession setup phase. Set `statement_cache_size=0` from the first async engine created.

---

### Pitfall 12: SQLite In-Memory Tests No Longer Valid After AsyncSession Migration

**What goes wrong:**
The existing test suite uses `SessionProvider(db_url="sqlite:///:memory:")` for unit and integration tests (18+ test files). After converting stores to `AsyncSession`, these tests cannot use SQLite in-memory for two reasons: (1) the async driver for SQLite (`aiosqlite`) has different behavior than asyncpg for PostgreSQL, and (2) FTS5/tsvector and vec0/pgvector have no common interface in SQLite in-memory mode. Tests that use FTS or vector search will either skip silently or crash.

**Why it happens:**
SQLite's timezone handling is different from PostgreSQL (naive vs aware datetimes), LIKE case sensitivity differs, and the test infrastructure that imports `sqlite_vec` is optional (skipped if not installed). The tests were designed for sync SQLite — they are not valid validators of async PostgreSQL behavior.

**How to avoid:**
- Migrate tests to `testcontainers[postgres]` for any test that touches stores, sessions, or search.
- Keep a SQLite in-memory path only for pure domain logic tests (no stores).
- The testcontainers fixture pattern for pytest is a session-scoped PostgreSQL container that runs all store tests against real PostgreSQL.
- CI must have Docker available. The existing `requirements-ci.txt` must add `testcontainers[postgres]` and `pytest-asyncio`.
- Do NOT attempt to keep SQLite as a "fast" fallback for store tests — the behavioral differences are too large to trust.

**Warning signs:**
- Tests pass with `sqlite:///:memory:` but requests fail in production with different results.
- Datetime comparison tests produce different results across environments.

**Phase to address:** Test infrastructure phase. Establish testcontainers fixture BEFORE converting the first store to async.

---

### Pitfall 13: pgvector Extension Not Registered in Alembic env.py

**What goes wrong:**
After installing `pgvector` and defining `vector` type columns in models, `alembic revision --autogenerate` emits:
```
SAWarning: Did not recognize type 'vector' of column 'embedding'
```
and generates an empty migration that appears to detect no changes. Subsequent attempts to apply the migration fail because the `vector` column was never created.

**Why it happens:**
Alembic's schema introspection does not know about custom PostgreSQL types like `vector` by default. Without registering the type in `env.py`, autogenerate treats the column as unknown and omits it from the diff.

**How to avoid:**
- In `alembic/env.py`'s `run_migrations_online()`, before `context.configure(...)`, add:
  ```python
  from pgvector.sqlalchemy import Vector
  connection.dialect.ischema_names["vector"] = Vector
  ```
- Also add `CREATE EXTENSION IF NOT EXISTS vector` in the first migration that uses the `vector` type.
- Write the `vector` column migration by hand — do not rely on autogenerate for it.

**Warning signs:**
- `alembic revision --autogenerate` produces no change for a model with a new `vector` column.
- `alembic check` says no changes detected even though `memory_items.embedding` is still `LargeBinary`.

**Phase to address:** pgvector integration phase. Set up the extension registration before writing the embedding migration.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Keep sync Session with `run_sync()` wrapper instead of native AsyncSession | No store rewrites, faster transition | Every DB call still blocks one greenlet thread; can't use true async DB features; still need asyncpg | Only as a temporary bridge during incremental migration; remove within same milestone |
| Migrate schema but not data (leave SQLite file in production) | Simpler milestone scope | Dual-write or data gap in production; users lose history | Never — v2.0 must include a data migration path for existing installations |
| Use SQLite in-memory for post-migration store tests | Test speed, no Docker dependency | Tests do not catch PostgreSQL-specific bugs (type coercion, LIKE sensitivity, FK enforcement) | Never after AsyncSession conversion |
| Leave `anyio.to_thread.run_sync` wrappers in MCP tools | Zero MCP changes needed during store migration | Silent return of coroutine objects; MCP tools return empty data | Never — remove immediately when each store is converted |
| Skip squashing old migrations before adding PG migrations | Saves 2-4 hours | Alembic head conflicts; harder to onboard new contributors; migration graph debugging nightmare | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| asyncpg + PgBouncer | Not disabling prepared statements | Pass `statement_cache_size=0` as a kwarg to `create_async_engine` connect_args |
| pgvector + Alembic | Relying on autogenerate for `vector` column | Register type in `env.py`, write migration by hand, add `CREATE EXTENSION IF NOT EXISTS vector` |
| tsvector + Alembic | Autogenerate loops on GIN expression indexes | Exclude tsvector indexes from autogenerate via `include_object` filter in `env.py` |
| ARQ worker + AsyncSession | Sharing one session across concurrent jobs | Create per-job session in `on_job_start`, destroy in `on_job_complete` hooks |
| Docker PG + asyncpg | Forgetting to wait for PG to be ready on container start | Use `pool_pre_ping=True` on engine + retry logic in startup hook |
| SQLite → PG data copy via pgloader | Foreign key violations stopping import mid-table | Run `PRAGMA integrity_check` on SQLite first; use `pgloader` with `CONTINUE ON ERROR` + violation report |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Using `ilike` on unindexed large text columns (title, abstract) without tsvector | Full table scan; queries >500ms as papers table grows | Add tsvector GIN index; use `to_tsquery` for search instead of `ilike` | At ~50K papers rows |
| No connection pool size limit on asyncpg engine | DB reports "too many connections"; asyncpg pool waits indefinitely | Set `pool_size=10, max_overflow=5` on `create_async_engine` | At high concurrency (>20 simultaneous requests) |
| Returning full relationship graphs via `selectinload` on list endpoints | N+1 converted to single SELECT IN, but result set is huge | Use `selectinload` only for needed relationships; add explicit `limit()` | Immediately on large datasets |
| Running Alembic migrations with `NullPool` in production | Each migration step opens and closes a connection; slow for 28 migrations | NullPool is correct for migrations; not a runtime concern | Migrations take >2 min but this is acceptable |
| pgvector ANN search without an index | Sequential scan through all embeddings | Create HNSW or IVFFlat index on `vector` column before enabling ANN search | At ~10K embedding rows |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Storing PostgreSQL credentials in `.env` committed to git | Credential leak | Use environment injection (Docker Compose env, CI secrets); `.env` is in `.gitignore` already |
| Running Alembic with a superuser in production | Migration can drop tables it should not touch | Create a limited `paperbot_migrator` role with only `CONNECT, CREATE TABLE, ALTER TABLE` rights |
| Not rotating `api_key_value` stored in `model_endpoints` table | Plaintext API key accessible to any DB reader | Encrypt with a master key or store only as `api_key_env` references; existing TODO in codebase |

---

## "Looks Done But Isn't" Checklist

- [ ] **AsyncSession conversion:** Store methods return `await`-able results — verify each store has zero remaining `with self._provider.session()` sync context managers.
- [ ] **Relationship loading:** `lazy="raise"` added to all relationships in `models.py` — verify by running the test suite and checking for no `MissingGreenlet` errors.
- [ ] **FTS migration:** `memory_store._search_fts5` and `document_index_store._search_chunk_ids_with_fts` replaced with tsvector implementations — verify by running keyword search and checking result count matches SQLite baseline.
- [ ] **sqlite-vec to pgvector:** `memory_store._search_vec` replaced with pgvector ANN search — verify by running vector search and checking cosine distances are plausible.
- [ ] **sqlite_master queries removed:** grep for `sqlite_master` in `memory_store.py` and `document_index_store.py` returns 0 results.
- [ ] **PRAGMA removed:** grep for `PRAGMA` in `src/` returns 0 results outside of test files.
- [ ] **anyio.to_thread.run_sync removed from MCP tools:** grep for `anyio.to_thread.run_sync` in `mcp/tools/` and `mcp/resources/` returns 0 results.
- [ ] **Text → JSONB with USING clause:** every migration file that changes a `_json` column from `Text` to `JSONB` contains `USING column::jsonb` in the `op.execute` call.
- [ ] **JSONB autogenerate imports:** every autogenerated migration file that uses `postgresql.JSONB` has `from sqlalchemy.dialects import postgresql` at the top.
- [ ] **ARQ worker lifecycle:** `WorkerSettings` has `on_job_start` and `on_job_complete` hooks; `startup` does NOT create a session, only a factory.
- [ ] **testcontainers in CI:** `requirements-ci.txt` includes `testcontainers[postgres]`; CI runner has Docker socket accessible.
- [ ] **pgvector extension registered:** `alembic/env.py` registers `Vector` in `connection.dialect.ischema_names` before any `context.configure` call.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| MissingGreenlet on lazy load in production | HIGH | Add `selectinload` to affected queries; redeploy; no data loss |
| Text → JSONB migration failed mid-run | HIGH | Restore from backup; fix migration with `USING` clause; re-run from last successful step |
| FK violation during SQLite → PG data import | MEDIUM | Identify orphaned rows from pgloader report; delete from SQLite; re-export; re-import |
| ARQ worker shared session corruption | HIGH | Stop worker; identify affected jobs from logs; replay failed jobs; add per-job session lifecycle |
| Alembic autogenerate loop on tsvector GIN index | LOW | Add `include_object` filter to `env.py`; delete spurious empty migration files |
| SQLite in-memory tests passing but PG failing | MEDIUM | Add testcontainers fixture; run failing tests against PG to reveal type/behavior mismatches |
| asyncpg prepared statement errors in production | MEDIUM | Add `statement_cache_size=0` to connect_args; redeploy; no data loss |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| MissingGreenlet on lazy relationships (#1) | Model schema refactoring (add `lazy="raise"` to all relationships) | Run full test suite; zero `MissingGreenlet` errors |
| is_active integer → boolean type change (#2) | Model refactoring (audit all `== 0` / `== 1` sites before column type change) | Integration test: read `is_active` field, assert `isinstance(result, bool)` |
| LIKE case sensitivity (#3) | Store migration (audit all `.like()` calls, convert to `.ilike()`) | Search regression test with mixed-case input |
| Text → JSONB without CAST (#4) | Alembic migration authoring (hand-write all type-change migrations) | Run migration against seeded test database; verify row counts unchanged |
| FTS5 sqlite_master queries (#5) | Store interface design (add `is_sqlite` guard or FTS port abstraction) | Start store with PostgreSQL URL; no `sqlite_master` errors |
| tsvector autogenerate loop (#6) | Alembic tooling setup (add `include_object` filter before writing tsvector migrations) | `alembic revision --autogenerate` produces empty file after stable schema |
| ARQ worker session scope (#7) | ARQ worker migration (add `on_job_start`/`on_job_complete` hooks) | Two concurrent ARQ jobs; each sees only its own committed rows |
| anyio.to_thread.run_sync left in MCP tools (#8) | MCP tool update (convert each tool immediately after its store) | MCP tool integration test returns populated data, not empty list |
| Alembic branch conflicts (#9) | Pre-migration setup (squash existing migrations to single baseline) | `alembic heads` returns exactly 1 head |
| Data migration FK violations (#10) | Data migration tooling (SQLite integrity check + pgloader with violation log) | Row counts match between SQLite export and PostgreSQL import |
| asyncpg prepared statements (#11) | AsyncSession setup (set `statement_cache_size=0` on first async engine) | High-concurrency load test against PostgreSQL returns no prepared statement errors |
| SQLite in-memory tests invalid (#12) | Test infrastructure (establish testcontainers fixture before first store conversion) | Store integration tests run against real PostgreSQL container in CI |
| pgvector type not registered (#13) | pgvector integration (register in `env.py` before first embedding migration) | `alembic revision --autogenerate` detects `vector` column as unchanged |

---

## Sources

- Codebase: `memory_store.py` — 20+ `sqlite_master` queries, `_ensure_fts5`, `_ensure_vec_table`, `sqlite_vec.load` calls
- Codebase: `document_index_store.py` — `_ensure_fts5`, `sqlite_master` queries, `ilike` fallback search
- Codebase: `paper_store.py` — `.ilike()` in 4 locations, `func.lower()` for title matching
- Codebase: `research_store.py` — `func.lower().like()` for search, `is_active == 1` / `== 0` in 5 locations
- Codebase: `sqlalchemy_db.py` — sync `sessionmaker`, `prepare_threshold: 0` for psycopg only
- Codebase: `models.py` — 84 `Text` JSON columns, `is_active: Mapped[int]`, all relationships default to `lazy="select"`
- Codebase: `alembic/versions/` — 28 migration files, existing branch merge at `4c71b28a2f67`
- Codebase: `mcp/tools/` — 16 uses of `anyio.to_thread.run_sync` wrapping sync store calls
- [SQLAlchemy AsyncIO docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) — `expire_on_commit=False`, `selectinload`, `MissingGreenlet` behavior
- [SQLAlchemy async discussion #9757](https://github.com/sqlalchemy/sqlalchemy/discussions/9757) — `greenlet_spawn has not been called`
- [SQLAlchemy async discussion #5923](https://github.com/sqlalchemy/sqlalchemy/discussions/5923) — sync and async coexistence
- [SQLAlchemy Boolean/SQLite docs](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html) — type affinity, boolean storage as 0/1
- [Alembic autogenerate docs](https://alembic.sqlalchemy.org/en/latest/autogenerate.html) — limitations: renames detected as add/drop, `compare_server_default` accuracy
- [Alembic issue #1390](https://github.com/sqlalchemy/alembic/issues/1390) — tsvector GIN index autogenerate false positive loop
- [Alembic issue #1324](https://github.com/sqlalchemy/alembic/discussions/1324) — pgvector type not recognized; `ischema_names` fix
- [Alembic issue #697](https://github.com/sqlalchemy/alembic/issues/697) — Text → JSON migration data loss
- [asyncpg FAQ](https://magicstack.github.io/asyncpg/current/faq.html) — prepared statement conflicts with PgBouncer
- [asyncpg issue #1058](https://github.com/MagicStack/asyncpg/issues/1058) — prepared statements despite disabled
- [ARQ + SQLAlchemy](https://wazaari.dev/blog/arq-sqlalchemy-done-right) — per-job session lifecycle with `on_job_start`/`on_job_complete`
- [Testcontainers Python](https://testcontainers.com/guides/getting-started-with-testcontainers-for-python/) — PostgreSQL fixture pattern for pytest
- [PostgreSQL case sensitivity](https://www.cybertec-postgresql.com/en/case-insensitive-pattern-matching-in-postgresql/) — LIKE vs ILIKE, migration implications
- [pgloader SQLite → PostgreSQL docs](https://pgloader.readthedocs.io/en/latest/ref/sqlite.html) — FK ordering, type casting, violation handling

---
*Pitfalls research for: PostgreSQL migration + async data layer + model refactoring (PaperBot v2.0)*
*Researched: 2026-03-14*
