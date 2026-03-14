# Phase 12: PG Infrastructure & Schema - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

PaperBot runs against PostgreSQL with a complete, PG-compatible schema — tsvector, JSONB, and pgvector columns in place — without crashing on any SQLite-only code path. All SQLite code is deleted (not guarded). Docker Compose provides the local dev environment. Alembic env.py supports async execution. Full tsvector search implementation and JSONB store cleanup land in this phase.

Note: This phase's scope is larger than originally planned. It absorbs JSONB model+store cleanup, full tsvector search implementation, and GIN indexes — work originally scoped to Phase 15. Phase 15 is reduced to vector search (HNSW index) + RRF hybrid search only.

</domain>

<decisions>
## Implementation Decisions

### SQLite removal
- One-way migration: SQLite is **dropped entirely** in Phase 12, not guarded
- Delete all SQLite-specific code: FTS5 virtual tables, sqlite-vec extension loading, sqlite_master queries, `ensure_sqlite_parent_dir`, `check_same_thread` connect args
- Delete all 22 old Alembic migration files from `alembic/versions/`
- No SQLite upgrade path — fresh PG only (no `alembic stamp` support for old migrations)
- Tests switch to PG in Phase 13 (testcontainers); Phase 12 uses GH Actions PG service container for CI

### Docker Compose
- Image: `pgvector/pgvector:pg17` (official pgvector image, PostgreSQL 17)
- PG only — no pgAdmin, no Adminer
- Named volume for data persistence across `docker-compose down/up`
- Standard port 5432 exposed to host
- Hardcoded credentials: user=paperbot, password=paperbot, database=paperbot
- pgvector extension created via `docker-entrypoint-initdb.d` SQL script (not Alembic)

### Alembic migrations
- Squash all 22 SQLite-era migrations into a single `0001_pg_baseline.py`
- Baseline creates the full PG-native schema from scratch (all tables, indexes, constraints)
- Single file — not split by domain
- All old migration files deleted from git

### JSONB columns
- All 84 JSON columns born as JSONB in the baseline migration (not Text → JSONB conversion)
- ORM models use native JSONB mapped type (`Mapped[dict]` with `JSONB` column type), not `Mapped[str]` with `Text`
- Full store code cleanup in Phase 12: remove all `json.loads()`/`json.dumps()` calls on these columns
- GIN indexes on all 84 JSONB columns using default operator class (supports @>, ?, ?|, ?& operators)

### pgvector embeddings
- pgvector `Vector(1536)` column on `memory_items` table only
- Drop the old `LargeBinary` embedding column (no side-by-side with old column)
- Fixed dimension 1536 (matching OpenAI ada-002); model change = Alembic migration

### tsvector full-text search
- Per-row language detection: Python-side `langdetect` library detects language on write, stores detected language in a column
- PG trigger uses the stored language column to populate tsvector via `to_tsvector(lang, content)`
- All PG built-in dictionaries supported (english, german, french, spanish, etc.)
- Fallback: 'simple' dictionary (no stemming, no stop words) when language is unknown or unsupported
- tsvector columns + GIN indexes on `memory_items` and `document_chunks`
- Full implementation in Phase 12 including search queries (not just schema)

### Alembic env.py
- Async execution path added proactively in Phase 12 (even though stores are still sync)
- Uses `psycopg` v3 (supports both sync and async modes natively)
- Single driver — no psycopg2/asyncpg split
- SQLite path completely removed from env.py

### SessionProvider consolidation
- Single shared Engine registered in DI container (`Container.instance()`)
- All stores resolve the shared Engine via DI (same pattern as LLMClient)
- Per-store SessionProviders eliminated in Phase 12

### Default DB URL and onboarding
- `DEFAULT_DB_URL` changed to `postgresql+psycopg://paperbot:paperbot@localhost:5432/paperbot`
- Update `env.example` and README with: 1) `docker-compose up`, 2) `alembic upgrade head`, 3) run
- Update `alembic.ini` default URL to match
- No setup script — docs-only onboarding

### CI continuity
- Phase 12 adds a minimal PostgreSQL service container in GitHub Actions CI
- Uses `postgres:17` (or pgvector equivalent) service block
- Tests connect via `PAPERBOT_DB_URL` env var pointing to CI PG instance
- Phase 13 adds testcontainers for local `pytest` — CI is never broken

### Phase 15 scope reduction
- Phase 15 reduced to: HNSW vector index on pgvector column + RRF hybrid search (combining tsvector + pgvector)
- JSONB cleanup, tsvector implementation, and GIN indexes are absorbed into Phase 12

### Claude's Discretion
- PG trigger implementation details (BEFORE INSERT vs AFTER INSERT, trigger function naming)
- langdetect → PG dictionary mapping table
- Exact GIN index naming convention
- psycopg v3 connection pool configuration (pool_size, max_overflow)
- alembic.ini logging configuration for PG
- docker-compose.yml healthcheck implementation

</decisions>

<specifics>
## Specific Ideas

- Clean break from SQLite — no dual-support, no guards, no legacy paths
- Phase 12 is intentionally front-loaded: schema, JSONB cleanup, tsvector full implementation, SessionProvider consolidation all land together so the codebase is PG-native from this point forward
- Hybrid tsvector population (Python langdetect + PG trigger) gives best of both worlds: accurate language detection with automatic tsvector consistency

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sqlalchemy_db.py`: SessionProvider and engine creation — to be refactored to PG-only shared Engine
- `alembic/env.py`: Already has psycopg connect_args support — extend with async path
- `models.py`: 46 ORM models with 84 `Text` JSON columns — convert to JSONB mapped type
- `DI Container` (`core/di/container.py`): `register()`/`resolve()` pattern — use for shared Engine registration

### Established Patterns
- DI container singleton for cross-cutting services (LLMClient, EventLog) — follow same pattern for shared Engine
- Stores import SessionProvider and create their own engines — pattern to be replaced with DI-resolved shared Engine
- `memory_store.py` has FTS5 and sqlite-vec code (~200 lines) — to be deleted and replaced with tsvector queries
- `document_index_store.py` has sqlite_master queries — to be deleted

### Integration Points
- `core/di/bootstrap.py`: Register shared Engine here
- All 17+ store files: Refactor constructor to accept Engine from DI instead of creating SessionProvider
- `alembic/versions/`: Delete all existing files, write single PG baseline
- `.github/workflows/ci.yml`: Add PostgreSQL service container
- `env.example`: Update default DB URL
- `pyproject.toml` or `requirements.txt`: Add psycopg[binary], langdetect dependencies

</code_context>

<deferred>
## Deferred Ideas

- Phase 15 still handles HNSW vector index and RRF hybrid search (tsvector + pgvector fusion)
- Phase 17 handles pgloader data migration from existing SQLite databases
- Roadmap update needed: Phase 15 scope description should be updated to reflect reduced scope

</deferred>

---

*Phase: 12-pg-infrastructure-schema*
*Context gathered: 2026-03-14*
