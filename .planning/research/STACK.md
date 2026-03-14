# Technology Stack

**Project:** PaperBot v2.0 — PostgreSQL Migration + Async Data Layer
**Researched:** 2026-03-14
**Confidence:** HIGH

---

## Principle: Surgical Additions Only

The existing stack handles every concern except async PostgreSQL access and PG-native
feature types. This document covers only the **new packages required for v2.0** and
exactly how they integrate with existing `SessionProvider`, `SQLAlchemy 2.0`, and
`alembic`. Nothing is added for its own sake.

---

## What Already Exists (Do NOT Re-add)

| Capability | Existing Package | Status |
|---|---|---|
| SQLAlchemy ORM | `SQLAlchemy>=2.0.0` | Installed. Already uses `future=True` mode. |
| Schema migrations | `alembic>=1.13.0` | Installed. 27 migrations. `env.py` already PG-aware. |
| psycopg3 sync driver | `psycopg[binary]>=3.2.0` | Installed. Used in `create_db_engine` for `prepare_threshold=0`. |
| SQLite vector search | `sqlite-vec>=0.1.6` | Installed. Optional extra — replaced by pgvector in PG. |

---

## New Additions Required

### Core: Async Driver

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `asyncpg` | `>=0.31.0` | Native async PostgreSQL binary-protocol driver | Fastest Python PG driver (5x faster than psycopg3 in async benchmarks). No libpq dependency — pure asyncio. SQLAlchemy async engine uses it via `postgresql+asyncpg://` URL. The existing `psycopg[binary]` stays for Alembic migrations (Alembic runs sync DDL; async drivers are not needed there). HIGH confidence — PyPI verified. |

### Core: SQLAlchemy Async Extensions

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `sqlalchemy[asyncio]` | `>=2.0.0` (install extra, not version bump) | Unlocks `create_async_engine`, `AsyncSession`, `async_sessionmaker` | SQLAlchemy's asyncio extension requires `greenlet` which is bundled via the `[asyncio]` extra. In SQLAlchemy 2.1+ (released Jan 2026) `greenlet` is no longer auto-installed — the extra is mandatory. Since `SQLAlchemy>=2.0.0` is already pinned, this is a re-install with the extra flag, not a version change. HIGH confidence — official SQLAlchemy 2.1 changelog verified. |

### Core: PG-Native Feature Types

| Technology | Version | Purpose | Why |
|---|---|---|---|
| `pgvector` | `>=0.4.2` | `Vector` column type for SQLAlchemy + pgvector PG extension | Replaces `sqlite-vec` LargeBinary blob approach. Provides typed `Vector(N)` mapped column, HNSW/IVFFlat index helpers, and cosine/L2/inner-product distance ops inside SQLAlchemy queries. Async-compatible via `register_vector_async` + `event.listens_for`. HIGH confidence — PyPI 0.4.2 verified, official pgvector-python repo confirmed SQLAlchemy 2.0 support. |

### Development Infrastructure

| Technology | Version | Purpose | Why |
|---|---|---|---|
| Docker image `pgvector/pgvector:pg17` | latest (PG 17.x) | Local dev PostgreSQL with pgvector bundled | Single image replaces `postgres:17` + manual `CREATE EXTENSION vector`. The official `pgvector/pgvector` Docker Hub image is maintained alongside the extension. PG 17.3+ required (17.0–17.2 have a symbol linking bug with pgvector). MEDIUM confidence — Docker Hub and pgvector GitHub verified. |

---

## Installation Changes

```bash
# 1. Add asyncpg (new dependency)
pip install "asyncpg>=0.31.0"

# 2. Re-install SQLAlchemy with asyncio extra to pull in greenlet
#    (required for SQLAlchemy 2.1+; safe on 2.0.x too)
pip install "sqlalchemy[asyncio]>=2.0.0"

# 3. Add pgvector Python type package (new dependency)
pip install "pgvector>=0.4.2"

# -- pyproject.toml changes --
# In [project].dependencies:
#   Change: "SQLAlchemy>=2.0.0"
#   To:     "SQLAlchemy[asyncio]>=2.0.0"
#
# Add to [project].dependencies:
#   "asyncpg>=0.31.0"
#
# Move sqlite-vec out of [project.optional-dependencies].search
# and add pgvector in its place for PG installs:
#   "pgvector>=0.4.2"
```

```bash
# Docker: local PG dev environment
docker run -d \
  --name paperbot-pg \
  -e POSTGRES_USER=paperbot \
  -e POSTGRES_PASSWORD=paperbot \
  -e POSTGRES_DB=paperbot \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# Set env var
export PAPERBOT_DB_URL="postgresql+asyncpg://paperbot:paperbot@localhost:5432/paperbot"

# Run migrations (Alembic uses sync psycopg3 — no change needed here)
alembic upgrade head
```

---

## Integration with Existing Code

### SessionProvider: Extend, Not Replace

The existing `SessionProvider` wraps a sync engine. The pattern is to add an **async
counterpart** `AsyncSessionProvider` in the same file, not to rewrite `SessionProvider`.
Sync sessions remain necessary for Alembic migrations, tests using `tmp_path` SQLite,
and any code that cannot easily be made async (e.g., ARQ workers running in threads).

```python
# New class — add to sqlalchemy_db.py alongside existing SessionProvider
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

def create_async_db_engine(db_url: Optional[str] = None):
    url = db_url or get_db_url()
    # postgresql+asyncpg:// URL required; sqlite URLs need aiosqlite (not needed here)
    return create_async_engine(url, pool_pre_ping=True)

class AsyncSessionProvider:
    def __init__(self, db_url: Optional[str] = None):
        self.engine = create_async_db_engine(db_url)
        self._factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    def session(self) -> AsyncSession:
        return self._factory()
```

Stores are then refactored one-by-one: each store's methods become `async def`, each
`with provider.session() as s:` becomes `async with provider.session() as s:`, and
`s.execute(select(...))` already returns `Result` in SQLAlchemy 2.0 — the await is
the only mechanical change per call site.

### Alembic: Stays Sync

Alembic's `upgrade()` / `downgrade()` functions must remain synchronous. The existing
`alembic/env.py` already uses `engine_from_config` with psycopg3 for PG URLs — no
change needed. Do NOT switch `env.py` to `async_engine_from_config` unless running
migrations programmatically inside a FastAPI lifespan (then use the `run_sync` pattern
with `conn.run_sync(run_upgrade, cfg)` to avoid event loop conflicts).

### JSONB: Replace `Text` + `json.dumps` Columns

Models currently serialize dicts to `Text` (e.g., `payload_json`, `metadata_json`,
`keywords_json`). On PG, migrate these to `JSONB` via Alembic `op.alter_column` +
`server_default='{}'`. The ORM mapping changes from `Text` to
`sqlalchemy.dialects.postgresql.JSONB`. Access via `model.payload` (dict) replaces
`json.loads(model.payload_json)`.

```python
# Before (SQLite Text)
from sqlalchemy import Text
payload_json: Mapped[str] = mapped_column(Text, default="{}")

# After (PG JSONB)
from sqlalchemy.dialects.postgresql import JSONB
payload: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
```

### tsvector: Replace FTS5 Virtual Tables

The `0019_memory_fts5` migration already guards on `dialect != "sqlite"` and explicitly
comments "Postgres uses pg_trgm / tsvector instead." The PG migration adds a new
Alembic revision that:

1. Adds a `search_vector tsvector` generated column (or trigger-maintained column) on
   `memory_items.content`.
2. Creates a GIN index: `CREATE INDEX ix_memory_items_fts ON memory_items USING gin(search_vector)`.
3. Adds a `before insert or update` trigger calling
   `to_tsvector('english', NEW.content)`.

```python
# In the Alembic upgrade function
from sqlalchemy.dialects.postgresql import TSVECTOR

op.add_column('memory_items',
    sa.Column('search_vector', TSVECTOR(), nullable=True))
op.create_index('ix_memory_items_fts', 'memory_items', ['search_vector'],
    postgresql_using='gin')
op.execute("""
    CREATE OR REPLACE FUNCTION memory_items_fts_update() RETURNS trigger AS $$
    BEGIN
        NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
""")
op.execute("""
    CREATE TRIGGER memory_items_fts_trigger
    BEFORE INSERT OR UPDATE OF content ON memory_items
    FOR EACH ROW EXECUTE FUNCTION memory_items_fts_update();
""")
```

### pgvector: Replace `LargeBinary` Embedding Columns

`MemoryItemModel.embedding` is currently `LargeBinary` (raw Float32 bytes for
sqlite-vec). On PG, this becomes `Vector(N)` from `pgvector.sqlalchemy`.

```python
# Before (sqlite-vec)
from sqlalchemy import LargeBinary
embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

# After (pgvector)
from pgvector.sqlalchemy import Vector
embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)
```

Register pgvector type codec on AsyncSession connections:

```python
from pgvector.psycopg import register_vector_async  # noqa — psycopg3 variant
from sqlalchemy import event

@event.listens_for(async_engine.sync_engine, "connect")
def connect(dbapi_connection, connection_record):
    dbapi_connection.run_async(register_vector_async)
```

---

## Data Migration Tooling

For migrating existing SQLite data to PG (existing users' `data/paperbot.db`):

**Use pgloader** — the standard open-source CLI for SQLite-to-PostgreSQL migrations.
It handles type coercion, sequences, and index recreation automatically.

```bash
# Install pgloader (system package, not a Python dep)
apt-get install pgloader  # or brew install pgloader on macOS

# Migrate schema + data in one command
pgloader sqlite:///data/paperbot.db postgresql://paperbot:paperbot@localhost/paperbot
```

pgloader is a **system-level tool for one-time data migration only** — it is not a
Python dependency and must not be added to `requirements.txt`. After pgloader copies
the data, run `alembic upgrade head` on the PG database to apply any PG-specific
migrations (tsvector columns, JSONB conversions, pgvector columns) that were skipped
on SQLite.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|---|---|---|---|
| Async PG driver | `asyncpg` | `psycopg3[asyncio]` | psycopg3 is already installed for sync use. asyncpg is ~5x faster in benchmarks and is the de facto standard for SQLAlchemy async PG. The official FastAPI full-stack template uses psycopg3 for its simplicity (single driver), but PaperBot already has psycopg3 for Alembic — asyncpg adds maximum async throughput without replacing the sync driver. |
| Vector type | `pgvector` | Raw `float[]` array column | `float[]` requires manual distance query SQL. `pgvector` provides typed `Vector(N)` with HNSW/IVFFlat index support and SQLAlchemy-integrated distance operators. No contest. |
| FTS in PG | `tsvector` + GIN trigger | `pg_trgm` trigram index | `pg_trgm` supports fuzzy/partial matching; `tsvector` with `to_tsvector` provides true stemmed, ranked BM25-style search. For the academic paper domain (keywords, abstracts), stemmed full-text search is better. `pg_trgm` can be added later as a supplement if substring matching is needed. |
| Local dev PG | `pgvector/pgvector:pg17` | `postgres:17` + manual extension install | The prebuilt image eliminates a `CREATE EXTENSION vector` step and avoids the need for OS-level pgvector compilation in CI. Zero extra setup cost. |
| JSONB migration | Rename column, drop old | Keep Text column + add JSONB side-by-side | Dual-write is more work and more error-prone. Alembic ALTER COLUMN with server_default cast is clean within a transaction. |
| Data migration tool | `pgloader` | Custom Python ETL script | pgloader handles type coercion, sequence reset, and index recreation automatically. A custom script would need to replicate all of that. pgloader is a one-time dev tool, not a runtime dependency. |

---

## What NOT to Install

| Library | Why Might You Think You Need It | Why You Don't |
|---|---|---|
| `aiosqlite` | "Need async SQLite for tests" | Tests use sync SQLite sessions via the existing `SessionProvider`. The async layer only activates for PG URLs. Do not add async SQLite complexity to the test suite — it adds zero value and breaks the sync/async separation. |
| `databases` (encode/databases) | "Thin async SQL layer" | Superseded by SQLAlchemy 2.0 async. Adds a second ORM-like layer on top of SQLAlchemy. Creates two competing DB abstractions in the same codebase. |
| `tortoise-orm` | "Pure async ORM" | Would require rewriting 46 models from scratch. SQLAlchemy 2.0 async is the right choice when you already have an SA codebase. |
| `alembic-utils` | "Helpers for PG-specific objects" | The tsvector triggers and functions are written once in raw SQL inside Alembic migrations. `alembic-utils` adds a dependency for a one-time setup task. |
| `psycopg2` | "Might still need it" | `psycopg[binary]>=3.2.0` (psycopg3) is already installed. psycopg2 is a separate, older package. Never install both. |
| `sqlmodel` | "Pydantic-integrated ORM" | SQLModel wraps SQLAlchemy with Pydantic models. Rewriting 46 SA models to SQLModel just to get Pydantic integration is not worth it. Pydantic models for API layer already exist separately. |

---

## Version Compatibility

| Package | Pin | Notes |
|---|---|---|
| `asyncpg>=0.31.0` | Lower-bound only | 0.31.0 (Nov 2025) adds Python 3.14 support. Requires Python ≥3.9. PaperBot CI tests on 3.10/3.11/3.12 — all compatible. |
| `sqlalchemy[asyncio]>=2.0.0` | As existing | SQLAlchemy 2.1.0b1 (Jan 2026) dropped Python 3.9 support. The existing `>=2.0.0` pin may resolve to 2.1.x on 3.10/3.11/3.12 environments. To stay on stable 2.0.x, consider pinning `>=2.0.0,<2.1.0` until 2.1 stable is released. |
| `pgvector>=0.4.2` | Lower-bound only | 0.4.2 is current (2025). Requires pgvector PG extension ≥0.5.0 for HNSW index support. `pgvector/pgvector:pg17` Docker image includes extension 0.8.x. |
| `psycopg[binary]>=3.2.0` | Unchanged | Stays. Used by Alembic's sync `engine_from_config` for DDL migrations. The `prepare_threshold=0` connect arg in `alembic/env.py` already handles PgBouncer compatibility. |
| PostgreSQL | 17.3+ | PG 17.0–17.2 have a symbol linking bug with pgvector. Use `pgvector/pgvector:pg17` which tracks latest PG 17 patch. |

---

## Sources

- [asyncpg PyPI — version 0.31.0 confirmed](https://pypi.org/project/asyncpg/) — HIGH confidence
- [asyncpg GitHub MagicStack/asyncpg — Python 3.9–3.14, PG 9.5–18](https://github.com/MagicStack/asyncpg) — HIGH confidence
- [SQLAlchemy 2.0 Asyncio Extension docs](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) — HIGH confidence
- [SQLAlchemy 2.1.0b1 release blog — greenlet no longer auto-installed](https://www.sqlalchemy.org/blog/2026/01/21/sqlalchemy-2.1.0b1-released/) — HIGH confidence
- [SQLAlchemy 2.1 migration guide — Python 3.10 minimum](https://www.sqlalchemy.org/docs/21/changelog/migration_21.html) — HIGH confidence
- [pgvector-python PyPI — version 0.4.2 confirmed](https://pypi.org/project/pgvector/) — HIGH confidence
- [pgvector-python SQLAlchemy integration docs](https://deepwiki.com/pgvector/pgvector-python/3.1-sqlalchemy-integration) — MEDIUM confidence
- [pgvector GitHub — Docker image pgvector/pgvector:pg17](https://github.com/pgvector/pgvector) — HIGH confidence
- [Alembic tsvector + JSONB migration patterns](https://berkkaraal.com/blog/2024/09/19/setup-fastapi-project-with-async-sqlalchemy-2-alembic-postgresql-and-docker/) — MEDIUM confidence
- [pgloader SQLite → PostgreSQL migration](https://pgloader.readthedocs.io/en/latest/ref/sqlite.html) — HIGH confidence
- [psycopg3 vs asyncpg comparison (2026)](https://fernandoarteaga.dev/blog/psycopg-vs-asyncpg/) — MEDIUM confidence
- Codebase: `src/paperbot/infrastructure/stores/sqlalchemy_db.py` — confirmed existing SessionProvider, psycopg3 connect args, future=True mode
- Codebase: `alembic/versions/0019_memory_fts5.py` — confirmed FTS5 guard: `if dialect != "sqlite": return`
- Codebase: `requirements.txt` + `pyproject.toml` — confirmed all existing deps and Python version matrix (3.10/3.11/3.12 in CI)
- Codebase: `src/paperbot/infrastructure/stores/models.py` — confirmed LargeBinary embedding column, Text JSON columns pattern

---

*Stack research for: PostgreSQL migration + async data layer + PG-native features*
*Researched: 2026-03-14*
