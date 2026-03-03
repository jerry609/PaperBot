"""memory_items embedding column + sqlite-vec virtual table

Revision ID: 0020_memory_embedding
Revises: 0019_memory_fts5
Create Date: 2026-03-03

Phase B of issue #161:
- Adds `embedding BLOB` column to memory_items for storing float32 vectors.
- Creates `vec_items` sqlite-vec virtual table (SQLite + sqlite-vec only).
- Creates sync triggers to keep vec_items in step with memory_items.embedding.

Graceful degradation: if sqlite-vec is not installed the column migration still
runs; only the virtual table and triggers are skipped.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0020_memory_embedding"
down_revision = "0019_memory_fts5"
branch_labels = None
depends_on = None

_EMBEDDING_DIM = 1536  # text-embedding-3-small

_ADD_COLUMN = "ALTER TABLE memory_items ADD COLUMN embedding BLOB"

_CREATE_VEC = f"""\
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
USING vec0(embedding float[{_EMBEDDING_DIM}])
"""

_POPULATE_VEC = """\
INSERT OR IGNORE INTO vec_items(rowid, embedding)
SELECT id, embedding FROM memory_items
WHERE embedding IS NOT NULL AND deleted_at IS NULL
"""

_TRIGGER_VEC_INSERT = """\
CREATE TRIGGER IF NOT EXISTS memory_items_vec_ai
AFTER INSERT ON memory_items
WHEN new.embedding IS NOT NULL
BEGIN
    INSERT OR REPLACE INTO vec_items(rowid, embedding) VALUES (new.id, new.embedding);
END
"""

_TRIGGER_VEC_UPDATE = """\
CREATE TRIGGER IF NOT EXISTS memory_items_vec_au
AFTER UPDATE OF embedding ON memory_items
BEGIN
    DELETE FROM vec_items WHERE rowid = old.id;
    INSERT OR IGNORE INTO vec_items(rowid, embedding)
    SELECT new.id, new.embedding WHERE new.embedding IS NOT NULL;
END
"""

_TRIGGER_VEC_DELETE = """\
CREATE TRIGGER IF NOT EXISTS memory_items_vec_ad
AFTER DELETE ON memory_items BEGIN
    DELETE FROM vec_items WHERE rowid = old.id;
END
"""

_DROP_VEC_TRIGGERS = """\
DROP TRIGGER IF EXISTS memory_items_vec_ai;
DROP TRIGGER IF EXISTS memory_items_vec_au;
DROP TRIGGER IF EXISTS memory_items_vec_ad;
"""
_DROP_VEC_TABLE = "DROP TABLE IF EXISTS vec_items"


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # 1. Add embedding column (all dialects — column is used for BLOB storage).
    try:
        bind.execute(sa.text(_ADD_COLUMN))
    except Exception:
        pass  # Column already exists.

    if dialect != "sqlite":
        return  # sqlite-vec is SQLite-only.

    # 2. Load sqlite-vec extension (best-effort).
    try:
        import sqlite_vec  # type: ignore
        raw_conn = bind.connection.dbapi_connection  # type: ignore
        raw_conn.enable_load_extension(True)
        sqlite_vec.load(raw_conn)
        raw_conn.enable_load_extension(False)
    except Exception:
        return  # sqlite-vec not installed — skip virtual table creation.

    # 3. Create vec_items virtual table + triggers.
    bind.execute(sa.text(_CREATE_VEC))
    bind.execute(sa.text(_POPULATE_VEC))
    bind.execute(sa.text(_TRIGGER_VEC_INSERT))
    bind.execute(sa.text(_TRIGGER_VEC_UPDATE))
    bind.execute(sa.text(_TRIGGER_VEC_DELETE))


def downgrade() -> None:
    bind = op.get_bind()

    # Drop vec infrastructure (SQLite only; ignore errors).
    if bind.dialect.name == "sqlite":
        try:
            bind.execute(sa.text(_DROP_VEC_TRIGGERS))
            bind.execute(sa.text(_DROP_VEC_TABLE))
        except Exception:
            pass

    # NOTE: SQLite doesn't support DROP COLUMN; leave the embedding column in place.
