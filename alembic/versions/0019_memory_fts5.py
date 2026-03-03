"""memory_items FTS5 full-text index

Revision ID: 0019_memory_fts5
Revises: b94c1a2be26e
Create Date: 2026-03-03

Creates a SQLite FTS5 virtual table for memory_items.content and three triggers
(after insert / after delete / after update) to keep it in sync.

FTS5 provides BM25-ranked full-text search; falls back to LIKE on non-SQLite
databases (Postgres) where FTS5 is not available.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0019_memory_fts5"
down_revision = "b94c1a2be26e"
branch_labels = None
depends_on = None

# --------------------------------------------------------------------------- #
# SQLite-only helpers                                                          #
# --------------------------------------------------------------------------- #

_CREATE_FTS = """\
CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts
USING fts5(content, tokenize='porter ascii');
"""

_POPULATE_FTS = """\
INSERT INTO memory_items_fts(rowid, content)
SELECT id, content FROM memory_items
WHERE deleted_at IS NULL;
"""

_TRIGGER_INSERT = """\
CREATE TRIGGER IF NOT EXISTS memory_items_fts_ai
AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_items_fts(rowid, content) VALUES (new.id, new.content);
END;
"""

_TRIGGER_DELETE = """\
CREATE TRIGGER IF NOT EXISTS memory_items_fts_ad
AFTER DELETE ON memory_items BEGIN
    DELETE FROM memory_items_fts WHERE rowid = old.id;
END;
"""

_TRIGGER_UPDATE = """\
CREATE TRIGGER IF NOT EXISTS memory_items_fts_au
AFTER UPDATE OF content ON memory_items BEGIN
    DELETE FROM memory_items_fts WHERE rowid = old.id;
    INSERT INTO memory_items_fts(rowid, content) VALUES (new.id, new.content);
END;
"""

_DROP_TRIGGERS = """\
DROP TRIGGER IF EXISTS memory_items_fts_ai;
DROP TRIGGER IF EXISTS memory_items_fts_ad;
DROP TRIGGER IF EXISTS memory_items_fts_au;
"""

_DROP_FTS = "DROP TABLE IF EXISTS memory_items_fts;"


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect != "sqlite":
        # FTS5 is SQLite-only; Postgres uses pg_trgm / tsvector instead.
        return

    bind.execute(sa.text(_CREATE_FTS))
    bind.execute(sa.text(_POPULATE_FTS))
    bind.execute(sa.text(_TRIGGER_INSERT))
    bind.execute(sa.text(_TRIGGER_DELETE))
    bind.execute(sa.text(_TRIGGER_UPDATE))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return

    bind.execute(sa.text(_DROP_TRIGGERS))
    bind.execute(sa.text(_DROP_FTS))
