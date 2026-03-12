"""document evidence index tables

Revision ID: 0024_document_evidence_index
Revises: 0023_intelligence_events
Create Date: 2026-03-12

Adds document intelligence persistence for explicit-ingest indexing:
- document_assets
- document_index_jobs
- document_chunks
- SQLite FTS5 index for chunk retrieval
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0024_document_evidence_index"
down_revision = "0023_intelligence_events"
branch_labels = None
depends_on = None

_CREATE_FTS = """\
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
USING fts5(heading, content, tokenize='porter ascii')
"""

_POPULATE_FTS = """\
INSERT INTO document_chunks_fts(rowid, heading, content)
SELECT id, heading, content FROM document_chunks
"""

_TRIGGER_INSERT = """\
CREATE TRIGGER IF NOT EXISTS document_chunks_fts_ai
AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, heading, content)
    VALUES (new.id, new.heading, new.content);
END
"""

_TRIGGER_DELETE = """\
CREATE TRIGGER IF NOT EXISTS document_chunks_fts_ad
AFTER DELETE ON document_chunks BEGIN
    DELETE FROM document_chunks_fts WHERE rowid = old.id;
END
"""

_TRIGGER_UPDATE = """\
CREATE TRIGGER IF NOT EXISTS document_chunks_fts_au
AFTER UPDATE OF heading, content ON document_chunks BEGIN
    DELETE FROM document_chunks_fts WHERE rowid = old.id;
    INSERT INTO document_chunks_fts(rowid, heading, content)
    VALUES (new.id, new.heading, new.content);
END
"""

_DROP_FTS = "DROP TABLE IF EXISTS document_chunks_fts"
_DROP_TRIGGERS = """\
DROP TRIGGER IF EXISTS document_chunks_fts_ai;
DROP TRIGGER IF EXISTS document_chunks_fts_ad;
DROP TRIGGER IF EXISTS document_chunks_fts_au;
"""


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("document_assets"):
        op.create_table(
            "document_assets",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("paper_id", sa.Integer(), nullable=False),
            sa.Column(
                "source_type", sa.String(length=32), nullable=False, server_default="paper_metadata"
            ),
            sa.Column("title", sa.Text(), nullable=False, server_default=""),
            sa.Column("locator_url", sa.String(length=512), nullable=True),
            sa.Column("checksum", sa.String(length=64), nullable=False, server_default=""),
            sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("paper_id", "source_type", name="uq_document_assets_paper_source"),
        )

    if not inspector.has_table("document_index_jobs"):
        op.create_table(
            "document_index_jobs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("paper_id", sa.Integer(), nullable=False),
            sa.Column("asset_id", sa.Integer(), nullable=True),
            sa.Column(
                "trigger_source", sa.String(length=64), nullable=False, server_default="manual"
            ),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="queued"),
            sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["asset_id"], ["document_assets.id"], ondelete="SET NULL"),
        )

    if not inspector.has_table("document_chunks"):
        op.create_table(
            "document_chunks",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("paper_id", sa.Integer(), nullable=False),
            sa.Column("asset_id", sa.Integer(), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("section", sa.String(length=128), nullable=False, server_default=""),
            sa.Column("heading", sa.String(length=255), nullable=False, server_default=""),
            sa.Column("content", sa.Text(), nullable=False, server_default=""),
            sa.Column("content_hash", sa.String(length=64), nullable=False, server_default=""),
            sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("embedding_json", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["asset_id"], ["document_assets.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("asset_id", "chunk_index", name="uq_document_chunks_asset_index"),
        )

    inspector = sa.inspect(bind)
    index_names = {
        index["name"]
        for table_name in ("document_assets", "document_index_jobs", "document_chunks")
        if inspector.has_table(table_name)
        for index in inspector.get_indexes(table_name)
    }

    indexes = [
        ("ix_document_assets_paper_id", "document_assets", ["paper_id"]),
        ("ix_document_assets_source_type", "document_assets", ["source_type"]),
        ("ix_document_assets_checksum", "document_assets", ["checksum"]),
        ("ix_document_index_jobs_paper_id", "document_index_jobs", ["paper_id"]),
        ("ix_document_index_jobs_asset_id", "document_index_jobs", ["asset_id"]),
        ("ix_document_index_jobs_trigger_source", "document_index_jobs", ["trigger_source"]),
        ("ix_document_index_jobs_status", "document_index_jobs", ["status"]),
        ("ix_document_index_jobs_enqueued_at", "document_index_jobs", ["enqueued_at"]),
        ("ix_document_index_jobs_updated_at", "document_index_jobs", ["updated_at"]),
        ("ix_document_chunks_paper_id", "document_chunks", ["paper_id"]),
        ("ix_document_chunks_asset_id", "document_chunks", ["asset_id"]),
        ("ix_document_chunks_chunk_index", "document_chunks", ["chunk_index"]),
        ("ix_document_chunks_section", "document_chunks", ["section"]),
        ("ix_document_chunks_content_hash", "document_chunks", ["content_hash"]),
        ("ix_document_chunks_created_at", "document_chunks", ["created_at"]),
        ("ix_document_chunks_updated_at", "document_chunks", ["updated_at"]),
        ("ix_document_chunks_paper_section", "document_chunks", ["paper_id", "section"]),
    ]
    for index_name, table_name, columns in indexes:
        if index_name in index_names:
            continue
        op.create_index(index_name, table_name, columns)

    if bind.dialect.name == "sqlite":
        bind.execute(sa.text(_CREATE_FTS))
        bind.execute(sa.text(_POPULATE_FTS))
        bind.execute(sa.text(_TRIGGER_INSERT))
        bind.execute(sa.text(_TRIGGER_DELETE))
        bind.execute(sa.text(_TRIGGER_UPDATE))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if bind.dialect.name == "sqlite":
        bind.execute(sa.text(_DROP_TRIGGERS))
        bind.execute(sa.text(_DROP_FTS))

    for table_name, index_names in (
        (
            "document_chunks",
            [
                "ix_document_chunks_paper_section",
                "ix_document_chunks_updated_at",
                "ix_document_chunks_created_at",
                "ix_document_chunks_content_hash",
                "ix_document_chunks_section",
                "ix_document_chunks_chunk_index",
                "ix_document_chunks_asset_id",
                "ix_document_chunks_paper_id",
            ],
        ),
        (
            "document_index_jobs",
            [
                "ix_document_index_jobs_updated_at",
                "ix_document_index_jobs_enqueued_at",
                "ix_document_index_jobs_status",
                "ix_document_index_jobs_trigger_source",
                "ix_document_index_jobs_asset_id",
                "ix_document_index_jobs_paper_id",
            ],
        ),
        (
            "document_assets",
            [
                "ix_document_assets_checksum",
                "ix_document_assets_source_type",
                "ix_document_assets_paper_id",
            ],
        ),
    ):
        if not inspector.has_table(table_name):
            continue
        existing_indexes = {index["name"] for index in inspector.get_indexes(table_name)}
        for index_name in index_names:
            if index_name in existing_indexes:
                op.drop_index(index_name, table_name=table_name)

    if inspector.has_table("document_chunks"):
        op.drop_table("document_chunks")
    if inspector.has_table("document_index_jobs"):
        op.drop_table("document_index_jobs")
    if inspector.has_table("document_assets"):
        op.drop_table("document_assets")
