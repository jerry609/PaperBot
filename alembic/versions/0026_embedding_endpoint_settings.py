"""add embedding endpoint settings table

Revision ID: 0026_embedding_endpoint_settings
Revises: 0025_password_reset_tokens, 0024_document_evidence_index
Create Date: 2026-03-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0026_embedding_endpoint_settings"
down_revision = ("0025_password_reset_tokens", "0024_document_evidence_index")
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "embedding_endpoints" not in inspector.get_table_names():
        op.create_table(
            "embedding_endpoints",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("scope", sa.String(length=32), nullable=False, server_default="default"),
            sa.Column("provider", sa.String(length=32), nullable=False, server_default="openai"),
            sa.Column("base_url", sa.String(length=512), nullable=True),
            sa.Column(
                "api_key_env",
                sa.String(length=64),
                nullable=False,
                server_default="PAPERBOT_EMBEDDING_API_KEY",
            ),
            sa.Column("api_key_value", sa.String(length=512), nullable=True),
            sa.Column(
                "model",
                sa.String(length=128),
                nullable=False,
                server_default="text-embedding-3-small",
            ),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("scope", name="uq_embedding_endpoints_scope"),
        )
        op.create_index("ix_embedding_endpoints_scope", "embedding_endpoints", ["scope"])
        op.create_index("ix_embedding_endpoints_provider", "embedding_endpoints", ["provider"])
        op.create_index("ix_embedding_endpoints_enabled", "embedding_endpoints", ["enabled"])


def downgrade() -> None:
    op.drop_index("ix_embedding_endpoints_enabled", table_name="embedding_endpoints")
    op.drop_index("ix_embedding_endpoints_provider", table_name="embedding_endpoints")
    op.drop_index("ix_embedding_endpoints_scope", table_name="embedding_endpoints")
    op.drop_table("embedding_endpoints")
