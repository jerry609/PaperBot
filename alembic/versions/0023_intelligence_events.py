"""Create intelligence_events table for community radar

Revision ID: 0023_intelligence_events
Revises: 0022_repro_experience_dedup
Create Date: 2026-03-10

Stores cached signals from Reddit, GitHub, HuggingFace and X for the
community radar / intelligence feed feature.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0023_intelligence_events"
down_revision = "0022_repro_experience_dedup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_exists = inspector.has_table("intelligence_events")

    if not table_exists:
        op.create_table(
            "intelligence_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(64), nullable=False, index=True),
            sa.Column("external_id", sa.String(255), nullable=False),
            sa.Column("source", sa.String(32), nullable=False, server_default="unknown", index=True),
            sa.Column("source_label", sa.String(64), nullable=False, server_default=""),
            sa.Column("kind", sa.String(64), nullable=False, server_default="signal", index=True),
            sa.Column("title", sa.Text(), nullable=False, server_default=""),
            sa.Column("summary", sa.Text(), nullable=False, server_default=""),
            sa.Column("url", sa.Text(), nullable=False, server_default=""),
            sa.Column("repo_full_name", sa.String(128), nullable=False, server_default="", index=True),
            sa.Column("author_name", sa.String(128), nullable=False, server_default="", index=True),
            sa.Column("keyword_hits_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("author_matches_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("repo_matches_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("metric_name", sa.String(64), nullable=False, server_default=""),
            sa.Column("metric_value", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("metric_delta", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("score", sa.Float(), nullable=False, server_default="0.0", index=True),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True, index=True),
            sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("payload_json", sa.Text(), nullable=False, server_default="{}"),
            sa.UniqueConstraint("user_id", "external_id", name="uq_intelligence_events_user_external"),
        )

    inspector = sa.inspect(bind)
    index_names = {index["name"] for index in inspector.get_indexes("intelligence_events")}

    if "ix_intelligence_events_user_score" not in index_names:
        op.create_index(
            "ix_intelligence_events_user_score",
            "intelligence_events",
            ["user_id", "score"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("intelligence_events"):
        return

    index_names = {index["name"] for index in inspector.get_indexes("intelligence_events")}
    if "ix_intelligence_events_user_score" in index_names:
        op.drop_index("ix_intelligence_events_user_score", table_name="intelligence_events")

    op.drop_table("intelligence_events")
