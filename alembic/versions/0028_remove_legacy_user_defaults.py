"""Remove legacy shared default user_id server defaults.

Revision ID: 0028_remove_legacy_user_defaults
Revises: 0027_global_paper_feedback
Create Date: 2026-03-14
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0028_remove_legacy_user_defaults"
down_revision = "0027_global_paper_feedback"
branch_labels = None
depends_on = None


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    if _has_column("repro_context_pack", "user_id"):
        with op.batch_alter_table("repro_context_pack") as batch_op:
            batch_op.alter_column(
                "user_id",
                existing_type=sa.String(length=64),
                server_default=None,
            )

    if _has_column("intelligence_events", "user_id"):
        with op.batch_alter_table("intelligence_events") as batch_op:
            batch_op.alter_column(
                "user_id",
                existing_type=sa.String(length=64),
                server_default=None,
            )


def downgrade() -> None:
    if _has_column("repro_context_pack", "user_id"):
        with op.batch_alter_table("repro_context_pack") as batch_op:
            batch_op.alter_column(
                "user_id",
                existing_type=sa.String(length=64),
                server_default="default",
            )

    if _has_column("intelligence_events", "user_id"):
        with op.batch_alter_table("intelligence_events") as batch_op:
            batch_op.alter_column(
                "user_id",
                existing_type=sa.String(length=64),
                server_default="default",
            )
