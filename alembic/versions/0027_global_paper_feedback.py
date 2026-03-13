"""Allow global paper feedback without an active track.

Revision ID: 0027_global_paper_feedback
Revises: 0026_embedding_endpoint_settings
Create Date: 2026-03-13
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0027_global_paper_feedback"
down_revision = "0026_embedding_endpoint_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "paper_feedback" not in inspector.get_table_names():
        return

    columns = {column["name"]: column for column in inspector.get_columns("paper_feedback")}
    track_column = columns.get("track_id")
    if not track_column or bool(track_column.get("nullable")):
        return

    with op.batch_alter_table("paper_feedback") as batch_op:
        batch_op.alter_column("track_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "paper_feedback" not in inspector.get_table_names():
        return

    columns = {column["name"]: column for column in inspector.get_columns("paper_feedback")}
    track_column = columns.get("track_id")
    if not track_column or not bool(track_column.get("nullable")):
        return

    null_count = conn.execute(sa.text("SELECT COUNT(*) FROM paper_feedback WHERE track_id IS NULL")).scalar()
    if null_count:
        raise RuntimeError("Cannot downgrade: global paper_feedback rows with NULL track_id exist")

    with op.batch_alter_table("paper_feedback") as batch_op:
        batch_op.alter_column("track_id", existing_type=sa.Integer(), nullable=False)
