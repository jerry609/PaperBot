"""Add password_reset_tokens table

Revision ID: 0025_password_reset_tokens
Revises: 0024_users
Create Date: 2026-03-10
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0025_password_reset_tokens"
down_revision = "0024_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "password_reset_tokens" not in inspector.get_table_names():
        op.create_table(
            "password_reset_tokens",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer, nullable=False),
            sa.Column("token", sa.String(64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("used", sa.Boolean, nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_prt_token", "password_reset_tokens", ["token"], unique=True)
        op.create_index("ix_prt_user_id", "password_reset_tokens", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_prt_user_id", table_name="password_reset_tokens")
    op.drop_index("ix_prt_token", table_name="password_reset_tokens")
    op.drop_table("password_reset_tokens")
