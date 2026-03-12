"""Add users table

Revision ID: 0024_users
Revises: 0023_intelligence_events
Create Date: 2026-03-09
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0024_users"
down_revision = "0023_intelligence_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "users" not in inspector.get_table_names():
        op.create_table(
            "users",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("email", sa.String(255), nullable=True),
            sa.Column("hashed_password", sa.String(255), nullable=True),
            sa.Column("github_id", sa.String(64), nullable=True),
            sa.Column("github_username", sa.String(128), nullable=True),
            sa.Column("display_name", sa.String(128), nullable=True),
            sa.Column("avatar_url", sa.String(512), nullable=True),
            sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "email IS NOT NULL OR github_id IS NOT NULL",
                name="ck_users_identity",
            ),
        )

    existing_indexes = {i["name"] for i in inspector.get_indexes("users")}
    if "uq_users_email" not in existing_indexes:
        op.create_index("uq_users_email", "users", ["email"], unique=True)
    if "uq_users_github_id" not in existing_indexes:
        op.create_index("uq_users_github_id", "users", ["github_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_users_github_id", table_name="users")
    op.drop_index("uq_users_email", table_name="users")
    op.drop_table("users")

