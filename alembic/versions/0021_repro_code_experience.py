"""repro_code_experience table for CodeMemory persistence

Revision ID: 0021_repro_code_experience
Revises: 0020_memory_embedding
Create Date: 2026-03-03

Issue #162: Persist CodeMemory experience data so it survives process restarts.
Creates the repro_code_experience table with indexes on paper_id and pack_id.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_repro_code_experience"
down_revision = "0020_memory_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repro_code_experience",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pack_id", sa.String(64), nullable=True),
        sa.Column("paper_id", sa.String(256), nullable=True),
        sa.Column("pattern_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("code_snippet", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_repro_code_experience_paper_id", "repro_code_experience", ["paper_id"])
    op.create_index("ix_repro_code_experience_pack_id", "repro_code_experience", ["pack_id"])
    op.create_index("ix_repro_code_experience_pattern_type", "repro_code_experience", ["pattern_type"])


def downgrade() -> None:
    op.drop_index("ix_repro_code_experience_pattern_type", table_name="repro_code_experience")
    op.drop_index("ix_repro_code_experience_pack_id", table_name="repro_code_experience")
    op.drop_index("ix_repro_code_experience_paper_id", table_name="repro_code_experience")
    op.drop_table("repro_code_experience")
