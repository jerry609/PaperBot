"""Add user_id column and dedup constraint to repro_code_experience

Revision ID: 0022_repro_experience_dedup
Revises: 0021_repro_code_experience
Create Date: 2026-03-04

Adds user_id for multi-tenant isolation and a UNIQUE constraint on
(user_id, paper_id, pattern_type, content) to prevent duplicate
experience records from accumulating across GenerationNode / VerificationNode
retries for the same paper.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0022_repro_experience_dedup"
down_revision = "0021_repro_code_experience"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add user_id column (nullable first, then set default, then NOT NULL)
    op.add_column(
        "repro_code_experience",
        sa.Column("user_id", sa.String(64), nullable=True),
    )
    op.execute("UPDATE repro_code_experience SET user_id = 'default' WHERE user_id IS NULL")
    op.alter_column("repro_code_experience", "user_id", nullable=False)
    op.create_index("ix_repro_code_experience_user_id", "repro_code_experience", ["user_id"])

    # Dedup unique constraint (user_id + paper_id + pattern_type + content)
    op.create_unique_constraint(
        "uq_repro_exp_user_paper_type_content",
        "repro_code_experience",
        ["user_id", "paper_id", "pattern_type", "content"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_repro_exp_user_paper_type_content",
        "repro_code_experience",
        type_="unique",
    )
    op.drop_index("ix_repro_code_experience_user_id", table_name="repro_code_experience")
    op.drop_column("repro_code_experience", "user_id")
