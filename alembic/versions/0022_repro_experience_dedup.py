"""Add UNIQUE constraint to repro_code_experience for deduplication

Revision ID: 0022_repro_experience_dedup
Revises: 0021_repro_code_experience
Create Date: 2026-03-04

Prevents duplicate experience records with the same (paper_id, pattern_type,
content) from being written by GenerationNode / VerificationNode retries.
"""
from __future__ import annotations

from alembic import op


revision = "0022_repro_experience_dedup"
down_revision = "0021_repro_code_experience"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_repro_exp_paper_type_content",
        "repro_code_experience",
        ["paper_id", "pattern_type", "content"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_repro_exp_paper_type_content",
        "repro_code_experience",
        type_="unique",
    )
