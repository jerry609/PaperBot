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


def _has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def _is_column_nullable(table_name: str, column_name: str) -> bool | None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for col in inspector.get_columns(table_name):
        if col["name"] == column_name:
            return bool(col.get("nullable", True))
    return None


def _has_index(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def _has_unique_constraint(table_name: str, constraint_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(
        uq.get("name") == constraint_name
        for uq in inspector.get_unique_constraints(table_name)
    )


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    # Add user_id column (nullable first, then backfill, then NOT NULL)
    if not _has_column("repro_code_experience", "user_id"):
        op.add_column(
            "repro_code_experience",
            sa.Column("user_id", sa.String(64), nullable=True),
        )

    op.execute("UPDATE repro_code_experience SET user_id = 'default' WHERE user_id IS NULL")

    if _is_column_nullable("repro_code_experience", "user_id"):
        if is_sqlite:
            with op.batch_alter_table("repro_code_experience") as batch_op:
                batch_op.alter_column(
                    "user_id",
                    existing_type=sa.String(length=64),
                    nullable=False,
                )
        else:
            op.alter_column(
                "repro_code_experience",
                "user_id",
                existing_type=sa.String(length=64),
                nullable=False,
            )

    if not _has_index("repro_code_experience", "ix_repro_code_experience_user_id"):
        op.create_index("ix_repro_code_experience_user_id", "repro_code_experience", ["user_id"])

    # Dedup unique constraint (user_id + paper_id + pattern_type + content)
    if not _has_unique_constraint("repro_code_experience", "uq_repro_exp_user_paper_type_content"):
        if is_sqlite:
            with op.batch_alter_table("repro_code_experience") as batch_op:
                batch_op.create_unique_constraint(
                    "uq_repro_exp_user_paper_type_content",
                    ["user_id", "paper_id", "pattern_type", "content"],
                )
        else:
            op.create_unique_constraint(
                "uq_repro_exp_user_paper_type_content",
                "repro_code_experience",
                ["user_id", "paper_id", "pattern_type", "content"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    if _has_unique_constraint("repro_code_experience", "uq_repro_exp_user_paper_type_content"):
        if is_sqlite:
            with op.batch_alter_table("repro_code_experience") as batch_op:
                batch_op.drop_constraint("uq_repro_exp_user_paper_type_content", type_="unique")
        else:
            op.drop_constraint(
                "uq_repro_exp_user_paper_type_content",
                "repro_code_experience",
                type_="unique",
            )

    if _has_index("repro_code_experience", "ix_repro_code_experience_user_id"):
        op.drop_index("ix_repro_code_experience_user_id", table_name="repro_code_experience")

    if _has_column("repro_code_experience", "user_id"):
        if is_sqlite:
            with op.batch_alter_table("repro_code_experience") as batch_op:
                batch_op.drop_column("user_id")
        else:
            op.drop_column("repro_code_experience", "user_id")
