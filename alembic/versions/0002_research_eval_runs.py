"""research eval runs

Revision ID: 0002_research_eval_runs
Revises: 0001_baseline_schema
Create Date: 2025-12-23

Adds:
- research_context_runs: context build events (routing + config)
- paper_impressions: recommended papers shown to user
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0002_research_eval_runs"
down_revision = "0001_baseline_schema"
branch_labels = None
depends_on = None


def _is_offline() -> bool:
    try:
        return bool(context.is_offline_mode())
    except Exception:
        return False


def _insp():
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return _insp().has_table(name)


def _get_indexes(table: str) -> set[str]:
    idx = set()
    for i in _insp().get_indexes(table):
        idx.add(str(i.get("name") or ""))
    return idx


def _create_index(name: str, table: str, cols: list[str]) -> None:
    if _is_offline():
        op.create_index(name, table, cols)
        return
    if name in _get_indexes(table):
        return
    op.create_index(name, table, cols)


def upgrade() -> None:
    if _is_offline():
        _upgrade_create_tables()
        return
    _upgrade_create_tables()
    _upgrade_create_indexes()


def _upgrade_create_tables() -> None:
    if _is_offline() or not _has_table("research_context_runs"):
        op.create_table(
            "research_context_runs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("track_id", sa.Integer(), sa.ForeignKey("research_tracks.id"), nullable=True),
            sa.Column("query", sa.Text(), server_default="", nullable=False),
            sa.Column("merged_query", sa.Text(), server_default="", nullable=False),
            sa.Column("stage", sa.String(length=16), server_default="auto", nullable=False),
            sa.Column("exploration_ratio", sa.Float(), server_default="0.0", nullable=False),
            sa.Column("diversity_strength", sa.Float(), server_default="0.0", nullable=False),
            sa.Column("routing_json", sa.Text(), server_default="{}", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if _is_offline() or not _has_table("paper_impressions"):
        op.create_table(
            "paper_impressions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "run_id", sa.Integer(), sa.ForeignKey("research_context_runs.id"), nullable=False
            ),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("track_id", sa.Integer(), sa.ForeignKey("research_tracks.id"), nullable=True),
            sa.Column("paper_id", sa.String(length=64), nullable=False),
            sa.Column("rank", sa.Integer(), server_default="0", nullable=False),
            sa.Column("score", sa.Float(), server_default="0.0", nullable=False),
            sa.Column("reasons_json", sa.Text(), server_default="[]", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("run_id", "paper_id", name="uq_paper_impression_run_paper"),
        )


def _upgrade_create_indexes() -> None:
    _create_index("ix_research_context_runs_user_id", "research_context_runs", ["user_id"])
    _create_index("ix_research_context_runs_track_id", "research_context_runs", ["track_id"])
    _create_index("ix_research_context_runs_stage", "research_context_runs", ["stage"])
    _create_index("ix_research_context_runs_created_at", "research_context_runs", ["created_at"])

    _create_index("ix_paper_impressions_run_id", "paper_impressions", ["run_id"])
    _create_index("ix_paper_impressions_user_id", "paper_impressions", ["user_id"])
    _create_index("ix_paper_impressions_track_id", "paper_impressions", ["track_id"])
    _create_index("ix_paper_impressions_paper_id", "paper_impressions", ["paper_id"])
    _create_index("ix_paper_impressions_rank", "paper_impressions", ["rank"])
    _create_index("ix_paper_impressions_created_at", "paper_impressions", ["created_at"])


def downgrade() -> None:
    op.drop_table("paper_impressions")
    op.drop_table("research_context_runs")
