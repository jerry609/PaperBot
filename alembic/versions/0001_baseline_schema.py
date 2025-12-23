"""baseline schema

Revision ID: 0001_baseline_schema
Revises: None
Create Date: 2025-12-23

This migration bootstraps the SQLite/Postgres schema used by PaperBot.

Notes:
- This repo previously relied on `Base.metadata.create_all()` for local dev.
- To keep upgrades safe for existing local DBs, the online-mode upgrade is best-effort and
  skips creating tables/columns that already exist.
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision = "0001_baseline_schema"
down_revision = None
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


def _get_columns(table: str) -> set[str]:
    cols = set()
    for c in _insp().get_columns(table):
        cols.add(str(c.get("name") or ""))
    return cols


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


def _add_column(table: str, column: sa.Column) -> None:
    if _is_offline():
        op.add_column(table, column)
        return
    if column.name in _get_columns(table):
        return
    op.add_column(table, column)


def upgrade() -> None:
    if _is_offline():
        # Offline mode: emit a full schema for a fresh DB.
        _upgrade_create_tables()
        return

    # Online mode: best-effort, idempotent-ish upgrade (local DBs may already have tables).
    _upgrade_create_tables()
    _upgrade_add_columns()
    _upgrade_create_indexes()


def _upgrade_create_tables() -> None:
    # ---- Core run/event tables
    if _is_offline() or not _has_table("agent_runs"):
        op.create_table(
            "agent_runs",
            sa.Column("run_id", sa.String(length=64), primary_key=True),
            sa.Column("workflow", sa.String(length=64), server_default="", nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("status", sa.String(length=32), server_default="running", nullable=False),
            sa.Column("executor_type", sa.String(length=32), nullable=True),
            sa.Column("timeout_seconds", sa.Integer(), nullable=True),
            sa.Column("paper_url", sa.String(length=512), nullable=True),
            sa.Column("paper_id", sa.String(length=64), nullable=True),
            sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
        )
        _create_index("ix_agent_runs_workflow", "agent_runs", ["workflow"])
        _create_index("ix_agent_runs_started_at", "agent_runs", ["started_at"])

    if _is_offline() or not _has_table("agent_events"):
        op.create_table(
            "agent_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("agent_runs.run_id"), nullable=False),
            sa.Column("trace_id", sa.String(length=64), server_default="", nullable=False),
            sa.Column("span_id", sa.String(length=32), server_default="", nullable=False),
            sa.Column("parent_span_id", sa.String(length=32), nullable=True),
            sa.Column("workflow", sa.String(length=64), server_default="", nullable=False),
            sa.Column("stage", sa.String(length=64), server_default="", nullable=False),
            sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
            sa.Column("agent_name", sa.String(length=128), server_default="", nullable=False),
            sa.Column("role", sa.String(length=32), server_default="", nullable=False),
            sa.Column("type", sa.String(length=64), server_default="", nullable=False),
            sa.Column("payload_json", sa.Text(), server_default="{}", nullable=False),
            sa.Column("metrics_json", sa.Text(), server_default="{}", nullable=False),
            sa.Column("tags_json", sa.Text(), server_default="{}", nullable=False),
            sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        )
        _create_index("ix_agent_events_run_id", "agent_events", ["run_id"])
        _create_index("ix_agent_events_trace_id", "agent_events", ["trace_id"])
        _create_index("ix_agent_events_ts", "agent_events", ["ts"])

    if _is_offline() or not _has_table("execution_logs"):
        op.create_table(
            "execution_logs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("agent_runs.run_id"), nullable=False),
            sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
            sa.Column("level", sa.String(length=16), server_default="info", nullable=False),
            sa.Column("message", sa.Text(), server_default="", nullable=False),
            sa.Column("source", sa.String(length=32), server_default="system", nullable=False),
        )
        _create_index("ix_execution_logs_run_id", "execution_logs", ["run_id"])
        _create_index("ix_execution_logs_ts", "execution_logs", ["ts"])

    if _is_offline() or not _has_table("resource_metrics"):
        op.create_table(
            "resource_metrics",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("agent_runs.run_id"), nullable=False),
            sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
            sa.Column("cpu_percent", sa.Float(), server_default="0", nullable=False),
            sa.Column("memory_mb", sa.Float(), server_default="0", nullable=False),
            sa.Column("memory_limit_mb", sa.Float(), server_default="4096", nullable=False),
        )
        _create_index("ix_resource_metrics_run_id", "resource_metrics", ["run_id"])
        _create_index("ix_resource_metrics_ts", "resource_metrics", ["ts"])

    if _is_offline() or not _has_table("runbook_steps"):
        op.create_table(
            "runbook_steps",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("agent_runs.run_id"), nullable=False),
            sa.Column("step_name", sa.String(length=64), server_default="", nullable=False),
            sa.Column("status", sa.String(length=32), server_default="running", nullable=False),
            sa.Column("executor_type", sa.String(length=32), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("command", sa.Text(), server_default="", nullable=False),
            sa.Column("exit_code", sa.Integer(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
        )
        _create_index("ix_runbook_steps_run_id", "runbook_steps", ["run_id"])
        _create_index("ix_runbook_steps_step_name", "runbook_steps", ["step_name"])
        _create_index("ix_runbook_steps_started_at", "runbook_steps", ["started_at"])

    if _is_offline() or not _has_table("artifacts"):
        op.create_table(
            "artifacts",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("agent_runs.run_id"), nullable=False),
            sa.Column("step_id", sa.Integer(), sa.ForeignKey("runbook_steps.id"), nullable=True),
            sa.Column("type", sa.String(length=32), server_default="", nullable=False),
            sa.Column("path_or_uri", sa.Text(), server_default="", nullable=False),
            sa.Column("mime", sa.String(length=128), nullable=True),
            sa.Column("size_bytes", sa.Integer(), nullable=True),
            sa.Column("sha256", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
        )
        _create_index("ix_artifacts_run_id", "artifacts", ["run_id"])
        _create_index("ix_artifacts_step_id", "artifacts", ["step_id"])
        _create_index("ix_artifacts_type", "artifacts", ["type"])
        _create_index("ix_artifacts_created_at", "artifacts", ["created_at"])

    # ---- Memory tables
    if _is_offline() or not _has_table("memory_sources"):
        op.create_table(
            "memory_sources",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("platform", sa.String(length=32), server_default="unknown", nullable=False),
            sa.Column("filename", sa.String(length=256), server_default="", nullable=False),
            sa.Column("sha256", sa.String(length=64), nullable=False),
            sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
            sa.Column("conversation_count", sa.Integer(), server_default="0", nullable=False),
            sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
        )
        _create_index("ix_memory_sources_user_id", "memory_sources", ["user_id"])
        _create_index("ix_memory_sources_platform", "memory_sources", ["platform"])
        _create_index("ix_memory_sources_sha256", "memory_sources", ["sha256"])
        _create_index("ix_memory_sources_ingested_at", "memory_sources", ["ingested_at"])

    if _is_offline() or not _has_table("memory_items"):
        op.create_table(
            "memory_items",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("workspace_id", sa.String(length=64), nullable=True),
            sa.Column("scope_type", sa.String(length=16), server_default="global", nullable=False),
            sa.Column("scope_id", sa.String(length=64), nullable=True),
            sa.Column("kind", sa.String(length=32), server_default="fact", nullable=False),
            sa.Column("content", sa.Text(), server_default="", nullable=False),
            sa.Column("content_hash", sa.String(length=64), nullable=False),
            sa.Column("confidence", sa.Float(), server_default="0.6", nullable=False),
            sa.Column("status", sa.String(length=16), server_default="approved", nullable=False),
            sa.Column("supersedes_id", sa.Integer(), sa.ForeignKey("memory_items.id"), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("use_count", sa.Integer(), server_default="0", nullable=False),
            sa.Column("pii_risk", sa.Integer(), server_default="0", nullable=False),
            sa.Column("tags_json", sa.Text(), server_default="[]", nullable=False),
            sa.Column("evidence_json", sa.Text(), server_default="{}", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("deleted_reason", sa.Text(), server_default="", nullable=False),
            sa.Column("source_id", sa.Integer(), sa.ForeignKey("memory_sources.id"), nullable=True),
            sa.UniqueConstraint("user_id", "content_hash", name="uq_memory_user_hash"),
        )
        _create_index("ix_memory_items_user_id", "memory_items", ["user_id"])
        _create_index("ix_memory_items_workspace_id", "memory_items", ["workspace_id"])
        _create_index("ix_memory_items_scope_type", "memory_items", ["scope_type"])
        _create_index("ix_memory_items_scope_id", "memory_items", ["scope_id"])
        _create_index("ix_memory_items_kind", "memory_items", ["kind"])
        _create_index("ix_memory_items_content_hash", "memory_items", ["content_hash"])
        _create_index("ix_memory_items_status", "memory_items", ["status"])
        _create_index("ix_memory_items_supersedes_id", "memory_items", ["supersedes_id"])
        _create_index("ix_memory_items_expires_at", "memory_items", ["expires_at"])
        _create_index("ix_memory_items_last_used_at", "memory_items", ["last_used_at"])
        _create_index("ix_memory_items_deleted_at", "memory_items", ["deleted_at"])
        _create_index("ix_memory_items_source_id", "memory_items", ["source_id"])
        _create_index("ix_memory_items_created_at", "memory_items", ["created_at"])
        _create_index("ix_memory_items_updated_at", "memory_items", ["updated_at"])

    if _is_offline() or not _has_table("memory_audit_log"):
        op.create_table(
            "memory_audit_log",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
            sa.Column("actor_id", sa.String(length=64), server_default="", nullable=False),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("workspace_id", sa.String(length=64), nullable=True),
            sa.Column("action", sa.String(length=32), server_default="", nullable=False),
            sa.Column("item_id", sa.Integer(), sa.ForeignKey("memory_items.id"), nullable=True),
            sa.Column("source_id", sa.Integer(), sa.ForeignKey("memory_sources.id"), nullable=True),
            sa.Column("detail_json", sa.Text(), server_default="{}", nullable=False),
        )
        _create_index("ix_memory_audit_log_ts", "memory_audit_log", ["ts"])
        _create_index("ix_memory_audit_log_actor_id", "memory_audit_log", ["actor_id"])
        _create_index("ix_memory_audit_log_user_id", "memory_audit_log", ["user_id"])
        _create_index("ix_memory_audit_log_workspace_id", "memory_audit_log", ["workspace_id"])
        _create_index("ix_memory_audit_log_action", "memory_audit_log", ["action"])
        _create_index("ix_memory_audit_log_item_id", "memory_audit_log", ["item_id"])
        _create_index("ix_memory_audit_log_source_id", "memory_audit_log", ["source_id"])

    # ---- Research tables
    if _is_offline() or not _has_table("research_tracks"):
        op.create_table(
            "research_tracks",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=128), server_default="", nullable=False),
            sa.Column("description", sa.Text(), server_default="", nullable=False),
            sa.Column("keywords_json", sa.Text(), server_default="[]", nullable=False),
            sa.Column("venues_json", sa.Text(), server_default="[]", nullable=False),
            sa.Column("methods_json", sa.Text(), server_default="[]", nullable=False),
            sa.Column("is_active", sa.Integer(), server_default="0", nullable=False),
            sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("user_id", "name", name="uq_research_tracks_user_name"),
        )
        _create_index("ix_research_tracks_user_id", "research_tracks", ["user_id"])
        _create_index("ix_research_tracks_is_active", "research_tracks", ["is_active"])
        _create_index("ix_research_tracks_archived_at", "research_tracks", ["archived_at"])
        _create_index("ix_research_tracks_created_at", "research_tracks", ["created_at"])
        _create_index("ix_research_tracks_updated_at", "research_tracks", ["updated_at"])

    if _is_offline() or not _has_table("research_tasks"):
        op.create_table(
            "research_tasks",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("track_id", sa.Integer(), sa.ForeignKey("research_tracks.id"), nullable=False),
            sa.Column("title", sa.Text(), server_default="", nullable=False),
            sa.Column("status", sa.String(length=16), server_default="todo", nullable=False),
            sa.Column("priority", sa.Integer(), server_default="0", nullable=False),
            sa.Column("paper_id", sa.String(length=64), nullable=True),
            sa.Column("paper_url", sa.String(length=512), nullable=True),
            sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("done_at", sa.DateTime(timezone=True), nullable=True),
        )
        _create_index("ix_research_tasks_track_id", "research_tasks", ["track_id"])
        _create_index("ix_research_tasks_status", "research_tasks", ["status"])
        _create_index("ix_research_tasks_priority", "research_tasks", ["priority"])
        _create_index("ix_research_tasks_paper_id", "research_tasks", ["paper_id"])
        _create_index("ix_research_tasks_created_at", "research_tasks", ["created_at"])
        _create_index("ix_research_tasks_updated_at", "research_tasks", ["updated_at"])
        _create_index("ix_research_tasks_done_at", "research_tasks", ["done_at"])

    if _is_offline() or not _has_table("research_milestones"):
        op.create_table(
            "research_milestones",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("track_id", sa.Integer(), sa.ForeignKey("research_tracks.id"), nullable=False),
            sa.Column("name", sa.String(length=128), server_default="", nullable=False),
            sa.Column("status", sa.String(length=16), server_default="todo", nullable=False),
            sa.Column("notes", sa.Text(), server_default="", nullable=False),
            sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        _create_index("ix_research_milestones_track_id", "research_milestones", ["track_id"])
        _create_index("ix_research_milestones_status", "research_milestones", ["status"])
        _create_index("ix_research_milestones_due_at", "research_milestones", ["due_at"])
        _create_index("ix_research_milestones_created_at", "research_milestones", ["created_at"])
        _create_index("ix_research_milestones_updated_at", "research_milestones", ["updated_at"])

    if _is_offline() or not _has_table("paper_feedback"):
        op.create_table(
            "paper_feedback",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("track_id", sa.Integer(), sa.ForeignKey("research_tracks.id"), nullable=False),
            sa.Column("paper_id", sa.String(length=64), nullable=False),
            sa.Column("action", sa.String(length=16), nullable=False),
            sa.Column("weight", sa.Float(), server_default="0", nullable=False),
            sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
            sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
        )
        _create_index("ix_paper_feedback_user_id", "paper_feedback", ["user_id"])
        _create_index("ix_paper_feedback_track_id", "paper_feedback", ["track_id"])
        _create_index("ix_paper_feedback_paper_id", "paper_feedback", ["paper_id"])
        _create_index("ix_paper_feedback_action", "paper_feedback", ["action"])
        _create_index("ix_paper_feedback_ts", "paper_feedback", ["ts"])
        _create_index("ix_paper_feedback_track_action_ts", "paper_feedback", ["track_id", "action", "ts"])

    if _is_offline() or not _has_table("research_track_embeddings"):
        op.create_table(
            "research_track_embeddings",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("track_id", sa.Integer(), sa.ForeignKey("research_tracks.id"), nullable=False),
            sa.Column("model", sa.String(length=64), server_default="text-embedding-3-small", nullable=False),
            sa.Column("text_hash", sa.String(length=64), server_default="", nullable=False),
            sa.Column("embedding_json", sa.Text(), server_default="[]", nullable=False),
            sa.Column("dim", sa.Integer(), server_default="0", nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("track_id", "model", name="uq_track_embedding_track_model"),
        )
        _create_index("ix_research_track_embeddings_track_id", "research_track_embeddings", ["track_id"])
        _create_index("ix_research_track_embeddings_model", "research_track_embeddings", ["model"])
        _create_index("ix_research_track_embeddings_text_hash", "research_track_embeddings", ["text_hash"])
        _create_index("ix_research_track_embeddings_updated_at", "research_track_embeddings", ["updated_at"])


def _upgrade_add_columns() -> None:
    # Best-effort additive upgrades for older local DBs.
    if not _has_table("memory_items"):
        return

    cols = _get_columns("memory_items")

    if "workspace_id" not in cols:
        _add_column("memory_items", sa.Column("workspace_id", sa.String(length=64), nullable=True))
    if "scope_type" not in cols:
        _add_column(
            "memory_items",
            sa.Column("scope_type", sa.String(length=16), server_default="global", nullable=False),
        )
    if "scope_id" not in cols:
        _add_column("memory_items", sa.Column("scope_id", sa.String(length=64), nullable=True))


def _upgrade_create_indexes() -> None:
    # A small set of composite indexes used by the read paths (best-effort).
    if not _has_table("memory_items"):
        return
    _create_index(
        "ix_memory_items_user_scope_status_deleted",
        "memory_items",
        ["user_id", "scope_type", "scope_id", "status", "deleted_at"],
    )


def downgrade() -> None:
    # Best-effort downgrade. SQLite can't reliably drop columns; we only drop tables we own.
    if not _is_offline():
        insp = _insp()
        if insp.has_table("research_track_embeddings"):
            op.drop_table("research_track_embeddings")
        if insp.has_table("paper_feedback"):
            op.drop_table("paper_feedback")
        if insp.has_table("research_milestones"):
            op.drop_table("research_milestones")
        if insp.has_table("research_tasks"):
            op.drop_table("research_tasks")
        if insp.has_table("research_tracks"):
            op.drop_table("research_tracks")
        return

    op.drop_table("research_track_embeddings")
    op.drop_table("paper_feedback")
    op.drop_table("research_milestones")
    op.drop_table("research_tasks")
    op.drop_table("research_tracks")
