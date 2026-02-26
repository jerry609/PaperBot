"""add_p2c_repro_context_pack_tables

Revision ID: b94c1a2be26e
Revises: 4c71b28a2f67
Create Date: 2026-02-26 16:34:35.636031

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = 'b94c1a2be26e'
down_revision = '4c71b28a2f67'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'repro_context_pack',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False, server_default='default'),
        sa.Column('project_id', sa.String(length=64), nullable=True),
        sa.Column('paper_id', sa.String(length=256), nullable=False),
        sa.Column('paper_title', sa.Text(), nullable=True),
        sa.Column('version', sa.String(length=16), nullable=False, server_default='v1'),
        sa.Column('depth', sa.String(length=16), nullable=False, server_default='standard'),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
        sa.Column('objective', sa.Text(), nullable=True),
        sa.Column('pack_json', sa.Text(), nullable=False, server_default='{}'),
        sa.Column('confidence_overall', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('warning_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_repro_context_pack_user_id', 'repro_context_pack', ['user_id'])
    op.create_index('ix_repro_context_pack_paper_id', 'repro_context_pack', ['paper_id'])
    op.create_index('ix_repro_context_pack_project_id', 'repro_context_pack', ['project_id'])
    op.create_index('ix_repro_context_pack_status', 'repro_context_pack', ['status'])
    op.create_index('ix_repro_context_pack_confidence_overall', 'repro_context_pack', ['confidence_overall'])
    op.create_index('ix_repro_context_pack_created_at', 'repro_context_pack', ['created_at'])

    op.create_table(
        'repro_context_stage_result',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('context_pack_id', sa.String(length=64), nullable=False),
        sa.Column('stage_name', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='completed'),
        sa.Column('result_json', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('duration_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['context_pack_id'], ['repro_context_pack.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_repro_context_stage_result_context_pack_id', 'repro_context_stage_result', ['context_pack_id'])
    op.create_index('ix_repro_context_stage_result_stage_name', 'repro_context_stage_result', ['stage_name'])
    op.create_index('ix_repro_context_stage_result_status', 'repro_context_stage_result', ['status'])

    op.create_table(
        'repro_context_evidence',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('context_pack_id', sa.String(length=64), nullable=False),
        sa.Column('evidence_type', sa.String(length=32), nullable=False),
        sa.Column('ref', sa.Text(), nullable=False, server_default=''),
        sa.Column('supports_json', sa.Text(), nullable=False, server_default='[]'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.ForeignKeyConstraint(['context_pack_id'], ['repro_context_pack.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_repro_context_evidence_context_pack_id', 'repro_context_evidence', ['context_pack_id'])
    op.create_index('ix_repro_context_evidence_evidence_type', 'repro_context_evidence', ['evidence_type'])

    op.create_table(
        'repro_context_feedback',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('context_pack_id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['context_pack_id'], ['repro_context_pack.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_repro_context_feedback_context_pack_id', 'repro_context_feedback', ['context_pack_id'])
    op.create_index('ix_repro_context_feedback_user_id', 'repro_context_feedback', ['user_id'])


def downgrade() -> None:
    op.drop_table('repro_context_feedback')
    op.drop_table('repro_context_evidence')
    op.drop_table('repro_context_stage_result')
    op.drop_table('repro_context_pack')
