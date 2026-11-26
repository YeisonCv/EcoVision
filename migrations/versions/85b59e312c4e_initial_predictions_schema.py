"""initial predictions schema

Revision ID: 85b59e312c4e
Revises: 
Create Date: 2026-11-24 23:39:18.344537

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '85b59e312c4e'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, primary_key=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=80), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_predictions_category', 'predictions', ['category'], unique=False)
    op.create_index('ix_predictions_created_at', 'predictions', ['created_at'], unique=False)


def downgrade():
    raise RuntimeError(
        "La migración inicial no se revierte automáticamente para proteger los registros de predictions. "
        "Aplica una migración correctiva hacia delante."
    )
