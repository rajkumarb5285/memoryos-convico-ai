"""Initial schema for MEMORYOS

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-30 19:25:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 0. Ensure vector extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. Users table
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # 2. Characters table
    op.create_table(
        "characters",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_characters_id", "characters", ["id"])
    op.create_index("ix_characters_owner_user_id", "characters", ["owner_user_id"])

    # 3. Conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("character_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_id", "conversations", ["id"])
    op.create_index("ix_conversations_owner_user_id", "conversations", ["owner_user_id"])
    op.create_index("ix_conversations_character_id", "conversations", ["character_id"])

    # 4. Messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("character_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_id", "messages", ["id"])
    op.create_index("ix_messages_owner_user_id", "messages", ["owner_user_id"])
    op.create_index("ix_messages_character_id", "messages", ["character_id"])
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_idempotency_key", "messages", ["idempotency_key"])
    op.create_index("idx_messages_owner_char_conv", "messages", ["owner_user_id", "character_id", "conversation_id", "created_at"])

    # 5. Memories table
    op.create_table(
        "memories",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("character_id", sa.String(length=36), nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=True),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("predicate", sa.String(length=100), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("memory_type", sa.String(length=50), nullable=False),
        sa.Column("domain", sa.String(length=50), nullable=False),
        sa.Column("polarity", sa.String(length=20), nullable=False, server_default="positive"),
        sa.Column("temporal_status", sa.String(length=20), nullable=False, server_default="current"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("source_message_id", sa.String(length=36), nullable=True),
        sa.Column("supersedes_memory_id", sa.String(length=36), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("valid_until IS NULL OR valid_until >= valid_from", name="check_valid_temporal_range"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["supersedes_memory_id"], ["memories.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_id", "memories", ["id"])
    op.create_index("ix_memories_owner_user_id", "memories", ["owner_user_id"])
    op.create_index("ix_memories_character_id", "memories", ["character_id"])
    op.create_index("ix_memories_conversation_id", "memories", ["conversation_id"])
    op.create_index("ix_memories_subject", "memories", ["subject"])
    op.create_index("ix_memories_predicate", "memories", ["predicate"])
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"])
    op.create_index("ix_memories_domain", "memories", ["domain"])
    op.create_index("ix_memories_temporal_status", "memories", ["temporal_status"])
    op.create_index("ix_memories_deleted_at", "memories", ["deleted_at"])
    op.create_index("idx_memories_owner_char_active", "memories", ["owner_user_id", "character_id", "deleted_at"])
    op.create_index("idx_memories_owner_char_temporal", "memories", ["owner_user_id", "character_id", "temporal_status"])
    op.create_index("idx_memories_domain_predicate", "memories", ["owner_user_id", "character_id", "domain", "predicate"])

    # Full text search index
    op.execute(
        "CREATE INDEX idx_memories_fts ON memories USING gin("
        "to_tsvector('english', subject || ' ' || predicate || ' ' || object || ' ' || coalesce(raw_text, ''))"
        ");"
    )

    # Vector index for pgvector cosine distance
    op.execute(
        "CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.drop_table("memories")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("characters")
    op.drop_table("users")

