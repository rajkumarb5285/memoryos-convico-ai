from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.character import Character
    from app.models.conversation import Conversation
    from app.models.user import User


class Memory(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "memories"

    owner_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    character_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    subject: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    predicate: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    object: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    memory_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # preference, location, employment, identity, plan, event
    domain: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # food, work, geography, personal, activity
    polarity: Mapped[str] = mapped_column(String(20), nullable=False, default="positive")  # positive, negative, neutral
    temporal_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="current", index=True
    )  # current, historical, future

    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source_message_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    supersedes_memory_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    embedding: Mapped[list | None] = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    owner: Mapped["User"] = relationship("User", back_populates="memories")
    character: Mapped["Character"] = relationship("Character", back_populates="memories")
    conversation: Mapped[Optional["Conversation"]] = relationship("Conversation", back_populates="memories")

    __table_args__ = (
        CheckConstraint(
            "valid_until IS NULL OR valid_until >= valid_from",
            name="check_valid_temporal_range",
        ),
        Index(
            "idx_memories_owner_char_active",
            "owner_user_id",
            "character_id",
            "deleted_at",
        ),
        Index(
            "idx_memories_owner_char_temporal",
            "owner_user_id",
            "character_id",
            "temporal_status",
        ),
        Index(
            "idx_memories_domain_predicate",
            "owner_user_id",
            "character_id",
            "domain",
            "predicate",
        ),
    )
