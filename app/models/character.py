from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.memory import Memory
    from app.models.user import User


class Character(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "characters"

    owner_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    owner: Mapped["User"] = relationship("User", back_populates="characters")
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="character", cascade="all, delete-orphan"
    )
    memories: Mapped[list["Memory"]] = relationship("Memory", back_populates="character", cascade="all, delete-orphan")
