from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.character import Character
    from app.models.memory import Memory
    from app.models.message import Message
    from app.models.user import User


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "conversations"

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
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    owner: Mapped["User"] = relationship("User", back_populates="conversations")
    character: Mapped["Character"] = relationship("Character", back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    memories: Mapped[list["Memory"]] = relationship("Memory", back_populates="conversation")
