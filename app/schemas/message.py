from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MessageCreate(BaseModel):
    role: str = "user"
    content: str
    idempotency_key: str | None = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_user_id: str
    character_id: str
    conversation_id: str
    role: str
    content: str
    idempotency_key: str | None = None
    created_at: datetime
