from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationCreate(BaseModel):
    character_id: str
    title: str | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_user_id: str
    character_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
