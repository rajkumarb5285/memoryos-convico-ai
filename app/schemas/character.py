from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CharacterCreate(BaseModel):
    name: str
    description: str | None = None


class CharacterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_user_id: str
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime
