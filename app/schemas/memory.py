from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.message import MessageCreate


class MemoryBase(BaseModel):
    subject: str
    predicate: str
    object: str
    raw_text: str | None = None
    memory_type: str = "fact"
    domain: str = "personal"
    polarity: str = "positive"
    temporal_status: str = "current"
    confidence: float = 1.0
    valid_from: datetime
    valid_until: datetime | None = None


class MemoryCreate(MemoryBase):
    character_id: str
    conversation_id: str | None = None
    source_message_id: str | None = None
    supersedes_memory_id: str | None = None
    metadata_json: dict[str, Any] | None = None


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_user_id: str
    character_id: str
    conversation_id: str | None = None
    subject: str
    predicate: str
    object: str
    raw_text: str | None = None
    memory_type: str
    domain: str
    polarity: str
    temporal_status: str
    valid_from: datetime
    valid_until: datetime | None = None
    confidence: float
    source_message_id: str | None = None
    supersedes_memory_id: str | None = None
    deleted_at: datetime | None = None
    version: int
    created_at: datetime
    updated_at: datetime


class MemoryIngestRequest(BaseModel):
    character_id: str
    conversation_id: str | None = None
    messages: list[MessageCreate]
    idempotency_key: str | None = None


class MemoryIngestResponse(BaseModel):
    status: str = "success"
    conversation_id: str
    character_id: str
    extracted_facts_count: int
    memories_created: list[MemoryResponse] = Field(default_factory=list)
    memories_updated: list[MemoryResponse] = Field(default_factory=list)
    historical_preserved: list[str] = Field(default_factory=list)
    idempotent_replay: bool = False


class MemoryFilterParams(BaseModel):
    character_id: str | None = None
    domain: str | None = None
    temporal_status: str | None = None
    predicate: str | None = None
    include_deleted: bool = False
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class PaginatedMemoryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[MemoryResponse]
