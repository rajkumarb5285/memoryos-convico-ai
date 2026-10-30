from datetime import datetime

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural language retrieval query (English or Hinglish)",
    )
    character_id: str = Field(..., description="Character scope to isolate memory")
    conversation_id: str | None = Field(None, description="Optional conversation context for follow-up rewriting")
    temporal_filter: str | None = Field("all", description="'current', 'historical', or 'all'")
    domain_filter: str | None = Field(None, description="Domain constraint if known")
    limit: int = Field(5, ge=1, le=50, description="Maximum number of evidence candidates")


class RetrievedEvidence(BaseModel):
    memory_id: str
    fact: str
    subject: str
    predicate: str
    object: str
    memory_type: str
    domain: str
    temporal_status: str
    valid_from: datetime
    valid_until: datetime | None = None
    confidence: float
    relevance_score: float
    retrieval_lanes: list[str]
    source_message_id: str | None = None
    raw_text: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    rewritten_query: str | None = None
    character_id: str
    insufficient_evidence: bool = False
    confidence_score: float
    results_count: int
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    answer_hint: str | None = None
