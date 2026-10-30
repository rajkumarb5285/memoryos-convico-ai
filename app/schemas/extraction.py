from datetime import datetime

from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    subject: str = Field(..., description="Subject of the asserted fact (e.g., 'user', 'sister')")
    predicate: str = Field(
        ...,
        description="Relation/predicate (e.g., 'likes', 'dislikes', 'lives_in', 'employed_at')",
    )
    object: str = Field(
        ...,
        description="Value/entity of the predicate (e.g., 'chai', 'Pune', 'Google')",
    )
    raw_text: str | None = Field(default=None, description="The original textual evidence span")
    memory_type: str = Field(
        default="fact",
        description="Category: 'preference', 'location', 'employment', 'identity', 'plan', 'activity'",
    )
    domain: str = Field(
        default="personal",
        description="Domain: 'food', 'work', 'geography', 'personal', 'activity', 'education'",
    )
    polarity: str = Field(default="positive", description="'positive', 'negative', or 'neutral'")
    temporal_status: str = Field(default="current", description="'current', 'historical', or 'future'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    valid_from: datetime | None = None
    valid_until: datetime | None = None


class ExtractionResult(BaseModel):
    is_assertion: bool = Field(
        default=True,
        description="True if the message contains assertive knowledge, False if question/noise",
    )
    language: str = Field(default="en", description="Detected language ('en', 'hinglish', etc.)")
    facts: list[ExtractedFact] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    raw_message: str
