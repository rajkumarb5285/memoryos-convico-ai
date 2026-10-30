from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_principal,
    verify_character_access,
    verify_conversation_access,
)
from app.cache.redis_service import cache_service
from app.core.logging import logger
from app.core.security import Principal, verify_user_ownership
from app.db.session import get_db
from app.extraction import get_memory_extractor
from app.lifecycle.resolver import lifecycle_resolver
from app.models.character import Character
from app.models.conversation import Conversation
from app.models.memory import Memory
from app.models.message import Message
from app.retrieval.coordinator import retrieval_coordinator
from app.schemas.character import CharacterCreate, CharacterResponse
from app.schemas.conversation import ConversationCreate, ConversationResponse
from app.schemas.memory import (
    MemoryIngestRequest,
    MemoryIngestResponse,
    MemoryResponse,
    PaginatedMemoryResponse,
)
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse

router = APIRouter()
extractor = get_memory_extractor()


@router.post("/character", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_character(
    char_in: CharacterCreate,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    """Create a new character memory scope for the authenticated user."""
    char = Character(
        owner_user_id=principal.user_id,
        name=char_in.name,
        description=char_in.description,
    )
    db.add(char)
    await db.commit()
    await db.refresh(char)
    return char


@router.post(
    "/conversation",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    conv_in: ConversationCreate,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation session for an authenticated user's character."""
    await verify_character_access(conv_in.character_id, principal, db)

    conv = Conversation(
        owner_user_id=principal.user_id,
        character_id=conv_in.character_id,
        title=conv_in.title,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.post(
    "/memory/ingest",
    response_model=MemoryIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_memory(
    req: MemoryIngestRequest,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest user messages, extract knowledge facts, resolve contradictions atomically,
    and persist durable memories with provenance.
    """
    # 1. Tenancy verification: Character must belong to authenticated principal
    await verify_character_access(req.character_id, principal, db)

    # 2. Conversation validation or automatic creation
    conv_id = req.conversation_id
    if conv_id:
        await verify_conversation_access(conv_id, req.character_id, principal, db)
    else:
        # Create an implicit conversation session if none specified
        conv = Conversation(
            owner_user_id=principal.user_id,
            character_id=req.character_id,
            title="Auto Conversation",
        )
        db.add(conv)
        await db.flush()
        conv_id = conv.id

    # 3. Idempotency Check
    if req.idempotency_key:
        idemp_stmt = select(Message).where(
            and_(
                Message.owner_user_id == principal.user_id,
                Message.character_id == req.character_id,
                Message.idempotency_key == req.idempotency_key,
            )
        )
        res = await db.execute(idemp_stmt)
        if res.scalars().first():
            logger.info(f"Idempotency hit for key: {req.idempotency_key}. Returning idempotent reply.")
            return MemoryIngestResponse(
                status="success",
                conversation_id=conv_id,
                character_id=req.character_id,
                extracted_facts_count=0,
                idempotent_replay=True,
            )

    all_created_memories = []
    all_updated_memories = []
    all_historical_preserved = []
    total_facts = 0

    # 4. Process each message atomically within database transaction
    for msg_in in req.messages:
        # Persist message record
        msg = Message(
            owner_user_id=principal.user_id,
            character_id=req.character_id,
            conversation_id=conv_id,
            role=msg_in.role,
            content=msg_in.content,
            idempotency_key=msg_in.idempotency_key or req.idempotency_key,
        )
        db.add(msg)
        await db.flush()

        # Update Redis short-term sliding context
        await cache_service.append_context_message(
            owner_user_id=principal.user_id,
            character_id=req.character_id,
            conversation_id=conv_id,
            role=msg_in.role,
            content=msg_in.content,
        )

        # Extract knowledge assertions
        extraction_res = await extractor.extract(msg_in.content)
        if extraction_res.is_assertion and extraction_res.facts:
            total_facts += len(extraction_res.facts)
            created, updated, hist = await lifecycle_resolver.ingest_facts(
                db=db,
                owner_user_id=principal.user_id,
                character_id=req.character_id,
                conversation_id=conv_id,
                source_message_id=msg.id,
                facts=extraction_res.facts,
            )
            all_created_memories.extend(created)
            all_updated_memories.extend(updated)
            all_historical_preserved.extend(hist)

    # Commit atomic transaction
    await db.commit()

    # Invalidate Redis retrieval cache for this character scope
    await cache_service.invalidate_scope(principal.user_id, req.character_id, conv_id)

    # Refresh newly created records
    for m in all_created_memories:
        await db.refresh(m)
    for m in all_updated_memories:
        await db.refresh(m)

    return MemoryIngestResponse(
        status="success",
        conversation_id=conv_id,
        character_id=req.character_id,
        extracted_facts_count=total_facts,
        memories_created=[MemoryResponse.model_validate(m) for m in all_created_memories],
        memories_updated=[MemoryResponse.model_validate(m) for m in all_updated_memories],
        historical_preserved=all_historical_preserved,
    )


@router.post("/memory/retrieve", response_model=RetrievalResponse)
async def retrieve_memory(
    req: RetrievalRequest,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve contextual memories using multi-lane Hybrid RAG (Structured, Lexical, Dense pgvector),
    Reciprocal Rank Fusion, and temporal reranking.
    """
    # Tenancy check: Ensure character belongs to principal
    await verify_character_access(req.character_id, principal, db)

    # Execute Hybrid RAG Coordinator
    return await retrieval_coordinator.retrieve(
        db=db,
        owner_user_id=principal.user_id,
        req=req,
    )


@router.get("/memory/{user_id}", response_model=PaginatedMemoryResponse)
async def get_user_memories(
    user_id: str,
    character_id: str | None = Query(None, description="Filter by character scope"),
    domain: str | None = Query(None, description="Filter by domain"),
    temporal_status: str | None = Query(None, description="Filter by temporal status (current/historical/future)"),
    predicate: str | None = Query(None, description="Filter by predicate"),
    include_deleted: bool = Query(False, description="Whether to include soft-deleted records"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    """
    List durable memories for a user.
    Strictly verifies ownership: requested user_id must match authenticated principal.
    """
    # Strict IDOR prevention: Forbidden if client attempts accessing another user's path
    verify_user_ownership(principal, user_id)

    conditions = [Memory.owner_user_id == principal.user_id]

    if not include_deleted:
        conditions.append(Memory.deleted_at.is_(None))

    if character_id:
        conditions.append(Memory.character_id == character_id)
    if domain:
        conditions.append(Memory.domain == domain)
    if temporal_status:
        conditions.append(Memory.temporal_status == temporal_status)
    if predicate:
        conditions.append(Memory.predicate == predicate)

    count_stmt = select(func.count(Memory.id)).where(and_(*conditions))
    total_res = await db.execute(count_stmt)
    total = total_res.scalar() or 0

    stmt = select(Memory).where(and_(*conditions)).order_by(Memory.updated_at.desc()).offset(offset).limit(limit)
    res = await db.execute(stmt)
    items = res.scalars().all()

    return PaginatedMemoryResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[MemoryResponse.model_validate(m) for m in items],
    )


@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    """
    Soft-delete a memory record, record tombstone timestamp, and invalidate caches.
    Strictly verifies memory ownership with row-level locking for concurrency safety.
    """
    stmt = (
        select(Memory)
        .where(
            and_(
                Memory.id == memory_id,
                Memory.owner_user_id == principal.user_id,
            )
        )
        .with_for_update()
    )
    res = await db.execute(stmt)
    memory = res.scalars().first()
    if not memory or memory.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or access denied",
        )

    # Soft delete
    now = datetime.now(UTC)
    memory.deleted_at = now
    memory.updated_at = now
    await db.commit()

    # Invalidate Redis cache
    await cache_service.invalidate_scope(
        owner_user_id=principal.user_id,
        character_id=memory.character_id,
        conversation_id=memory.conversation_id,
    )

    return {"status": "deleted", "memory_id": memory_id, "deleted_at": now.isoformat()}
