from datetime import UTC, datetime

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.embeddings.service import embedding_service
from app.models.memory import Memory
from app.schemas.extraction import ExtractedFact


class MemoryLifecycleResolver:
    """
    Manages the lifecycle, normalization, deduplication, contradiction resolution,
    and temporal supersession of conversational memories.
    """

    # Predicates representing singular current states that supersede previous current states
    SINGULAR_PREDICATES = {
        "lives_in": "geography",
        "employed_at": "work",
        "name_is": "personal",
        "diet": "food",
        "occupation_is": "work",
    }

    def normalize_fact(self, fact: ExtractedFact) -> ExtractedFact:
        """Normalize entities, subjects, and predicates."""
        norm_subj = fact.subject.strip().lower()
        norm_pred = fact.predicate.strip().lower()
        norm_obj = fact.object.strip()

        # Canonicalize predicate aliases
        if norm_pred in ["live_in", "reside_in", "resides_in", "lived_in"]:
            norm_pred = "lives_in"
        elif norm_pred in ["work_at", "works_at", "employed_by", "worked_at"]:
            norm_pred = "employed_at"
        elif norm_pred in ["like", "loves", "enjoy"]:
            norm_pred = "likes"
        elif norm_pred in ["dislike", "hate", "hates"]:
            norm_pred = "dislikes"

        return ExtractedFact(
            subject=norm_subj,
            predicate=norm_pred,
            object=norm_obj,
            raw_text=fact.raw_text,
            memory_type=fact.memory_type,
            domain=fact.domain,
            polarity=fact.polarity.lower(),
            temporal_status=fact.temporal_status.lower(),
            confidence=fact.confidence,
            valid_from=fact.valid_from or datetime.now(UTC),
            valid_until=fact.valid_until,
        )

    async def ingest_facts(
        self,
        db: AsyncSession,
        owner_user_id: str,
        character_id: str,
        conversation_id: str | None,
        source_message_id: str | None,
        facts: list[ExtractedFact],
    ) -> tuple[list[Memory], list[Memory], list[str]]:
        """
        Process and atomically persist extracted facts with contradiction resolution
        and temporal supersession.

        Returns:
            Tuple of (created_memories, updated_memories, historical_preserved_ids)
        """
        now = datetime.now(UTC)
        created_memories: list[Memory] = []
        updated_memories: list[Memory] = []
        historical_preserved: list[str] = []

        for raw_fact in facts:
            fact = self.normalize_fact(raw_fact)

            # 1. Exact Duplicate Check (same subject, predicate, object case-insensitively, polarity, scope, not deleted)
            dup_stmt = select(Memory).where(
                and_(
                    Memory.owner_user_id == owner_user_id,
                    Memory.character_id == character_id,
                    Memory.subject == fact.subject,
                    Memory.predicate == fact.predicate,
                    func.lower(Memory.object) == func.lower(fact.object),
                    Memory.polarity == fact.polarity,
                    Memory.temporal_status == fact.temporal_status,
                    Memory.deleted_at.is_(None),
                )
            )
            dup_res = await db.execute(dup_stmt)
            existing_dup = dup_res.scalars().first()
            if existing_dup:
                logger.info(f"Duplicate fact detected for memory ID {existing_dup.id}. Skipping duplicate creation.")
                continue

            # 2. Contradiction & Supersession Check
            # Check A: Direct Polarity Conflict on Preferences (e.g. "likes coffee" vs "dislikes coffee")
            if fact.predicate in ["likes", "dislikes"]:
                opposite_pred = "dislikes" if fact.predicate == "likes" else "likes"
                conflict_stmt = select(Memory).where(
                    and_(
                        Memory.owner_user_id == owner_user_id,
                        Memory.character_id == character_id,
                        Memory.subject == fact.subject,
                        Memory.predicate == opposite_pred,
                        func.lower(Memory.object) == func.lower(fact.object),
                        Memory.temporal_status == "current",
                        Memory.deleted_at.is_(None),
                    )
                )
                conflict_res = await db.execute(conflict_stmt)
                conflicting_pref = conflict_res.scalars().first()
                if conflicting_pref:
                    # Supersede older opposite preference
                    conflicting_pref.temporal_status = "historical"
                    conflicting_pref.valid_until = now
                    conflicting_pref.updated_at = now
                    updated_memories.append(conflicting_pref)
                    historical_preserved.append(conflicting_pref.id)

                    # Build textual representation for embedding
                    embed_text = f"{fact.subject} {fact.predicate} {fact.object}"
                    vector = embedding_service.embed_text(embed_text)

                    new_mem = Memory(
                        owner_user_id=owner_user_id,
                        character_id=character_id,
                        conversation_id=conversation_id,
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        raw_text=fact.raw_text,
                        memory_type=fact.memory_type,
                        domain=fact.domain,
                        polarity=fact.polarity,
                        temporal_status="current",
                        valid_from=now,
                        valid_until=None,
                        confidence=fact.confidence,
                        source_message_id=source_message_id,
                        supersedes_memory_id=conflicting_pref.id,
                        embedding=vector,
                        version=conflicting_pref.version + 1,
                    )
                    db.add(new_mem)
                    created_memories.append(new_mem)
                    continue

            # Check B: Singular Predicates (e.g., lives_in, employed_at)
            # If a new "current" fact arrives for a singular predicate, supersede any previous "current" fact
            if fact.predicate in self.SINGULAR_PREDICATES and fact.temporal_status == "current":
                prior_stmt = select(Memory).where(
                    and_(
                        Memory.owner_user_id == owner_user_id,
                        Memory.character_id == character_id,
                        Memory.subject == fact.subject,
                        Memory.predicate == fact.predicate,
                        Memory.temporal_status == "current",
                        Memory.deleted_at.is_(None),
                    )
                )
                prior_res = await db.execute(prior_stmt)
                prior_memories = prior_res.scalars().all()

                superseded_id = None
                for prior_mem in prior_memories:
                    if prior_mem.object.lower() != fact.object.lower():
                        prior_mem.temporal_status = "historical"
                        prior_mem.valid_until = now
                        prior_mem.updated_at = now
                        updated_memories.append(prior_mem)
                        historical_preserved.append(prior_mem.id)
                        superseded_id = prior_mem.id
                        logger.info(
                            f"Superseded memory {prior_mem.id} ({prior_mem.predicate}: {prior_mem.object}) with new fact ({fact.object})"
                        )

                embed_text = f"{fact.subject} {fact.predicate} {fact.object}"
                vector = embedding_service.embed_text(embed_text)

                new_mem = Memory(
                    owner_user_id=owner_user_id,
                    character_id=character_id,
                    conversation_id=conversation_id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object=fact.object,
                    raw_text=fact.raw_text,
                    memory_type=fact.memory_type,
                    domain=fact.domain,
                    polarity=fact.polarity,
                    temporal_status=fact.temporal_status,
                    valid_from=fact.valid_from or now,
                    valid_until=fact.valid_until,
                    confidence=fact.confidence,
                    source_message_id=source_message_id,
                    supersedes_memory_id=superseded_id,
                    embedding=vector,
                    version=1,
                )
                db.add(new_mem)
                created_memories.append(new_mem)
                continue

            # Check C: Standard Ingestion (e.g. historical facts or multi-valued facts)
            embed_text = f"{fact.subject} {fact.predicate} {fact.object}"
            vector = embedding_service.embed_text(embed_text)

            new_mem = Memory(
                owner_user_id=owner_user_id,
                character_id=character_id,
                conversation_id=conversation_id,
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                raw_text=fact.raw_text,
                memory_type=fact.memory_type,
                domain=fact.domain,
                polarity=fact.polarity,
                temporal_status=fact.temporal_status,
                valid_from=fact.valid_from or now,
                valid_until=fact.valid_until or (now if fact.temporal_status == "historical" else None),
                confidence=fact.confidence,
                source_message_id=source_message_id,
                supersedes_memory_id=None,
                embedding=vector,
                version=1,
            )
            db.add(new_mem)
            created_memories.append(new_mem)

        await db.flush()
        return created_memories, updated_memories, historical_preserved


lifecycle_resolver = MemoryLifecycleResolver()
