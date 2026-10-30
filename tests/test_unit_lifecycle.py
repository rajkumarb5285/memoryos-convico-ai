import uuid

import pytest
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.extraction.extractor import DeterministicMemoryExtractor
from app.lifecycle.resolver import lifecycle_resolver
from app.models.character import Character
from app.models.memory import Memory
from app.models.user import User

extractor = DeterministicMemoryExtractor()


@pytest.mark.asyncio
async def test_temporal_lifecycle_mumbai_pune_mumbai():
    """
    Test Section 3.1:
    1. 'I lived in Mumbai.'
    2. 'I now live in Pune.'
    3. 'I moved back to Mumbai.'
    Expected:
    After step 2: Pune is current, Mumbai is historical.
    After step 3: Mumbai is current, Pune is historical.
    """
    user_id = f"test-seq-{uuid.uuid4().hex[:8]}"
    char_id = f"char-seq-{uuid.uuid4().hex[:8]}"

    async with AsyncSessionLocal() as db:
        u = User(id=user_id, username=user_id, hashed_password="pw")
        db.add(u)
        c = Character(id=char_id, owner_user_id=user_id, name="TestChar")
        db.add(c)
        await db.commit()

        # Step 1: "I lived in Mumbai."
        r1 = await extractor.extract("I lived in Mumbai.")
        await lifecycle_resolver.ingest_facts(
            db=db,
            owner_user_id=user_id,
            character_id=char_id,
            conversation_id=None,
            source_message_id=None,
            facts=r1.facts,
        )
        await db.commit()

        # Step 2: "I now live in Pune."
        r2 = await extractor.extract("I now live in Pune.")
        await lifecycle_resolver.ingest_facts(
            db=db,
            owner_user_id=user_id,
            character_id=char_id,
            conversation_id=None,
            source_message_id=None,
            facts=r2.facts,
        )
        await db.commit()

        # Verify after step 2: Pune is current, Mumbai is historical
        stmt2 = select(Memory).where(
            Memory.owner_user_id == user_id,
            Memory.character_id == char_id,
            Memory.deleted_at.is_(None),
        )
        mems2 = (await db.execute(stmt2)).scalars().all()
        pune_mem = next((m for m in mems2 if m.object == "Pune"), None)
        mumbai_mem = next((m for m in mems2 if m.object == "Mumbai"), None)
        assert pune_mem is not None and pune_mem.temporal_status == "current"
        assert mumbai_mem is not None and mumbai_mem.temporal_status == "historical"

        # Step 3: "I moved back to Mumbai."
        r3 = await extractor.extract("I moved back to Mumbai.")
        await lifecycle_resolver.ingest_facts(
            db=db,
            owner_user_id=user_id,
            character_id=char_id,
            conversation_id=None,
            source_message_id=None,
            facts=r3.facts,
        )
        await db.commit()

        # Verify after step 3: Mumbai is current, Pune is historical
        stmt3 = select(Memory).where(
            Memory.owner_user_id == user_id,
            Memory.character_id == char_id,
            Memory.deleted_at.is_(None),
        )
        mems3 = (await db.execute(stmt3)).scalars().all()
        current_mems = [m for m in mems3 if m.temporal_status == "current"]
        historical_mems = [m for m in mems3 if m.temporal_status == "historical"]

        assert len(current_mems) == 1
        assert current_mems[0].object == "Mumbai"
        assert any(m.object == "Pune" for m in historical_mems)


@pytest.mark.asyncio
async def test_preference_updates_lifecycle():
    """
    Test Section 3.2:
    1. 'I like coffee.'
    2. 'I do not like coffee.'
    3. 'I like coffee again.' (or 'I like coffee.')
    Expected:
    Latest preference reflects valid update with previous records preserved historically.
    """
    user_id = f"test-pref-{uuid.uuid4().hex[:8]}"
    char_id = f"char-pref-{uuid.uuid4().hex[:8]}"

    async with AsyncSessionLocal() as db:
        u = User(id=user_id, username=user_id, hashed_password="pw")
        db.add(u)
        c = Character(id=char_id, owner_user_id=user_id, name="TestChar")
        db.add(c)
        await db.commit()

        # 1. Like coffee
        r1 = await extractor.extract("I like coffee.")
        await lifecycle_resolver.ingest_facts(
            db=db,
            owner_user_id=user_id,
            character_id=char_id,
            conversation_id=None,
            source_message_id=None,
            facts=r1.facts,
        )
        await db.commit()

        # 2. Dislike coffee
        r2 = await extractor.extract("I do not like coffee.")
        await lifecycle_resolver.ingest_facts(
            db=db,
            owner_user_id=user_id,
            character_id=char_id,
            conversation_id=None,
            source_message_id=None,
            facts=r2.facts,
        )
        await db.commit()

        # 3. Like coffee again
        r3 = await extractor.extract("I like coffee.")
        await lifecycle_resolver.ingest_facts(
            db=db,
            owner_user_id=user_id,
            character_id=char_id,
            conversation_id=None,
            source_message_id=None,
            facts=r3.facts,
        )
        await db.commit()

        stmt = select(Memory).where(
            Memory.owner_user_id == user_id,
            Memory.character_id == char_id,
            Memory.deleted_at.is_(None),
        )
        mems = (await db.execute(stmt)).scalars().all()

        current_pref = [m for m in mems if m.temporal_status == "current"]
        historical_prefs = [m for m in mems if m.temporal_status == "historical"]

        assert len(current_pref) == 1
        assert current_pref[0].predicate == "likes"
        assert current_pref[0].polarity == "positive"
        assert current_pref[0].object == "coffee"
        assert len(historical_prefs) >= 2
