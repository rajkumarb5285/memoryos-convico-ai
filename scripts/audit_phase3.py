import asyncio
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.models.character import Character
from app.extraction.extractor import DeterministicMemoryExtractor
from app.lifecycle.resolver import lifecycle_resolver
import uuid

async def audit_phase3():
    ext = DeterministicMemoryExtractor()
    async with AsyncSessionLocal() as db:
        user_id = f"audit-{uuid.uuid4().hex[:8]}"
        u = User(id=user_id, username=f"u_{uuid.uuid4().hex[:6]}", email=f"{user_id}@test.com", hashed_password="pw")
        db.add(u)
        await db.flush()

        char_id = f"char-{uuid.uuid4().hex[:8]}"
        c = Character(id=char_id, owner_user_id=user_id, name="Char")
        db.add(c)
        await db.flush()

        seq = [
            "I lived in Mumbai.",
            "I now live in Pune.",
            "I moved back to Mumbai."
        ]

        for s in seq:
            res = await ext.extract(s)
            print(f"Step '{s}': {len(res.facts)} facts")
            for f in res.facts:
                print(f"  extracted: {f.predicate}={f.object}, temp={f.temporal_status}")
            created, updated, hist = await lifecycle_resolver.ingest_facts(
                db=db,
                owner_user_id=user_id,
                character_id=char_id,
                conversation_id=None,
                source_message_id=None,
                facts=res.facts
            )
            await db.flush()

        # Check DB state
        from sqlalchemy import select
        from app.models.memory import Memory
        stmt = select(Memory).where(Memory.owner_user_id == user_id, Memory.deleted_at.is_(None))
        mems = (await db.execute(stmt)).scalars().all()
        print("\nFinal memories in DB:")
        for m in mems:
            print(f"  Memory {m.predicate}={m.object}: status={m.temporal_status}, valid_from={m.valid_from}, valid_until={m.valid_until}, supersedes={m.supersedes_memory_id}")

        # Test 3.2: Preference updates
        pref_seq = [
            "I like coffee.",
            "I do not like coffee.",
            "I like coffee."
        ]
        print("\nTesting preference updates:")
        for p in pref_seq:
            res = await ext.extract(p)
            print(f"Step '{p}': {len(res.facts)} facts")
            for f in res.facts:
                print(f"  extracted: {f.predicate}={f.object}, pol={f.polarity}")
            await lifecycle_resolver.ingest_facts(
                db=db,
                owner_user_id=user_id,
                character_id=char_id,
                conversation_id=None,
                source_message_id=None,
                facts=res.facts
            )
            await db.flush()

        stmt = select(Memory).where(Memory.owner_user_id == user_id, Memory.deleted_at.is_(None), Memory.predicate.in_(["likes", "dislikes"]))
        mems = (await db.execute(stmt)).scalars().all()
        print("\nFinal preferences in DB:")
        for m in mems:
            print(f"  Memory {m.predicate}={m.object}: pol={m.polarity}, status={m.temporal_status}, valid_from={m.valid_from}, valid_until={m.valid_until}")

if __name__ == "__main__":
    asyncio.run(audit_phase3())
