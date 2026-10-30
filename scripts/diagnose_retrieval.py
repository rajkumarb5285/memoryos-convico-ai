import asyncio
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal
from app.extraction.extractor import DeterministicMemoryExtractor
from app.lifecycle.resolver import lifecycle_resolver
from app.models.character import Character
from app.models.user import User
from app.retrieval.coordinator import retrieval_coordinator
from app.schemas.retrieval import RetrievalRequest

async def main():
    async with AsyncSessionLocal() as db:
        uid = f"diag_{int(time.time())}"
        cid = f"char_{int(time.time())}"
        user = User(id=uid, username=uid, hashed_password="pw")
        db.add(user)
        char = Character(id=cid, owner_user_id=uid, name="Diag")
        db.add(char)
        await db.commit()

        extractor = DeterministicMemoryExtractor()
        baseline_utterances = [
            "Main Indore mein rehta hoon.",
            "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai.",
            "I recently moved to Pune.",
            "I worked at Google before joining Microsoft.",
            "My sister studies at Stanford University.",
            "I will travel to Japan next year.",
            "I am now a vegetarian.",
        ]
        for utt in baseline_utterances:
            ext = await extractor.extract(utt)
            if ext.facts:
                await lifecycle_resolver.ingest_facts(
                    db=db, owner_user_id=uid, character_id=cid, conversation_id=None, source_message_id=None, facts=ext.facts
                )
        await db.commit()

        queries = [
            ("Where do I live now?", "Pune"),
            ("Where did I live before?", "Indore"),
            ("Which city did I live in previously?", "Indore"),
            ("Where is my current home?", "Pune"),
            ("What company do I work for now?", "Microsoft"),
            ("Where did I work before Microsoft?", "Google"),
            ("What beverage do I like?", "chai"),
            ("What do I think about coffee?", "coffee"),
            ("Where does my sister study?", "Stanford"),
            ("Where do I plan to travel?", "Japan"),
            ("What is my diet?", "Vegetarian"),
            ("Where does my behen study?", "Stanford"),
        ]

        print(f"Total queries: {len(queries)}")
        for q, exp in queries:
            res = await retrieval_coordinator.retrieve(
                db=db, owner_user_id=uid, req=RetrievalRequest(query=q, character_id=cid, limit=5)
            )
            print(f"\n=== QUERY: {q} (Expected: {exp}) ===")
            for idx, ev in enumerate(res.evidence, 1):
                is_rel = exp.lower() in ev.object.lower()
                print(f"  [{idx}] rel={is_rel} score={ev.relevance_score:.4f} lanes={ev.retrieval_lanes} status={ev.temporal_status} fact='{ev.fact}'")

if __name__ == "__main__":
    asyncio.run(main())
