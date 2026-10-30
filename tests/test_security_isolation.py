import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from app.cache.redis_service import cache_service
from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.main import app
from app.models.character import Character
from app.models.conversation import Conversation
from app.models.memory import Memory
from app.models.user import User


@pytest.fixture
async def setup_two_users_and_characters():
    """Create two distinct users with distinct characters for isolation tests."""
    async with AsyncSessionLocal() as session:
        uid_a = f"u_a_{uuid.uuid4().hex[:8]}"
        uid_b = f"u_b_{uuid.uuid4().hex[:8]}"
        cid_a1 = f"c_a1_{uuid.uuid4().hex[:8]}"
        cid_a2 = f"c_a2_{uuid.uuid4().hex[:8]}"
        cid_b = f"c_b_{uuid.uuid4().hex[:8]}"
        conv_id_a = f"conv_a_{uuid.uuid4().hex[:8]}"
        conv_id_b = f"conv_b_{uuid.uuid4().hex[:8]}"
        mem_id_a = f"mem_a_{uuid.uuid4().hex[:8]}"
        mem_id_b = f"mem_b_{uuid.uuid4().hex[:8]}"

        now = datetime.now(UTC)

        # User A
        user_a = User(id=uid_a, username=f"user_{uid_a}", hashed_password="hashed_a")
        session.add(user_a)
        char_a1 = Character(id=cid_a1, owner_user_id=uid_a, name="Hero A1")
        session.add(char_a1)
        char_a2 = Character(id=cid_a2, owner_user_id=uid_a, name="Hero A2")
        session.add(char_a2)

        # User B
        user_b = User(id=uid_b, username=f"user_{uid_b}", hashed_password="hashed_b")
        session.add(user_b)
        char_b = Character(id=cid_b, owner_user_id=uid_b, name="Villain B")
        session.add(char_b)

        # Conversations
        conv_a = Conversation(id=conv_id_a, owner_user_id=uid_a, character_id=cid_a1, title="Conv A")
        conv_b = Conversation(id=conv_id_b, owner_user_id=uid_b, character_id=cid_b, title="Conv B")
        session.add(conv_a)
        session.add(conv_b)

        # Memories
        mem_a = Memory(
            id=mem_id_a,
            owner_user_id=uid_a,
            character_id=cid_a1,
            conversation_id=conv_id_a,
            subject="user_a",
            predicate="lives_in",
            object="Secret Bunker A",
            valid_from=now,
            confidence=1.0,
            memory_type="location",
            domain="geography",
        )
        mem_b = Memory(
            id=mem_id_b,
            owner_user_id=uid_b,
            character_id=cid_b,
            conversation_id=conv_id_b,
            subject="user_b",
            predicate="lives_in",
            object="Secret Base B",
            valid_from=now,
            confidence=1.0,
            memory_type="location",
            domain="geography",
        )
        session.add(mem_a)
        session.add(mem_b)

        await session.commit()

        token_a = create_access_token(user_id=uid_a, username=f"user_{uid_a}")
        token_b = create_access_token(user_id=uid_b, username=f"user_{uid_b}")

        return {
            "user_a": uid_a,
            "char_a": cid_a1,
            "char_a2": cid_a2,
            "conv_a": conv_id_a,
            "mem_a": mem_id_a,
            "token_a": token_a,
            "user_b": uid_b,
            "char_b": cid_b,
            "conv_b": conv_id_b,
            "mem_b": mem_id_b,
            "token_b": token_b,
        }


@pytest.mark.asyncio
async def test_attack_1_user_a_reading_user_b_memory(setup_two_users_and_characters):
    """Attack 1: User A reading User B's memory via GET /memory/{user_id}"""
    fixture = setup_two_users_and_characters
    headers_a = {"Authorization": f"Bearer {fixture['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get(f"/memory/{fixture['user_b']}", headers=headers_a)
        # Must be rejected with HTTP 403 Forbidden!
        assert res.status_code == 403


@pytest.mark.asyncio
async def test_attack_2_user_a_deleting_user_b_memory(setup_two_users_and_characters):
    """Attack 2: User A deleting User B's memory via DELETE /memory/{memory_id}"""
    fixture = setup_two_users_and_characters
    headers_a = {"Authorization": f"Bearer {fixture['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.delete(f"/memory/{fixture['mem_b']}", headers=headers_a)
        assert res.status_code in [403, 404]


@pytest.mark.asyncio
async def test_attack_3_character_a_accessing_character_b_under_same_user(
    setup_two_users_and_characters,
):
    """Attack 3: Character A accessing Character A2's memory under the same user"""
    fixture = setup_two_users_and_characters
    headers_a = {"Authorization": f"Bearer {fixture['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/memory/retrieve",
            headers=headers_a,
            json={
                "query": "Where do I live?",
                "character_id": fixture["char_a2"],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert not any(ev["memory_id"] == fixture["mem_a"] for ev in data.get("evidence", []))


@pytest.mark.asyncio
async def test_attack_4_forged_memory_id(setup_two_users_and_characters):
    """Attack 4: A forged non-existent memory ID in a deletion request"""
    fixture = setup_two_users_and_characters
    headers_a = {"Authorization": f"Bearer {fixture['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.delete("/memory/forged-non-existent-id", headers=headers_a)
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_attack_5_cache_isolation_across_characters():
    """Attack 5: Cache hit returning another character's memory"""
    key_char1 = cache_service.build_context_key("user-1", "char-1", "conv-1")
    key_char2 = cache_service.build_context_key("user-1", "char-2", "conv-1")

    assert key_char1 != key_char2
    assert "char-1" in key_char1 and "char-2" in key_char2


@pytest.mark.asyncio
async def test_attack_6_vector_lane_isolation(setup_two_users_and_characters):
    """Attack 6: Vector result belonging to another owner must be filtered out at DB level"""
    fixture = setup_two_users_and_characters
    from app.retrieval.dense import dense_lane

    async with AsyncSessionLocal() as session:
        results = await dense_lane.retrieve(
            db=session,
            owner_user_id=fixture["user_a"],
            character_id=fixture["char_a"],
            query="Secret Base B",
            limit=10,
        )
        assert not any(m.id == fixture["mem_b"] for m in results)


@pytest.mark.asyncio
async def test_attack_7_lexical_lane_isolation(setup_two_users_and_characters):
    """Attack 7: Lexical result belonging to another owner must be filtered out at DB level"""
    fixture = setup_two_users_and_characters
    from app.retrieval.lexical import lexical_lane

    async with AsyncSessionLocal() as session:
        results = await lexical_lane.retrieve(
            db=session,
            owner_user_id=fixture["user_a"],
            character_id=fixture["char_a"],
            query="Secret Base B",
            limit=10,
        )
        assert not any(m.id == fixture["mem_b"] for m in results)


@pytest.mark.asyncio
async def test_attack_8_structured_lane_isolation(setup_two_users_and_characters):
    """Attack 8: Structured lookup with missing/different character filter"""
    fixture = setup_two_users_and_characters
    from app.retrieval.structured import structured_lane

    async with AsyncSessionLocal() as session:
        results = await structured_lane.retrieve(
            db=session,
            owner_user_id=fixture["user_a"],
            character_id=fixture["char_a2"],
            query="Where do I live?",
            limit=10,
        )
        assert not any(m.id == fixture["mem_a"] for m in results)


@pytest.mark.asyncio
async def test_attack_9_cross_scope_conversation_access(setup_two_users_and_characters):
    """Attack 9: Cross-scope access through foreign conversation IDs in ingest"""
    fixture = setup_two_users_and_characters
    headers_a = {"Authorization": f"Bearer {fixture['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/memory/ingest",
            headers=headers_a,
            json={
                "character_id": fixture["char_a"],
                "conversation_id": fixture["conv_b"],
                "messages": [{"role": "user", "content": "I live in Mumbai"}],
            },
        )
        assert res.status_code in [403, 404]


@pytest.mark.asyncio
async def test_attack_10_concurrent_deletion_safety(setup_two_users_and_characters):
    """Attack 10: Concurrent deletion requests on the same memory"""
    fixture = setup_two_users_and_characters
    headers_a = {"Authorization": f"Bearer {fixture['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task1 = client.delete(f"/memory/{fixture['mem_a']}", headers=headers_a)
        task2 = client.delete(f"/memory/{fixture['mem_a']}", headers=headers_a)

        res1, res2 = await asyncio.gather(task1, task2)
        statuses = [res1.status_code, res2.status_code]
        assert 200 in statuses
        assert 404 in statuses
