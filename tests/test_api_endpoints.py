import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def authenticated_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Register user
        username = f"user_{uuid.uuid4().hex[:8]}"
        password = "strong_password_123"
        reg_res = await client.post(
            "/auth/register",
            json={"username": username, "password": password},
        )
        assert reg_res.status_code == 201
        user_data = reg_res.json()
        user_id = user_data["id"]

        # Login to get token
        token_res = await client.post(
            "/auth/token",
            json={"username": username, "password": password},
        )
        assert token_res.status_code == 200
        token = token_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create character
        char_res = await client.post(
            "/character",
            headers=headers,
            json={"name": "Test Character", "description": "Character for API test"},
        )
        assert char_res.status_code == 201
        char_id = char_res.json()["id"]

        yield {
            "client": client,
            "user_id": user_id,
            "headers": headers,
            "char_id": char_id,
        }


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["ok", "degraded"]
        assert data["service"] == "memoryos"
        assert "checks" in data
        assert data["checks"]["database"] == "healthy"


@pytest.mark.asyncio
async def test_full_memory_lifecycle_via_api(authenticated_client):
    client = authenticated_client["client"]
    headers = authenticated_client["headers"]
    char_id = authenticated_client["char_id"]
    user_id = authenticated_client["user_id"]

    # 1. Ingest Day 1 Location: "Main Indore mein rehta hoon."
    ingest1 = await client.post(
        "/memory/ingest",
        headers=headers,
        json={
            "character_id": char_id,
            "messages": [{"role": "user", "content": "Main Indore mein rehta hoon."}],
        },
    )
    assert ingest1.status_code == 201
    data1 = ingest1.json()
    assert data1["extracted_facts_count"] >= 1
    assert len(data1["memories_created"]) >= 1
    mem_indore = data1["memories_created"][0]
    assert mem_indore["object"] == "Indore"
    assert mem_indore["temporal_status"] == "current"

    # 2. Ingest Day 2 Preference: "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai."
    ingest2 = await client.post(
        "/memory/ingest",
        headers=headers,
        json={
            "character_id": char_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai.",
                }
            ],
        },
    )
    assert ingest2.status_code == 201
    data2 = ingest2.json()
    assert data2["extracted_facts_count"] >= 2

    # 3. Ingest Day 3 Location update: "I recently moved to Pune."
    # Expect: Indore becomes historical, Pune becomes current!
    ingest3 = await client.post(
        "/memory/ingest",
        headers=headers,
        json={
            "character_id": char_id,
            "messages": [{"role": "user", "content": "I recently moved to Pune."}],
        },
    )
    assert ingest3.status_code == 201
    data3 = ingest3.json()
    assert len(data3["memories_created"]) >= 1
    assert mem_indore["id"] in data3["historical_preserved"]

    # 4. Ingest Question: "Do you know someone at Google?"
    # Expect: 0 memories created (assertions only, not questions!)
    ingest_q = await client.post(
        "/memory/ingest",
        headers=headers,
        json={
            "character_id": char_id,
            "messages": [{"role": "user", "content": "Do you know someone at Google?"}],
        },
    )
    assert ingest_q.status_code == 201
    assert ingest_q.json()["extracted_facts_count"] == 0

    # 5. Retrieve Current Location: "Where do I live now?"
    ret_curr = await client.post(
        "/memory/retrieve",
        headers=headers,
        json={
            "query": "Where do I live now?",
            "character_id": char_id,
        },
    )
    assert ret_curr.status_code == 200
    ret_curr_data = ret_curr.json()
    assert not ret_curr_data["insufficient_evidence"]
    assert ret_curr_data["evidence"][0]["object"] == "Pune"
    assert ret_curr_data["evidence"][0]["temporal_status"] == "current"

    # 6. Retrieve Historical Location: "Where did I live before?"
    ret_hist = await client.post(
        "/memory/retrieve",
        headers=headers,
        json={
            "query": "Where did I live before?",
            "character_id": char_id,
        },
    )
    assert ret_hist.status_code == 200
    ret_hist_data = ret_hist.json()
    assert not ret_hist_data["insufficient_evidence"]
    assert ret_hist_data["evidence"][0]["object"] == "Indore"
    assert ret_hist_data["evidence"][0]["temporal_status"] == "historical"

    # 7. Retrieve Insufficient Evidence Query
    ret_missing = await client.post(
        "/memory/retrieve",
        headers=headers,
        json={
            "query": "What is my favorite spaceship brand?",
            "character_id": char_id,
        },
    )
    assert ret_missing.status_code == 200
    assert ret_missing.json()["insufficient_evidence"] is True

    # 8. List Memories via GET /memory/{user_id}
    list_res = await client.get(
        f"/memory/{user_id}?character_id={char_id}",
        headers=headers,
    )
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 3

    # 9. Soft-delete a memory via DELETE /memory/{memory_id}
    del_target_id = mem_indore["id"]
    del_res = await client.delete(
        f"/memory/{del_target_id}",
        headers=headers,
    )
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify deleted memory is excluded by default from GET /memory/{user_id}
    list_after_del = await client.get(
        f"/memory/{user_id}?character_id={char_id}",
        headers=headers,
    )
    assert not any(m["id"] == del_target_id for m in list_after_del.json()["items"])


@pytest.mark.asyncio
async def test_idempotent_ingest(authenticated_client):
    client = authenticated_client["client"]
    headers = authenticated_client["headers"]
    char_id = authenticated_client["char_id"]

    idemp_key = f"key_{uuid.uuid4().hex}"
    payload = {
        "character_id": char_id,
        "idempotency_key": idemp_key,
        "messages": [{"role": "user", "content": "I like dark chocolate"}],
    }

    # First ingest
    res1 = await client.post("/memory/ingest", headers=headers, json=payload)
    assert res1.status_code == 201
    assert res1.json()["idempotent_replay"] is False

    # Second ingest with same idempotency key
    res2 = await client.post("/memory/ingest", headers=headers, json=payload)
    assert res2.status_code == 201
    assert res2.json()["idempotent_replay"] is True
