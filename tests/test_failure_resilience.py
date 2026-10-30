import pytest
from httpx import ASGITransport, AsyncClient

from app.cache.redis_service import cache_service
from app.main import app


@pytest.mark.asyncio
async def test_unauthenticated_request_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/memory/ingest",
            json={
                "character_id": "any-id",
                "messages": [{"role": "user", "content": "I live in Pune"}],
            },
        )
        assert res.status_code == 401
        assert "Missing Authorization header" in res.json()["detail"]


@pytest.mark.asyncio
async def test_malformed_token_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/memory/ingest",
            headers={"Authorization": "Bearer not-a-valid-jwt-token"},
            json={
                "character_id": "any-id",
                "messages": [{"role": "user", "content": "I live in Pune"}],
            },
        )
        assert res.status_code == 401


@pytest.mark.asyncio
async def test_redis_graceful_degradation():
    """Verify system remains operational and does not crash when Redis is unavailable."""
    # Temporarily set Redis client to simulate an unreachable Redis host
    original_url = cache_service._url
    cache_service._url = "redis://nonexistent-host:9999"
    cache_service._client = None
    cache_service._is_available = True

    try:
        # Context calls should return empty list without throwing an uncaught exception
        ctx = await cache_service.get_context("u1", "c1", "conv1")
        assert ctx == []

        # Invalidation should silently degrade without throwing
        await cache_service.invalidate_scope("u1", "c1")

        # Health check should return False/degraded
        health = await cache_service.check_health()
        assert health is False
    finally:
        # Restore configuration
        cache_service._url = original_url
        cache_service._client = None
        cache_service._is_available = True
