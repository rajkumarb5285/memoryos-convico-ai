import asyncio
import json
from typing import Any

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger


class RedisCacheService:
    """
    Manages short-term conversation context, retrieval caching, and TTL-based
    expiration with graceful offline degradation.
    """

    def __init__(self, redis_url: str | None = None):
        self._url = redis_url or settings.REDIS_URL
        self._client: redis.Redis | None = None
        self._is_available: bool = True

    async def get_client(self) -> redis.Redis | None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is not None and getattr(self, "_loop", None) != current_loop:
            self._client = None

        if self._client is None and self._is_available and current_loop is not None:
            try:
                self._loop = current_loop
                self._client = redis.from_url(
                    self._url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=settings.REDIS_MAX_CONNECTIONS,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0,
                )
                await self._client.ping()
            except Exception as e:
                logger.warning(f"Redis unavailable at {self._url}: {e}. Proceeding in degraded cache mode.")
                self._client = None
                self._is_available = False
        return self._client

    def build_context_key(self, owner_user_id: str, character_id: str, conversation_id: str) -> str:
        """Collision-safe namespaced key for short-term conversation context."""
        return f"memory_context:{owner_user_id}:{character_id}:{conversation_id}"

    def build_retrieval_key(self, owner_user_id: str, character_id: str, query: str) -> str:
        """Collision-safe namespaced key for retrieval caching."""
        sanitized_query = "".join(c for c in query.lower() if c.isalnum() or c in " _-")[:64]
        return f"retrieval_cache:{owner_user_id}:{character_id}:{sanitized_query}"

    async def get_context(self, owner_user_id: str, character_id: str, conversation_id: str) -> list[dict[str, Any]]:
        client = await self.get_client()
        if not client:
            return []
        try:
            key = self.build_context_key(owner_user_id, character_id, conversation_id)
            data = await client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to read context from Redis: {e}")
        return []

    async def set_context(
        self,
        owner_user_id: str,
        character_id: str,
        conversation_id: str,
        messages: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> None:
        client = await self.get_client()
        if not client:
            return
        try:
            key = self.build_context_key(owner_user_id, character_id, conversation_id)
            serialized = json.dumps(messages)
            await client.set(key, serialized, ex=ttl or settings.REDIS_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to write context to Redis: {e}")

    async def append_context_message(
        self,
        owner_user_id: str,
        character_id: str,
        conversation_id: str,
        role: str,
        content: str,
        limit: int = 10,
    ) -> None:
        existing = await self.get_context(owner_user_id, character_id, conversation_id)
        existing.append({"role": role, "content": content})
        if len(existing) > limit:
            existing = existing[-limit:]
        await self.set_context(owner_user_id, character_id, conversation_id, existing)

    async def invalidate_scope(self, owner_user_id: str, character_id: str, conversation_id: str | None = None) -> None:
        """Invalidate caches for a user and character after mutations."""
        client = await self.get_client()
        if not client:
            return
        try:
            if conversation_id:
                conv_key = self.build_context_key(owner_user_id, character_id, conversation_id)
                await client.delete(conv_key)

            # Invalidate cached retrieval queries for this character
            pattern = f"retrieval_cache:{owner_user_id}:{character_id}:*"
            keys = []
            async for k in client.scan_iter(match=pattern, count=100):
                keys.append(k)
            if keys:
                await client.delete(*keys)
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache: {e}")

    async def check_health(self) -> bool:
        try:
            client = await self.get_client()
            if client:
                res = await client.ping()
                return bool(res)
            return False
        except Exception:
            return False


cache_service = RedisCacheService()
