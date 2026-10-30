from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.redis_service import cache_service
from app.db.session import get_db

router = APIRouter()


@router.get("/health")
async def health_check(response: Response, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Health check probe: tests process liveness and readiness of database and Redis.
    Never leaks sensitive credentials or internal configuration strings.
    """
    db_status = "unhealthy"
    try:
        await db.execute(text("SELECT 1;"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    redis_status = "healthy" if await cache_service.check_health() else "degraded"

    is_ready = db_status == "healthy"
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if is_ready else "degraded",
        "service": "memoryos",
        "checks": {
            "database": db_status,
            "cache": redis_status,
        },
    }
