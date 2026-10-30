from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.service import embedding_service
from app.models.memory import Memory


class DenseSemanticRetrievalLane:
    """Lane C: Dense vector semantic search using pgvector cosine similarity."""

    async def retrieve(
        self,
        db: AsyncSession,
        owner_user_id: str,
        character_id: str,
        query: str,
        temporal_filter: str | None = "all",
        limit: int = 10,
    ) -> list[Memory]:
        query_vector = embedding_service.embed_text(query)

        conditions = [
            Memory.owner_user_id == owner_user_id,
            Memory.character_id == character_id,
            Memory.deleted_at.is_(None),
            Memory.embedding.is_not(None),
        ]

        if temporal_filter and temporal_filter != "all":
            conditions.append(Memory.temporal_status == temporal_filter)

        # In pgvector, cosine_distance operator is <=>
        stmt = (
            select(Memory)
            .where(and_(*conditions))
            .order_by(Memory.embedding.cosine_distance(query_vector))
            .limit(limit)
        )

        res = await db.execute(stmt)
        return list(res.scalars().all())


dense_lane = DenseSemanticRetrievalLane()
