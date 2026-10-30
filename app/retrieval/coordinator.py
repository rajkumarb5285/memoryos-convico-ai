from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.redis_service import cache_service
from app.retrieval.dense import dense_lane
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import lexical_lane
from app.retrieval.reranker import reranker
from app.retrieval.structured import structured_lane
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse


class RetrievalCoordinator:
    """Coordinates hybrid multi-lane RAG retrieval, RRF fusion, and reranking."""

    async def _resolve_follow_up_query(
        self,
        query: str,
        owner_user_id: str,
        character_id: str,
        conversation_id: str | None,
    ) -> tuple[str, bool]:
        """
        Rewrite conversational follow-up questions using recent context.
        e.g., 'What about before?' -> 'Where did I live before?' or 'Where did I work before?'
        """
        low = query.strip().lower()
        is_follow_up = any(
            low.startswith(p)
            for p in [
                "what about",
                "and where",
                "which one",
                "and before",
                "before that",
                "aur pehle",
            ]
        )
        if not is_follow_up or not conversation_id:
            return query, False

        recent_msgs = await cache_service.get_context(owner_user_id, character_id, conversation_id)
        if not recent_msgs:
            return query, False

        # Look for the last user question or topic in recent context
        for msg in reversed(recent_msgs):
            content = msg.get("content", "").lower()
            if any(k in content for k in ["live", "city", "rehta", "indore", "pune"]):
                if "before" in low or "pehle" in low:
                    return "Where did I live before?", True
            elif any(k in content for k in ["work", "company", "google", "microsoft", "job"]):
                if "before" in low or "pehle" in low:
                    return "Where did I work before?", True

        return query, False

    async def retrieve(
        self,
        db: AsyncSession,
        owner_user_id: str,
        req: RetrievalRequest,
    ) -> RetrievalResponse:
        resolved_query, was_rewritten = await self._resolve_follow_up_query(
            req.query, owner_user_id, req.character_id, req.conversation_id
        )

        # 1. Execute all three retrieval lanes independently
        structured_results = await structured_lane.retrieve(
            db=db,
            owner_user_id=owner_user_id,
            character_id=req.character_id,
            query=resolved_query,
            temporal_filter=req.temporal_filter,
            domain_filter=req.domain_filter,
            limit=req.limit * 2,
        )

        lexical_results = await lexical_lane.retrieve(
            db=db,
            owner_user_id=owner_user_id,
            character_id=req.character_id,
            query=resolved_query,
            temporal_filter=req.temporal_filter,
            limit=req.limit * 2,
        )

        dense_results = await dense_lane.retrieve(
            db=db,
            owner_user_id=owner_user_id,
            character_id=req.character_id,
            query=resolved_query,
            temporal_filter=req.temporal_filter,
            limit=req.limit * 2,
        )

        ranked_lanes = {
            "structured": structured_results,
            "lexical": lexical_results,
            "dense": dense_results,
        }

        # 2. Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(ranked_lanes)

        # 3. Multi-factor Reranker
        evidence, insufficient, top_confidence = reranker.rerank(
            query=resolved_query,
            candidates=fused_candidates,
            limit=req.limit,
        )

        # Construct answer hint from highest ranked evidence
        answer_hint = None
        if evidence and not insufficient:
            top_ev = evidence[0]
            answer_hint = f"{top_ev.subject} {top_ev.predicate} {top_ev.object} ({top_ev.temporal_status})"

        return RetrievalResponse(
            query=req.query,
            rewritten_query=resolved_query if was_rewritten else None,
            character_id=req.character_id,
            insufficient_evidence=insufficient,
            confidence_score=top_confidence,
            results_count=len(evidence),
            evidence=evidence,
            answer_hint=answer_hint,
        )


retrieval_coordinator = RetrievalCoordinator()
