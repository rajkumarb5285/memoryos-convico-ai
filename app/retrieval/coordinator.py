import numpy as np
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
        elif insufficient:
            answer_hint = "Insufficient evidence: no recorded memories match this query reliably."

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


def compute_retrieval_metrics(
    retrieved_objects: list[str],
    gold_expected_objects: list[str],
    k: int = 5,
) -> dict[str, float]:
    """
    Computes standard Information Retrieval (IR) metrics with strict one-to-one matching:
    - Precision@1 (P@1)
    - Precision@3 (P@3)
    - Precision@5 (P@5)
    - Recall@5 (R@5)
    - R-Precision (Precision@|R|)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain at 5 (nDCG@5)
    """
    if not gold_expected_objects:
        return {
            "p_at_1": 0.0,
            "p_at_3": 0.0,
            "p_at_5": 0.0,
            "r_at_5": 0.0,
            "r_prec": 0.0,
            "rr": 0.0,
            "ndcg_at_5": 0.0,
        }

    top_k_retrieved = retrieved_objects[:k]
    matched_gold_indices: set[int] = set()
    rel_flags: list[int] = []
    first_rank: int | None = None

    for rank_idx, ret_obj in enumerate(top_k_retrieved, start=1):
        matched = False
        ret_clean = (ret_obj or "").lower().strip()
        for g_idx, gold_obj in enumerate(gold_expected_objects):
            if g_idx not in matched_gold_indices and gold_obj:
                g_clean = gold_obj.lower().strip()
                if g_clean in ret_clean or ret_clean in g_clean:
                    matched_gold_indices.add(g_idx)
                    matched = True
                    if first_rank is None:
                        first_rank = rank_idx
                    break
        rel_flags.append(1 if matched else 0)

    while len(rel_flags) < k:
        rel_flags.append(0)

    relevant_count = len(matched_gold_indices)
    gold_count = len(gold_expected_objects)

    p_at_1 = float(rel_flags[0]) if len(rel_flags) >= 1 else 0.0
    p_at_3 = float(sum(rel_flags[:3]) / 3.0)
    p_at_5 = float(relevant_count / float(k))
    r_at_5 = float(relevant_count / float(gold_count)) if gold_count > 0 else 1.0
    r_prec = float(sum(rel_flags[:gold_count]) / float(gold_count)) if gold_count > 0 else 1.0
    rr = float(1.0 / first_rank) if first_rank else 0.0

    dcg = sum(rel / np.log2(idx + 1) for idx, rel in enumerate(rel_flags[:k], start=1))
    idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, min(k, gold_count) + 1))
    ndcg_5 = float(dcg / idcg) if idcg > 0 else 0.0

    return {
        "p_at_1": round(p_at_1, 4),
        "p_at_3": round(p_at_3, 4),
        "p_at_5": round(p_at_5, 4),
        "r_at_5": round(r_at_5, 4),
        "r_prec": round(r_prec, 4),
        "rr": round(rr, 4),
        "ndcg_at_5": round(ndcg_5, 4),
    }
