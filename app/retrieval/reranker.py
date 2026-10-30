import re

import numpy as np

from app.core.config import settings
from app.embeddings.service import embedding_service
from app.retrieval.fusion import FusionCandidate
from app.schemas.retrieval import RetrievedEvidence


class DeterministicReranker:
    """
    Reranks RRF candidates using multi-factor normalized scoring,
    real dense cosine similarity, explicit temporal alignment,
    and hallucination suppression.
    """

    def _detect_query_temporal_intent(self, query: str) -> str:
        low = query.lower()
        if any(
            w in low
            for w in [
                "before",
                "previously",
                "pehle",
                "used to",
                "past",
                "earlier",
                "history",
            ]
        ):
            return "historical"
        if any(w in low for w in ["now", "currently", "present", "ab", "today", "recent"]):
            return "current"
        return "neutral"

    def _compute_keyword_overlap(self, query: str, cand: FusionCandidate) -> float:
        q_words = set(re.findall(r"\b[A-Za-z0-9]+\b", query.lower()))
        # Remove common question filler words
        stopwords = {
            "what",
            "where",
            "which",
            "when",
            "how",
            "who",
            "did",
            "do",
            "does",
            "is",
            "are",
            "my",
            "i",
            "the",
            "a",
            "an",
            "in",
            "at",
            "about",
        }
        q_content = q_words - stopwords
        if not q_content:
            q_content = q_words

        m = cand.memory
        mem_text = f"{m.subject} {m.predicate} {m.object} {m.raw_text or ''}".lower().replace("_", " ")
        m_words = set(re.findall(r"\b[A-Za-z0-9]+\b", mem_text))

        if not q_content or not m_words:
            return 0.0

        # Root/stem match for content words with minimum length safeguard
        overlap_count = 0
        for qw in q_content:
            if any(
                qw == mw or (len(qw) >= 4 and len(mw) >= 4 and (qw.startswith(mw) or mw.startswith(qw)))
                for mw in m_words
            ):
                overlap_count += 1

        return overlap_count / len(q_content)

    def _compute_cosine_sim(self, query_vec: list[float], mem_embedding: list[float] | None) -> float:
        if not mem_embedding or not query_vec:
            return 0.0
        try:
            # Both vectors are unit-normalized
            v1 = np.array(query_vec, dtype=np.float32)
            v2 = np.array(mem_embedding, dtype=np.float32)
            sim = float(np.dot(v1, v2))
            return max(0.0, min(1.0, sim))
        except Exception:
            return 0.0

    def rerank(
        self,
        query: str,
        candidates: list[FusionCandidate],
        limit: int = 5,
    ) -> tuple[list[RetrievedEvidence], bool, float]:
        if not candidates:
            return [], True, 0.0

        temporal_intent = self._detect_query_temporal_intent(query)
        query_vec = embedding_service.embed_text(query)

        scored_evidence: list[tuple[float, RetrievedEvidence]] = []

        for cand in candidates:
            mem = cand.memory

            # 1. Real Dense Cosine Similarity
            semantic_sim = self._compute_cosine_sim(query_vec, mem.embedding)

            # 2. Keyword overlap
            overlap = self._compute_keyword_overlap(query, cand)

            # 3. Multi-lane presence reward
            # Candidates present in multiple lanes (e.g. structured AND dense) are much more confident
            lane_ratio = len(cand.lanes) / 3.0

            # 4. Temporal alignment feature
            if temporal_intent == "historical":
                temporal_score = 1.0 if mem.temporal_status == "historical" else 0.2
            elif temporal_intent == "current":
                temporal_score = 1.0 if mem.temporal_status == "current" else 0.25
            else:
                temporal_score = 0.85 if mem.temporal_status == "current" else 0.70

            # 5. Extraction confidence
            conf = float(mem.confidence)

            # Combined multi-factor score: weighted primarily by true semantic similarity and keyword overlap
            final_score = 0.40 * semantic_sim + 0.25 * overlap + 0.15 * lane_ratio + 0.10 * temporal_score + 0.10 * conf

            # If candidate only appeared in dense lane with zero keyword overlap and semantic similarity < 0.65,
            # it is an uncorroborated nearest neighbor in a sparse vector space — penalize heavily
            if overlap == 0.0 and cand.lanes == {"dense"} and semantic_sim < 0.65:
                final_score *= 0.30
            elif overlap == 0.0 and semantic_sim < 0.50:
                final_score *= 0.35

            fact_str = f"{mem.subject} {mem.predicate} {mem.object}"
            ev = RetrievedEvidence(
                memory_id=mem.id,
                fact=fact_str,
                subject=mem.subject,
                predicate=mem.predicate,
                object=mem.object,
                memory_type=mem.memory_type,
                domain=mem.domain,
                temporal_status=mem.temporal_status,
                valid_from=mem.valid_from,
                valid_until=mem.valid_until,
                confidence=mem.confidence,
                relevance_score=round(final_score, 4),
                retrieval_lanes=sorted(list(cand.lanes)),
                source_message_id=mem.source_message_id,
                raw_text=mem.raw_text,
            )
            scored_evidence.append((final_score, ev))

        # Sort by final score descending
        scored_evidence.sort(key=lambda x: x[0], reverse=True)
        top_items = [ev for _, ev in scored_evidence[:limit]]
        top_score = scored_evidence[0][0] if scored_evidence else 0.0

        # Check threshold for insufficient evidence
        insufficient = top_score < settings.RETRIEVAL_CONFIDENCE_THRESHOLD or len(top_items) == 0

        return top_items, insufficient, round(top_score, 4)


reranker = DeterministicReranker()
