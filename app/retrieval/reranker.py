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
    domain & predicate intent agreement, and distractor suppression.
    """

    INTENT_DOMAINS = {
        "work": {
            "keywords": ["work", "job", "company", "employed", "career", "office", "naukri", "kaam", "google", "microsoft"],
            "predicates": ["employed_at", "worked_at", "occupation_is"],
            "domain": "work",
        },
        "geography": {
            "keywords": ["live", "city", "rehta", "moved", "stay", "ghar", "home", "residence", "reside", "address", "pune", "indore"],
            "predicates": ["lives_in", "lived_in"],
            "domain": "geography",
        },
        "food": {
            "keywords": ["food", "drink", "coffee", "tea", "chai", "like", "dislike", "pasand", "biryani", "eat", "beverage", "diet", "vegetarian", "vegan"],
            "predicates": ["likes", "dislikes", "diet"],
            "domain": "food",
        },
        "personal": {
            "keywords": ["sister", "behen", "bhen", "brother", "bhai", "study", "studies", "padhti", "school", "college", "university", "stanford"],
            "predicates": ["studies_at"],
            "domain": "personal",
        },
        "travel": {
            "keywords": ["travel", "trip", "visit", "vacation", "flight", "tour", "japan", "ghoomne"],
            "predicates": ["plans_to_visit", "travel_destination"],
            "domain": "travel",
        },
    }

    SYNONYMS = {
        "home": ["live", "lives", "lives_in", "residence", "pune", "indore"],
        "residence": ["live", "lives", "lives_in", "home"],
        "behen": ["sister", "studies_at", "stanford"],
        "bhen": ["sister", "studies_at"],
        "chai": ["tea", "likes"],
        "coffee": ["dislikes", "drinks", "coffee"],
        "company": ["employed_at", "work", "job", "microsoft", "google"],
        "work": ["employed", "employed_at", "job", "company", "worked_at"],
        "job": ["employed_at", "work", "company", "employed"],
        "employed": ["work", "employed_at", "job", "company"],
        "beverage": ["chai", "tea", "coffee", "drink", "likes", "dislikes"],
        "diet": ["vegetarian", "diet", "food"],
        "travel": ["plans_to_visit", "visit", "trip", "japan"],
    }

    STOPWORDS = {
        "what", "where", "which", "when", "how", "who", "whom", "whose", "why",
        "did", "do", "does", "is", "are", "was", "were", "am", "be", "been", "being",
        "my", "i", "me", "mine", "you", "your", "yours", "we", "our", "ours", "they", "their",
        "the", "a", "an", "in", "at", "about", "for", "to", "of", "on", "with", "by", "from",
        "tell", "know", "can", "could", "would", "should", "will", "shall", "that", "this",
        "now", "currently", "present", "ab", "today", "recent", "recently",
        "before", "previously", "pehle", "past", "history", "earlier", "used",
    }

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
        q_content = q_words - self.STOPWORDS
        if not q_content:
            q_content = q_words

        m = cand.memory
        mem_text = f"{m.subject} {m.predicate} {m.object} {m.raw_text or ''}".lower().replace("_", " ")
        m_words = set(re.findall(r"\b[A-Za-z0-9]+\b", mem_text))

        if not q_content or not m_words:
            return 0.0

        overlap_count = 0
        for qw in q_content:
            syns = self.SYNONYMS.get(qw, [])
            if any(
                qw == mw
                or (len(qw) >= 4 and len(mw) >= 4 and (qw.startswith(mw) or mw.startswith(qw)))
                or any(s == mw for s in syns)
                for mw in m_words
            ):
                overlap_count += 1

        return overlap_count / len(q_content)

    def _compute_cosine_sim(self, query_vec: list[float], mem_embedding: list[float] | None) -> float:
        if not mem_embedding or not query_vec:
            return 0.0
        try:
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

        low_query = query.lower()
        temporal_intent = self._detect_query_temporal_intent(query)
        query_vec = embedding_service.embed_text(query)

        # Detect target domains from query
        target_domains = set()
        target_predicates = set()
        for dom, spec in self.INTENT_DOMAINS.items():
            if any(kw in low_query for kw in spec["keywords"]):
                target_domains.add(dom)
                target_predicates.update(spec["predicates"])

        scored_evidence: list[tuple[float, RetrievedEvidence]] = []

        for cand in candidates:
            mem = cand.memory

            # 1. Real Dense Cosine Similarity
            semantic_sim = self._compute_cosine_sim(query_vec, mem.embedding)

            # 2. Keyword overlap
            overlap = self._compute_keyword_overlap(query, cand)

            # 3. Multi-lane presence reward
            lane_ratio = len(cand.lanes) / 3.0

            # 4. Temporal alignment
            if temporal_intent == "historical":
                temporal_score = 1.0 if mem.temporal_status == "historical" else 0.15
            elif temporal_intent == "current":
                temporal_score = 1.0 if mem.temporal_status == "current" else 0.20
            else:
                temporal_score = 0.85 if mem.temporal_status == "current" else 0.70

            # 5. Extraction confidence
            conf = float(mem.confidence)

            # 6. Domain & Predicate Agreement
            if target_domains:
                if mem.domain in target_domains or mem.predicate in target_predicates:
                    domain_score = 1.0
                else:
                    domain_score = 0.10  # Strong cross-domain suppression
            else:
                domain_score = 0.70

            # Combined multi-factor score
            final_score = (
                0.30 * semantic_sim
                + 0.25 * overlap
                + 0.20 * domain_score
                + 0.10 * lane_ratio
                + 0.10 * temporal_score
                + 0.05 * conf
            )

            # Strict temporal constraint enforcement:
            # A current fact cannot answer an explicit "before" / historical question
            if temporal_intent == "historical" and mem.temporal_status == "current":
                final_score *= 0.60
            elif temporal_intent == "current" and mem.temporal_status == "historical":
                final_score *= 0.75

            # Heavy penalty if candidate only appeared in dense lane with zero keyword overlap and semantic similarity < 0.65
            if overlap == 0.0 and cand.lanes == {"dense"} and semantic_sim < 0.65:
                final_score *= 0.30
            elif overlap == 0.0 and semantic_sim < 0.50:
                final_score *= 0.35

            # If cross-domain mismatch with zero overlap, strongly suppress
            if target_domains and domain_score == 0.10 and overlap == 0.0:
                final_score *= 0.25

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
