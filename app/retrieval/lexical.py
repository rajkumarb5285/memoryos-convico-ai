import re

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory


class LexicalRetrievalLane:
    """Lane B: Lexical search using PostgreSQL full-text search and keyword matching."""

    # Hinglish transliteration synonym map to assist lexical matching
    HINGLISH_SYNONYMS = {
        "behen": ["sister", "bhen", "bahan"],
        "bhen": ["sister", "behen"],
        "bhai": ["brother"],
        "chai": ["tea"],
        "pehle": ["before", "previously", "past"],
        "rehta": ["live", "lives", "lived"],
        "padhti": ["study", "studies"],
        "pasand": ["like", "likes", "preference"],
    }

    def _expand_query_terms(self, query: str) -> list[str]:
        words = re.findall(r"\b[A-Za-z0-9]+\b", query.lower())
        terms = set(words)
        for w in words:
            if w in self.HINGLISH_SYNONYMS:
                terms.update(self.HINGLISH_SYNONYMS[w])
        return [t for t in terms if len(t) > 2]

    async def retrieve(
        self,
        db: AsyncSession,
        owner_user_id: str,
        character_id: str,
        query: str,
        temporal_filter: str | None = "all",
        limit: int = 10,
    ) -> list[Memory]:
        terms = self._expand_query_terms(query)
        if not terms:
            return []

        conditions = [
            Memory.owner_user_id == owner_user_id,
            Memory.character_id == character_id,
            Memory.deleted_at.is_(None),
        ]

        if temporal_filter and temporal_filter != "all":
            conditions.append(Memory.temporal_status == temporal_filter)

        # Build term ILIKE clauses
        term_filters = []
        for term in terms[:6]:  # top terms
            pattern = f"%{term}%"
            term_filters.append(Memory.subject.ilike(pattern))
            term_filters.append(Memory.predicate.ilike(pattern))
            term_filters.append(Memory.object.ilike(pattern))
            term_filters.append(Memory.raw_text.ilike(pattern))

        conditions.append(or_(*term_filters))

        stmt = select(Memory).where(and_(*conditions)).order_by(Memory.updated_at.desc()).limit(limit)
        res = await db.execute(stmt)
        return list(res.scalars().all())


lexical_lane = LexicalRetrievalLane()
