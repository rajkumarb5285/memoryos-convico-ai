from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory


class StructuredRetrievalLane:
    """Lane A: Structured exact entity, predicate, domain, and temporal status retrieval."""

    def _extract_intent_filters(self, query: str):
        low = query.lower()
        predicates = []
        domains = []
        temporal = None

        if any(
            w in low
            for w in ["live", "city", "rehta", "moved", "stay", "ghar", "home", "residence", "reside", "address"]
        ):
            predicates.extend(["lives_in", "lived_in"])
            domains.append("geography")

        if any(
            w in low
            for w in [
                "work",
                "job",
                "company",
                "employed",
                "career",
                "office",
                "kaam",
                "naukri",
                "google",
                "microsoft",
            ]
        ):
            predicates.extend(["employed_at", "worked_at", "occupation_is"])
            domains.append("work")

        if any(
            w in low
            for w in [
                "food",
                "drink",
                "coffee",
                "tea",
                "chai",
                "like",
                "dislike",
                "pasand",
                "biryani",
                "eat",
                "beverage",
                "diet",
                "vegetarian",
                "vegan",
            ]
        ):
            predicates.extend(["likes", "dislikes", "diet"])
            domains.append("food")

        if any(
            w in low
            for w in [
                "sister",
                "behen",
                "bhen",
                "brother",
                "bhai",
                "study",
                "studies",
                "padhti",
                "school",
                "college",
                "university",
                "stanford",
            ]
        ):
            predicates.extend(["studies_at", "employed_at"])
            domains.append("personal")

        if any(w in low for w in ["travel", "trip", "visit", "journey", "vacation", "japan", "ghoomne"]):
            predicates.extend(["plans_to_visit", "travel_destination"])
            domains.append("travel")

        if any(w in low for w in ["name", "naam", "who am i", "identity"]):
            predicates.append("name_is")
            domains.append("personal")

        if any(w in low for w in ["before", "previously", "pehle", "past", "history", "used to"]):
            temporal = "historical"
        elif any(w in low for w in ["now", "currently", "ab", "present", "recent"]):
            temporal = "current"

        return predicates, domains, temporal

    async def retrieve(
        self,
        db: AsyncSession,
        owner_user_id: str,
        character_id: str,
        query: str,
        temporal_filter: str | None = "all",
        domain_filter: str | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        predicates, domains, detected_temporal = self._extract_intent_filters(query)

        # Structured lane requires at least one structured constraint
        if not predicates and not domains and not domain_filter:
            return []

        conditions = [
            Memory.owner_user_id == owner_user_id,
            Memory.character_id == character_id,
            Memory.deleted_at.is_(None),
        ]

        if domain_filter:
            conditions.append(Memory.domain == domain_filter)
        elif domains:
            conditions.append(Memory.domain.in_(domains))

        if predicates:
            conditions.append(Memory.predicate.in_(predicates))

        target_temporal = temporal_filter if temporal_filter and temporal_filter != "all" else detected_temporal
        if target_temporal:
            conditions.append(Memory.temporal_status == target_temporal)

        stmt = select(Memory).where(and_(*conditions)).order_by(Memory.updated_at.desc()).limit(limit)
        res = await db.execute(stmt)
        return list(res.scalars().all())


structured_lane = StructuredRetrievalLane()
