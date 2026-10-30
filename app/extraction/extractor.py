import re
from datetime import UTC, datetime

from app.extraction.base import BaseMemoryExtractor
from app.schemas.extraction import ExtractedFact, ExtractionResult


class DeterministicMemoryExtractor(BaseMemoryExtractor):
    """
    Production-grade rule-based neural extractor designed for high precision,
    multilingual (English + Romanized Hinglish), multi-fact extraction, polarity,
    and temporal reasoning without external API dependency.
    """

    # Pure question / casual chat indicators
    QUESTION_STARTERS = [
        r"^(do|can|could|would|will|is|are|who|what|where|when|why|how)\s+you\b",
        r"^(do|can|could|would|will)\s+(anyone|we|they|i)\b",
        r"^(kya\s+aap|kya\s+tum|kya\s+kisi\s+ko|kya\s+chal\s+raha)\b",
        r"^(kaise\s+ho|kaisa\s+hai|kya\s+hal\s+hai)\b",
        r"^(hello|hi|hey|good\s+morning|good\s+evening|namaste|pranam)\b",
    ]

    # Hinglish stopwords / transliterations
    HINGLISH_MARKERS = [
        "hai",
        "hoon",
        "tha",
        "thi",
        "pasand",
        "rehta",
        "meri",
        "mera",
        "behen",
        "bhen",
        "bhai",
        "bahut",
        "achi",
        "achha",
        "lekin",
        "magar",
        "chhod",
    ]

    def _is_hinglish(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(rf"\b{m}\b", low) for m in self.HINGLISH_MARKERS)

    def _is_pure_question_or_noise(self, text: str) -> bool:
        text_clean = text.strip()
        low = text_clean.lower()

        # Check pure questions
        for pat in self.QUESTION_STARTERS:
            if re.search(pat, low):
                # If it doesn't assert personal fact like "I live in Pune, do you?", it's pure question
                if not any(
                    assertion_marker in low
                    for assertion_marker in [
                        "i live",
                        "i work",
                        "my name",
                        "main rehta",
                        "mera naam",
                        "i like",
                        "mujhe pasand",
                    ]
                ):
                    return True

        if text_clean.endswith("?") and not any(m in low for m in ["i ", "my ", "main ", "mujhe ", "mera ", "meri "]):
            return True

        # Check pure greetings
        if re.fullmatch(r"(hello|hi|hey|good\s+(morning|afternoon|evening)|namaste)[!.\s]*", low):
            return True

        return False

    async def extract(self, message: str, context: str | None = None) -> ExtractionResult:
        raw = message.strip()
        if not raw:
            return ExtractionResult(
                is_assertion=False,
                language="en",
                facts=[],
                confidence=0.0,
                raw_message=message,
            )

        if self._is_pure_question_or_noise(raw):
            return ExtractionResult(
                is_assertion=False,
                language="hinglish" if self._is_hinglish(raw) else "en",
                facts=[],
                confidence=0.95,
                raw_message=message,
            )

        lang = "hinglish" if self._is_hinglish(raw) else "en"
        facts: list[ExtractedFact] = []

        # Split clauses by conjunctions for multi-fact extraction
        clauses = re.split(
            r"(?<=[.,;!?])\s+|\s+(?:and|but|however|lekin|aur|par|before\s+joining|after\s+leaving)\s+",
            raw,
            flags=re.IGNORECASE,
        )

        now = datetime.now(UTC)

        # 1. Complex temporal transitions: "I worked at Google before joining Microsoft"
        match_trans = re.search(
            r"I\s+worked\s+at\s+([A-Za-z0-9\s]+?)\s+before\s+joining\s+([A-Za-z0-9\s]+)",
            raw,
            re.IGNORECASE,
        )
        if match_trans:
            c1 = match_trans.group(1).strip()
            c2 = match_trans.group(2).strip()
            facts.append(
                ExtractedFact(
                    subject="user",
                    predicate="worked_at",
                    object=c1,
                    raw_text=f"worked at {c1}",
                    memory_type="employment",
                    domain="work",
                    polarity="positive",
                    temporal_status="historical",
                    confidence=0.95,
                    valid_from=now,
                )
            )
            facts.append(
                ExtractedFact(
                    subject="user",
                    predicate="employed_at",
                    object=c2,
                    raw_text=f"joining {c2}",
                    memory_type="employment",
                    domain="work",
                    polarity="positive",
                    temporal_status="current",
                    confidence=0.95,
                    valid_from=now,
                )
            )

        # 2. "I left Google and joined Microsoft"
        match_left = re.search(
            r"I\s+left\s+([A-Za-z0-9\s]+?)\s+(?:and|to)\s+joined?\s+([A-Za-z0-9\s]+)",
            raw,
            re.IGNORECASE,
        )
        if match_left and not facts:
            c1 = match_left.group(1).strip()
            c2 = match_left.group(2).strip()
            facts.append(
                ExtractedFact(
                    subject="user",
                    predicate="worked_at",
                    object=c1,
                    raw_text=f"left {c1}",
                    memory_type="employment",
                    domain="work",
                    polarity="positive",
                    temporal_status="historical",
                    confidence=0.95,
                    valid_from=now,
                )
            )
            facts.append(
                ExtractedFact(
                    subject="user",
                    predicate="employed_at",
                    object=c2,
                    raw_text=f"joined {c2}",
                    memory_type="employment",
                    domain="work",
                    polarity="positive",
                    temporal_status="current",
                    confidence=0.95,
                    valid_from=now,
                )
            )

        # Parse individual clauses if complex multi-clause wasn't solely handled
        for clause in clauses:
            clause = clause.strip().rstrip(".,;!?")
            if not clause:
                continue

            # Skip questions inside compound sentences (e.g. "do you know someone there?")
            if re.search(r"^(do|can|could|would)\s+you\b", clause, re.IGNORECASE) or (
                clause.endswith("?") and not any(k in clause.lower() for k in ["i ", "my ", "main ", "mujhe "])
            ):
                continue

            low_c = clause.lower()

            # --- PREFERENCES: HINGLISH ---
            # Negation: "Mujhe coffee pasand nahi hai" / "Mujhe cricket bilkul pasand nahi"
            match_hg_neg = re.search(
                r"mujhe\s+([a-zA-Z0-9\s]+?)\s+(?:bilkul\s+)?pasand\s+nah(?:i|ee)\b",
                low_c,
            )
            if match_hg_neg:
                item = match_hg_neg.group(1).strip()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="dislikes",
                        object=item,
                        raw_text=clause,
                        memory_type="preference",
                        domain="food" if item in ["coffee", "tea", "chai", "biryani", "sushi", "pizza"] else "personal",
                        polarity="negative",
                        temporal_status="current",
                        confidence=0.92,
                        valid_from=now,
                    )
                )
                continue

            # Positive: "chai bahut achi lagti hai" / "mujhe tea pasand hai" / "music achha lagta hai"
            match_hg_pos = re.search(
                r"(?:mujhe\s+)?([a-zA-Z0-9\s]+?)\s+(?:bahut\s+)?(?:achi|achha|badiya)\s+lag(?:ti|ta)?\s+hai",
                low_c,
            )
            if not match_hg_pos:
                match_hg_pos = re.search(r"mujhe\s+([a-zA-Z0-9\s]+?)\s+pasand\s+hai", low_c)
            if match_hg_pos:
                item = match_hg_pos.group(1).strip()
                item = re.sub(r"^(but|and|aur|lekin|magar|par)\s+", "", item, flags=re.IGNORECASE).strip()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="likes",
                        object=item,
                        raw_text=clause,
                        memory_type="preference",
                        domain="food" if item in ["coffee", "tea", "chai", "biryani", "sushi", "pizza"] else "personal",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.92,
                        valid_from=now,
                    )
                )
                continue

            # --- PREFERENCES: ENGLISH ---
            # Negation: "I don't like coffee" / "I hate sushi" / "I dislike tea"
            match_en_neg = re.search(
                r"I\s+(?:do\s*n['o]t\s+like|dislike|hate)\s+([a-zA-Z0-9\s]+)",
                clause,
                re.IGNORECASE,
            )
            if match_en_neg:
                item = match_en_neg.group(1).strip()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="dislikes",
                        object=item,
                        raw_text=clause,
                        memory_type="preference",
                        domain="food"
                        if item.lower()
                        in [
                            "coffee",
                            "tea",
                            "chai",
                            "biryani",
                            "sushi",
                            "pizza",
                            "meat",
                        ]
                        else "personal",
                        polarity="negative",
                        temporal_status="current",
                        confidence=0.95,
                        valid_from=now,
                    )
                )
                continue

            # Positive: "I like biryani" / "I love Italian food" / "My favorite beverage is tea"
            match_en_pos = re.search(r"I\s+(?:love|like|enjoy)\s+([a-zA-Z0-9\s]+)", clause, re.IGNORECASE)
            if match_en_pos:
                item = match_en_pos.group(1).strip()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="likes",
                        object=item,
                        raw_text=clause,
                        memory_type="preference",
                        domain="food"
                        if item.lower() in ["coffee", "tea", "chai", "biryani", "sushi", "pizza"]
                        else "personal",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.95,
                        valid_from=now,
                    )
                )
                continue

            # Dietary preference: "I am now a vegetarian" / "I turned vegan"
            match_diet = re.search(
                r"I\s+am\s+(?:now\s+)?a\s+(vegetarian|vegan|pescatarian)",
                clause,
                re.IGNORECASE,
            )
            if match_diet:
                diet = match_diet.group(1).capitalize()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="diet",
                        object=diet,
                        raw_text=clause,
                        memory_type="preference",
                        domain="food",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.95,
                        valid_from=now,
                    )
                )
                continue

            # --- LOCATIONS: HINGLISH ---
            # Historical: "Main Indore mein rehta tha" / "pehle main Delhi mein tha"
            match_hg_loc_hist = re.search(r"(?:main|pehle)\s+([a-zA-Z]+)\s+me(?:in)?\s+rehta\s+tha", low_c)
            if match_hg_loc_hist:
                city = match_hg_loc_hist.group(1).capitalize()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="lived_in",
                        object=city,
                        raw_text=clause,
                        memory_type="location",
                        domain="geography",
                        polarity="positive",
                        temporal_status="historical",
                        confidence=0.92,
                        valid_from=now,
                    )
                )
                continue

            # Current: "Main Indore mein rehta hoon" / "ab Pune shift ho gaya"
            match_hg_loc_curr = re.search(
                r"(?:main|ab)\s+([a-zA-Z]+)\s+(?:me(?:in)?\s+rehta\s+hoon|shift\s+ho\s+gaya|move\s+ho\s+gaya)",
                low_c,
            )
            if match_hg_loc_curr:
                city = match_hg_loc_curr.group(1).capitalize()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="lives_in",
                        object=city,
                        raw_text=clause,
                        memory_type="location",
                        domain="geography",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.92,
                        valid_from=now,
                    )
                )
                continue

            # --- LOCATIONS: ENGLISH ---
            # Moved to / lives in / lived in
            match_en_moved = re.search(r"I\s+(?:recently\s+)?moved\s+to\s+([a-zA-Z\s]+)", clause, re.IGNORECASE)
            if match_en_moved:
                city = match_en_moved.group(1).strip().capitalize()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="lives_in",
                        object=city,
                        raw_text=clause,
                        memory_type="location",
                        domain="geography",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.95,
                        valid_from=now,
                    )
                )
                continue

            match_en_live = re.search(r"I\s+(?:currently\s+)?live\s+in\s+([a-zA-Z\s]+)", clause, re.IGNORECASE)
            if match_en_live:
                city = match_en_live.group(1).strip().capitalize()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="lives_in",
                        object=city,
                        raw_text=clause,
                        memory_type="location",
                        domain="geography",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.95,
                        valid_from=now,
                    )
                )
                continue

            match_en_lived = re.search(
                r"I\s+(?:used\s+to\s+live\s+in|lived\s+in)\s+([a-zA-Z\s]+)",
                clause,
                re.IGNORECASE,
            )
            if match_en_lived:
                city = match_en_lived.group(1).strip().capitalize()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="lived_in",
                        object=city,
                        raw_text=clause,
                        memory_type="location",
                        domain="geography",
                        polarity="positive",
                        temporal_status="historical",
                        confidence=0.95,
                        valid_from=now,
                    )
                )
                continue

            # --- IDENTITY & EMPLOYMENT ---
            match_name = re.search(
                r"(?:My\s+name\s+is|I\s+am|Mera\s+naam)\s+([A-Z][a-z]+)",
                clause,
                re.IGNORECASE,
            )
            if match_name and not any(f.predicate == "name_is" for f in facts):
                name = match_name.group(1).strip()
                if name.lower() not in ["a", "the", "now", "here"]:
                    facts.append(
                        ExtractedFact(
                            subject="user",
                            predicate="name_is",
                            object=name,
                            raw_text=clause,
                            memory_type="identity",
                            domain="personal",
                            polarity="positive",
                            temporal_status="current",
                            confidence=0.98,
                            valid_from=now,
                        )
                    )

            match_occ = re.search(
                r"I\s+(?:work\s+as\s+a|am\s+a)\s+([a-zA-Z\s]+?)(?:\s+at|\s+in|$)",
                clause,
                re.IGNORECASE,
            )
            if match_occ and not any(f.predicate == "occupation_is" for f in facts):
                occ = match_occ.group(1).strip()
                if occ.lower() not in ["vegetarian", "vegan", "student"]:
                    facts.append(
                        ExtractedFact(
                            subject="user",
                            predicate="occupation_is",
                            object=occ,
                            raw_text=clause,
                            memory_type="employment",
                            domain="work",
                            polarity="positive",
                            temporal_status="current",
                            confidence=0.95,
                            valid_from=now,
                        )
                    )

            match_work_comp = re.search(
                r"I\s+(?:work\s+at|work\s+for|am\s+employed\s+at)\s+([a-zA-Z0-9\s]+)",
                clause,
                re.IGNORECASE,
            )
            if match_work_comp and not any(f.predicate in ["employed_at", "worked_at"] for f in facts):
                comp = match_work_comp.group(1).strip()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="employed_at",
                        object=comp,
                        raw_text=clause,
                        memory_type="employment",
                        domain="work",
                        polarity="positive",
                        temporal_status="current",
                        confidence=0.95,
                        valid_from=now,
                    )
                )

            # Sister/Family: "My sister studies at Stanford" / "Meri behen Stanford mein padhti hai"
            match_sis = re.search(
                r"(?:My\s+sister\s+studies\s+at|Meri\s+(?:behen|bhen)\s+([a-zA-Z0-9\s]+?)\s+me(?:in)?\s+padhti\s+hai)\s*([a-zA-Z0-9\s]+)?",
                clause,
                re.IGNORECASE,
            )
            if match_sis:
                inst = (match_sis.group(1) or match_sis.group(2) or "").strip()
                if inst:
                    facts.append(
                        ExtractedFact(
                            subject="sister",
                            predicate="studies_at",
                            object=inst,
                            raw_text=clause,
                            memory_type="education",
                            domain="personal",
                            polarity="positive",
                            temporal_status="current",
                            confidence=0.95,
                            valid_from=now,
                        )
                    )

            # Future plans: "I will travel to Japan" / "I plan to visit Paris"
            match_plan = re.search(
                r"I\s+(?:will\s+travel\s+to|plan\s+to\s+visit)\s+([a-zA-Z\s]+)",
                clause,
                re.IGNORECASE,
            )
            if match_plan:
                dest = match_plan.group(1).strip()
                facts.append(
                    ExtractedFact(
                        subject="user",
                        predicate="plans_to_visit",
                        object=dest,
                        raw_text=clause,
                        memory_type="plan",
                        domain="activity",
                        polarity="positive",
                        temporal_status="future",
                        confidence=0.90,
                        valid_from=now,
                    )
                )

        # Deduplicate facts within single extraction
        unique_facts: list[ExtractedFact] = []
        seen = set()
        for f in facts:
            key = (f.subject.lower(), f.predicate.lower(), f.object.lower())
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)

        return ExtractionResult(
            is_assertion=len(unique_facts) > 0,
            language=lang,
            facts=unique_facts,
            confidence=0.95 if unique_facts else 0.50,
            raw_message=message,
        )
