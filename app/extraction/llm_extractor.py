import json
import re
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.extraction.base import BaseMemoryExtractor
from app.extraction.extractor import DeterministicMemoryExtractor
from app.schemas.extraction import ExtractedFact, ExtractionResult

SYSTEM_PROMPT = """You are an advanced neural memory extraction engine for conversational AI (MEMORYOS).
Your goal is to extract personal knowledge assertions and memories about the user from dialogue.
Support both English and Romanized Hinglish (Hindi written in Latin script, e.g., 'mujhe chai pasand hai', 'mai pehle Bangalore me rehta tha').

Guidelines:
1. Distinguish ASSERTIONS from QUESTIONS and NOISE:
   - "Do you know anyone at Google?" -> NOT an assertion (is_assertion: false, facts: [])
   - "Hi, how are you doing today?" -> NOT an assertion (is_assertion: false, facts: [])
   - "I work as a software engineer" -> ASSERTION (is_assertion: true)
   - "mujhe chai pasand nahi hai" -> ASSERTION (is_assertion: true, polarity: "dislikes")
2. For temporal changes:
   - "I worked at Microsoft before joining Google":
     - Fact 1: predicate="employed_at", object="Microsoft", temporal_status="historical"
     - Fact 2: predicate="employed_at", object="Google", temporal_status="current"
3. Domains: "preference", "career", "location", "family", "lifestyle", "general"
4. Polarity: "likes", "dislikes", "neutral"
5. Temporal Status: "current", "historical"

Output valid JSON matching this schema:
{
  "is_assertion": true,
  "confidence": 0.95,
  "facts": [
    {
      "subject": "user",
      "predicate": "employed_at",
      "object": "Google",
      "polarity": "likes",
      "temporal_status": "current",
      "domain": "career",
      "confidence": 0.95
    }
  ]
}
If the message contains no user personal assertions, return {"is_assertion": false, "confidence": 0.9, "facts": []}.
"""


class GroqMemoryExtractor(BaseMemoryExtractor):
    """
    LLM-powered memory extraction using Groq's high-speed inference API.
    Gracefully falls back to DeterministicMemoryExtractor if offline or rate limited.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._fallback_extractor = DeterministicMemoryExtractor()
        self._endpoint = "https://api.groq.com/openai/v1/chat/completions"

    async def extract(self, message: str, context: str | None = None) -> ExtractionResult:
        clean_msg = message.strip()
        if not clean_msg:
            return ExtractionResult(is_assertion=False, confidence=1.0, facts=[], raw_message="")

        if not self.api_key:
            return await self._fallback_extractor.extract(message, context)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            if context:
                messages.append({"role": "system", "content": f"Recent Conversation Context:\n{context}"})
            messages.append({"role": "user", "content": clean_msg})

            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.0,
            }

            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(self._endpoint, headers=headers, json=payload)

            if resp.status_code != 200:
                logger.warning(f"Groq API returned HTTP {resp.status_code}. Falling back to deterministic extractor.")
                return await self._fallback_extractor.extract(message, context)

            data = resp.json()
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract JSON block
            parsed_json = self._parse_json_payload(raw_content)
            if not parsed_json or not isinstance(parsed_json, dict):
                return await self._fallback_extractor.extract(message, context)

            is_assertion = bool(parsed_json.get("is_assertion", False))
            overall_confidence = float(parsed_json.get("confidence", 0.9))
            facts_data = parsed_json.get("facts", [])

            extracted_facts: list[ExtractedFact] = []
            for item in facts_data:
                if not isinstance(item, dict):
                    continue
                extracted_facts.append(
                    ExtractedFact(
                        subject=str(item.get("subject", "user")).lower().strip(),
                        predicate=str(item.get("predicate", "statement")).lower().strip().replace(" ", "_"),
                        object=str(item.get("object", "")).strip(),
                        polarity=str(item.get("polarity", "neutral")).lower(),
                        temporal_status=str(item.get("temporal_status", "current")).lower(),
                        domain=str(item.get("domain", "general")).lower(),
                        confidence=float(item.get("confidence", 0.9)),
                    )
                )

            # If LLM extracted 0 facts on an assertion, use deterministic corroboration
            if is_assertion and not extracted_facts:
                return await self._fallback_extractor.extract(message, context)

            return ExtractionResult(
                is_assertion=is_assertion,
                confidence=overall_confidence,
                facts=extracted_facts,
                raw_message=clean_msg,
            )

        except Exception as e:
            logger.warning(f"Groq memory extraction error: {e!s}. Falling back to deterministic extractor.")
            return await self._fallback_extractor.extract(message, context)

    def _parse_json_payload(self, text: str) -> dict[str, Any] | None:
        """Robustly extract JSON object from LLM response text."""
        # Try direct load
        text_strip = text.strip()
        try:
            return json.loads(text_strip)
        except Exception:
            pass

        # Try markdown block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_strip, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try first '{' to last '}'
        start = text_strip.find("{")
        end = text_strip.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text_strip[start : end + 1])
            except Exception:
                pass

        return None
