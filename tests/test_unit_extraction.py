import pytest

from app.extraction.extractor import DeterministicMemoryExtractor

extractor = DeterministicMemoryExtractor()


@pytest.mark.asyncio
async def test_hinglish_preference_and_negation():
    msg = "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai."
    res = await extractor.extract(msg)

    assert res.is_assertion is True
    assert res.language == "hinglish"
    assert len(res.facts) >= 2

    coffee_fact = next((f for f in res.facts if "coffee" in f.object.lower()), None)
    assert coffee_fact is not None
    assert coffee_fact.polarity == "negative"
    assert coffee_fact.predicate == "dislikes"

    chai_fact = next(
        (f for f in res.facts if "chai" in f.object.lower() or "tea" in f.object.lower()),
        None,
    )
    assert chai_fact is not None
    assert chai_fact.polarity == "positive"
    assert chai_fact.predicate == "likes"


@pytest.mark.asyncio
async def test_employment_transition_temporal():
    msg = "I worked at Google before joining Microsoft."
    res = await extractor.extract(msg)

    assert res.is_assertion is True
    assert len(res.facts) >= 2

    google_fact = next((f for f in res.facts if "google" in f.object.lower()), None)
    assert google_fact is not None
    assert google_fact.temporal_status == "historical"
    assert google_fact.predicate == "worked_at"

    msft_fact = next((f for f in res.facts if "microsoft" in f.object.lower()), None)
    assert msft_fact is not None
    assert msft_fact.temporal_status == "current"
    assert msft_fact.predicate == "employed_at"


@pytest.mark.asyncio
async def test_question_rejection_no_assertion():
    # Prompt: "Do you know someone at Google? must not automatically create a memory asserting that the user works at Google"
    msg = "Do you know someone at Google?"
    res = await extractor.extract(msg)

    assert res.is_assertion is False
    assert len(res.facts) == 0


@pytest.mark.asyncio
async def test_casual_greeting_noise_rejection():
    msg = "Hello! Good morning, how are you doing today?"
    res = await extractor.extract(msg)

    assert res.is_assertion is False
    assert len(res.facts) == 0


@pytest.mark.asyncio
async def test_location_hinglish_and_english():
    res_hg = await extractor.extract("Main Indore mein rehta hoon.")
    assert res_hg.is_assertion is True
    assert any(f.predicate == "lives_in" and "indore" in f.object.lower() for f in res_hg.facts)

    res_en = await extractor.extract("I recently moved to Pune.")
    assert res_en.is_assertion is True
    assert any(f.predicate == "lives_in" and "pune" in f.object.lower() for f in res_en.facts)


@pytest.mark.asyncio
async def test_family_social_extraction():
    res = await extractor.extract("My sister studies at Stanford University.")
    assert res.is_assertion is True
    sis_fact = next((f for f in res.facts if f.subject == "sister"), None)
    assert sis_fact is not None
    assert sis_fact.predicate == "studies_at"
    assert "stanford" in sis_fact.object.lower()


@pytest.mark.asyncio
async def test_future_plan_extraction():
    res = await extractor.extract("I will travel to Japan next year.")
    assert res.is_assertion is True
    plan_fact = next((f for f in res.facts if f.memory_type == "plan"), None)
    assert plan_fact is not None
    assert plan_fact.temporal_status == "future"
    assert "japan" in plan_fact.object.lower()


@pytest.mark.asyncio
async def test_dietary_preference():
    res = await extractor.extract("I am now a vegetarian.")
    assert res.is_assertion is True
    diet_fact = next((f for f in res.facts if f.predicate == "diet"), None)
    assert diet_fact is not None
    assert "vegetarian" in diet_fact.object.lower()
