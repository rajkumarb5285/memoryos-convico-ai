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


@pytest.mark.asyncio
async def test_hinglish_spelling_variants_and_compact_preferences():
    # 1. Spelling variant 'nahin'
    res1 = await extractor.extract("Mujhe coffee pasand nahin hai.")
    assert res1.is_assertion is True
    assert any(f.predicate == "dislikes" and "coffee" in f.object.lower() for f in res1.facts)

    # 2. Spelling variant 'nhi'
    res2 = await extractor.extract("Mujhe coffee pasand nhi hai.")
    assert res2.is_assertion is True
    assert any(f.predicate == "dislikes" and "coffee" in f.object.lower() for f in res2.facts)

    # 3. Compact preference: "Chai pasand hai, coffee nahi."
    res3 = await extractor.extract("Chai pasand hai, coffee nahi.")
    assert res3.is_assertion is True
    chai = next((f for f in res3.facts if "chai" in f.object.lower()), None)
    coffee = next((f for f in res3.facts if "coffee" in f.object.lower()), None)
    assert chai is not None and chai.polarity == "positive"
    assert coffee is not None and coffee.polarity == "negative"

    # 4. Multi-clause with 'lekin'
    res4 = await extractor.extract("Mujhe chai pasand hai lekin coffee pasand nahi hai.")
    assert res4.is_assertion is True
    assert len(res4.facts) == 2


@pytest.mark.asyncio
async def test_filter_hypotheticals_and_self_questions():
    # Questions about self should not create memories
    res1 = await extractor.extract("Do I like coffee?")
    assert res1.is_assertion is False
    assert len(res1.facts) == 0

    # Hypothetical conditionals should not create memories
    res2 = await extractor.extract("If I moved to Delhi, I would look for a job.")
    assert res2.is_assertion is False
    assert len(res2.facts) == 0

    # Question about 3rd party
    res3 = await extractor.extract("Do you know someone at Google?")
    assert res3.is_assertion is False
    assert len(res3.facts) == 0


@pytest.mark.asyncio
async def test_location_sequence_extraction():
    # Test "I lived in Mumbai", "I now live in Pune", "I moved back to Mumbai"
    r1 = await extractor.extract("I lived in Mumbai.")
    assert r1.facts[0].predicate == "lives_in"
    assert r1.facts[0].temporal_status == "historical"
    assert r1.facts[0].object == "Mumbai"

    r2 = await extractor.extract("I now live in Pune.")
    assert r2.facts[0].predicate == "lives_in"
    assert r2.facts[0].temporal_status == "current"
    assert r2.facts[0].object == "Pune"

    r3 = await extractor.extract("I moved back to Mumbai.")
    assert r3.facts[0].predicate == "lives_in"
    assert r3.facts[0].temporal_status == "current"
    assert r3.facts[0].object == "Mumbai"
