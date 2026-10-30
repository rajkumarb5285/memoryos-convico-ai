from datetime import UTC, datetime

from app.models.memory import Memory
from app.retrieval.fusion import FusionCandidate, reciprocal_rank_fusion
from app.retrieval.reranker import reranker


def test_rrf_formula_and_contribution():
    mem1 = Memory(id="mem-1", subject="user", predicate="lives_in", object="Pune")
    mem2 = Memory(id="mem-2", subject="user", predicate="lived_in", object="Indore")

    lanes = {
        "structured": [mem1, mem2],
        "dense": [mem2, mem1],
    }

    # mem1 has rank 1 in structured, rank 2 in dense
    # score(mem1) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # mem2 has rank 2 in structured, rank 1 in dense
    # score(mem2) = 1/(60+2) + 1/(60+1) = 0.032522 (tie, then tie-broken by ID)

    candidates = reciprocal_rank_fusion(lanes, k=60)
    assert len(candidates) == 2
    assert "structured" in candidates[0].lanes
    assert "dense" in candidates[0].lanes
    assert round(candidates[0].rrf_score, 4) == round(candidates[1].rrf_score, 4)


def test_temporal_reranking_historical_query():
    from app.embeddings.service import embedding_service

    now = datetime.now(UTC)
    curr_mem = Memory(
        id="mem-curr",
        owner_user_id="u1",
        character_id="c1",
        subject="user",
        predicate="lives_in",
        object="Pune",
        temporal_status="current",
        valid_from=now,
        confidence=0.95,
        memory_type="location",
        domain="geography",
    )
    curr_mem.embedding = embedding_service.embed_text("user lives in Pune")

    hist_mem = Memory(
        id="mem-hist",
        owner_user_id="u1",
        character_id="c1",
        subject="user",
        predicate="lived_in",
        object="Indore",
        temporal_status="historical",
        valid_from=now,
        confidence=0.95,
        memory_type="location",
        domain="geography",
    )
    hist_mem.embedding = embedding_service.embed_text("user lived in Indore")

    cands = [
        FusionCandidate(memory=curr_mem, rrf_score=0.03, lanes={"structured", "dense"}),
        FusionCandidate(memory=hist_mem, rrf_score=0.03, lanes={"structured", "dense"}),
    ]

    # Query: "Which city did I live in before?" -> Historical query!
    evidence, insufficient, top_score = reranker.rerank("Which city did I live in before?", cands, limit=2)
    assert not insufficient
    assert len(evidence) == 2
    # Historical fact should be ranked first!
    assert evidence[0].memory_id == "mem-hist"
    assert evidence[0].object == "Indore"


def test_temporal_reranking_current_query():
    from app.embeddings.service import embedding_service

    now = datetime.now(UTC)
    curr_mem = Memory(
        id="mem-curr",
        owner_user_id="u1",
        character_id="c1",
        subject="user",
        predicate="lives_in",
        object="Pune",
        temporal_status="current",
        valid_from=now,
        confidence=0.95,
        memory_type="location",
        domain="geography",
    )
    curr_mem.embedding = embedding_service.embed_text("user lives in Pune")

    hist_mem = Memory(
        id="mem-hist",
        owner_user_id="u1",
        character_id="c1",
        subject="user",
        predicate="lived_in",
        object="Indore",
        temporal_status="historical",
        valid_from=now,
        confidence=0.95,
        memory_type="location",
        domain="geography",
    )
    hist_mem.embedding = embedding_service.embed_text("user lived in Indore")

    cands = [
        FusionCandidate(memory=curr_mem, rrf_score=0.03, lanes={"structured", "dense"}),
        FusionCandidate(memory=hist_mem, rrf_score=0.03, lanes={"structured", "dense"}),
    ]

    # Query: "Where do I live now?" -> Current query!
    evidence, insufficient, top_score = reranker.rerank("Where do I live now?", cands, limit=2)
    assert not insufficient
    assert evidence[0].memory_id == "mem-curr"
    assert evidence[0].object == "Pune"


def test_insufficient_evidence_when_empty():
    evidence, insufficient, top_score = reranker.rerank("What is my passport number?", [], limit=5)
    assert insufficient is True
    assert len(evidence) == 0
    assert top_score == 0.0
