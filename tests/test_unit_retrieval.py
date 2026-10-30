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


def test_precision_and_recall_at_5_hand_calculated():
    """
    Unit test verifying Precision@5 and Recall@5 calculation formulas with hand-calculated examples.
    Formula 1: Precision@5 = relevant_in_top_5 / actual_returned_in_top_5
    Formula 2: Recall@5 = relevant_in_top_5 / expected_relevant_total
    """

    def calc_metrics_for_query(top5_relevant_count: int, top5_total_returned: int, expected_relevant_total: int):
        p_at_5 = (top5_relevant_count / top5_total_returned) if top5_total_returned > 0 else 0.0
        r_at_5 = (top5_relevant_count / expected_relevant_total) if expected_relevant_total > 0 else 1.0
        return p_at_5, r_at_5

    # Case 1: 4 results returned in top 5, exactly 2 relevant out of 2 expected
    p1, r1 = calc_metrics_for_query(top5_relevant_count=2, top5_total_returned=4, expected_relevant_total=2)
    assert p1 == 0.50  # 2/4
    assert r1 == 1.00  # 2/2

    # Case 2: 5 results returned, 3 relevant out of 4 expected
    p2, r2 = calc_metrics_for_query(top5_relevant_count=3, top5_total_returned=5, expected_relevant_total=4)
    assert p2 == 0.60  # 3/5
    assert r2 == 0.75  # 3/4

    # Case 3: Empty results (insufficient evidence) with 1 expected
    p3, r3 = calc_metrics_for_query(top5_relevant_count=0, top5_total_returned=0, expected_relevant_total=1)
    assert p3 == 0.00
    assert r3 == 0.00

    # Macro averages
    macro_p = (p1 + p2 + p3) / 3.0
    macro_r = (r1 + r2 + r3) / 3.0
    assert round(macro_p, 4) == 0.3667
    assert round(macro_r, 4) == 0.5833
