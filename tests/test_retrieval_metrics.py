from app.retrieval.coordinator import compute_retrieval_metrics


def test_five_relevant_results():
    retrieved = ["pune", "indore", "mumbai", "delhi", "bangalore"]
    gold = ["pune", "indore", "mumbai", "delhi", "bangalore"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_3"] == 1.0
    assert m["p_at_5"] == 1.0
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0
    assert m["rr"] == 1.0
    assert m["ndcg_at_5"] == 1.0


def test_one_relevant_result_among_five():
    retrieved = ["pune", "tea", "coffee", "google", "japan"]
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0
    assert m["rr"] == 1.0
    assert m["ndcg_at_5"] == 1.0


def test_no_relevant_results_among_five():
    retrieved = ["tea", "coffee", "google", "japan", "stanford"]
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 0.0
    assert m["p_at_5"] == 0.0
    assert m["r_at_5"] == 0.0
    assert m["r_prec"] == 0.0
    assert m["rr"] == 0.0
    assert m["ndcg_at_5"] == 0.0


def test_fewer_than_five_results_returned():
    retrieved = ["pune", "indore"]
    gold = ["pune", "indore"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_3"] == round(2.0 / 3.0, 4)
    assert m["p_at_5"] == 0.4
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0
    assert m["rr"] == 1.0


def test_duplicate_returned_predictions():
    # If retriever accidentally returns duplicates of the same fact, only one should receive credit
    retrieved = ["pune", "pune", "pune", "pune", "pune"]
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0
    assert m["rr"] == 1.0


def test_multiple_relevant_memories_for_one_query():
    retrieved = ["microsoft", "google", "apple", "netflix", "meta"]
    gold = ["microsoft", "google"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_3"] == round(2.0 / 3.0, 4)
    assert m["p_at_5"] == 0.4
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0
    assert m["rr"] == 1.0


def test_empty_relevance_judgments():
    retrieved = ["pune", "indore"]
    gold = []
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_5"] == 0.0
    assert m["r_at_5"] == 0.0
    assert m["rr"] == 0.0
    assert m["ndcg_at_5"] == 0.0


def test_multiple_predictions_matching_same_expected_fact():
    # Substring overlap: "pune" and "pune city" should not both match gold ["pune"]
    retrieved = ["pune", "pune city", "delhi", "mumbai", "chennai"]
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0


def test_late_rank_relevant_result():
    retrieved = ["apple", "google", "microsoft", "amazon", "pune"]
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 0.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 1.0
    assert m["rr"] == 0.2  # Rank 5 -> RR = 1/5 = 0.2
    assert m["ndcg_at_5"] < 1.0


def test_empty_retrieved_results():
    retrieved = []
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 0.0
    assert m["p_at_3"] == 0.0
    assert m["p_at_5"] == 0.0
    assert m["r_at_5"] == 0.0
    assert m["r_prec"] == 0.0
    assert m["rr"] == 0.0
    assert m["ndcg_at_5"] == 0.0


def test_empty_string_and_none_retrieved_no_match():
    retrieved = ["", None, "   ", "pune"]
    gold = ["pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 0.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 1.0
    assert m["rr"] == 0.25  # Rank 4


def test_duplicate_expected_facts_deduplicated():
    retrieved = ["pune", "delhi", "mumbai", "chennai", "kolkata"]
    gold = ["pune", "pune", "Pune"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 1.0
    assert m["r_prec"] == 1.0


def test_substring_false_positive_prevention():
    # 'in' should NOT match 'indore', 'go' should NOT match 'google', 'a' should NOT match 'japan'
    retrieved = ["in", "go", "a", "no", "pan"]
    gold = ["indore", "google", "japan"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 0.0
    assert m["p_at_5"] == 0.0
    assert m["r_at_5"] == 0.0
    assert m["rr"] == 0.0


def test_one_retrieved_matches_at_most_one_gold():
    # If a compound retrieved item contains multiple gold keywords, it consumes only one gold entry
    retrieved = ["pune and indore", "delhi", "mumbai", "chennai", "kolkata"]
    gold = ["pune", "indore"]
    m = compute_retrieval_metrics(retrieved, gold, k=5)
    assert m["p_at_1"] == 1.0
    assert m["p_at_5"] == 0.2
    assert m["r_at_5"] == 0.5  # Only 1 of 2 gold facts matched
