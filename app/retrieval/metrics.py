import numpy as np


def compute_retrieval_metrics(
    retrieved_objects: list[str],
    gold_expected_objects: list[str],
    k: int = 5,
) -> dict[str, float]:
    """
    Computes standard Information Retrieval (IR) metrics with strict one-to-one matching:
    - Precision@1 (P@1)
    - Precision@3 (P@3)
    - Precision@5 (P@5)
    - Recall@5 (R@5)
    - R-Precision (Precision@|R|)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain at 5 (nDCG@5)

    Conforms to standard TREC / Cranfield evaluation methodology:
    - Strict one-to-one matching: A retrieved prediction can match at most one gold item.
    - Fixed cutoff K: P@5 denominator is strictly 5.
    - Zero division safety on empty gold sets.
    """
    if not gold_expected_objects:
        return {
            "p_at_1": 0.0,
            "p_at_3": 0.0,
            "p_at_5": 0.0,
            "r_at_5": 0.0,
            "r_prec": 0.0,
            "rr": 0.0,
            "ndcg_at_5": 0.0,
        }

    # Clean retrieved items and limit to k
    top_k_retrieved = retrieved_objects[:k]

    # Strict one-to-one fact matching
    matched_gold_indices: set[int] = set()
    rel_flags: list[int] = []
    first_rank: int | None = None

    for rank_idx, ret_obj in enumerate(top_k_retrieved, start=1):
        matched = False
        ret_clean = (ret_obj or "").lower().strip()
        for g_idx, gold_obj in enumerate(gold_expected_objects):
            if g_idx not in matched_gold_indices and gold_obj:
                g_clean = gold_obj.lower().strip()
                if g_clean in ret_clean or ret_clean in g_clean:
                    matched_gold_indices.add(g_idx)
                    matched = True
                    if first_rank is None:
                        first_rank = rank_idx
                    break
        rel_flags.append(1 if matched else 0)

    # Pad binary relevance flags to length k
    while len(rel_flags) < k:
        rel_flags.append(0)

    relevant_count = len(matched_gold_indices)
    gold_count = len(gold_expected_objects)

    p_at_1 = float(rel_flags[0]) if len(rel_flags) >= 1 else 0.0
    p_at_3 = float(sum(rel_flags[:3]) / 3.0)
    p_at_5 = float(relevant_count / float(k))
    r_at_5 = float(relevant_count / float(gold_count)) if gold_count > 0 else 1.0
    r_prec = float(sum(rel_flags[:gold_count]) / float(gold_count)) if gold_count > 0 else 1.0
    rr = float(1.0 / first_rank) if first_rank else 0.0

    # Discounted Cumulative Gain (DCG@5) and Ideal DCG (IDCG@5)
    dcg = sum(rel / np.log2(idx + 1) for idx, rel in enumerate(rel_flags[:k], start=1))
    idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, min(k, gold_count) + 1))
    ndcg_5 = float(dcg / idcg) if idcg > 0 else 0.0

    return {
        "p_at_1": round(p_at_1, 4),
        "p_at_3": round(p_at_3, 4),
        "p_at_5": round(p_at_5, 4),
        "r_at_5": round(r_at_5, 4),
        "r_prec": round(r_prec, 4),
        "rr": round(rr, 4),
        "ndcg_at_5": round(ndcg_5, 4),
    }
