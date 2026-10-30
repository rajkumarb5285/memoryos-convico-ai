from dataclasses import dataclass, field

from app.models.memory import Memory


@dataclass
class FusionCandidate:
    memory: Memory
    rrf_score: float = 0.0
    lanes: set[str] = field(default_factory=set)
    best_rank: int = 999


def reciprocal_rank_fusion(
    ranked_lanes: dict[str, list[Memory]],
    k: int = 60,
) -> list[FusionCandidate]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion:
    RRF(d) = Σ_i 1 / (k + rank_i(d))
    """
    candidates: dict[str, FusionCandidate] = {}

    for lane_name, lane_memories in ranked_lanes.items():
        for rank, mem in enumerate(lane_memories, start=1):
            if mem.id not in candidates:
                candidates[mem.id] = FusionCandidate(memory=mem)

            cand = candidates[mem.id]
            cand.rrf_score += 1.0 / (k + rank)
            cand.lanes.add(lane_name)
            cand.best_rank = min(cand.best_rank, rank)

    # Sort deterministically: highest rrf_score, then tie-break by ID
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda c: (c.rrf_score, c.memory.id),
        reverse=True,
    )
    return sorted_candidates
