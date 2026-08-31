from __future__ import annotations

from collections import defaultdict


# rrf merges by rank position
def rrf(ranked_lists: list[list[str]], weights: list[float] | None = None, k: int = 60) -> list[str]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, asin in enumerate(ranked, start=1):
            scores[asin] += weight / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
