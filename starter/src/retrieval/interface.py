from __future__ import annotations

import os
import sys

from starter.src.dialog.state import SlotState
from starter.src.retrieval.fusion import rrf
from starter.src.retrieval.rerank import rerank, set_catalog
from starter.src.retrieval.sparse import BM25Index

RETRIEVAL = os.getenv("RETRIEVAL", "hybrid")

# (sparse, dense) weights; buying is literal, browsing is vague
WEIGHTS = {"buying": [1.0, 0.15], "browsing": [1.0, 0.40]}

_sparse: BM25Index | None = None
_dense = None


def init(catalog_path: str) -> None:
    global _sparse, _dense
    set_catalog(catalog_path)
    _sparse = BM25Index(catalog_path)
    _dense = None
    if RETRIEVAL == "sparse":
        return
    try:
        from starter.src.retrieval.dense import DenseIndex

        _dense = DenseIndex()
    except Exception as exc:
        print(f"dense retrieval unavailable ({exc}); using BM25 only", file=sys.stderr)


# Retrieval seam: dialog side calls this, retrieval side implements it.
def retrieve(query: str, slots: SlotState, track: str, top_k: int) -> list[str]:
    if _dense is None:
        ranked = _sparse.search(query, top_k)
    elif RETRIEVAL == "dense":
        ranked = _dense.search(query, top_k)
    else:
        lists = [_sparse.search(query, top_k), _dense.search(query, top_k)]
        ranked = rrf(lists, WEIGHTS.get(track, [1.0, 0.25]))[:top_k]
    return rerank(query, ranked)


def catalog_ids() -> set[str]:
    return _sparse.asins
