from __future__ import annotations

from starter.src.dialog.state import SlotState
from starter.src.retrieval.sparse import BM25Index

_index: BM25Index | None = None


def init(catalog_path: str) -> None:
    global _index
    _index = BM25Index(catalog_path)


# Retrieval seam: dialog side calls this, retrieval side implements it.
def retrieve(query: str, slots: SlotState, track: str, top_k: int) -> list[str]:
    return _index.search(query, top_k)
