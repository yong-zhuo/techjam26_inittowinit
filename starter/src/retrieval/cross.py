from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from starter.src.index_build import ASSETS
from starter.src.retrieval.rerank import describe
from starter.src.retrieval.sparse import CATALOG

CROSS_MODEL = os.getenv("CROSS_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_TOP = int(os.getenv("CROSS_TOP", "20"))

_model = None
_docs: dict[str, str] | None = None
_live = os.getenv("CROSS", "0") == "1"


def _load_docs() -> dict[str, str]:
    global _docs
    if _docs is None:
        _docs = {}
        with Path(CATALOG).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                _docs[str(product["parent_asin"])] = describe(product)
    return _docs


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(CROSS_MODEL, cache_folder=str(ASSETS), max_length=256)
    return _model


def rerank(query: str, ranked: list[str]) -> list[str]:
    global _live
    if not _live or len(ranked) < 2:
        return ranked
    window, tail = ranked[:CROSS_TOP], ranked[CROSS_TOP:]
    try:
        docs = _load_docs()
        scores = _load_model().predict([(query, docs.get(a, a)) for a in window])
    except Exception as exc:
        print(f"cross-encoder disabled ({exc}); using fused order", file=sys.stderr)
        _live = False
        return ranked
    order = sorted(range(len(window)), key=lambda i: -scores[i])
    return [window[i] for i in order] + tail
