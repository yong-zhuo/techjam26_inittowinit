from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from submission.src.index_build import ASSETS, MODEL_NAME, MODEL_REVISION

# bge was trained with this on queries only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class DenseIndex:
    def __init__(self, assets: str | Path = ASSETS) -> None:
        self.assets = Path(assets)
        vectors_path = self.assets / "embeddings.npy"
        asins_path = self.assets / "asins.json"
        if not vectors_path.exists() or not asins_path.exists():
            raise RuntimeError(
                f"No embedding index in {self.assets}. "
                "Run: python -m submission.src.index_build"
            )
        self.vectors = np.load(vectors_path)
        self.asins = json.loads(asins_path.read_text(encoding="utf-8"))
        if len(self.asins) != self.vectors.shape[0]:
            raise RuntimeError(
                f"index is corrupt: {self.vectors.shape[0]} vectors but {len(self.asins)} asins"
            )
        self.model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, cache_folder=str(self.assets))

    def encode(self, query: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    def search(self, query: str, top_k: int) -> list[str]:
        if not query.strip():
            return []
        scores = self.vectors @ self.encode(query)
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.asins[i] for i in idx]
