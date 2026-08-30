from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

CATALOG = os.getenv("CATALOG_PATH", "data/catalog.jsonl")

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Column order below is parent_asin, title, categories, features, details, store, description.
FIELD_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class BM25Index:
    def __init__(self, catalog_path: str | Path = CATALOG) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.asins: set[str] = set()
        self._build()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                self.asins.add(str(product["parent_asin"]))
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def search(self, query: str, top_k: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(query)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        weights = ", ".join(str(weight) for weight in FIELD_WEIGHTS)
        rows = self.connection.execute(
            f"SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {weights}) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [str(row[0]) for row in rows]
