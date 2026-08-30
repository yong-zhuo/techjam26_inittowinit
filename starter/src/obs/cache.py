from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(os.getenv("CACHE_DIR", "starter/assets/llm_cache"))


def key(*parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(cache_key: str) -> dict | None:
    path = CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def put(cache_key: str, value: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(value), encoding="utf-8")
    except Exception:
        pass
