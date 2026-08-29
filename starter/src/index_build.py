from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from starter.src.retrieval.sparse import _text

CATALOG = os.getenv("CATALOG_PATH", "data/catalog.jsonl")
ASSETS = Path("starter/assets")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Title first so it survives the model's 512-token truncation.
FIELD_SETS = {
    "title": ("title",),
    "title_cat": ("title", "categories"),
    "title_cat_feat": ("title", "categories", "features"),
    "all": ("title", "categories", "features", "description"),
}


def product_text(product: dict, fields: tuple[str, ...]) -> str:
    return " ".join(part for part in (_text(product.get(f)) for f in fields) if part)


def load_catalog(fields: tuple[str, ...]) -> tuple[list[str], list[str]]:
    asins, texts = [], []
    with Path(CATALOG).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asins.append(str(product["parent_asin"]))
            texts.append(product_text(product, fields))
    return asins, texts


def load_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, cache_folder=str(ASSETS))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load encoder weights ({exc}).\n"
            "This step requires network access on first run. See the README setup section."
        ) from exc


def is_current(fields_name: str) -> bool:
    meta_path = ASSETS / "meta.json"
    if not all((ASSETS / f).exists() for f in ("embeddings.npy", "asins.json", "meta.json")):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        meta.get("model") == MODEL_NAME
        and meta.get("revision") == MODEL_REVISION
        and meta.get("fields") == fields_name
    )


def build(fields_name: str, batch_size: int) -> None:
    fields = FIELD_SETS[fields_name]
    ASSETS.mkdir(parents=True, exist_ok=True)

    asins, texts = load_catalog(fields)
    print(f"{len(asins)} products, fields={fields_name}")

    model = load_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(ASSETS / "embeddings.npy", vectors)
    (ASSETS / "asins.json").write_text(json.dumps(asins), encoding="utf-8")
    (ASSETS / "meta.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "revision": MODEL_REVISION,
                "fields": fields_name,
                "rows": len(asins),
                "dim": int(vectors.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {vectors.shape} to {ASSETS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", default="title_cat", choices=sorted(FIELD_SETS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.force and is_current(args.fields):
        print(f"index already built for fields={args.fields}; use --force to rebuild")
        return
    build(args.fields, args.batch_size)


if __name__ == "__main__":
    main()
