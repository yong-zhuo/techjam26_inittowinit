from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from starter.src.obs import cache
from starter.src.retrieval.sparse import CATALOG

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ENABLED = os.getenv("RERANK", "1") != "0"
MODEL = os.getenv("RERANK_MODEL", "")
TOP = int(os.getenv("RERANK_TOP", "20"))
# candidates covered by the sliding pass; equal to TOP means a single window
DEPTH = int(os.getenv("RERANK_DEPTH", "20"))
EFFORT = os.getenv("RERANK_EFFORT", "low")
TITLE_CHARS = 80

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|suede|wool|spandex|silk|rayon|denim|mesh|"
    r"rubber|acrylic|sterling silver|silver|gold|stainless steel|alloy|canvas|linen)\b",
    re.I,
)

PROMPT = """A shopper is searching for one specific product. Everything they have said:

{conversation}

Numbered candidates:
{candidates}

Rank by the details that DISTINGUISH these candidates from each other - model, pattern,
cut, silhouette, closure, size, specific materials. Attributes shared by nearly every
candidate carry no information; ignore them. A stated hard requirement the candidate
clearly fails should push it down. Do not reward brand fame or popularity.

Reorder ALL candidate numbers, best first.
Reply with only the numbers, comma-separated. Include every number exactly once."""

_titles: dict[str, str] | None = None
_usage = {"prompt_tokens": 0, "completion_tokens": 0}
_live = ENABLED and bool(MODEL)


def describe(product: dict) -> str:
    parts = [str(product.get("title") or "")[:TITLE_CHARS]]

    categories = [str(c) for c in (product.get("categories") or [])]
    if categories:
        parts.append("/".join(categories[-2:]))

    blob = " ".join(
        [str(product.get("title") or ""), " ".join(str(f) for f in (product.get("features") or []))]
    )
    material = MATERIAL_RE.search(blob)
    if material:
        parts.append(material.group(1).lower())

    # short factual bullets only
    snippets = [s for f in (product.get("features") or []) if 3 <= len(s := str(f).strip()) <= 60]
    parts.extend(snippets[:2])

    if product.get("price") not in (None, ""):
        parts.append(f"${product['price']}")

    return " | ".join(parts)


def _load_titles() -> dict[str, str]:
    global _titles
    if _titles is None:
        _titles = {}
        with Path(CATALOG).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                _titles[str(product["parent_asin"])] = describe(product)
    return _titles


def pop_usage() -> dict[str, int]:
    used = dict(_usage)
    _usage["prompt_tokens"] = _usage["completion_tokens"] = 0
    return used


def parse_order(reply: str, size: int) -> list[int]:
    seen, order = set(), []
    for token in re.findall(r"\d+", reply):
        index = int(token) - 1
        if 0 <= index < size and index not in seen:
            seen.add(index)
            order.append(index)
    order.extend(i for i in range(size) if i not in seen)
    return order


def _call(conversation: str, candidates: list[str]) -> str:
    import litellm

    # providers vary in which params they accept
    litellm.drop_params = True

    listing = "\n".join(
        f"{i}. {_load_titles().get(asin, asin)}" for i, asin in enumerate(candidates, start=1)
    )
    prompt = PROMPT.format(conversation=conversation.strip()[:1500], candidates=listing)
    cache_key = cache.key(MODEL, EFFORT, prompt)
    hit = cache.get(cache_key)
    if hit:
        return hit["reply"]

    extra = {"reasoning_effort": EFFORT} if EFFORT else {}
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        **extra,
    )
    reply = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    if usage is not None:
        _usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        _usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
    cache.put(cache_key, {"reply": reply})
    return reply


# windows slide back to front so a promoted item can keep climbing
def rerank(conversation: str, ranked: list[str]) -> list[str]:
    global _live
    if not _live or len(ranked) < 2:
        return ranked
    depth = min(DEPTH, len(ranked))
    head, tail = list(ranked[:depth]), ranked[depth:]
    step = max(1, TOP // 2)
    for start in reversed(range(0, max(1, depth - TOP + 1), step)):
        window = head[start:start + TOP]
        if len(window) < 2:
            continue
        try:
            order = parse_order(_call(conversation, window), len(window))
        except Exception as exc:
            print(f"rerank disabled ({exc}); using fused order", file=sys.stderr)
            _live = False
            break
        head[start:start + TOP] = [window[i] for i in order]
    return head + tail
