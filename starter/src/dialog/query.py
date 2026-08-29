from __future__ import annotations

import os
import re

from starter.src.dialog.state import SlotState

PROFILE_SEED = os.getenv("PROFILE_SEED", "0")

SLOT_FIELDS = ("category", "color", "material", "style", "use_case", "brand", "budget")
BUYING_CUES = re.compile(r"\$|\b\d+(?:\.\d+)?\b|\bsize\b|\bbudget\b|\bbrand\b", re.I)


def rewrite(history: list[str], slots: SlotState, profile: dict | None = None) -> str:
    parts = list(history)
    if PROFILE_SEED == "1" and profile:
        parts.extend(str(tag) for tag in profile.get("preference_tags") or [])
    return " ".join(parts)


def route(message: str, slots: SlotState) -> str:
    filled = sum(1 for field in SLOT_FIELDS if getattr(slots, field) is not None)
    if filled >= 3 or BUYING_CUES.search(message):
        return "buying"
    return "browsing"
