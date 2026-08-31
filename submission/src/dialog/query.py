from __future__ import annotations

import os
import re

from submission.src.dialog.state import SlotState

PROFILE_SEED = os.getenv("PROFILE_SEED", "0")

BUYING_CUES = re.compile(r"\$|\b\d+(?:\.\d+)?\b|\bsize\b|\bbudget\b|\bbrand\b", re.I)


def rewrite(history: list[str], slots: SlotState, profile: dict | None = None) -> str:
    parts = list(history)
    if PROFILE_SEED == "1" and profile:
        parts.extend(str(tag) for tag in profile.get("preference_tags") or [])
    return " ".join(parts)


# Slot-count was dropped from this check: only color and material are ever
# extracted (state.py), so a "filled >= 3" threshold could never be reached
# and was silently dead code. Widening extraction to make it reachable is the
# retrieval owner's call - route()'s only consumer is their RRF weighting -
# so this stays a pure keyword check until that's decided.
def route(message: str, slots: SlotState) -> str:
    if BUYING_CUES.search(message):
        return "buying"
    return "browsing"
