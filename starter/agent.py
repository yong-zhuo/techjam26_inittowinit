from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from starter.src.dialog.query import rewrite, route
from starter.src.dialog.state import SlotState
from starter.src.retrieval import interface
from starter.src.retrieval.sparse import CATALOG

MAX_TURNS = 10
ASK_POLICY = os.getenv("ASK_POLICY", "other")
EXPLORE = os.getenv("EXPLORE", "1")
DEPTH = 120


@dataclass
class Session:
    slots: SlotState = field(default_factory=SlotState)
    history: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    last_ranked: list[str] = field(default_factory=list)
    shown: set[str] = field(default_factory=set)


def choose_attribute(session: Session, turn: int) -> str | None:
    if ASK_POLICY == "none" or turn >= MAX_TURNS:
        return None
    return "other"


def phrase(attribute: str | None) -> str:
    if attribute is None:
        return "Here are the closest matches I found."
    if attribute == "other":
        return "Here are the closest matches so far. What else matters to you?"
    return f"Here are the closest matches so far. Any preference on {attribute}?"


class Agent:
    def __init__(self, catalog_path: str | Path = CATALOG) -> None:
        interface.init(str(catalog_path))
        self._sessions: dict[str, Session] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = Session(profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self._sessions.setdefault(session_id, Session())
        session.history.append(user_message)
        slots = session.slots
        if slots.is_override(user_message):
            slots.erase_conflicting(user_message)
            session.last_ranked = []
            session.shown.clear()
        slots.update(user_message)

        query = rewrite(session.history, slots, session.profile)
        track = route(user_message, slots)
        ranked = self._rank(query, user_message, session, track, top_k)
        attribute = choose_attribute(session, turn)
        return {
            "message": phrase(attribute),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _rank(self, query: str, latest: str, session: Session, track: str, top_k: int) -> list[str]:
        depth = DEPTH if EXPLORE == "1" else top_k
        ranked = self._search(query, session, track, depth)
        if not ranked:
            ranked = self._search(latest, session, track, depth)
        candidates = [asin for asin in dict.fromkeys(ranked) if isinstance(asin, str) and asin]
        fresh = [asin for asin in candidates if asin not in session.shown] if EXPLORE == "1" else candidates
        unique = fresh[:top_k]
        for source in (candidates, session.last_ranked):
            for asin in source:
                if len(unique) >= top_k:
                    break
                if asin not in unique:
                    unique.append(asin)
        if unique:
            session.last_ranked = unique
            session.shown.update(unique)
        return unique

    def _search(self, query: str, session: Session, track: str, top_k: int) -> list[str]:
        try:
            return interface.retrieve(query, session.slots, track, top_k)
        except Exception:
            return []
