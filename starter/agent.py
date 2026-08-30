from __future__ import annotations

import atexit
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from starter.src.dialog.query import rewrite, route
from starter.src.dialog.state import SlotState
from starter.src.retrieval import interface
from starter.src.retrieval.sparse import CATALOG

MAX_TURNS = 10
ASK_POLICY = os.getenv("ASK_POLICY", "other")
EXPLORE = os.getenv("EXPLORE", "1")
DEPTH = int(os.getenv("DEPTH", "120"))
RESULTS = os.getenv("RESULTS", "full")

# Ordered by how often each type appears in the public set: 96%, 76%, 26%, 9%, 4%, 2%.
ATTRIBUTE_ORDER = ("feature", "material", "color", "style", "size", "use_case")
NO_MORE = re.compile(r"^i don't have (?:a|an additional) preference", re.I)

_failures = {"search": 0, "respond": 0}
_errors: list[str] = []
_fallback: list[str] = []


def _record(where: str, exc: Exception) -> None:
    _failures[where] += 1
    if len(_errors) < 5:
        _errors.append(f"{where}: {type(exc).__name__}: {exc}")


def _report() -> None:
    total = sum(_failures.values())
    print(f"[agent] exceptions caught: {total} "
          f"(search={_failures['search']}, respond={_failures['respond']})", file=sys.stderr)
    for detail in _errors:
        print(f"[agent]   {detail}", file=sys.stderr)


atexit.register(_report)


@dataclass
class Session:
    slots: SlotState = field(default_factory=SlotState)
    history: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    last_ranked: list[str] = field(default_factory=list)
    shown: set[str] = field(default_factory=set)
    ask_index: int = 0
    last_ask: str | None = None


def choose_attribute(session: Session, turn: int) -> str | None:
    if ASK_POLICY == "none" or turn >= MAX_TURNS:
        return None
    if ASK_POLICY != "targeted" or session.ask_index >= len(ATTRIBUTE_ORDER):
        return "other"
    return ATTRIBUTE_ORDER[session.ask_index]


def slots_filled(slots: SlotState) -> int:
    return sum(1 for field_name in ("color", "material") if getattr(slots, field_name) is not None)


def result_count(session: Session, turn: int, top_k: int) -> int:
    if RESULTS == "turn1_small":
        return top_k if turn > 1 else min(top_k, 3)
    if RESULTS == "ramp_turn":
        return max(1, min(top_k, 2 + turn))
    if RESULTS == "ramp_facts":
        return max(1, min(top_k, 3 + 3 * slots_filled(session.slots)))
    return top_k


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
        self._catalog_ids: set[str] = set()
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                self._catalog_ids.add(str(json.loads(line)["parent_asin"]))
        if not _fallback:
            try:
                warmup = interface.retrieve("clothing shoes", SlotState(), "browsing", 10)
                _fallback.extend(asin for asin in warmup if isinstance(asin, str) and asin in self._catalog_ids)
            except Exception as exc:
                _record("search", exc)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = Session(profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception as exc:
            _record("respond", exc)
            session = self._sessions.get(session_id)
            ranked = list(session.last_ranked) if session and session.last_ranked else list(_fallback)
            return {
                "message": "Here are the closest matches I found.",
                "ask_attribute": None if turn >= MAX_TURNS or ASK_POLICY == "none" else "other",
                "recommendations": [{"parent_asin": asin} for asin in ranked],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self._sessions.setdefault(session_id, Session())
        session.history.append(user_message)
        if session.last_ask and NO_MORE.match(user_message.strip()):
            session.ask_index += 1
        slots = session.slots
        if slots.is_override(user_message):
            slots.erase_conflicting(user_message)
            session.last_ranked = []
            session.shown.clear()
        slots.update(user_message)

        query = rewrite(session.history, slots, session.profile)
        track = route(user_message, slots)
        count = result_count(session, turn, top_k)
        ranked = self._rank(query, user_message, session, track, top_k, count)
        attribute = choose_attribute(session, turn)
        session.last_ask = attribute
        return {
            "message": phrase(attribute),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _rank(self, query: str, latest: str, session: Session, track: str, top_k: int, count: int) -> list[str]:
        depth = DEPTH if EXPLORE == "1" else top_k
        ranked = self._search(query, session, track, depth)
        if not ranked:
            ranked = self._search(latest, session, track, depth)
        candidates = [
            asin for asin in dict.fromkeys(ranked)
            if isinstance(asin, str) and asin and asin in self._catalog_ids
        ]
        fresh = [asin for asin in candidates if asin not in session.shown] if EXPLORE == "1" else candidates
        unique = fresh[:count]
        for source in (candidates, session.last_ranked, _fallback):
            for asin in source:
                if len(unique) >= count:
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
        except Exception as exc:
            _record("search", exc)
            return []
