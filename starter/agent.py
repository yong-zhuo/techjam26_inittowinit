from __future__ import annotations

from pathlib import Path

from starter.src.dialog.state import SlotState
from starter.src.retrieval import interface
from starter.src.retrieval.sparse import CATALOG


class Agent:
    def __init__(self, catalog_path: str | Path = CATALOG) -> None:
        interface.init(str(catalog_path))
        self._slots: dict[str, SlotState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._slots[session_id] = SlotState()

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._slots:
            raise RuntimeError("reset must be called before respond")
        slots = self._slots[session_id]
        if slots.is_override(user_message):
            slots.erase_conflicting(user_message)
        slots.update(user_message)
        ranked = interface.retrieve(user_message, slots, "buying", top_k)
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": asin} for asin in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
