from __future__ import annotations

import re
from dataclasses import dataclass, field

# Vocabulary mirrors the evaluator's intent-card regexes.
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
CONTRADICT = ("actually", "instead", "no wait", "rather than", "forget", "don't want", "scratch that")

COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)


def extract_all(message: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    colors = list(dict.fromkeys(match.lower() for match in COLOR_RE.findall(message)))
    materials = list(dict.fromkeys(match.lower() for match in MATERIAL_RE.findall(message)))
    if colors:
        found["color"] = colors
    if materials:
        found["material"] = materials
    return found


def extract(message: str) -> dict[str, str]:
    return {key: values[0] for key, values in extract_all(message).items()}


# for dialogue state tracking
@dataclass
class SlotState:
    category: str | None = None
    color: str | None = None
    material: str | None = None
    style: str | None = None
    use_case: str | None = None
    brand: str | None = None
    budget: float | None = None
    rejected: list[str] = field(default_factory=list)

    def update(self, message: str) -> None:
        for key, value in extract(message).items():
            if getattr(self, key) is None:
                setattr(self, key, value)

    def is_override(self, message: str) -> bool:
        lexical = any(cue in message.lower() for cue in CONTRADICT)
        conflict = any(
            getattr(self, key) is not None and getattr(self, key) not in values
            for key, values in extract_all(message).items()
        )
        return lexical or conflict

    def erase_conflicting(self, message: str) -> list[str]:
        erased = []
        for key, values in extract_all(message).items():
            current = getattr(self, key)
            if current is not None and current not in values:
                self.rejected.append(current)
                setattr(self, key, None)
                erased.append(key)
        return erased
