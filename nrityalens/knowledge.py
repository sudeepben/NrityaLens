from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeaningEntry:
    dance_form: str
    label: str
    type: str
    keywords: list[str]
    meaning: str
    feedback_focus: list[str]


def load_meanings(path: Path) -> list[MeaningEntry]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [MeaningEntry(**record) for record in records]


def find_meaning(entries: list[MeaningEntry], label: str | None) -> MeaningEntry | None:
    if not label:
        return None

    normalized = label.casefold()
    for entry in entries:
        if entry.label.casefold() == normalized:
            return entry
    return None
