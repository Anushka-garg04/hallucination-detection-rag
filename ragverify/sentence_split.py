import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    merged = []
    for p in parts:
        if merged and len(p.split()) <= 2:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged
