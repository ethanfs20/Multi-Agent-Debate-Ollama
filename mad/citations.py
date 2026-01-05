from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class Source:
    sid: str
    title: str
    url: str
    snippet: str

def trim(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def format_sources_block(sources: List[Source], *, max_snippet_chars: int = 900) -> str:
    if not sources:
        return "SOURCES: (none)\n"

    lines = ["SOURCES (cite like [S1] next to the sentence you used it for):"]
    for s in sources:
        lines.append(f"[{s.sid}] {s.title} — {s.url}\n{trim(s.snippet, max_snippet_chars)}")
    return "\n".join(lines) + "\n"
