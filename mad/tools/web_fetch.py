from __future__ import annotations

import re
from urllib.request import Request, urlopen

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def _strip_big_json(text: str) -> str:
    # Remove very large {...} blobs (common on modern sites; JSON-LD, app state, etc.)
    return re.sub(r"(?s)\{.{2000,}?\}", " ", text)

def fetch_url_text(url: str, *, timeout_s: int = 20, max_bytes: int = 250_000, max_chars: int = 12_000) -> str:
    req = Request(url, headers={"User-Agent": "MAD-Ollama/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read(max_bytes).decode("utf-8", errors="replace")

    raw = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw)
    text = _TAG_RE.sub(" ", raw)
    text = _strip_big_json(text)
    text = _WS_RE.sub(" ", text).strip()

    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text
