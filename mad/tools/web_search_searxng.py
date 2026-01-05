from __future__ import annotations

import json
import urllib.parse
from urllib.request import Request, urlopen
from typing import Any, Dict, List

def searxng_search(base_url: str, query: str, *, top_k: int = 6, timeout_s: int = 20) -> List[Dict[str, Any]]:
    q = urllib.parse.quote(query)
    url = f"{base_url.rstrip('/')}/search?q={q}&format=json"
    req = Request(url, headers={"User-Agent": "MAD-Ollama/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    results = data.get("results") or []
    out: List[Dict[str, Any]] = []
    for r in results[: max(1, int(top_k))]:
        out.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "snippet": (r.get("content") or r.get("snippet") or "").strip(),
        })
    return out
