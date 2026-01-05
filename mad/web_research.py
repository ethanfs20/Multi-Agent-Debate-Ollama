from __future__ import annotations

import json
import re
from typing import List
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .citations import Source, format_sources_block


def _http_get_json(url: str, timeout: int) -> dict:
    req = Request(url, headers={"User-Agent": "MAD/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as e:
        raise RuntimeError(f"HTTPError {e.code}: {e.reason}")
    except URLError as e:
        raise RuntimeError(f"URLError: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Bad JSON from SearXNG: {e}")


def searxng_search(
    *,
    query: str,
    base_url: str,
    topk: int,
    timeout: int,
) -> List[Source]:
    params = {
        "q": query,
        "format": "json",
        "language": "en",
        "safesearch": 0,
    }

    url = base_url.rstrip("/") + "/search?" + urlencode(params)
    data = _http_get_json(url, timeout=timeout)

    results = data.get("results", [])[:topk]
    sources: List[Source] = []

    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        link = (r.get("url") or "").strip()
        snippet = (r.get("content") or "").strip()

        if not title or not link:
            continue

        # Normalize whitespace
        snippet = re.sub(r"\s+", " ", snippet)

        sources.append(
            Source(
                sid=f"S{i}",
                title=title,
                url=link,
                snippet=snippet,
            )
        )

    return sources


def build_evidence_pack(
    *,
    question: str,
    searxng_url: str,
    topk: int,
    timeout: int,
) -> str:
    """
    Runs a SearXNG query and returns a formatted SOURCES block.
    Never raises — always degrades safely.
    """
    try:
        sources = searxng_search(
            query=question,
            base_url=searxng_url,
            topk=topk,
            timeout=timeout,
        )
    except Exception as e:
        return f"SOURCES: (web search failed: {e})\n"

    return format_sources_block(sources)
