from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SEARCH_LINK_RE = re.compile(r"(google\.com/search\?|duckduckgo\.com/\?q=|bing\.com/search\?)", re.IGNORECASE)
_CONSTRAINT_LEAK_RE = re.compile(r"\b(say only:|output only:|respond only:)\b", re.IGNORECASE)
_REFUSAL_RE = re.compile(r"\b(i cannot|i can't|i won'?t|i am unable|cannot provide)\b", re.IGNORECASE)

# very rough stopwords for keyword overlap checks
_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","is","are","was","were",
    "be","been","being","that","this","it","they","them","their","we","you","your","i","my",
    "from","by","at","not","but","if","then","than","so","can","could","should","would"
}


@dataclass
class QualityFlags:
    off_topic: bool = False
    constraint_leak: bool = False
    refusal: bool = False
    contains_url: bool = False
    contains_search_links: bool = False
    likely_truncated: bool = False
    wrong_task: bool = False
    notes: List[str] = None  # type: ignore


def _keywords(s: str) -> Counter:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", (s or "").lower())
    words = [w for w in words if w not in _STOP]
    return Counter(words)


def _keyword_overlap_ratio(a: str, b: str) -> float:
    ka = _keywords(a)
    kb = _keywords(b)
    if not ka or not kb:
        return 0.0
    inter = sum((ka & kb).values())
    denom = max(1, sum(ka.values()))
    return inter / denom


def detect_truncation(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    # Common truncation signatures: cut mid-bullet, dangling "•", unfinished "Claim:", etc.
    if t.endswith(("-", "*", "•", "+", ":", ",")):
        return True
    if re.search(r"\b(Claim|Evidence|Limits|What they found)\s*:\s*$", t):
        return True
    # Ending with a partial sentence and no punctuation
    if len(t) > 120 and t[-1].isalnum() and not re.search(r"[.!?]\s*$", t):
        return True
    return False


def detect_wrong_task(agent_key: str, question: str, output: str) -> bool:
    q = (question or "").lower()
    o = (output or "").lower()
    # If question doesn't mention UBI but output starts analyzing UBI, flag.
    if "ubi" not in q and "universal basic income" not in q:
        if "universal basic income" in o or re.search(r"\bubi\b", o):
            return True
    # Similar for totally different topic anchors
    # (extend this list later based on what you see)
    drift_terms = ["student loan", "gun control", "abortion", "immigration", "bitcoin", "crypto"]
    if not any(term in q for term in drift_terms):
        if any(term in o for term in drift_terms):
            # only mark wrong_task if overlap with question is low
            if _keyword_overlap_ratio(question, output) < 0.10:
                return True
    return False


def assess_quality(*, agent_key: str, stage_label: str, question: str, constrained_run: bool, output: str) -> QualityFlags:
    notes: List[str] = []
    out = output or ""

    contains_url = bool(_URL_RE.search(out))
    contains_search_links = bool(_SEARCH_LINK_RE.search(out))
    if contains_search_links:
        notes.append("Contains search links (not valid citations).")
    elif contains_url:
        notes.append("Contains URLs (may not be desired for training).")

    constraint_leak = (not constrained_run) and bool(_CONSTRAINT_LEAK_RE.search(out))
    if constraint_leak:
        notes.append("Constraint-leak language present despite unconstrained user question.")

    refusal = bool(_REFUSAL_RE.search(out))
    if refusal:
        notes.append("Refusal / non-answer language detected.")

    likely_truncated = detect_truncation(out)
    if likely_truncated:
        notes.append("Output looks truncated/cut off (likely num_predict limit).")

    wrong_task = detect_wrong_task(agent_key, question, out)
    if wrong_task:
        notes.append("Output appears to address a different task/topic than the user question.")

    # Off-topic heuristic: overlap too low AND not an obvious schema output agent
    overlap = _keyword_overlap_ratio(question, out)
    off_topic = overlap < 0.08 and not any(k in stage_label.lower() for k in ["judge", "referee"])
    if off_topic:
        notes.append(f"Low keyword overlap with question (overlap={overlap:.2f}).")

    return QualityFlags(
        off_topic=off_topic,
        constraint_leak=constraint_leak,
        refusal=refusal,
        contains_url=contains_url,
        contains_search_links=contains_search_links,
        likely_truncated=likely_truncated,
        wrong_task=wrong_task,
        notes=notes,
    )


def summarize_quality(call_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate flags across calls
    totals = Counter()
    worst: List[Tuple[int, str, Dict[str, Any]]] = []

    for idx, r in enumerate(call_records):
        q = r.get("quality", {}) or {}
        for k, v in q.items():
            if k == "notes":
                continue
            if v is True:
                totals[k] += 1

        score = 0
        score += 3 if q.get("wrong_task") else 0
        score += 3 if q.get("constraint_leak") else 0
        score += 2 if q.get("refusal") else 0
        score += 2 if q.get("likely_truncated") else 0
        score += 1 if q.get("contains_search_links") else 0
        score += 1 if q.get("off_topic") else 0

        if score >= 3:
            worst.append((score, r.get("stage_label", f"call_{idx}"), r))

    worst.sort(key=lambda x: (-x[0], x[1]))
    worst = worst[:10]

    return {
        "counts": dict(totals),
        "worst_calls": [
            {
                "score": s,
                "stage_label": label,
                "agent_key": rec.get("agent_key"),
                "model_used": rec.get("model_used"),
                "quality": rec.get("quality"),
            }
            for (s, label, rec) in worst
        ],
    }


def format_quality_report(summary: Dict[str, Any]) -> str:
    counts = summary.get("counts", {}) or {}
    worst = summary.get("worst_calls", []) or []

    lines = []
    lines.append("=== QUALITY REPORT ===")
    if not counts:
        lines.append("No quality flags triggered.")
    else:
        lines.append("Flag counts:")
        for k in sorted(counts.keys()):
            lines.append(f"- {k}: {counts[k]}")
    if worst:
        lines.append("")
        lines.append("Worst calls (top issues):")
        for w in worst:
            lines.append(f"- [{w['score']}] {w['stage_label']} ({w['agent_key']}, {w['model_used']})")
            q = w.get("quality", {}) or {}
            notes = q.get("notes", []) or []
            for n in notes[:6]:
                lines.append(f"    • {n}")
    return "\n".join(lines)
