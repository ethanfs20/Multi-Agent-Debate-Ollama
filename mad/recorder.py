from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, safe_filename, sha256_text, utc_now_iso


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    # append-only; flush so you don't lose data on crash
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def default_redactions() -> List[Dict[str, str]]:
    # simple patterns you can extend
    return [
        {"pattern": r"(?i)\b(authorization:\s*bearer\s+)[A-Za-z0-9._-]+\b", "replace": r"\1<REDACTED>"},
        {"pattern": r"\bghp_[A-Za-z0-9]{20,}\b", "replace": "<REDACTED_GITHUB_TOKEN>"},
        {"pattern": r"\bAKIA[0-9A-Z]{16}\b", "replace": "<REDACTED_AWS_KEY>"},
        {"pattern": r"(?i)\b(api[_-]?key\s*[:=]\s*)[A-Za-z0-9._-]+\b", "replace": r"\1<REDACTED>"},
    ]


def apply_redactions(text: str, rules: List[Dict[str, str]]) -> str:
    import re
    out = text or ""
    for r in rules:
        out = re.sub(r["pattern"], r["replace"], out)
    return out


class RunRecorder:
    def __init__(self, runs_dir: str, run_id: str, redact: bool) -> None:
        self.run_id = run_id
        self.runs_dir = runs_dir
        self.redact = redact
        self.redaction_rules = default_redactions()

        day = utc_now_iso()[:10]  # YYYY-MM-DD
        self.run_path = os.path.join(runs_dir, day, safe_filename(run_id))
        ensure_dir(self.run_path)

        self.calls_path = os.path.join(self.run_path, "calls.jsonl")
        self.dataset_path = os.path.join(self.run_path, "dataset.jsonl")
        self.run_json_path = os.path.join(self.run_path, "run.json")
        self.quality_json_path = os.path.join(self.run_path, "quality.json")

        self._call_records: List[Dict[str, Any]] = []

    def _maybe_redact(self, s: str) -> str:
        if not self.redact:
            return s
        return apply_redactions(s, self.redaction_rules)

    def record_call(self, record: Dict[str, Any], save_calls: bool, save_dataset: bool) -> None:
        # redact big text fields (prompts + output)
        if self.redact:
            p = record.get("prompts", {}) or {}
            for k in ["system", "user", "full_prompt"]:
                if k in p and isinstance(p[k], str):
                    p[k] = self._maybe_redact(p[k])
            record["prompts"] = p

            out = record.get("output", {}) or {}
            if "text" in out and isinstance(out["text"], str):
                out["text"] = self._maybe_redact(out["text"])
            record["output"] = out

        # store in memory for quality summary
        self._call_records.append(record)

        if save_calls:
            _append_jsonl(self.calls_path, record)

        if save_dataset:
            # trainable sample
            ds = {
                "task": "mad_call",
                "agent": record.get("agent_key"),
                "messages": [
                    {"role": "system", "content": record.get("prompts", {}).get("system", "")},
                    {"role": "user", "content": record.get("prompts", {}).get("user", "")},
                ],
                "target": record.get("output", {}).get("text", ""),
                "meta": {
                    "run_id": record.get("run_id"),
                    "stage_label": record.get("stage_label"),
                    "round": record.get("round"),
                    "model_used": record.get("model_used"),
                    "quality": record.get("quality"),
                },
            }
            _append_jsonl(self.dataset_path, ds)

    def write_run_summary(self, summary: Dict[str, Any]) -> None:
        _atomic_write_json(self.run_json_path, summary)

    def write_quality(self, quality_summary: Dict[str, Any]) -> None:
        _atomic_write_json(self.quality_json_path, quality_summary)

    def get_call_records(self) -> List[Dict[str, Any]]:
        return list(self._call_records)

    def location(self) -> str:
        return self.run_path
