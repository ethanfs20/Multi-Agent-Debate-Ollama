from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


def is_strict_constraint_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return ("say only:" in q) or ("output only:" in q) or ("respond only:" in q)


def build_prompt(system: str, user: str) -> str:
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", (s or "").strip())
    return s[:120] if len(s) > 120 else s


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def critique_has_evidence_flags(text: str) -> bool:
    t = (text or "").lower()
    flags = [
        "evidence missing", "missing evidence", "overstated", "over-claimed", "unsupported",
        "cannot support", "not supported", "likely wrong", "factually", "misleading",
        "hallucinat", "no evidence", "weak evidence", "unverified", "incorrect",
    ]
    return any(f in t for f in flags)


def mode_preset(mode: str) -> Tuple[str, str]:
    mode = (mode or "general").lower().strip()
    if mode == "neteng":
        return (
            "Domain: networking. Include concrete Cisco IOS examples and show commands when relevant.",
            "Verifier: show commands + expected outputs + troubleshooting checklist.",
        )
    if mode == "bash":
        return (
            "Domain: Bash/Linux scripting. Use safe patterns, input validation, and clear errors.",
            "Verifier: test plan, edge cases, shellcheck notes.",
        )
    return (
        "Domain: general. Be accurate, practical, and aligned to user constraints.",
        "Verifier: concrete checks to validate correctness.",
    )


def read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def get_host_env_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.path.abspath(os.sys.executable),
        "cwd": os.path.abspath(os.getcwd()),
        "git_commit": get_git_commit_short(),
    }
