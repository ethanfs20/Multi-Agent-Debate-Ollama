from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def http_post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTPError {e.code}: {body or e.reason}")
    except URLError as e:
        raise RuntimeError(f"URLError: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Bad JSON from Ollama: {e}")


def generate(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    top_p: float,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature, "top_p": top_p},
    }
    data = http_post_json(url, payload, timeout=timeout)
    return (data.get("response") or "").strip()


def generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    top_p: float,
    label: str,
    transcript_path: Optional[str],
    quiet: bool,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": temperature, "top_p": top_p},
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    parts: List[str] = []
    tf = None
    try:
        if transcript_path:
            from pathlib import Path
            Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
            tf = open(transcript_path, "w", encoding="utf-8", errors="replace")
            tf.write(f"===== {label} ({model}) =====\n\n")

        if not quiet:
            print(f"\n===== {label} ({model}) =====\n", flush=True)

        with urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("response", "")
                if chunk:
                    parts.append(chunk)
                    if tf:
                        tf.write(chunk)
                        tf.flush()
                    if not quiet:
                        print(chunk, end="", flush=True)
                if obj.get("done", False):
                    break

        if tf:
            tf.write("\n\n===== END =====\n")
        if not quiet:
            print("\n\n===== END =====\n", flush=True)

        return "".join(parts).strip()
    finally:
        if tf:
            tf.close()
