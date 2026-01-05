from __future__ import annotations

import json
import os
from typing import Optional

from .utils import sha256_text


def cache_key(run_id: str, stage_label: str, model: str, prompt: str,
              num_predict: int, temperature: float, top_p: float) -> str:
    blob = f"{run_id}\n{stage_label}\n{model}\n{num_predict}\n{temperature}\n{top_p}\n{prompt}"
    return sha256_text(blob)


def cache_get(cache_dir: str, key: str) -> Optional[str]:
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("response")
    except Exception:
        return None


def cache_put(cache_dir: str, key: str, response: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"response": response}, f, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
