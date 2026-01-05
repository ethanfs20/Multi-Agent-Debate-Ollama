#!/usr/bin/env python3
"""
mad/cli.py

CLI entrypoint for Multi-Agent Debate (MAD).

Key behavior:
- Loads config from ./config/defaults.json, ./config/agents.json, ./config/schemas.json
- Builds RunConfig with ONLY fields that RunConfig is expected to support
- Calls run_mad(config, model_map, schemas)
- Accepts --web flags for future use, but currently ignores them (since the pipeline
  does not implement retrieval end-to-end yet).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from mad.orchestrator import run_mad
from mad.types import RunConfig


def _repo_root() -> Path:
    # mad/cli.py -> repo root is parent of "mad" package directory
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object/dict.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def _load_configs(root: Path) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    cfg_dir = root / "config"
    defaults_path = cfg_dir / "defaults.json"
    agents_path = cfg_dir / "agents.json"
    schemas_path = cfg_dir / "schemas.json"

    defaults = _read_json(defaults_path)

    agents_obj = _read_json(agents_path)
    models = agents_obj.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{agents_path} must contain an object key 'models' (dict).")
    # Ensure all values are strings
    model_map: Dict[str, str] = {}
    for k, v in models.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"{agents_path}: models must be string->string. Bad entry: {k}={v!r}")
        model_map[k] = v

    schemas = _read_json(schemas_path)
    # Schemas content is expected to be string->string, but we keep it permissive and validate lightly
    schemas_map: Dict[str, str] = {}
    for k, v in schemas.items():
        if isinstance(k, str) and isinstance(v, str):
            schemas_map[k] = v
        else:
            # keep keys that are strings with non-string values out
            pass

    return defaults, model_map, schemas_map


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mad", description="Policy-grade Multi-Agent Debate (Ollama HTTP)")

    # Required
    p.add_argument("--question", required=True, help="User question/prompt for the debate.")

    # Core overrides
    p.add_argument("--mode", default=None, choices=["general", "neteng", "bash"], help="Domain preset.")
    p.add_argument("--num-predict", type=int, default=None, help="Max tokens for model generation.")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling.")
    p.add_argument("--timeout", type=int, default=None, help="HTTP timeout seconds.")
    p.add_argument("--retries", type=int, default=None, help="Retries per call.")
    p.add_argument("--max-rounds", type=int, default=None, help="Max debate rounds.")

    # Streaming
    p.add_argument("--stream", action="store_true", help="Stream model output to console.")
    p.add_argument("--stream-only-final", action="store_true", help="Only stream the final judge output.")

    # Base URL (Ollama)
    p.add_argument("--base-url", default=None, help="Ollama base URL (default from config).")

    # Paths
    p.add_argument("--runs-dir", default=None, help="Directory for run artifacts.")
    p.add_argument("--cache-dir", default=None, help="Directory for cache (or empty to disable).")
    p.add_argument("--transcripts-dir", default=None, help="Directory for transcripts (or empty to disable).")
    p.add_argument("--log", default=None, help="Legacy top-level log file path.")

    # Behavior toggles (optional)
    p.add_argument("--concise", action="store_true", help="Prefer concise outputs (if supported by prompts).")
    p.add_argument("--parallel", action="store_true", help="Run A and C in parallel if supported.")
    p.add_argument("--a-samples", type=int, default=None, help="Number of A samples (1-5).")

    p.add_argument("--no-save-calls", action="store_true", help="Do not write calls.jsonl / dataset.jsonl.")
    p.add_argument("--no-save-dataset", action="store_true", help="Do not write dataset.jsonl.")
    p.add_argument("--no-quality-report", action="store_true", help="Do not print quality report.")
    p.add_argument("--no-redact-secrets", action="store_true", help="Do not redact secrets in saved logs.")

    # === WEB SEARCH (not wired yet) ===
    p.add_argument("--web", action="store_true", help="(Reserved) Enable web research. Currently ignored.")
    p.add_argument("--searxng-url", default="http://127.0.0.1:8080", help="(Reserved) SearXNG URL.")
    p.add_argument("--web-topk", type=int, default=8, help="(Reserved) Top-k web results.")
    p.add_argument("--web-fetch", type=int, default=3, help="(Reserved) Fetch count per query.")

    return p


def _coalesce(override: Any, fallback: Any) -> Any:
    return fallback if override is None else override


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    root = _repo_root()

    try:
        defaults, model_map, schemas = _load_configs(root)
    except Exception as e:
        print(f"[error] Config load failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Pull defaults (from config/defaults.json)
    # Only set fields that are expected to exist on RunConfig.
    # (If you later add fields to RunConfig, add them here too.)
    cfg = RunConfig(
        question=str(args.question),

        mode=_coalesce(args.mode, defaults.get("mode", "general")),
        num_predict=int(_coalesce(args.num_predict, defaults.get("num_predict", 4096))),
        temperature=float(_coalesce(args.temperature, defaults.get("temperature", 0.2))),
        top_p=float(_coalesce(args.top_p, defaults.get("top_p", 0.9))),
        timeout=int(_coalesce(args.timeout, defaults.get("timeout", 260))),
        retries=int(_coalesce(args.retries, defaults.get("retries", 1))),
        max_rounds=int(_coalesce(args.max_rounds, defaults.get("max_rounds", 3))),

        stream=bool(args.stream or defaults.get("stream", False)),
        stream_only_final=bool(args.stream_only_final or defaults.get("stream_only_final", False)),

        base_url=str(_coalesce(args.base_url, defaults.get("base_url", "http://127.0.0.1:11434"))),

        runs_dir=str(_coalesce(args.runs_dir, defaults.get("runs_dir", "runs"))),
        cache_dir=str(_coalesce(args.cache_dir, defaults.get("cache_dir", "cache"))) if _coalesce(args.cache_dir, defaults.get("cache_dir", "cache")) else None,
        transcripts_dir=str(_coalesce(args.transcripts_dir, defaults.get("transcripts_dir", "transcripts"))) if _coalesce(args.transcripts_dir, defaults.get("transcripts_dir", "transcripts")) else None,
        log_file=str(_coalesce(args.log, defaults.get("log_file", "mad_log.json"))),

        concise=bool(args.concise or defaults.get("concise", False)),
        parallel=bool(args.parallel or defaults.get("parallel", False)),
        a_samples=int(_coalesce(args.a_samples, defaults.get("a_samples", 1))),

        save_calls=not bool(args.no_save_calls) and bool(defaults.get("save_calls", True)),
        save_dataset=not bool(args.no_save_dataset) and bool(defaults.get("save_dataset", True)),
        print_quality_report=not bool(args.no_quality_report) and bool(defaults.get("print_quality_report", True)),
        redact_secrets=not bool(args.no_redact_secrets) and bool(defaults.get("redact_secrets", True)),

         web=bool(args.web),
    searxng_url=str(args.searxng_url),
    web_topk=int(args.web_topk),
    web_fetch=int(args.web_fetch),
    )

    # Run
    try:
        run_mad(cfg, model_map, schemas)
    except KeyboardInterrupt:
        print("\n[info] Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[error] MAD run failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
