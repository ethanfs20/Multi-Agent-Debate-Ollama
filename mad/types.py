from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    model: str
    system: str


@dataclass(frozen=True)
class CallResult:
    text: str
    elapsed_s: float
    cache_hit: bool
    model_used: str


@dataclass
class RunConfig:
    question: str
    mode: str = "general"
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 260
    retries: int = 1
    max_rounds: int = 3
    concise: bool = False
    num_predict: int = 260
    temperature: float = 0.2
    top_p: float = 0.9
    stream: bool = False
    stream_only_final: bool = False
    parallel: bool = False
    a_samples: int = 1
    cache_dir: Optional[str] = "cache"
    transcripts_dir: Optional[str] = "transcripts"
    log_file: str = "mad_log.json"
    json_output: bool = False

    # NEW
    runs_dir: str = "runs"
    save_calls: bool = True
    save_dataset: bool = True
    print_quality_report: bool = True
    redact_secrets: bool = True

    # model overrides
    model_a: str = "llama3.1:latest"
    model_b: str = "qwen2.5:latest"
    model_c: str = "mistral:latest"
    model_j: str = "llama3.1:latest"
    model_r: str = "qwen2.5:latest"
    model_v: str = "qwen2.5:latest"
    model_e: str = "qwen2.5:latest"
    fallback_c: Optional[str] = "qwen2.5:latest"

    # paths to config files (optional)
    defaults_path: Optional[str] = None
    agents_path: Optional[str] = None
    schemas_path: Optional[str] = None
    # --- WEB RESEARCH ---
    web: bool = False
    searxng_url: str = "http://127.0.0.1:8080"
    web_topk: int = 8
    web_fetch: int = 3
