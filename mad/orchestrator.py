# mad/orchestrator.py
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:
    ThreadPoolExecutor = None  # type: ignore

from .types import CallResult, RunConfig
from .utils import (
    build_prompt,
    critique_has_evidence_flags,
    extract_json_object,
    is_strict_constraint_question,
    safe_filename,
    utc_now_iso,
    get_host_env_info,
)
from .cache import cache_get, cache_key, cache_put
from .ollama_http import generate, generate_stream
from .prompts import agent_system_prompts, judge_instruction_block, make_sys_common
from .utils import mode_preset
from .agents import build_agents
from .quality import assess_quality, summarize_quality, format_quality_report
from .recorder import RunRecorder

# NEW: web research (SearXNG -> SOURCES block)
from .web_research import build_evidence_pack


def _tpath(transcripts_dir: Optional[str], label: str) -> Optional[str]:
    if not transcripts_dir:
        return None
    os.makedirs(transcripts_dir, exist_ok=True)
    return os.path.join(transcripts_dir, f"{safe_filename(label)}.txt")


def _max_num_predict() -> int:
    """
    Single safety cap for num_predict across the whole run.
    Override with env var:
        MAD_MAX_NUM_PREDICT=16384
    """
    raw = os.getenv("MAD_MAX_NUM_PREDICT", "8192").strip()
    try:
        v = int(raw)
    except Exception:
        v = 8192
    # sane bounds
    return max(256, min(v, 200000))


def _cap_num_predict(n: int) -> int:
    return max(16, min(int(n), _max_num_predict()))


def run_call(
    *,
    run_id: str,
    stage_label: str,
    base_url: str,
    model: str,
    fallback_model: Optional[str],
    prompt: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    top_p: float,
    retries: int,
    backoff: float,
    stream: bool,
    label: str,
    transcript_path: Optional[str],
    quiet_stream: bool,
    cache_dir: Optional[str],
    constrained_run: bool,
) -> CallResult:
    t0 = time.time()

    key = None
    if cache_dir:
        key = cache_key(run_id, stage_label, model, prompt, num_predict, temperature, top_p)
        hit = cache_get(cache_dir, key)
        if hit is not None:
            return CallResult(text=hit, elapsed_s=time.time() - t0, cache_hit=True, model_used=model)

    last_err: Optional[Exception] = None

    def attempt(m: str) -> str:
        if stream:
            return generate_stream(
                base_url,
                m,
                prompt,
                timeout,
                num_predict,
                temperature,
                top_p,
                label=label,
                transcript_path=transcript_path,
                quiet=quiet_stream,
            )
        return generate(base_url, m, prompt, timeout, num_predict, temperature, top_p)

    for i in range(retries + 1):
        try:
            out = attempt(model)
            if cache_dir and key and (not constrained_run):
                cache_put(cache_dir, key, out)
            return CallResult(text=out, elapsed_s=time.time() - t0, cache_hit=False, model_used=model)
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (2**i))

    if fallback_model:
        try:
            out = attempt(fallback_model)
            if cache_dir and (not constrained_run):
                key2 = cache_key(run_id, stage_label, fallback_model, prompt, num_predict, temperature, top_p)
                cache_put(cache_dir, key2, out)
            return CallResult(text=out, elapsed_s=time.time() - t0, cache_hit=False, model_used=fallback_model)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Call failed for {label} (model={model}). Last error: {last_err}")


def run_mad(config: RunConfig, model_map: Dict[str, str], schemas: Dict[str, str]) -> Dict[str, Any]:
    question = config.question.strip()
    run_id = uuid.uuid4().hex
    constrained_run = is_strict_constraint_question(question)

    recorder = RunRecorder(config.runs_dir, run_id, redact=config.redact_secrets)

    # === NEW: Web research evidence pack (SearXNG) ===
    # IMPORTANT: do NOT add evidence to strict "say only" / constrained runs
    evidence_block = ""
    if getattr(config, "web", False) and not constrained_run:
        print("[*] Web research (SearXNG)...", flush=True)
        evidence_block = build_evidence_pack(
            question=question,
            searxng_url=getattr(config, "searxng_url", "http://127.0.0.1:8080"),
            topk=int(getattr(config, "web_topk", 8)),
            timeout=int(getattr(config, "timeout", 260)),
        )

    evidence_suffix = ("\n\n" + evidence_block) if evidence_block else ""
    question_with_evidence = question + evidence_suffix

    domain, verifier_style = mode_preset(config.mode)
    sys_common = make_sys_common(domain, config.concise)

    systems = agent_system_prompts(sys_common, verifier_style, schemas)
    agents = build_agents(model_map, systems)

    timings: Dict[str, Any] = {"steps": []}

    def call_agent(
        agent_key: str,
        user_text: str,
        label: str,
        stream_ok: bool,
        round_num: int,
        fallback_model: Optional[str] = None,
        override_num_predict: Optional[int] = None,
        override_temperature: Optional[float] = None,
        override_top_p: Optional[float] = None,
        cache_ok: bool = True,
        quiet_stream: bool = False,
    ) -> CallResult:
        a = agents[agent_key]
        full_prompt = build_prompt(a.system, user_text)

        effective_np = _cap_num_predict(
            override_num_predict if override_num_predict is not None else config.num_predict
        )
        effective_temp = override_temperature if override_temperature is not None else config.temperature
        effective_top_p = override_top_p if override_top_p is not None else config.top_p

        ts_start = utc_now_iso()
        res = run_call(
            run_id=run_id,
            stage_label=label,
            base_url=config.base_url,
            model=a.model,
            fallback_model=fallback_model,
            prompt=full_prompt,
            timeout=config.timeout,
            num_predict=effective_np,
            temperature=effective_temp,
            top_p=effective_top_p,
            retries=config.retries,
            backoff=1.5,
            stream=stream_ok,
            label=label,
            transcript_path=_tpath(config.transcripts_dir, label),
            quiet_stream=quiet_stream,
            cache_dir=(config.cache_dir if (cache_ok and not constrained_run) else None),
            constrained_run=constrained_run,
        )
        ts_end = utc_now_iso()

        timings["steps"].append(
            {
                "label": label,
                "agent": agent_key,
                "model": res.model_used,
                "seconds": round(res.elapsed_s, 3),
                "cache_hit": res.cache_hit,
            }
        )

        qflags = assess_quality(
            agent_key=agent_key,
            stage_label=label,
            question=question,
            constrained_run=constrained_run,
            output=res.text,
        )

        # Full call record for training/debug
        record = {
            "run_id": run_id,
            "round": round_num,
            "stage_label": label,
            "agent_key": agent_key,
            "agent_name": a.name,
            "model_requested": a.model,
            "model_used": res.model_used,
            "fallback_model": fallback_model,
            "params": {
                "num_predict": effective_np,
                "temperature": effective_temp,
                "top_p": effective_top_p,
                "timeout": config.timeout,
                "retries": config.retries,
                "max_num_predict_cap": _max_num_predict(),
            },
            "cache": {"enabled": bool(config.cache_dir and cache_ok and not constrained_run), "hit": res.cache_hit},
            "timing": {"ts_start": ts_start, "ts_end": ts_end, "elapsed_s": round(res.elapsed_s, 6)},
            "prompts": {
                "system": a.system,
                "user": user_text,
                "full_prompt": full_prompt,
            },
            "output": {"text": res.text},
            "quality": qflags.__dict__,
        }

        recorder.record_call(record, save_calls=config.save_calls, save_dataset=config.save_dataset)
        return res

    print("[*] Starting policy-grade MAD (HTTP)...", flush=True)

    # Warm-up
    _ = call_agent(
        "A",
        "Reply with exactly: READY",
        "Warm-up",
        config.stream and not config.stream_only_final,
        round_num=0,
        override_num_predict=16,
        override_temperature=0.0,
        override_top_p=1.0,
        cache_ok=False,
    )

    def a_best(prompt_user: str, label_prefix: str, round_num: int) -> str:
        n = max(1, min(int(config.a_samples), 5))
        if n == 1:
            return call_agent(
                "A",
                prompt_user,
                f"{label_prefix} (A)",
                config.stream and not config.stream_only_final,
                round_num=round_num,
            ).text

        cands: List[str] = []
        for i in range(n):
            r = call_agent(
                "A",
                prompt_user,
                f"{label_prefix} (A sample {i+1}/{n})",
                False,
                round_num=round_num,
                override_temperature=max(0.35, config.temperature),
                quiet_stream=True,
            )
            cands.append(r.text)

        pick_in = "USER:\n" + question + "\n\n" + "\n\n---\n\n".join([f"[{i+1}]\n{t}" for i, t in enumerate(cands)])
        pick = call_agent(
            "AJ",
            pick_in,
            f"{label_prefix} (A pick)",
            config.stream and not config.stream_only_final,
            round_num=round_num,
            override_num_predict=_cap_num_predict(config.num_predict + 60),
        )
        return pick.text

    a_text = c_text = critique = judge_draft = sanitized = final_fixed = ""
    referee_obj: Optional[Dict[str, Any]] = None
    redteam_text = verifier_text = ef_text = ea_text = ""

    max_rounds = max(1, min(int(config.max_rounds), 5))
    clean = False

    for rnd in range(1, max_rounds + 1):
        print(f"[*] Round {rnd}/{max_rounds}", flush=True)

        if rnd == 1:
            a_user = question_with_evidence
            c_user = question_with_evidence
        else:
            a_user = (
                f"Question:\n{question}\n\n"
                f"Evidence:\n{evidence_block or '(none)'}\n\n"
                f"Prior:\n{a_text}\n\nCritique:\n{critique}\n\nRevise to fix issues."
            )
            c_user = (
                f"Question:\n{question}\n\n"
                f"Evidence:\n{evidence_block or '(none)'}\n\n"
                f"Prior:\n{c_text}\n\nCritique:\n{critique}\n\nRevise with DIFFERENT framing than A."
            )

        do_stream_ac = config.stream and not config.stream_only_final

        if config.parallel and ThreadPoolExecutor is not None:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {
                    ex.submit(a_best, a_user, f"Round {rnd}", rnd): "A",
                    ex.submit(
                        lambda: call_agent(
                            "C",
                            c_user,
                            f"Round {rnd} (C)",
                            do_stream_ac,
                            round_num=rnd,
                            fallback_model=model_map.get("fallback_c"),
                        ).text
                    ): "C",
                }
                out: Dict[str, str] = {}
                for f in as_completed(futs):
                    out[futs[f]] = f.result()
            a_text = out["A"]
            c_text = out["C"]
        else:
            a_text = a_best(a_user, f"Round {rnd}", rnd)
            c_text = call_agent(
                "C",
                c_user,
                f"Round {rnd} (C)",
                do_stream_ac,
                round_num=rnd,
                fallback_model=model_map.get("fallback_c"),
            ).text

        print("[*] Skeptic critique...", flush=True)
        crit_in = (
            f"USER question:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"A:\n{a_text}\n\nC:\n{c_text}\n\n"
            "Start with: VERDICT: CLEAN or VERDICT: ISSUES\n"
            "Next: DIVERSITY: OK or DIVERSITY: BAD\n"
            "Then bullet issues."
        )
        critique = call_agent(
            "B",
            crit_in,
            f"Round {rnd} (B critique)",
            config.stream and not config.stream_only_final,
            round_num=rnd,
            override_num_predict=_cap_num_predict(config.num_predict + 140),
        ).text

        verdict_clean = bool(re.search(r"VERDICT:\s*CLEAN", critique, re.I))
        diversity_bad = bool(re.search(r"DIVERSITY:\s*BAD", critique, re.I))
        evidence_flags = critique_has_evidence_flags(critique)
        clean = verdict_clean and (not diversity_bad) and (not evidence_flags)

        print("[*] Judge synthesis...", flush=True)
        j_in = judge_instruction_block(
            question_with_evidence,
            a_text,
            c_text,
            critique,
            schemas,
        )
        judge_draft = call_agent(
            "J",
            j_in,
            f"Round {rnd} (Judge)",
            config.stream,
            round_num=rnd,
            override_num_predict=_cap_num_predict(config.num_predict + 240),
        ).text

        print("[*] Sanitizer (no overclaim)...", flush=True)
        s_in = (
            f"USER QUESTION:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Candidate answer:\n{judge_draft}\n\nRewrite per rules."
        )
        sanitized = call_agent(
            "S",
            s_in,
            f"Round {rnd} (Sanitizer)",
            config.stream and not config.stream_only_final,
            round_num=rnd,
            override_num_predict=_cap_num_predict(config.num_predict + 240),
            override_temperature=0.1,
            override_top_p=0.95,
        ).text

        print("[*] Referee compliance...", flush=True)
        r_in = (
            f"USER request:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Candidate answer:\n{sanitized}\n\nReturn STRICT JSON only."
        )
        r_res = call_agent(
            "R",
            r_in,
            f"Round {rnd} (Referee)",
            False,
            round_num=rnd,
            override_num_predict=_cap_num_predict(280),
            override_temperature=0.0,
            override_top_p=1.0,
            quiet_stream=True,
        ).text

        referee_obj = extract_json_object(r_res) or {
            "is_constrained": False,
            "verdict": "PASS",
            "violations": ["Referee JSON parse failed; using Sanitizer output."],
            "fixed_output": sanitized,
        }

        fixed = referee_obj.get("fixed_output")
        final_fixed = fixed.strip() if isinstance(fixed, str) and fixed.strip() else sanitized.strip()

        if not final_fixed or final_fixed.strip().lower() == "none":
            final_fixed = sanitized.strip() or judge_draft.strip()

        if str(referee_obj.get("verdict", "")).upper() == "FAIL":
            clean = False

        if clean:
            break

    is_constrained = bool(referee_obj and referee_obj.get("is_constrained"))

    # Post-final extras (unless constrained)
    if not is_constrained:
        print("[*] Red Team...", flush=True)
        t_in = (
            f"USER QUESTION:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Final answer:\n{final_fixed}\n\nFind the 3 most dangerous simplifications."
        )
        redteam_text = call_agent(
            "T",
            t_in,
            "Red Team",
            config.stream and not config.stream_only_final,
            round_num=max_rounds + 1,
            override_num_predict=_cap_num_predict(config.num_predict + 140),
            override_temperature=0.15,
            override_top_p=0.95,
        ).text.strip()

        print("[*] Verifier...", flush=True)
        v_in = (
            f"USER:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Final:\n{final_fixed}\n\n"
            "Provide an executable verification checklist. Each item must include:\n"
            "- What to check\n- Where/how to check\n- What outcome would confirm vs weaken\n"
        )
        verifier_text = call_agent(
            "V",
            v_in,
            "Verifier",
            config.stream and not config.stream_only_final,
            round_num=max_rounds + 1,
            override_num_predict=_cap_num_predict(config.num_predict + 140),
            override_temperature=0.1,
            override_top_p=0.95,
        ).text.strip()

        print("[*] Evidence (FOR)...", flush=True)
        ef_in = (
            f"USER QUESTION:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Final answer:\n{final_fixed}\n\n"
            "Extract the PRO-side claims and rate evidence per claim. Stay on-topic."
        )
        ef_text = call_agent(
            "EF",
            ef_in,
            "Evidence FOR",
            config.stream and not config.stream_only_final,
            round_num=max_rounds + 1,
            override_num_predict=_cap_num_predict(config.num_predict + 240),
            override_temperature=0.1,
            override_top_p=0.95,
        ).text.strip()

        print("[*] Evidence (AGAINST)...", flush=True)
        ea_in = (
            f"USER QUESTION:\n{question}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Final answer:\n{final_fixed}\n\n"
            "Extract the ANTI-side claims and rate evidence per claim (including evidence contradicting pro claims if any). Stay on-topic."
        )
        ea_text = call_agent(
            "EA",
            ea_in,
            "Evidence AGAINST",
            config.stream and not config.stream_only_final,
            round_num=max_rounds + 1,
            override_num_predict=_cap_num_predict(config.num_predict + 240),
            override_temperature=0.1,
            override_top_p=0.95,
        ).text.strip()

    out = final_fixed
    if redteam_text:
        out += "\n\nRed Team:\n" + redteam_text
    if verifier_text:
        out += "\n\nVerification:\n" + verifier_text
    if ef_text:
        out += "\n\nEvidence (FOR):\n" + ef_text
    if ea_text:
        out += "\n\nEvidence (AGAINST):\n" + ea_text

    total = sum(s.get("seconds", 0.0) for s in timings["steps"])

    # Quality summary from recorded calls
    call_records = recorder.get_call_records()
    quality_summary = summarize_quality(call_records)
    recorder.write_quality(quality_summary)

    if config.print_quality_report:
        print(format_quality_report(quality_summary), flush=True)

    run_summary: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "question": question,
        "constrained_run": constrained_run,
        "final": final_fixed,
        "referee": referee_obj,
        "timings": timings,
        "total_step_seconds": round(total, 3),
        "models": model_map,
        "config": {
            "mode": config.mode,
            "timeout": config.timeout,
            "retries": config.retries,
            "max_rounds": config.max_rounds,
            "num_predict": config.num_predict,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": config.stream,
            "stream_only_final": config.stream_only_final,
            "parallel": config.parallel,
            "a_samples": config.a_samples,
            "save_calls": config.save_calls,
            "save_dataset": config.save_dataset,
            "redact_secrets": config.redact_secrets,
            "max_num_predict_cap": _max_num_predict(),
            # NEW (web)
            "web": bool(getattr(config, "web", False)),
            "searxng_url": str(getattr(config, "searxng_url", "")),
            "web_topk": int(getattr(config, "web_topk", 0)),
            "web_fetch": int(getattr(config, "web_fetch", 0)),
        },
        "env": get_host_env_info(),
        "paths": {
            "run_dir": recorder.location(),
            "calls_jsonl": os.path.join(recorder.location(), "calls.jsonl"),
            "dataset_jsonl": os.path.join(recorder.location(), "dataset.jsonl"),
            "quality_json": os.path.join(recorder.location(), "quality.json"),
        },
        "quality_summary": quality_summary,
    }
    recorder.write_run_summary(run_summary)

    # also write your legacy top-level log_file (kept)
    try:
        with open(config.log_file, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("[warn] Could not write log:", e)

    return {
        "run_id": run_id,
        "final": final_fixed,
        "combined_output": out,
        "constrained": is_constrained,
        "timings": timings,
        "log_file": config.log_file,
        "total_step_seconds": round(total, 3),
        "run_dir": recorder.location(),
        "quality_summary": quality_summary,
    }
