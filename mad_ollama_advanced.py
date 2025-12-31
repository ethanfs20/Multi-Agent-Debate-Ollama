#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:
    ThreadPoolExecutor = None  # type: ignore


@dataclass
class Agent:
    key: str
    name: str
    model: str
    system: str


@dataclass
class CallResult:
    text: str
    elapsed_s: float
    cache_hit: bool
    model_used: str


# -----------------------------
# Helpers
# -----------------------------
def is_strict_constraint_question(q: str) -> bool:
    """
    Detect smoke-test style constraints like: 'Say ONLY: ok' or 'Output ONLY: X'.
    We treat these as constrained runs and avoid cache reuse across them.
    """
    q = (q or "").strip().lower()
    return ("say only:" in q) or ("output only:" in q) or ("respond only:" in q)


def build_prompt(system: str, user: str) -> str:
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return s[:120] if len(s) > 120 else s


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
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


# -----------------------------
# Ollama HTTP
# -----------------------------
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


def ollama_generate(base_url: str, model: str, prompt: str, timeout: int,
                    num_predict: int, temperature: float, top_p: float) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature, "top_p": top_p},
    }
    data = http_post_json(url, payload, timeout=timeout)
    return (data.get("response") or "").strip()


def ollama_generate_stream(base_url: str, model: str, prompt: str, timeout: int,
                           num_predict: int, temperature: float, top_p: float,
                           label: str, transcript_path: Optional[str], quiet: bool) -> str:
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
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
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


# -----------------------------
# Cache
# -----------------------------
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


# -----------------------------
# Core call wrapper
# -----------------------------
def run_call(*, run_id: str, stage_label: str,
             base_url: str, model: str, fallback_model: Optional[str], prompt: str,
             timeout: int, num_predict: int, temperature: float, top_p: float,
             retries: int, backoff: float, stream: bool, label: str,
             transcript_path: Optional[str], quiet_stream: bool,
             cache_dir: Optional[str]) -> CallResult:
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
            return ollama_generate_stream(
                base_url, m, prompt, timeout, num_predict, temperature, top_p,
                label=label, transcript_path=transcript_path, quiet=quiet_stream
            )
        return ollama_generate(base_url, m, prompt, timeout, num_predict, temperature, top_p)

    for i in range(retries + 1):
        try:
            out = attempt(model)
            if cache_dir and key:
                cache_put(cache_dir, key, out)
            return CallResult(text=out, elapsed_s=time.time() - t0, cache_hit=False, model_used=model)
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (2 ** i))

    if fallback_model:
        try:
            out = attempt(fallback_model)
            if cache_dir:
                key2 = cache_key(run_id, stage_label, fallback_model, prompt, num_predict, temperature, top_p)
                cache_put(cache_dir, key2, out)
            return CallResult(text=out, elapsed_s=time.time() - t0, cache_hit=False, model_used=fallback_model)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Call failed for {label} (model={model}). Last error: {last_err}")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Advanced Local MAD (HTTP) for Ollama.")
    ap.add_argument("--question", required=True)
    ap.add_argument("--mode", default="general", choices=["general", "neteng", "bash"])
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--concise", action="store_true")
    ap.add_argument("--num-predict", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--stream-only-final", action="store_true")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--a-samples", type=int, default=1)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--transcripts-dir", default="transcripts")
    ap.add_argument("--log", default="mad_log.json")
    ap.add_argument("--json", action="store_true")

    # Model overrides
    ap.add_argument("--model-a", default="llama3.1:latest")
    ap.add_argument("--model-b", default="qwen2.5:latest")
    ap.add_argument("--model-c", default="mistral:latest")
    ap.add_argument("--model-j", default="llama3.1:latest")
    ap.add_argument("--model-r", default="qwen2.5:latest")
    ap.add_argument("--model-v", default="qwen2.5:latest")
    ap.add_argument("--model-e", default="qwen2.5:latest")
    ap.add_argument("--fallback-c", default="qwen2.5:latest")
    args = ap.parse_args()

    question = args.question.strip()
    run_id = uuid.uuid4().hex
    constrained_run = is_strict_constraint_question(question)

    domain, verifier_style = mode_preset(args.mode)
    concise = " Prefer brevity unless the user asked for detail." if args.concise else ""
    sys_common = (
        "Obey the USER exactly. If they require strict output (e.g., 'Say ONLY: ok'), output exactly that and nothing else. "
        "Never use a generic template unless asked. Do not invent citations or numeric results. "
        + domain + concise
    )

    agents: Dict[str, Agent] = {
        "A": Agent("A", "Agent A (Proposer)", args.model_a, sys_common + " You propose the best direct answer."),
        "B": Agent("B", "Agent B (Skeptic)", args.model_b, sys_common + " You critique compliance/correctness and missing points."),
        "C": Agent(
            "C", "Agent C (Alternative)", args.model_c,
            sys_common
            + " You MUST use a meaningfully different framing and assumptions than Agent A. "
              "If A emphasizes benefits, emphasize costs/risks; if A is macro, go micro; if A is abstract, be concrete. "
              "Do not repeat Agent A's structure."
        ),
        "J": Agent("J", "Judge", args.model_j, sys_common + " You synthesize the final answer that best obeys the USER."),
        "R": Agent(
            "R", "Referee (Compliance)", args.model_r,
            "Return STRICT JSON only with keys: is_constrained(bool), verdict(PASS|FAIL), violations(list), fixed_output(string|null). "
            "If constrained, fixed_output must be exactly required output. If not constrained, minimal edits to comply."
        ),
        "V": Agent("V", "Verifier", args.model_v, sys_common + " Produce a short verification checklist. " + verifier_style),
        "E": Agent(
            "E", "Evidence Rater", args.model_e,
            "Evaluate empirical support strength for the FINAL answer. Do not invent numbers. "
            "Output:\n1) Key claims\n2) Evidence strength per claim (High/Medium/Low/Unclear)\n3) What would raise confidence\n4) Pitfalls"
        ),
        "AJ": Agent("AJ", "A-Sample Judge", args.model_j, "Pick the best candidate among drafts. Return ONLY the best draft verbatim."),
    }

    timings: Dict[str, Any] = {"steps": []}
    cache_dir = (args.cache_dir or "").strip() or None
    transcripts_dir = (args.transcripts_dir or "").strip() or None

    def tpath(label: str) -> Optional[str]:
        if not transcripts_dir:
            return None
        os.makedirs(transcripts_dir, exist_ok=True)
        return os.path.join(transcripts_dir, f"{safe_filename(label)}.txt")

    def call_agent(agent_key: str, user_text: str, label: str, stream_ok: bool,
                   fallback_model: Optional[str] = None,
                   override_num_predict: Optional[int] = None,
                   override_temperature: Optional[float] = None,
                   override_top_p: Optional[float] = None,
                   cache_ok: bool = True,
                   quiet_stream: bool = False) -> CallResult:
        a = agents[agent_key]
        prompt = build_prompt(a.system, user_text)

        res = run_call(
            run_id=run_id,
            stage_label=label,
            base_url=args.base_url,
            model=a.model,
            fallback_model=fallback_model,
            prompt=prompt,
            timeout=args.timeout,
            num_predict=override_num_predict if override_num_predict is not None else args.num_predict,
            temperature=override_temperature if override_temperature is not None else args.temperature,
            top_p=override_top_p if override_top_p is not None else args.top_p,
            retries=args.retries,
            backoff=1.5,
            stream=stream_ok,
            label=label,
            transcript_path=tpath(label),
            quiet_stream=quiet_stream,
            # Never cache constrained smoke tests; prevents "Say ONLY: ok" contamination.
            cache_dir=(cache_dir if (cache_ok and not constrained_run) else None),
        )

        timings["steps"].append({
            "label": label, "agent": agent_key, "model": res.model_used,
            "seconds": round(res.elapsed_s, 3), "cache_hit": res.cache_hit
        })
        return res

    print("[*] Starting MAD (HTTP)...", flush=True)

    # Warm-up (fail fast)
    _ = call_agent(
        "A",
        "Reply with exactly: READY",
        "Warm-up",
        args.stream and not args.stream_only_final,
        override_num_predict=16,
        override_temperature=0.0,
        override_top_p=1.0,
        cache_ok=False,
    )

    def a_best(prompt_user: str, label_prefix: str) -> str:
        n = max(1, min(int(args.a_samples), 5))
        if n == 1:
            return call_agent("A", prompt_user, f"{label_prefix} (A)",
                              args.stream and not args.stream_only_final).text

        cands: List[str] = []
        for i in range(n):
            r = call_agent("A", prompt_user, f"{label_prefix} (A sample {i+1}/{n})", False,
                           override_temperature=max(0.35, args.temperature),
                           quiet_stream=True)
            cands.append(r.text)

        pick_in = "USER:\n" + question + "\n\n" + "\n\n---\n\n".join([f"[{i+1}]\n{t}" for i, t in enumerate(cands)])
        pick = call_agent("AJ", pick_in, f"{label_prefix} (A pick)",
                          args.stream and not args.stream_only_final,
                          override_num_predict=min(260, args.num_predict + 40))
        return pick.text

    a_text = c_text = critique = judge_draft = final_fixed = ""
    referee_obj: Optional[Dict[str, Any]] = None
    verifier_text = evidence_text = ""

    max_rounds = max(1, min(int(args.max_rounds), 5))
    clean = False

    for rnd in range(1, max_rounds + 1):
        print(f"[*] Round {rnd}/{max_rounds}", flush=True)

        if rnd == 1:
            a_user = question
            c_user = question
        else:
            a_user = f"Question:\n{question}\n\nPrior:\n{a_text}\n\nCritique:\n{critique}\n\nRevise to fix issues."
            c_user = f"Question:\n{question}\n\nPrior:\n{c_text}\n\nCritique:\n{critique}\n\nRevise with DIFFERENT framing than A."

        do_stream_ac = args.stream and not args.stream_only_final

        if args.parallel and ThreadPoolExecutor is not None:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {
                    ex.submit(a_best, a_user, f"Round {rnd}"): "A",
                    ex.submit(lambda: call_agent("C", c_user, f"Round {rnd} (C)", do_stream_ac,
                                                 fallback_model=(args.fallback_c or None)).text): "C",
                }
                out = {}
                for f in as_completed(futs):
                    out[futs[f]] = f.result()
            a_text = out["A"]
            c_text = out["C"]
        else:
            a_text = a_best(a_user, f"Round {rnd}")
            c_text = call_agent("C", c_user, f"Round {rnd} (C)", do_stream_ac,
                                fallback_model=(args.fallback_c or None)).text

        print("[*] Skeptic critique...", flush=True)
        crit_in = (
            f"USER question:\n{question}\n\nA:\n{a_text}\n\nC:\n{c_text}\n\n"
            "Start with: VERDICT: CLEAN or VERDICT: ISSUES\n"
            "Next: DIVERSITY: OK or DIVERSITY: BAD\n"
            "Then bullet issues (constraints first)."
        )
        critique = call_agent(
            "B",
            crit_in,
            f"Round {rnd} (B critique)",
            args.stream and not args.stream_only_final,
            override_num_predict=min(320, args.num_predict + 120),
        ).text

        verdict_clean = bool(re.search(r"VERDICT:\s*CLEAN", critique, re.I))
        diversity_bad = bool(re.search(r"DIVERSITY:\s*BAD", critique, re.I))
        clean = verdict_clean and (not diversity_bad)

        print("[*] Judge synthesis...", flush=True)

        # Strong schema: Claim -> Evidence -> Limits
        j_in = (
            f"USER QUESTION:\n{question}\n\n"
            f"AGENT A DRAFT:\n{a_text}\n\n"
            f"AGENT C DRAFT:\n{c_text}\n\n"
            f"SKEPTIC CRITIQUE:\n{critique}\n\n"
            "TASK:\n"
            "Write the best possible final answer that directly answers the user.\n\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "1) Strongest arguments FOR (3–5 bullets)\n"
            "2) Strongest arguments AGAINST (3–5 bullets)\n"
            "3) Empirical evidence map (MANDATORY)\n"
            "   - For EACH bullet above, include:\n"
            "     • Claim: <copy the bullet>\n"
            "     • Evidence: <what real-world pilots/studies exist or what type of evidence exists>\n"
            "     • What they found: <directional summary only; no made-up numbers>\n"
            "     • Limits: <duration/scale/selection/external validity issues>\n"
            "4) What evidence would change the conclusion (3–6 bullets)\n\n"
            "RULES:\n"
            "- Do NOT include or repeat any unrelated constraints (e.g., 'Say ONLY: ok') unless present in the user question above.\n"
            "- If you cannot name a specific study/pilot confidently, say 'Evidence exists but I can’t name it reliably' rather than inventing.\n"
        )

        judge_draft = call_agent(
            "J",
            j_in,
            f"Round {rnd} (Judge)",
            args.stream,
            override_num_predict=min(420, args.num_predict + 200),
        ).text

        print("[*] Referee compliance...", flush=True)
        r_in = f"USER request:\n{question}\n\nCandidate answer:\n{judge_draft}\n\nReturn STRICT JSON only."
        r_res = call_agent(
            "R",
            r_in,
            f"Round {rnd} (Referee)",
            False,
            override_num_predict=280,
            override_temperature=0.0,
            override_top_p=1.0,
            quiet_stream=True,
        ).text

        referee_obj = extract_json_object(r_res) or {
            "is_constrained": False,
            "verdict": "PASS",
            "violations": ["Referee JSON parse failed; using Judge draft."],
            "fixed_output": judge_draft,
        }

        fixed = referee_obj.get("fixed_output")
        final_fixed = fixed.strip() if isinstance(fixed, str) and fixed.strip() else judge_draft.strip()

        # Hard guard against "None"
        if not final_fixed or final_fixed.strip().lower() == "none":
            final_fixed = judge_draft.strip()

        if str(referee_obj.get("verdict", "")).upper() == "FAIL":
            clean = False

        if clean:
            break

    is_constrained = bool(referee_obj and referee_obj.get("is_constrained"))

    if not is_constrained:
        print("[*] Verifier...", flush=True)
        v_in = (
            f"USER:\n{question}\n\nFinal:\n{final_fixed}\n\n"
            "Provide a verification checklist that is executable. "
            "Each item must include: what to check, where to check it, and what outcome would confirm vs weaken the claim."
        )
        verifier_text = call_agent(
            "V",
            v_in,
            "Verifier",
            args.stream and not args.stream_only_final,
            override_num_predict=min(280, args.num_predict + 60),
            override_temperature=0.1,
            override_top_p=0.95,
        ).text.strip()

        print("[*] Evidence weighting...", flush=True)
        e_in = (
            f"USER:\n{question}\n\nFinal:\n{final_fixed}\n\n"
            "Evaluate evidence strength per claim.\n"
            "For each claim, name 1–2 example programs/experiments if you can; if not, say so.\n"
            "Separate 'evidence for mechanism' vs 'evidence at scale'. Do NOT invent numbers.\n"
        )
        evidence_text = call_agent(
            "E",
            e_in,
            "Evidence weighting",
            args.stream and not args.stream_only_final,
            override_num_predict=min(360, args.num_predict + 140),
            override_temperature=0.1,
            override_top_p=0.95,
        ).text.strip()

    out = final_fixed
    if verifier_text:
        out += "\n\nVerification:\n" + verifier_text
    if evidence_text:
        out += "\n\nEvidence weighting:\n" + evidence_text

    run_log = {
        "run_id": run_id,
        "question": question,
        "constrained_run": constrained_run,
        "final": final_fixed,
        "referee": referee_obj,
        "timings": timings,
    }
    try:
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("[warn] Could not write log:", e, file=sys.stderr)

    if args.json:
        print(json.dumps({
            "run_id": run_id,
            "final": final_fixed,
            "verification": verifier_text,
            "evidence_weighting": evidence_text,
            "constrained": is_constrained,
            "timings": timings,
            "log_file": args.log
        }, ensure_ascii=False))
    else:
        print("\n" + out)
        total = sum(s.get("seconds", 0.0) for s in timings["steps"])
        print("\n---\nTiming summary:")
        for s in timings["steps"]:
            hit = " (cache)" if s.get("cache_hit") else ""
            print(f"- {s['label']}: {s['seconds']}s [{s['model']}]{hit}")
        print(f"Total (sum of steps): {round(total, 3)}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
