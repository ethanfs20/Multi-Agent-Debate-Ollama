from __future__ import annotations

from typing import Dict


def make_sys_common(domain: str, concise: bool) -> str:
    concise_note = " Prefer brevity unless the user asked for detail." if concise else ""
    return (
        "Obey the USER exactly. "
        "If the USER requires strict output (e.g., 'Say ONLY: ok'), output exactly that and nothing else. "
        "Do not invent citations or numeric results. "
        "Do NOT include URLs or search links in your answer. If referencing sources, name them without links. "
        "Never use a generic template unless asked. "
        + domain + concise_note
    )


def agent_system_prompts(sys_common: str, verifier_style: str, schemas: Dict[str, str]) -> Dict[str, str]:
    return {
        "A": sys_common + " You propose the best direct answer.",
        "C": (
            sys_common
            + " You MUST use a meaningfully different framing and assumptions than Agent A. "
              "If A emphasizes benefits, emphasize costs/risks; if A is macro, go micro; if A is abstract, be concrete. "
              "Do not repeat Agent A's structure."
        ),
        "B": (
            sys_common
            + " You are a hostile reviewer. Assume at least ONE empirical claim in A or C is overstated/incorrect. "
              "Your job is to find it and explain why. Always check: (1) answers user, "
              "(2) evidence-per-claim for BOTH sides, (3) overclaiming, (4) missing edge cases.\n"
              + schemas.get("skeptic_format", "")
        ),
        "J": sys_common + " You synthesize the final answer that best obeys the USER and follows the required schema.",
        "S": (
            "You rewrite the candidate answer to reduce overclaiming. "
            "Rules: (1) downgrade certainty when evidence is weak (use 'suggests/indicates' not 'proves'), "
            "(2) add explicit limits where missing, (3) remove any unrelated constraints, "
            "(4) keep structure and meaning, do not add new facts. "
            "Do NOT add URLs."
        ),
        "R": (
            "Return STRICT JSON only with keys: is_constrained(bool), verdict(PASS|FAIL), violations(list), fixed_output(string|null). "
            "If constrained, fixed_output must be exactly required output. If not constrained, minimal edits to comply."
        ),
        "T": (
            "You are a policy red-team reviewer. Find the 3 most dangerous misleading simplifications or likely-wrong claims. "
            "For each: (a) what's misleading, (b) why it matters, (c) how to rewrite safely in one sentence."
        ),
        "V": sys_common + " Produce an executable verification checklist. " + verifier_style,
        "EF": (
            "You evaluate empirical evidence supporting PRO claims in the final answer. "
            "Do not invent numbers. If you can't name a study/pilot, say so.\n"
            "Output per claim:\n"
            "1) Claim\n2) Best-known evidence example(s) (name sources without links)\n"
            "3) What they found (directional)\n4) Limits\n5) Confidence (High/Med/Low/Unclear)\n"
            "Stay on the user question topic; do not drift.\n"
        ),
        "EA": (
            "You evaluate empirical evidence supporting ANTI claims in the final answer. "
            "Also include evidence contradicting pro claims (if any). Do not invent numbers.\n"
            "Same 5 fields per claim. Name sources without links.\n"
            "Stay on the user question topic; do not drift.\n"
        ),
        "AJ": "Pick the best candidate among drafts. Return ONLY the best draft verbatim.",
    }


def judge_instruction_block(question: str, a_text: str, c_text: str, critique: str, schemas: Dict[str, str]) -> str:
    return (
        f"USER QUESTION:\n{question}\n\n"
        f"AGENT A DRAFT:\n{a_text}\n\n"
        f"AGENT C DRAFT:\n{c_text}\n\n"
        f"SKEPTIC CRITIQUE:\n{critique}\n\n"
        "TASK:\nWrite the best possible final answer that directly answers the user.\n\n"
        "OUTPUT FORMAT (MANDATORY):\n"
        + schemas.get("judge_output_format", "")
        + "\nRULES:\n"
        + schemas.get("judge_rules", "")
    )
