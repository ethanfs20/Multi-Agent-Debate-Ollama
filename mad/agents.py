from __future__ import annotations

from typing import Dict
from .types import AgentSpec


def build_agents(model_map: Dict[str, str], systems: Dict[str, str]) -> Dict[str, AgentSpec]:
    return {
        "A": AgentSpec("A", "Agent A (Proposer)", model_map["model_a"], systems["A"]),
        "C": AgentSpec("C", "Agent C (Alternative)", model_map["model_c"], systems["C"]),
        "B": AgentSpec("B", "Agent B (Skeptic)", model_map["model_b"], systems["B"]),
        "J": AgentSpec("J", "Judge", model_map["model_j"], systems["J"]),
        "S": AgentSpec("S", "Sanitizer (No Overclaim)", model_map["model_b"], systems["S"]),
        "R": AgentSpec("R", "Referee (Compliance)", model_map["model_r"], systems["R"]),
        "T": AgentSpec("T", "Red Team (Policy Risk)", model_map["model_b"], systems["T"]),
        "V": AgentSpec("V", "Verifier", model_map["model_v"], systems["V"]),
        "EF": AgentSpec("EF", "Evidence Rater (FOR)", model_map["model_e"], systems["EF"]),
        "EA": AgentSpec("EA", "Evidence Rater (AGAINST)", model_map["model_e"], systems["EA"]),
        "AJ": AgentSpec("AJ", "A-Sample Judge", model_map["model_j"], systems["AJ"]),
    }
