"""
Serveur MCP HalluGuard - OBLIGATOIRE (exigence prof).

Protocole : stdio (FastMCP Python).
Expose 4 tools :
  - verify_claim(claim, evidences, node_type)  -> {score, label, confidence, hallucination_type}
  - check_temporal_coherence(claim)            -> {conflict, fact_id, score}
  - add_fact(fact, confidence, step)           -> {status, fact_id}
  - get_belief_state()                         -> JSON complet BeliefState

Chaque appel est journalise dans results/logs/mcp_calls.log (JSONL).
"""

from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- path setup pour imports depuis sous-process ---
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP
from src.halluguard.verifier import LightweightVerifier
from src.memory.persistent_memory import PersistentMemory

# ------------------------------------------------------------------ #
# Initialisation                                                        #
# ------------------------------------------------------------------ #

mcp = FastMCP("HalluGuard")

_verifier = LightweightVerifier()                          # chargement NLI paresseux
_mem = PersistentMemory(memory_dir=str(_ROOT / "data" / "memory"))
_belief = _mem.load_or_create("mcp_session")               # rechargement si session existante

_MCP_LOG = _ROOT / "results" / "logs" / "mcp_calls.log"
_MCP_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log(tool: str, args: dict, result: dict) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": tool,
        "args": {k: (v[:80] if isinstance(v, str) else v) for k, v in args.items()},
        "result": result,
    }
    with _MCP_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ #
# Tool 1 : verify_claim                                                 #
# ------------------------------------------------------------------ #

@mcp.tool()
def verify_claim(
    claim: str,
    evidences: Optional[list] = None,
    node_type: str = "generation",
) -> dict:
    """
    Verifie si claim est coherent avec les evidences (M1 NLI).

    Returns: score, label ("correct"|"hallucinated"), confidence,
             hallucination_type (T1-T5|null), insufficient_evidence, latency_ms.
    """
    result = _verifier.verify(claim, evidences or [], node_type=node_type)
    _log("verify_claim", {"claim": claim, "n_evidences": len(evidences or []), "node_type": node_type}, result)
    return result


# ------------------------------------------------------------------ #
# Tool 2 : check_temporal_coherence                                     #
# ------------------------------------------------------------------ #

@mcp.tool()
def check_temporal_coherence(claim: str) -> dict:
    """
    Verifie la coherence de claim contre le BeliefState (M2).

    Returns: conflict (bool), fact_id (str|null), score (float).
    """
    result = _belief.check_temporal_coherence(claim, verifier=_verifier)
    _log("check_temporal_coherence", {"claim": claim}, result)
    return result


# ------------------------------------------------------------------ #
# Tool 3 : add_fact                                                     #
# ------------------------------------------------------------------ #

@mcp.tool()
def add_fact(fact: str, confidence: float = 0.9, step: int = 1) -> dict:
    """
    Ajoute un fait valide dans le BeliefState persistant.

    Returns: status ("added"|"evicted_oldest"), fact_id.
    """
    before = len(_belief)
    fact_id = _belief.add_fact(content=fact, confidence=confidence, step=step)
    after = len(_belief)
    status = "evicted_oldest" if after <= before else "added"
    _mem.save(_belief)
    result = {"status": status, "fact_id": fact_id}
    _log("add_fact", {"fact": fact, "confidence": confidence, "step": step}, result)
    return result


# ------------------------------------------------------------------ #
# Tool 4 : get_belief_state                                             #
# ------------------------------------------------------------------ #

@mcp.tool()
def get_belief_state() -> dict:
    """
    Retourne le BeliefState complet de la session MCP courante.

    Returns: session_id, active_facts, max_facts, facts list.
    """
    summary = _belief.summary()
    summary["facts"] = [
        {
            "fact_id": f.fact_id,
            "content": f.content,
            "confidence": f.confidence,
            "step": f.step,
            "status": f.status,
        }
        for f in _belief.get_active_facts()
    ]
    _log("get_belief_state", {}, {"session_id": summary["session_id"], "active_facts": summary["active_facts"]})
    return summary


# ------------------------------------------------------------------ #
# Point d'entree stdio                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mcp.run(transport="stdio")
