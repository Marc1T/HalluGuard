"""
Experimentation Variante C — M1 + M2 + MCP + memoire persistante.

Etapes :
  1. Pipeline Variante C sur 3 questions (NLI inter-sources + BeliefState + appels MCP)
  2. Demonstration des 4 tools MCP (add_fact, get_belief_state, verify_claim, check_temporal)
  3. Test de persistance : Session 1 -> Q1 -> sauvegarde -> Session 2 -> Q2 -> verification

Lancer : python src/experiments/variantC.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASELINE_FILE = ROOT / "results" / "baseline_latency.json"
VARIANT_B_FILE = ROOT / "results" / "variantB_results.json"
RESULTS_FILE  = ROOT / "results" / "variantC_results.json"
LOG_PATH      = str(ROOT / "results" / "logs" / "variantC.log")
MCP_LOG       = ROOT / "results" / "logs" / "mcp_calls.log"
SEPARATOR     = "-" * 60

QUESTIONS = [
    "Qu'est-ce qu'un LLM ?",
    "Comment fonctionne le RAG ?",
    "Quelles sont les limites des agents IA ?",
]

SESSION_C = "variantC_exp"


# ------------------------------------------------------------------ #
# Proxy MCP direct (sans subprocess stdio)                             #
# ------------------------------------------------------------------ #

class DirectMCPProxy:
    """
    Appelle les fonctions du serveur MCP directement (sans protocole stdio).
    Partage le verifier deja charge pour eviter un double chargement NLI.
    """

    def __init__(self):
        import importlib
        self._mod = importlib.import_module("src.mcp_servers.halluguard_mcp")

    def inject_verifier(self, verifier) -> None:
        """Injecte le verifier deja charge dans le module MCP pour eviter double-load."""
        self._mod._verifier = verifier

    def call_tool(self, tool_name: str, args: dict):
        fn = getattr(self._mod, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool MCP inconnu : {tool_name}")
        return fn(**args)


# ------------------------------------------------------------------ #
# Etape 1 — Pipeline Variante C (session complete)                     #
# ------------------------------------------------------------------ #

def run_variant_c(session_id: str, questions: list[str]) -> tuple[list[dict], object]:
    """Construit et lance le pipeline Variante C. Retourne (results, middleware)."""
    mcp_proxy = DirectMCPProxy()

    from src.agents.rag_agent import build_rag_pipeline

    pipeline, middleware = build_rag_pipeline(
        variant="C",
        session_id=session_id,
        policy_mode="log",
    )

    # Passer le proxy MCP au middleware
    middleware.mcp_client = mcp_proxy
    middleware.policy.log_path = Path(LOG_PATH)
    middleware.policy.log_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for q in questions:
        state = {
            "query": q,
            "documents": [],
            "reasoning": "",
            "tool_output": "",
            "answer": "",
            "halluguard_events": [],
        }
        t0 = time.time()
        final = pipeline.invoke(state)
        latency = round(time.time() - t0, 2)

        results.append({
            "query": q,
            "answer": final["answer"],
            "latency_seconds": latency,
            "documents_used": len(final.get("documents", [])),
            "halluguard_events": final.get("halluguard_events", []),
            "variant": "C",
        })

    # Injecter le verifier charge dans le proxy MCP (pour eviter double-load)
    mcp_proxy.inject_verifier(middleware.verifier)

    # Sauvegarder la session
    middleware.save_session()

    return results, middleware, mcp_proxy


# ------------------------------------------------------------------ #
# Etape 2 — Demonstration des 4 tools MCP                             #
# ------------------------------------------------------------------ #

def demonstrate_mcp_tools(mcp_proxy: DirectMCPProxy, sample_answer: str) -> dict:
    """Appelle les 4 tools MCP et retourne les resultats."""
    print("\n  [Tool 1] add_fact")
    r1 = mcp_proxy.call_tool("add_fact", {
        "fact": "Les LLM utilisent l'architecture Transformer avec attention multi-tetes.",
        "confidence": 0.95,
        "step": 1,
    })
    print(f"    -> {r1}")

    print("  [Tool 2] get_belief_state")
    r2 = mcp_proxy.call_tool("get_belief_state", {})
    print(f"    -> session={r2.get('session_id')}, active_facts={r2.get('active_facts')}")

    print("  [Tool 3] verify_claim (NLI deja charge)")
    r3 = mcp_proxy.call_tool("verify_claim", {
        "claim": "The Eiffel Tower is in London.",
        "evidences": ["The Eiffel Tower is located in Paris, France."],
        "node_type": "generation",
    })
    print(f"    -> label={r3.get('label')}, score={r3.get('score')}, type={r3.get('hallucination_type')}")

    print("  [Tool 4] check_temporal_coherence")
    r4 = mcp_proxy.call_tool("check_temporal_coherence", {
        "claim": "The Eiffel Tower is in London.",
    })
    print(f"    -> conflict={r4.get('conflict')}, score={r4.get('score')}")

    return {"add_fact": r1, "get_belief_state": r2, "verify_claim": r3,
            "check_temporal_coherence": r4}


# ------------------------------------------------------------------ #
# Etape 3 — Test de persistance memoire                               #
# ------------------------------------------------------------------ #

def test_persistence() -> dict:
    """
    Simule deux sessions separees :
    - Session 1 : pose Q1, sauvegarde le BeliefState
    - Session 2 : recharge depuis disque, pose Q2, verifie que Q1 est present
    """
    from src.agents.rag_agent import build_rag_pipeline
    from src.memory.persistent_memory import PersistentMemory

    PERSIST_SESSION = "persistence_test"
    mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))

    # ---- Session 1 : Q1 ----
    print("\n  [Session 1] Lancement avec Q1...")
    mem.delete(PERSIST_SESSION)  # partir d'un etat propre

    pipeline1, mw1 = build_rag_pipeline(
        variant="C", session_id=PERSIST_SESSION, policy_mode="log"
    )
    state1 = {
        "query": QUESTIONS[0],
        "documents": [], "reasoning": "", "tool_output": "", "answer": "",
        "halluguard_events": [],
    }
    final1 = pipeline1.invoke(state1)
    mw1.save_session()

    facts_after_q1 = len(mw1.belief.get_active_facts())
    session_file = mem.path_for(PERSIST_SESSION)
    print(f"  -> Reponse generee, {facts_after_q1} faits sauvegardes dans {session_file.name}")

    # ---- Session 2 : recharge depuis disque ----
    print("  [Session 2] Rechargement depuis disque, lancement avec Q2...")
    pipeline2, mw2 = build_rag_pipeline(
        variant="C", session_id=PERSIST_SESSION, policy_mode="log"
    )
    facts_loaded = mw2.belief.get_active_facts()
    print(f"  -> {len(facts_loaded)} faits recharges depuis disque")

    # Verifier que les faits de Q1 sont bien la
    q1_content = final1["answer"][:50].strip()
    facts_contents = [f.content[:50] for f in facts_loaded]

    state2 = {
        "query": QUESTIONS[1],
        "documents": [], "reasoning": "", "tool_output": "", "answer": "",
        "halluguard_events": [],
    }
    final2 = pipeline2.invoke(state2)
    mw2.save_session()

    facts_after_q2 = len(mw2.belief.get_active_facts())
    print(f"  -> Session 2 complete, {facts_after_q2} faits au total")

    persistence_ok = len(facts_loaded) >= facts_after_q1
    print(f"  [PERSISTENCE] {'OK' if persistence_ok else 'ECHEC'} : "
          f"{len(facts_loaded)} faits retrouves apres rechargement (attendu >= {facts_after_q1})")

    return {
        "persistence_ok": persistence_ok,
        "facts_after_q1": facts_after_q1,
        "facts_reloaded": len(facts_loaded),
        "facts_after_q2": facts_after_q2,
        "session_file": str(session_file),
    }


# ------------------------------------------------------------------ #
# Overhead vs A                                                        #
# ------------------------------------------------------------------ #

def compute_overhead(results_c: list[dict], baseline: dict) -> list[dict]:
    bmap = {r["query"]: r["latency_seconds"] for r in baseline["questions"]}
    for r in results_c:
        lat_a = bmap.get(r["query"])
        if lat_a and lat_a > 0:
            r["latency_A_seconds"] = lat_a
            r["overhead_pct"] = round((r["latency_seconds"] - lat_a) / lat_a * 100, 1)
        else:
            r["latency_A_seconds"] = None
            r["overhead_pct"] = None
    return results_c


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - Experimentation Variante C (M1 + M2 + MCP)")
    print("  Middleware complet : NLI + BeliefState + memoire persistante")
    print("=" * 60)

    # Charger les baselines
    if not BASELINE_FILE.exists():
        print(f"ERREUR : {BASELINE_FILE} introuvable.")
        sys.exit(1)
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    print(f"\n  Baseline A : latence moy. {baseline['avg_latency_seconds']}s")

    # ------------------------------------------------------------------ #
    # 1. Pipeline Variante C                                               #
    # ------------------------------------------------------------------ #
    print(f"\n[1/3] Pipeline Variante C sur {len(QUESTIONS)} questions...")
    print(SEPARATOR)

    # Reset log
    Path(LOG_PATH).write_text("", encoding="utf-8")

    results_c, middleware, mcp_proxy = run_variant_c(SESSION_C, QUESTIONS)
    results_c = compute_overhead(results_c, baseline)

    for i, r in enumerate(results_c, 1):
        overhead = r.get("overhead_pct")
        overhead_str = f"+{overhead}%" if overhead and overhead >= 0 else f"{overhead}%"
        print(f"\nQ{i} : {r['query']}")
        preview = r["answer"][:250] + ("..." if len(r["answer"]) > 250 else "")
        print(f"  Reponse  : {preview}")
        print(f"  Latence C: {r['latency_seconds']}s  |  A: {r.get('latency_A_seconds')}s  |  Overhead: {overhead_str}")
        events = r.get("halluguard_events", [])
        if events:
            for ev in events:
                print(f"  -> HalluGuard: node={ev.get('node_id')} | type={ev.get('hallucination_type')} | score={ev.get('score')}")

    # BeliefState apres pipeline
    bs = middleware.belief
    active = bs.get_active_facts()
    print(f"\n  BeliefState : {len(active)} faits actifs apres 3 questions")

    # ------------------------------------------------------------------ #
    # 2. Demonstration 4 tools MCP                                         #
    # ------------------------------------------------------------------ #
    print(f"\n[2/3] Demonstration des 4 tools MCP...")
    print(SEPARATOR)
    sample = results_c[0]["answer"]
    mcp_demo = demonstrate_mcp_tools(mcp_proxy, sample)

    # Log MCP
    if MCP_LOG.exists():
        lines = [l for l in MCP_LOG.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        print(f"\n  mcp_calls.log : {len(lines)} entrees totales")
        for line in lines[-6:]:
            e = json.loads(line)
            print(f"  {e['timestamp']} | {e['tool']:<28} | result={str(e['result'])[:50]}")

    # ------------------------------------------------------------------ #
    # 3. Test de persistance                                               #
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Test de persistance memoire...")
    print(SEPARATOR)
    persistence = test_persistence()

    # ------------------------------------------------------------------ #
    # Sauvegarde resultats                                                 #
    # ------------------------------------------------------------------ #
    latencies = [r["latency_seconds"] for r in results_c]
    overheads = [r["overhead_pct"] for r in results_c if r.get("overhead_pct") is not None]
    stats = middleware.overhead_stats()

    payload = {
        "variant": "C",
        "description": "M1 + M2 + MCP + memoire persistante",
        "questions": results_c,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "avg_overhead_pct": round(sum(overheads) / len(overheads), 1) if overheads else None,
        "middleware_stats": stats,
        "belief_state_facts": len(active),
        "mcp_tools_demo": {k: str(v)[:100] for k, v in mcp_demo.items()},
        "persistence_test": persistence,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  Resultats sauvegardes   : {RESULTS_FILE.name}")
    print(f"  Latence moyenne C       : {payload['avg_latency_seconds']}s")
    print(f"  Overhead moyen vs A     : {payload['avg_overhead_pct']}%")
    print(f"  Persistance memoire     : {'OK' if persistence['persistence_ok'] else 'ECHEC'}")
    print(f"{'='*60}")
