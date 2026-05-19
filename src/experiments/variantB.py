"""
Experimentation Variante B — M1 uniquement (NLI inter-sources, sans BeliefState).

- Charge les latences baseline depuis results/baseline_latency.json
- Lance le pipeline LangGraph + HalluGuardMiddleware (variant=B) sur 3 questions
- Mesure la latence et l'overhead vs Variante A
- Affiche les evenements HalluGuard loggues
- Sauvegarde dans results/variantB_results.json

Lancer : python src/experiments/variantB.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASELINE_FILE = ROOT / "results" / "baseline_latency.json"
RESULTS_FILE  = ROOT / "results" / "variantB_results.json"
LOG_PATH      = str(ROOT / "results" / "logs" / "variantB.log")
SEPARATOR     = "-" * 60

QUESTIONS = [
    "Qu'est-ce qu'un LLM ?",
    "Comment fonctionne le RAG ?",
    "Quelles sont les limites des agents IA ?",
]


# ------------------------------------------------------------------ #
# Pipeline Variante B                                                   #
# ------------------------------------------------------------------ #

def run_variant_b() -> tuple[list[dict], object]:
    """Lance le pipeline Variante B sur les 3 questions. Retourne (results, middleware)."""
    from src.agents.rag_agent import build_rag_pipeline

    pipeline, middleware = build_rag_pipeline(
        variant="B",
        session_id="variantB_exp",
        policy_mode="log",
    )

    # Injecter le log_path dedie dans la policy du middleware
    middleware.policy.log_path = Path(LOG_PATH)
    middleware.policy.log_path.parent.mkdir(parents=True, exist_ok=True)
    # Reinitialiser le log pour cet experiment
    middleware.policy.log_path.write_text("", encoding="utf-8")

    results = []
    for q in QUESTIONS:
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
            "variant": "B",
        })

    return results, middleware


# ------------------------------------------------------------------ #
# Overhead vs Variante A                                               #
# ------------------------------------------------------------------ #

def compute_overhead(results_b: list[dict], baseline: dict) -> list[dict]:
    """Calcule l'overhead (latB - latA) / latA * 100 pour chaque question."""
    baseline_map = {r["query"]: r["latency_seconds"] for r in baseline["questions"]}
    for r in results_b:
        lat_a = baseline_map.get(r["query"])
        if lat_a and lat_a > 0:
            overhead_pct = round((r["latency_seconds"] - lat_a) / lat_a * 100, 1)
            r["latency_A_seconds"] = lat_a
            r["overhead_pct"] = overhead_pct
        else:
            r["latency_A_seconds"] = None
            r["overhead_pct"] = None
    return results_b


# ------------------------------------------------------------------ #
# Affichage log HalluGuard                                             #
# ------------------------------------------------------------------ #

def show_halluguard_log(log_path: str) -> list[dict]:
    p = Path(log_path)
    if not p.exists() or p.stat().st_size == 0:
        print("  (aucun evenement HalluGuard loggue)")
        return []

    entries = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))

    print(f"  Entrees log : {len(entries)}")
    for e in entries:
        corrected = e.get("corrected", False)
        print(
            f"  {e.get('timestamp')} | noeud={e.get('node_id'):<12} "
            f"| type={e.get('hallucination_type')} "
            f"| score={e.get('score')} "
            f"| corrige={corrected}"
        )
    return entries


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - Experimentation Variante B (M1 uniquement)")
    print("  Middleware : LightweightVerifier NLI inter-sources")
    print("=" * 60)

    # Charger la baseline
    if not BASELINE_FILE.exists():
        print(f"ERREUR : {BASELINE_FILE} introuvable. Lancer baseline_A.py d'abord.")
        sys.exit(1)
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    print(f"\n  Baseline Variante A chargee ({len(baseline['questions'])} questions)")
    print(f"  Latence moyenne A : {baseline['avg_latency_seconds']}s\n")

    # Pipeline
    print("[1/2] Pipeline Variante B (premier appel NLI = chargement modele)...")
    print(SEPARATOR)

    results_b, middleware = run_variant_b()
    results_b = compute_overhead(results_b, baseline)

    # Affichage resultats
    for i, r in enumerate(results_b, 1):
        overhead = r.get("overhead_pct")
        overhead_str = f"+{overhead}%" if overhead and overhead >= 0 else f"{overhead}%"
        print(f"\nQ{i} : {r['query']}")
        preview = r["answer"][:250] + ("..." if len(r["answer"]) > 250 else "")
        print(f"  Reponse    : {preview}")
        print(f"  Latence B  : {r['latency_seconds']}s")
        print(f"  Latence A  : {r.get('latency_A_seconds')}s")
        print(f"  Overhead   : {overhead_str}")
        events = r.get("halluguard_events", [])
        print(f"  Evenements HalluGuard : {len(events)}")
        for ev in events:
            print(f"    -> node={ev.get('node_id')} | type={ev.get('hallucination_type')} | score={ev.get('score')}")

    # Log HalluGuard
    print(f"\n[2/2] Log HalluGuard ({LOG_PATH.split('/')[-1]}) :")
    print(SEPARATOR)
    log_entries = show_halluguard_log(LOG_PATH)

    # Stats middleware
    stats = middleware.overhead_stats()
    print(f"\n  Overhead middleware : {stats['calls']} appels NLI")
    print(f"  Avg overhead/appel : {stats['avg_ms']}ms")
    print(f"  Max overhead/appel : {stats['max_ms']}ms")

    # Sauvegarde
    latencies_b = [r["latency_seconds"] for r in results_b]
    overheads = [r["overhead_pct"] for r in results_b if r.get("overhead_pct") is not None]
    payload = {
        "variant": "B",
        "description": "M1 uniquement — NLI inter-sources, sans BeliefState",
        "questions": results_b,
        "avg_latency_seconds": round(sum(latencies_b) / len(latencies_b), 2),
        "avg_overhead_pct": round(sum(overheads) / len(overheads), 1) if overheads else None,
        "middleware_stats": stats,
        "halluguard_log_entries": log_entries,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"  Resultats sauvegardes : {RESULTS_FILE.name}")
    print(f"  Latence moyenne B     : {payload['avg_latency_seconds']}s")
    print(f"  Overhead moyen vs A   : {payload['avg_overhead_pct']}%")
    print(f"{'='*60}")
