"""
demo_pipeline.py — Demonstration automatique de bout en bout.

Lance 3 verifications via le protocole A2A (RAGAgent -> HalluGuard).
Ne necessite pas Mistral/Ollama — utilise le verifier NLI directement.
Appelé par demo.bat (etape [3/5]).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agents.a2a_protocol import (
    HALLUGUARD_AGENT_CARD,
    RAG_AGENT_CARD,
    delegate_verification,
)
from src.halluguard.belief_state import BeliefState
from src.memory.persistent_memory import PersistentMemory


def _reset_mcp_belief() -> None:
    """Remet a zero le BeliefState du serveur MCP pour la demo."""
    try:
        import importlib
        mod = importlib.import_module("src.mcp_servers.halluguard_mcp")
        mod._belief = BeliefState(session_id="demo_mcp_session")
    except Exception:
        pass

# ------------------------------------------------------------------ #
# 3 questions de demonstration                                          #
# ------------------------------------------------------------------ #

DEMO_CASES = [
    {
        "id": 1,
        "question": "Quelle est l'architecture des LLM modernes ?",
        "claim": (
            "Les LLM modernes utilisent l'architecture CNN (Convolutional Neural Network) "
            "pour traiter le texte sequence par sequence."
        ),
        "evidences": [
            "Les LLM utilisent l'architecture Transformer avec mecanisme d'attention multi-tetes "
            "(Vaswani et al., 2017, Google Brain).",
            "GPT-3 est base sur le Transformer decodeur avec 96 couches d'attention et 175 milliards de parametres.",
        ],
        "node_type": "generation",
        "expected": "hallucinated (T3 — reasoning)",
    },
    {
        "id": 2,
        "question": "Quand a ete publie GPT-3 et par qui ?",
        "claim": (
            "GPT-3 a ete publie en 2018 avec 13 milliards de parametres par Google DeepMind."
        ),
        "evidences": [
            "GPT-3 a ete publie en mai 2020 par OpenAI avec 175 milliards de parametres.",
            "GPT-3 a ete entraine sur 570 Go de texte internet pour un cout de 4,6 millions de dollars.",
        ],
        "node_type": "generation",
        "expected": "hallucinated (T4 — causale/generation)",
    },
    {
        "id": 3,
        "question": "Qu'est-ce que le RAG et comment reduit-il les hallucinations ?",
        "claim": (
            "Le RAG (Retrieval-Augmented Generation) reduit les hallucinations de 30 a 60 pourcent "
            "en ancrant la generation dans des documents verifiables recuperes par ChromaDB."
        ),
        "evidences": [
            "Lewis et al. (2020, NeurIPS, Facebook AI Research) introduisent le RAG.",
            "Le RAG reduit les hallucinations de 30 a 60% selon le domaine en ancrant "
            "la generation dans des documents recuperes verifiables.",
            "ChromaDB utilise l'indexation HNSW avec un rappel superieur a 95%.",
        ],
        "node_type": "retrieval",
        "expected": "correct",
    },
]

# ------------------------------------------------------------------ #
# Affichage                                                             #
# ------------------------------------------------------------------ #

def sep(char: str = "-", n: int = 64) -> None:
    print(char * n)


def show_belief_state(belief, title: str) -> None:
    facts = belief.get_active_facts()
    s = belief.summary()
    print(f"\n  [{title}]")
    print(f"  session={s['session_id']} | actifs={s['active_facts']}/{s['max_facts']}")
    if not facts:
        print("  (vide)")
    else:
        for f in facts[:4]:
            print(f"  {f.fact_id[:8]} conf={f.confidence:.2f} | {f.content[:65]}...")


def show_result(case: dict, result: dict, elapsed_ms: float) -> None:
    m1_hallucinated = result["label"] == "hallucinated"
    m2_conflict = bool(result.get("m2_conflict", False))
    is_correct_case = "correct" in case["expected"] and "hallucinated" not in case["expected"]

    if m1_hallucinated:
        tag = "[M1 HALLUCINATION DETECTEE]"
    elif m2_conflict:
        tag = "[M2 CONFLIT TEMPOREL]"
    else:
        tag = "[CORRECT]"

    correct_pred = m1_hallucinated == (not is_correct_case)
    sep("*")
    print(f"  RESULTAT — Question {case['id']}/3")
    sep("*")
    print(f"  >> {tag}")
    print(f"  M1 (NLI)  : {result['label']} | score={result['score']:.4f}")
    print(f"  M2 (BS)   : conflit={m2_conflict}")
    print(f"  Type HG   : {result['hallucination_type'] or 'N/A'}")
    print(f"  Task      : {result['task_id']} ({result['task_status']})")
    print(f"  Latence   : {elapsed_ms:.0f} ms")
    print(f"  Attendu   : {case['expected']}")
    print(f"  M1 pred   : {'OK' if correct_pred else 'faux positif/negatif'}")
    sep("*")


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    print()
    sep("=")
    print("  HalluGuard — Pipeline de demonstration (3 questions)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sep("=")

    # Agents enregistres
    print()
    print("  AGENTS ENREGISTRES (protocole A2A Google 2024)")
    print(f"  RAGAgent   -> url={RAG_AGENT_CARD.url} | protocol={RAG_AGENT_CARD.protocol}")
    print(f"  HalluGuard -> url={HALLUGUARD_AGENT_CARD.url} | protocol={HALLUGUARD_AGENT_CARD.protocol}")
    print(f"  Skills     : {', '.join(s.id for s in HALLUGUARD_AGENT_CARD.skills)}")

    # Reset le BeliefState MCP pour partir d'un etat propre
    _reset_mcp_belief()

    # BeliefState local pour la session demo
    mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))
    belief = BeliefState(session_id="demo_pipeline_session")
    show_belief_state(belief, "BeliefState AVANT les 3 questions")

    # Run cases
    results = []
    for case in DEMO_CASES:
        print()
        sep()
        print(f"  Question {case['id']}/3 : {case['question']}")
        print(f"  Claim    : {case['claim'][:90]}...")
        print(f"  Node     : {case['node_type']}")
        print(f"  Evidences: {len(case['evidences'])} document(s) recupere(s) depuis ChromaDB")
        print(f"  Delegation: RAGAgent -> HalluGuard | skill=verify_claim")
        print("  Traitement en cours...")

        t0 = time.time()
        result = delegate_verification(
            claim=case["claim"],
            evidences=case["evidences"],
            node_type=case["node_type"],
            from_agent="RAGAgent",
        )
        elapsed_ms = (time.time() - t0) * 1000

        show_result(case, result, elapsed_ms)
        results.append(result)

        # Enrichir le BeliefState avec les affirmations correctes
        if result["label"] == "correct":
            belief.add_fact(
                content=case["claim"],
                confidence=result["score"],
                step=case["id"],
            )
            mem.save(belief)

    # BeliefState apres
    show_belief_state(belief, "BeliefState APRES les 3 questions")

    # Recapitulatif
    print()
    sep("=")
    print("  RECAPITULATIF")
    sep("=")
    m1_detected = sum(1 for r in results if r["label"] == "hallucinated")
    print(f"  Questions traitees       : {len(results)}")
    print(f"  Hallucinations detectees (M1): {m1_detected}/{len(results)}")
    print(f"  Taux de detection demo   : {m1_detected/len(results)*100:.0f}%")

    # Metriques benchmark
    print()
    sep()
    print("  METRIQUES BENCHMARK (results/resultats_comparatifs.json)")
    sep()
    try:
        bench_path = ROOT / "results" / "resultats_comparatifs.json"
        with bench_path.open(encoding="utf-8") as f:
            d = json.load(f)
        v = d["variants"]
        print(f"  Benchmark : {d['benchmark']} ({d['n_scenarios']} scenarios)")
        print(f"  Variante A (baseline)  : {v['A']['metrics']['detection_rate_pct']:.1f}% detection | {v['A']['metrics']['avg_latency_ms']:.0f} ms")
        print(f"  Variante B (NLI M1)    : {v['B']['metrics']['detection_rate_pct']:.1f}% detection | {v['B']['metrics']['avg_latency_ms']:.1f} ms")
        print(f"  Variante C (M1+M2+MCP) : {v['C']['metrics']['detection_rate_pct']:.1f}% detection | {v['C']['metrics']['avg_latency_ms']:.1f} ms")
        print(f"  PBR@1  B={v['B']['metrics']['pbr1_pct']:.1f}%  C={v['C']['metrics']['pbr1_pct']:.1f}%")
        print(f"  Gain A->C : +{v['C']['metrics']['detection_rate_pct']:.1f} points de detection")
    except Exception as e:
        print(f"  (impossible de lire les metriques : {e})")

    print()
    sep("=")
    print("  Demo terminee.")
    print(f"  Log A2A : results/logs/a2a_exchanges.log")
    print(f"  Log MCP : results/logs/mcp_calls.log")
    sep("=")
    print()


if __name__ == "__main__":
    main()
