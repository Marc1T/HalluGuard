"""
Benchmark HaluEval-Agentic — compare les 3 variantes (A, B, C) sur les scénarios JSONL.

Métriques par variante :
  - detection_rate_pct : % hallucinations détectées
  - pbr1_pct          : % bloquées en delta_steps <= 1 (PBR@1)
  - avg_latency_ms    : latence moyenne HalluGuard par scénario
  - avg_delta_steps   : nb étapes moyen de propagation avant détection

Usage :
  python src/tests/benchmark_runner.py           # 10 scénarios (test rapide)
  python src/tests/benchmark_runner.py --n 60    # 60 scénarios (benchmark complet)
  python src/tests/benchmark_runner.py --all     # alias --n 60
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"
RESULTS_FILE   = ROOT / "results" / "resultats_comparatifs.json"
SEPARATOR      = "-" * 60

# Faits connus injectés dans le BeliefState de Variante C pour simuler une session active
KNOWN_FACTS = [
    "Les LLM utilisent l'architecture Transformer avec mécanisme d'attention multi-têtes.",
    "ChromaDB est une base de données vectorielle open-source pour les pipelines RAG.",
    "Le RAG fonctionne en trois étapes : indexation des documents, récupération, génération.",
    "L'architecture Transformer a été introduite en 2017 dans Attention is All You Need.",
    "LangChain et LangGraph sont des frameworks pour construire des pipelines RAG agentiques.",
    "Mistral est un LLM open-source récent qui peut fonctionner en local via Ollama.",
    "Les embeddings sont des représentations vectorielles denses du sens sémantique.",
    "Le RLHF utilise des retours humains pour aligner les LLM sur les préférences utilisateurs.",
    "Le fine-tuning reste une technique importante et complémentaire au prompting en contexte.",
    "Les agents IA souffrent de limitations dans la planification à long terme.",
    "Le mécanisme d'attention standard est O(n²) en complexité spatiale.",
    "Les hallucinations sont un problème fondamental de tous les LLM.",
    "Un pipeline RAG de qualité nécessite un chunking approprié des documents source.",
    "La similarité cosinus est la mesure standard pour comparer des embeddings vectoriels.",
    "Le BeliefState peut maintenir jusqu'à 50 faits actifs avec politique d'expulsion.",
]


# ------------------------------------------------------------------ #
# Chargement scénarios                                                  #
# ------------------------------------------------------------------ #

def load_scenarios(n: Optional[int] = None) -> List[dict]:
    if not SCENARIOS_FILE.exists():
        print(f"ERREUR : {SCENARIOS_FILE} introuvable.")
        sys.exit(1)
    scenarios = []
    for line in SCENARIOS_FILE.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            scenarios.append(json.loads(line))
    if n is not None:
        scenarios = scenarios[:n]
    return scenarios


# ------------------------------------------------------------------ #
# Variante A — baseline sans détection                                  #
# ------------------------------------------------------------------ #

def run_variant_a(scenarios: List[dict]) -> List[dict]:
    results = []
    for s in scenarios:
        results.append({
            "id": s["id"],
            "detected": False,
            "delta_steps": s["length"],
            "latency_ms": 0.0,
            "label": "no_check",
            "score": None,
        })
    return results


# ------------------------------------------------------------------ #
# Variante B — M1 uniquement (NLI inter-sources)                        #
# ------------------------------------------------------------------ #

def run_variant_b(scenarios: List[dict], verifier) -> List[dict]:
    results = []
    for i, s in enumerate(scenarios, 1):
        t0 = time.time()
        r = verifier.verify(s["claim"], s["evidences"], node_type=s["node_type"])
        lat = (time.time() - t0) * 1000
        detected = r["label"] == "hallucinated"
        results.append({
            "id": s["id"],
            "detected": detected,
            "delta_steps": 0 if detected else s["length"],
            "latency_ms": round(lat, 1),
            "label": r["label"],
            "score": r["score"],
            "hallucination_type_detected": r.get("hallucination_type"),
        })
        print(f"  [{i:02d}/{len(scenarios)}] {s['id']} | {s['node_type']:<10} | "
              f"label={r['label']:<12} | score={r['score']:.3f} | detected={detected}")
    return results


# ------------------------------------------------------------------ #
# Variante C — M1 + M2 (BeliefState temporal)                          #
# ------------------------------------------------------------------ #

def _build_belief_for_benchmark(verifier):
    from src.memory.persistent_memory import PersistentMemory
    mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))
    mem.delete("benchmark_c")
    belief = mem.load_or_create("benchmark_c")
    for i, fact in enumerate(KNOWN_FACTS):
        belief.add_fact(content=fact, confidence=0.95, step=i + 1)
    return belief


def run_variant_c(scenarios: List[dict], verifier) -> List[dict]:
    belief = _build_belief_for_benchmark(verifier)
    results = []
    for i, s in enumerate(scenarios, 1):
        t0 = time.time()
        r = verifier.verify(s["claim"], s["evidences"], node_type=s["node_type"])
        try:
            m2 = belief.check_temporal_coherence(s["claim"][:512], verifier=verifier)
            m2_conflict = bool(m2.get("conflict", False))
        except Exception:
            m2_conflict = False
        lat = (time.time() - t0) * 1000
        detected = r["label"] == "hallucinated" or m2_conflict
        results.append({
            "id": s["id"],
            "detected": detected,
            "delta_steps": 0 if detected else s["length"],
            "latency_ms": round(lat, 1),
            "label": r["label"],
            "score": r["score"],
            "m2_conflict": m2_conflict,
            "hallucination_type_detected": r.get("hallucination_type"),
        })
        m2_str = " +M2" if m2_conflict else "    "
        print(f"  [{i:02d}/{len(scenarios)}] {s['id']} | {s['node_type']:<10} | "
              f"label={r['label']:<12}{m2_str} | score={r['score']:.3f} | detected={detected}")
    return results


# ------------------------------------------------------------------ #
# Métriques                                                             #
# ------------------------------------------------------------------ #

def compute_metrics(results: List[dict], scenarios: List[dict]) -> dict:
    hallucinated_ids = {s["id"] for s in scenarios if s["expected_label"] == "hallucinated"}
    h_results = [r for r in results if r["id"] in hallucinated_ids]

    detected_count = sum(1 for r in h_results if r["detected"])
    detection_rate = round(detected_count / len(h_results) * 100, 1) if h_results else 0.0

    pbr1_count = sum(1 for r in h_results if r["detected"] and r["delta_steps"] <= 1)
    pbr1_rate = round(pbr1_count / len(h_results) * 100, 1) if h_results else 0.0

    latencies = [r["latency_ms"] for r in results]
    avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    deltas = [r["delta_steps"] for r in h_results]
    avg_delta = round(sum(deltas) / len(deltas), 2) if deltas else 0.0

    return {
        "total_scenarios": len(scenarios),
        "hallucinated_scenarios": len(h_results),
        "detected": detected_count,
        "detection_rate_pct": detection_rate,
        "pbr1_pct": pbr1_rate,
        "avg_latency_ms": avg_lat,
        "avg_delta_steps": avg_delta,
    }


def metrics_by_type(results: List[dict], scenarios: List[dict]) -> dict:
    """Métriques détaillées par type T1-T5."""
    s_by_id = {s["id"]: s for s in scenarios}
    by_type: dict[str, list] = {}
    for r in results:
        s = s_by_id.get(r["id"], {})
        htype = s.get("hallucination_type", "?")
        by_type.setdefault(htype, []).append(r)

    out = {}
    for htype, rs in sorted(by_type.items()):
        detected = sum(1 for r in rs if r["detected"])
        out[htype] = {
            "total": len(rs),
            "detected": detected,
            "detection_rate_pct": round(detected / len(rs) * 100, 1),
        }
    return out


# ------------------------------------------------------------------ #
# Affichage résumé                                                      #
# ------------------------------------------------------------------ #

def print_summary(variant: str, metrics: dict, by_type: dict) -> None:
    print(f"\n  Variante {variant} :")
    print(f"    Detection rate  : {metrics['detection_rate_pct']}%")
    print(f"    PBR@1           : {metrics['pbr1_pct']}%")
    print(f"    Avg latency     : {metrics['avg_latency_ms']} ms")
    print(f"    Avg delta steps : {metrics['avg_delta_steps']}")
    if by_type:
        print(f"    Par type        : ", end="")
        parts = [f"{t}={v['detection_rate_pct']}%" for t, v in sorted(by_type.items())]
        print(" | ".join(parts))


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluEval-Agentic benchmark runner")
    parser.add_argument("--n", type=int, default=10, help="Nombre de scénarios (défaut: 10)")
    parser.add_argument("--all", action="store_true", help="Lance sur les 60 scénarios complets")
    args = parser.parse_args()

    n_scenarios = 60 if args.all else args.n

    print("=" * 60)
    print("  HalluGuard — Benchmark HaluEval-Agentic")
    print(f"  Scénarios : {n_scenarios} | Variantes : A, B, C")
    print("=" * 60)

    scenarios = load_scenarios(n=n_scenarios)
    print(f"\n  {len(scenarios)} scénarios chargés depuis {SCENARIOS_FILE.name}")

    # ---- Distribution ----
    by_type = {}
    by_len = {}
    for s in scenarios:
        by_type[s["hallucination_type"]] = by_type.get(s["hallucination_type"], 0) + 1
        by_len[s["length"]] = by_len.get(s["length"], 0) + 1
    type_str = " ".join(f"{k}:{v}" for k, v in sorted(by_type.items()))
    len_str  = " ".join(f"len={k}:{v}" for k, v in sorted(by_len.items()))
    print(f"  Types   : {type_str}")
    print(f"  Lengths : {len_str}")

    # ============================================================== #
    # Variante A                                                       #
    # ============================================================== #
    print(f"\n[1/3] Variante A — baseline (aucune détection)")
    print(SEPARATOR)
    results_a = run_variant_a(scenarios)
    metrics_a = compute_metrics(results_a, scenarios)
    btype_a   = metrics_by_type(results_a, scenarios)
    print_summary("A", metrics_a, btype_a)

    # ============================================================== #
    # Chargement NLI (partagé B + C)                                   #
    # ============================================================== #
    print(f"\n  Chargement du modèle NLI (cross-encoder/nli-MiniLM2-L6-H768)...")
    from src.halluguard.verifier import LightweightVerifier
    verifier = LightweightVerifier()
    t_load = time.time()
    # Warmup pour forcer le chargement du modèle
    _ = verifier.verify("warmup", ["warmup evidence"], node_type="generation")
    print(f"  Modèle NLI chargé en {round(time.time() - t_load, 1)}s")

    # ============================================================== #
    # Variante B                                                       #
    # ============================================================== #
    print(f"\n[2/3] Variante B — M1 NLI inter-sources")
    print(SEPARATOR)
    results_b = run_variant_b(scenarios, verifier)
    metrics_b = compute_metrics(results_b, scenarios)
    btype_b   = metrics_by_type(results_b, scenarios)
    print_summary("B", metrics_b, btype_b)

    # ============================================================== #
    # Variante C                                                       #
    # ============================================================== #
    print(f"\n[3/3] Variante C — M1 + M2 (BeliefState {len(KNOWN_FACTS)} faits pré-chargés)")
    print(SEPARATOR)
    results_c = run_variant_c(scenarios, verifier)
    metrics_c = compute_metrics(results_c, scenarios)
    btype_c   = metrics_by_type(results_c, scenarios)
    print_summary("C", metrics_c, btype_c)

    # ============================================================== #
    # Tableau comparatif                                               #
    # ============================================================== #
    print(f"\n{'='*60}")
    print("  RÉSUMÉ COMPARATIF")
    print(f"{'='*60}")
    print(f"  {'Métrique':<28} {'A':>8} {'B':>8} {'C':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Detection Rate (%)':<28} {metrics_a['detection_rate_pct']:>8} {metrics_b['detection_rate_pct']:>8} {metrics_c['detection_rate_pct']:>8}")
    print(f"  {'PBR@1 (%)':<28} {metrics_a['pbr1_pct']:>8} {metrics_b['pbr1_pct']:>8} {metrics_c['pbr1_pct']:>8}")
    print(f"  {'Avg Latency (ms)':<28} {metrics_a['avg_latency_ms']:>8} {metrics_b['avg_latency_ms']:>8} {metrics_c['avg_latency_ms']:>8}")
    print(f"  {'Avg Delta Steps':<28} {metrics_a['avg_delta_steps']:>8} {metrics_b['avg_delta_steps']:>8} {metrics_c['avg_delta_steps']:>8}")
    print(f"{'='*60}")

    # ============================================================== #
    # Sauvegarde                                                       #
    # ============================================================== #
    payload = {
        "benchmark": "HaluEval-Agentic",
        "n_scenarios": len(scenarios),
        "scenarios_file": str(SCENARIOS_FILE),
        "variants": {
            "A": {
                "description": "Baseline RAG sans HalluGuard",
                "metrics": metrics_a,
                "by_type": btype_a,
                "results": results_a,
            },
            "B": {
                "description": "HalluGuard M1 uniquement — NLI inter-sources",
                "metrics": metrics_b,
                "by_type": btype_b,
                "results": results_b,
            },
            "C": {
                "description": "HalluGuard complet — M1 + M2 BeliefState + MCP",
                "metrics": metrics_c,
                "by_type": btype_c,
                "results": results_c,
            },
        },
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Résultats sauvegardés : {RESULTS_FILE.name}")
    print(f"  Scénarios évalués     : {len(scenarios)}")
