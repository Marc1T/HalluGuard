"""
regenerate_scores.py — Source unique des scores NLI bruts (axe contradiction cohérent).

Pour chacun des 90 scénarios (60 hallucinés + 30 corrects) :
  - p_contradiction / p_entailment / p_neutral (probabilités softmax du cross-encoder)
  - label_argmax : règle historique (P(contra) > P(entail))
  - latence verify() mesurée à chaud (médiane sur R répétitions, modèle déjà chargé)

Sortie : results/scores_raw.json  → consommé par calibration.py, conformal.py,
make_summary.py. Aucune valeur n'est inventée : tout provient du modèle réel.

Déterministe (cross-encoder non stochastique). Lancer :
  venv\\Scripts\\python.exe src/experiments/regenerate_scores.py
"""
from __future__ import annotations
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"
OUT_FILE = ROOT / "results" / "scores_raw.json"
R_REPEATS = 5  # répétitions pour la latence à chaud


def load_scenarios() -> list[dict]:
    rows = []
    for line in SCENARIOS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    from src.halluguard.verifier import LightweightVerifier

    scenarios = load_scenarios()
    verifier = LightweightVerifier()

    # Warmup (force le chargement du modèle hors mesure de latence)
    t_load = time.time()
    verifier.verify("warmup", ["warmup evidence"], node_type="generation")
    load_s = round(time.time() - t_load, 1)
    print(f"Modèle NLI chargé/réchauffé en {load_s}s — {len(scenarios)} scénarios")

    results = []
    for i, s in enumerate(scenarios, 1):
        # Mesure de latence à chaud : médiane sur R répétitions
        lats = []
        out = None
        for _ in range(R_REPEATS):
            out = verifier.verify(s["claim"], s["evidences"], node_type=s["node_type"])
            lats.append(out["latency_ms"])
        results.append({
            "id": s["id"],
            "node_type": s["node_type"],
            "hallucination_type": s.get("hallucination_type"),
            "expected_label": s["expected_label"],
            "is_hallucinated": s["expected_label"] == "hallucinated",
            "p_contradiction": out["p_contradiction"],
            "p_entailment": out["p_entailment"],
            "p_neutral": out["p_neutral"],
            "label_argmax": out["label"],            # règle historique
            "latency_ms_median": round(statistics.median(lats), 1),
        })
        if i % 15 == 0 or i == len(scenarios):
            print(f"  [{i:02d}/{len(scenarios)}] {s['id']} "
                  f"p_contra={out['p_contradiction']:.3f} -> {out['label']}")

    all_lat = [r["latency_ms_median"] for r in results]
    all_lat.sort()
    q1 = all_lat[len(all_lat) // 4]
    q3 = all_lat[(3 * len(all_lat)) // 4]
    payload = {
        "description": "Scores NLI bruts par scénario — axe P(contradiction) cohérent pour tous. "
                       "Source unique pour calibration, conformal et métriques.",
        "model": "cross-encoder/nli-MiniLM2-L6-H768",
        "n_scenarios": len(results),
        "r_repeats_latency": R_REPEATS,
        "latency_overhead_ms": {
            "median": round(statistics.median(all_lat), 1),
            "mean": round(statistics.mean(all_lat), 1),
            "iqr": [q1, q3],
            "min": min(all_lat),
            "max": max(all_lat),
        },
        "results": results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lo = payload["latency_overhead_ms"]
    print(f"\nLatence verify() à chaud : médiane {lo['median']} ms "
          f"(IQR {lo['iqr']}, min {lo['min']}, max {lo['max']})")
    print(f"Écrit : {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
