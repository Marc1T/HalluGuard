"""
baseline_llm_judge.py — Baseline externe LLM-as-judge (C8).

Compare HalluGuard M1 (NLI léger, 117 M params) à une baseline forte : demander
directement à un LLM (Mistral) de juger si l'affirmation est contredite par
l'évidence, sur les MÊMES 90 scénarios. Réponse attendue : OUI (hallucinée) / NON.

But : montrer que le vérificateur NLI léger est compétitif (P/R/F1) face à un
LLM-juge bien plus lourd et coûteux, pour une latence très inférieure.

Back-end : API Mistral (clé dans .env). Lancer :
  venv\\Scripts\\python.exe src/experiments/baseline_llm_judge.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"
OUT_FILE = ROOT / "results" / "baseline_llm_judge.json"
MODEL = "open-mistral-7b"

PROMPT = (
    "Tu es un vérificateur factuel. Voici une évidence de référence et une affirmation.\n"
    "Évidence : {evidence}\n"
    "Affirmation : {claim}\n\n"
    "L'affirmation est-elle FAUSSE ou CONTREDITE par l'évidence ? "
    "Réponds par UN SEUL MOT : OUI (si contredite/fausse) ou NON (si cohérente)."
)


def load_scenarios() -> list[dict]:
    return [json.loads(l) for l in SCENARIOS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge(evidence: str, claim: str, key: str) -> tuple[bool, float]:
    import requests
    t0 = time.time()
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": MODEL, "temperature": 0.0, "max_tokens": 5,
              "messages": [{"role": "user", "content": PROMPT.format(evidence=evidence, claim=claim)}]},
        timeout=60,
    )
    r.raise_for_status()
    lat = (time.time() - t0) * 1000
    txt = r.json()["choices"][0]["message"]["content"].strip().lower()
    detected = txt.startswith("oui") or "oui" in txt[:6]   # OUI = hallucinée
    return detected, lat


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        sys.exit("MISTRAL_API_KEY absent (.env).")

    scenarios = load_scenarios()
    results, lats = [], []
    print(f"Baseline LLM-juge ({MODEL}) sur {len(scenarios)} scénarios...")
    for i, s in enumerate(scenarios, 1):
        ev = s["evidences"][0] if s.get("evidences") else ""
        detected, lat = judge(ev, s["claim"], key)
        lats.append(lat)
        results.append({"id": s["id"], "expected_label": s["expected_label"],
                        "is_hallucinated": s["expected_label"] == "hallucinated",
                        "detected": detected, "latency_ms": round(lat, 1)})
        if i % 15 == 0 or i == len(scenarios):
            print(f"  [{i:02d}/{len(scenarios)}] {s['id']} detected={detected}")

    tp = sum(1 for r in results if r["is_hallucinated"] and r["detected"])
    fn = sum(1 for r in results if r["is_hallucinated"] and not r["detected"])
    fp = sum(1 for r in results if not r["is_hallucinated"] and r["detected"])
    tn = sum(1 for r in results if not r["is_hallucinated"] and not r["detected"])
    recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0.0
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) else None
    f1 = round(2 * precision * recall / (precision + recall), 1) if precision and recall else 0.0
    import statistics as st
    payload = {
        "description": "Baseline LLM-as-judge (Mistral) sur les 90 scénarios, comparée à M1 NLI.",
        "model": MODEL, "n_scenarios": len(results),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "recall_pct": recall, "precision_pct": precision, "f1_pct": f1,
        "avg_latency_ms": round(st.mean(lats), 1),
        "median_latency_ms": round(st.median(lats), 1),
        "results": results,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLLM-juge : recall {recall}% | precision {precision}% | F1 {f1}% "
          f"| latence médiane {payload['median_latency_ms']} ms")
    print(f"(rappel : HalluGuard M1 = recall 76,7 / precision 90,2 / F1 82,9 / latence ~40 ms)")
    print(f"Écrit : {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
