"""
m2_threshold_analysis.py — Analyse de sensibilite du seuil M2 (BeliefState).

Problème : à seuil 0.50, M2 génère 24 FP sur 30 scénarios corrects.
Stratégie : un seul passage NLI pour collecter les scores bruts M1 et M2,
puis sweep de seuils a posteriori (pas de relance du modèle).

Sorties :
  Console : tableau comparatif par seuil
  results/m2_threshold_analysis.json
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"
OUT_FILE       = ROOT / "results" / "m2_threshold_analysis.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

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

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ── Chargement des 90 scénarios ───────────────────────────────────────────────
scenarios = []
for line in SCENARIOS_FILE.read_text(encoding="utf-8").strip().splitlines():
    if line.strip():
        scenarios.append(json.loads(line))

print(f"  {len(scenarios)} scénarios chargés")
n_hall = sum(1 for s in scenarios if s["expected_label"] == "hallucinated")
n_corr = sum(1 for s in scenarios if s["expected_label"] == "correct")
print(f"  Hallucinés : {n_hall} | Corrects : {n_corr}")

# ── Chargement NLI ────────────────────────────────────────────────────────────
print("\n  Chargement du modèle NLI...")
from src.halluguard.verifier import LightweightVerifier

verifier = LightweightVerifier()
t0 = time.time()
_ = verifier.verify("warmup", ["warmup"], node_type="generation")
print(f"  Modèle chargé en {round(time.time()-t0, 1)}s")

# ── Construction du BeliefState (identique au benchmark) ─────────────────────
from src.memory.persistent_memory import PersistentMemory
mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))
mem.delete("m2_analysis")
belief = mem.load_or_create("m2_analysis")
for i, fact in enumerate(KNOWN_FACTS):
    belief.add_fact(content=fact, confidence=0.95, step=i + 1)

# ── Passage unique : collecte des scores bruts M1 et M2 ──────────────────────
print("\n  Collecte des scores M1 + M2 (passage unique)...")
raw_results = []
for i, s in enumerate(scenarios, 1):
    # M1
    r_m1 = verifier.verify(s["claim"], s["evidences"], node_type=s["node_type"])
    m1_detected = r_m1["label"] == "hallucinated"

    # M2 — on récupère le score brut SANS appliquer le seuil
    try:
        r_m2 = belief.check_temporal_coherence(s["claim"][:512], verifier=verifier)
        m2_raw_score = r_m2["score"]
    except Exception:
        m2_raw_score = 0.0

    raw_results.append({
        "id":             s["id"],
        "expected_label": s["expected_label"],
        "node_type":      s["node_type"],
        "m1_detected":    m1_detected,
        "m1_score":       r_m1["score"],
        "m2_raw_score":   m2_raw_score,
    })
    if i % 10 == 0:
        print(f"    [{i:02d}/{len(scenarios)}]", end="\r")

print(f"    [{len(scenarios)}/{len(scenarios)}] terminé")

# ── Sweep de seuils ───────────────────────────────────────────────────────────
def compute_metrics_at_threshold(raw, threshold):
    tp = fp = tn = fn = 0
    for r in raw:
        detected = r["m1_detected"] or (r["m2_raw_score"] > threshold)
        is_hall  = r["expected_label"] == "hallucinated"
        if is_hall and detected:      tp += 1
        elif is_hall and not detected: fn += 1
        elif not is_hall and detected: fp += 1
        else:                          tn += 1
    n_h = tp + fn
    n_c = fp + tn
    recall    = round(tp / n_h * 100, 1) if n_h > 0 else 0.0
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else None
    f1 = round(2 * precision * recall / (precision + recall), 1) if (precision and recall > 0) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "recall": recall, "precision": precision, "f1": f1}

# Variante B seule (M1 seulement, référence)
metrics_b = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
for r in raw_results:
    is_hall = r["expected_label"] == "hallucinated"
    if is_hall and r["m1_detected"]:      metrics_b["TP"] += 1
    elif is_hall and not r["m1_detected"]: metrics_b["FN"] += 1
    elif not is_hall and r["m1_detected"]: metrics_b["FP"] += 1
    else:                                   metrics_b["TN"] += 1
tp_b = metrics_b["TP"]; fp_b = metrics_b["FP"]
rec_b  = round(tp_b / (tp_b + metrics_b["FN"]) * 100, 1)
prec_b = round(tp_b / (tp_b + fp_b) * 100, 1) if (tp_b + fp_b) > 0 else None
f1_b   = round(2 * prec_b * rec_b / (prec_b + rec_b), 1) if (prec_b and rec_b > 0) else 0.0
metrics_b.update({"recall": rec_b, "precision": prec_b, "f1": f1_b})

results_by_threshold = {}
for thr in THRESHOLDS:
    results_by_threshold[str(thr)] = compute_metrics_at_threshold(raw_results, thr)

# ── Trouver le seuil optimal (F1 max) ────────────────────────────────────────
best_f1   = -1.0
best_thr  = None
for thr in THRESHOLDS:
    m = results_by_threshold[str(thr)]
    if m["f1"] > best_f1:
        best_f1  = m["f1"]
        best_thr = thr

# ── Affichage ─────────────────────────────────────────────────────────────────
SEP = "=" * 72
print(f"\n{SEP}")
print("  ANALYSE DU SEUIL M2 — Variante C (M1 + M2)")
print(f"  90 scénarios : {n_hall} hallucinés + {n_corr} corrects")
print(SEP)
print(f"\n  Référence — Variante B (M1 seul, seuil M2 ignoré) :")
print(f"    TP={metrics_b['TP']}  FP={metrics_b['FP']}  TN={metrics_b['TN']}  FN={metrics_b['FN']}")
print(f"    Rappel={metrics_b['recall']}%  Précision={metrics_b['precision']}%  F1={metrics_b['f1']}%")
print()
print(f"  {'Seuil':>8} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} | {'Rappel':>7} {'Précis.':>8} {'F1':>7} | {'FP count':>8}")
print(f"  {'-'*68}")
for thr in THRESHOLDS:
    m   = results_by_threshold[str(thr)]
    tag = " <-- OPTIMAL" if thr == best_thr else ""
    prec_str = f"{m['precision']}%" if m['precision'] is not None else "N/A"
    print(f"  {thr:>8.2f} | {m['TP']:>4} {m['FP']:>4} {m['TN']:>4} {m['FN']:>4} | "
          f"{m['recall']:>6.1f}% {prec_str:>8} {m['f1']:>6.1f}% | {m['FP']:>8}{tag}")
print()
print(f"  Seuil optimal : {best_thr}  (F1 = {best_f1}%)")
print(SEP)

# ── Sauvegarde ────────────────────────────────────────────────────────────────
output = {
    "benchmark": "m2_threshold_sweep",
    "n_scenarios": len(scenarios),
    "n_hallucinated": n_hall,
    "n_correct": n_corr,
    "variant_b_reference": metrics_b,
    "thresholds": results_by_threshold,
    "optimal_threshold": best_thr,
    "optimal_f1": best_f1,
    "raw_scores_sample": raw_results[:5],
}
OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  Résultats sauvegardés : {OUT_FILE}")
