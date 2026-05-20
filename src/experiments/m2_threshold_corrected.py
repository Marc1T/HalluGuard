"""
m2_threshold_corrected.py — Sweep de seuils M2 avec setup production correct.

CORRECTION DU SETUP :
  Ancien setup (biaise) : BeliefState pre-charge avec 15 faits IA/NLP generiques
  qui recoupent semantiquement les 30 scenarios corrects => faux positifs parasites.

  Nouveau setup (production-fidele) : BeliefState frais par scenario.
  Les evidences du scenario sont ajoutees au BeliefState pour simuler
  que le pipeline a etabli les faits corrects dans un tour precedent
  de la session. M2 teste si le claim contredit ces faits etablis.

  Pour les scenarios hallucines  : claim contredit l'evidence => M2 devrait
  detecter => TP.
  Pour les scenarios corrects    : claim paraphrase l'evidence => NLI devrait
  retourner entailment/neutre => pas de conflit => TN.

Sorties :
  Console : tableau comparatif (3 colonnes : B, C ancien, C nouveau)
  results/m2_threshold_corrected.json
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

SCENARIOS_FILE  = ROOT / "data" / "halueval" / "scenarios.jsonl"
PREV_RESULTS    = ROOT / "results" / "m2_threshold_analysis.json"
OUT_FILE        = ROOT / "results" / "m2_threshold_corrected.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ── Chargement des 90 scenarios ───────────────────────────────────────────────
scenarios = []
for line in SCENARIOS_FILE.read_text(encoding="utf-8").strip().splitlines():
    if line.strip():
        scenarios.append(json.loads(line))

n_hall = sum(1 for s in scenarios if s["expected_label"] == "hallucinated")
n_corr = sum(1 for s in scenarios if s["expected_label"] == "correct")
print(f"  {len(scenarios)} scenarios charges : {n_hall} hallucines + {n_corr} corrects")

# ── Chargement NLI ────────────────────────────────────────────────────────────
print("\n  Chargement du modele NLI...")
from src.halluguard.verifier import LightweightVerifier
from src.halluguard.belief_state import BeliefState

verifier = LightweightVerifier()
t0 = time.time()
_ = verifier.verify("warmup", ["warmup"], node_type="generation")
print(f"  Modele charge en {round(time.time()-t0, 1)}s")

# ── Passage unique : scores M1 + M2 (BeliefState frais par scenario) ─────────
# Pour chaque scenario :
#   1. BeliefState vide
#   2. Ajouter les evidences du scenario comme faits etablis (step=0)
#   3. Tester le claim contre ce BeliefState via M2
# Cela simule : "le pipeline a recupere les faits corrects dans un tour precedent,
# puis le LLM produit un claim -- est-il coherent avec ce que le systeme sait ?"
print("\n  Collecte M1 + M2 (BeliefState frais par scenario)...")
raw_results = []
for i, s in enumerate(scenarios, 1):
    # M1 : NLI claim vs evidences
    r_m1 = verifier.verify(s["claim"], s["evidences"], node_type=s["node_type"])
    m1_detected = r_m1["label"] == "hallucinated"

    # M2 : BeliefState frais, on injecte les evidences du scenario
    belief = BeliefState(session_id=f"sc_{s['id']}")
    for j, ev in enumerate(s["evidences"]):
        belief.add_fact(content=ev, confidence=0.95, step=j)

    try:
        r_m2 = belief.check_temporal_coherence(s["claim"][:512], verifier=verifier)
        m2_raw_score = r_m2["score"]
    except Exception:
        m2_raw_score = 0.0

    raw_results.append({
        "id":             s["id"],
        "expected_label": s["expected_label"],
        "node_type":      s["node_type"],
        "hallucination_type": s.get("hallucination_type"),
        "m1_detected":    m1_detected,
        "m1_score":       r_m1["score"],
        "m2_raw_score":   m2_raw_score,
    })
    if i % 10 == 0:
        print(f"    [{i:02d}/{len(scenarios)}]", end="\r")

print(f"    [{len(scenarios)}/{len(scenarios)}] termine")

# ── Fonctions de metriques ────────────────────────────────────────────────────
def compute_at_threshold(raw, thr):
    tp = fp = tn = fn = 0
    for r in raw:
        detected = r["m1_detected"] or (r["m2_raw_score"] > thr)
        is_hall  = r["expected_label"] == "hallucinated"
        if   is_hall and     detected: tp += 1
        elif is_hall and not detected: fn += 1
        elif not is_hall and detected: fp += 1
        else:                          tn += 1
    n_h = tp + fn
    rec  = round(tp / n_h * 100, 1) if n_h > 0 else 0.0
    prec = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else None
    f1   = round(2 * prec * rec / (prec + rec), 1) if (prec and rec > 0) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "recall": rec, "precision": prec, "f1": f1}

# Variante B (M1 seul)
def m1_only(raw):
    tp = fp = tn = fn = 0
    for r in raw:
        is_hall = r["expected_label"] == "hallucinated"
        if   is_hall and     r["m1_detected"]: tp += 1
        elif is_hall and not r["m1_detected"]: fn += 1
        elif not is_hall and r["m1_detected"]: fp += 1
        else:                                   tn += 1
    n_h = tp + fn
    rec  = round(tp / n_h * 100, 1) if n_h > 0 else 0.0
    prec = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else None
    f1   = round(2 * prec * rec / (prec + rec), 1) if (prec and rec > 0) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "recall": rec, "precision": prec, "f1": f1}

metrics_b = m1_only(raw_results)

# Sweep nouveau setup
new_results = {str(thr): compute_at_threshold(raw_results, thr) for thr in THRESHOLDS}

# Optimal nouveau setup
best_f1_new  = max(m["f1"] for m in new_results.values())
best_thr_new = next(thr for thr in THRESHOLDS if new_results[str(thr)]["f1"] == best_f1_new)

# ── Chargement des resultats ancien setup pour comparaison ────────────────────
old_results = {}
if PREV_RESULTS.exists():
    prev = json.loads(PREV_RESULTS.read_text(encoding="utf-8"))
    for thr in THRESHOLDS:
        k = str(thr)
        if k in prev.get("thresholds", {}):
            old_results[k] = prev["thresholds"][k]

# ── Affichage comparatif ──────────────────────────────────────────────────────
SEP = "=" * 90
print(f"\n{SEP}")
print("  COMPARAISON SETUP ANCIEN vs NOUVEAU --- Sweep seuil M2")
print(f"  {len(scenarios)} scenarios : {n_hall} hallucines + {n_corr} corrects")
print(SEP)

# Reference B
rb = metrics_b
print(f"\n  Reference --- Variante B (M1 seul, pas de M2) :")
print(f"  TP={rb['TP']} FP={rb['FP']} TN={rb['TN']} FN={rb['FN']} | "
      f"Rappel={rb['recall']}% Precision={rb['precision']}% F1={rb['f1']}%")

print()
hdr = (f"  {'Seuil':>6} || "
       f"{'--- ANCIEN (KNOWN_FACTS) ---':^38} || "
       f"{'--- NOUVEAU (BeliefState/scenario) ---':^38}")
sub = (f"  {'':>6} || "
       f"{'Rappel':>7} {'Prec':>7} {'F1':>7} {'FP':>5} {'TN':>5} || "
       f"{'Rappel':>7} {'Prec':>7} {'F1':>7} {'FP':>5} {'TN':>5}")
print(hdr)
print(sub)
print(f"  {'-'*88}")

for thr in THRESHOLDS:
    k = str(thr)
    n = new_results[k]
    prec_n = f"{n['precision']}%" if n['precision'] is not None else "N/A"
    tag_n  = " <-OPT" if thr == best_thr_new else ""

    if k in old_results:
        o = old_results[k]
        prec_o = f"{o['precision']}%" if o.get('precision') is not None else "N/A"
        line_o = f"  {thr:>6.2f} || {o['recall']:>6.1f}% {prec_o:>7} {o['f1']:>6.1f}% {o['FP']:>5} {o['TN']:>5} "
    else:
        line_o = f"  {thr:>6.2f} || {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>5} {'N/A':>5} "

    print(f"{line_o}|| {n['recall']:>6.1f}% {prec_n:>7} {n['f1']:>6.1f}% {n['FP']:>5} {n['TN']:>5}{tag_n}")

print()
print(f"  Optimal nouveau setup : seuil={best_thr_new}  F1={best_f1_new}%")
m_opt = new_results[str(best_thr_new)]
print(f"  A ce seuil : TP={m_opt['TP']} FP={m_opt['FP']} TN={m_opt['TN']} FN={m_opt['FN']}")
beats_b = "OUI" if best_f1_new > metrics_b["f1"] else "NON"
print(f"  M2 bat M1 seul en F1 ? {beats_b}  (B={metrics_b['f1']}% vs C_opt={best_f1_new}%)")
print(SEP)

# ── Analyse par type d'hallucination (seuil optimal) ─────────────────────────
print(f"\n  Detail par type (seuil optimal {best_thr_new}) :")
types = ["T1", "T2", "T3", "T4", "T5", "correct"]
for t in types:
    subset = [r for r in raw_results if
              (r.get("hallucination_type") == t or
               (t == "correct" and r["expected_label"] == "correct"))]
    if not subset:
        continue
    detected = sum(1 for r in subset
                   if r["m1_detected"] or r["m2_raw_score"] > best_thr_new)
    is_hall_set = [r for r in subset if r["expected_label"] == "hallucinated"]
    is_corr_set = [r for r in subset if r["expected_label"] == "correct"]
    if is_hall_set:
        tp_ = sum(1 for r in is_hall_set
                  if r["m1_detected"] or r["m2_raw_score"] > best_thr_new)
        print(f"    {t} : {tp_}/{len(is_hall_set)} detectes "
              f"({round(tp_/len(is_hall_set)*100,1)}%)")
    else:
        fp_ = sum(1 for r in is_corr_set
                  if r["m1_detected"] or r["m2_raw_score"] > best_thr_new)
        print(f"    {t} (correct) : {fp_}/{len(is_corr_set)} faux positifs "
              f"({round(fp_/len(is_corr_set)*100,1)}% FPR)")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
output = {
    "setup": "production-faithful (BeliefState fresh per scenario, evidence injected)",
    "n_scenarios": len(scenarios),
    "n_hallucinated": n_hall,
    "n_correct": n_corr,
    "variant_b_reference": metrics_b,
    "thresholds_new_setup": new_results,
    "optimal_threshold_new": best_thr_new,
    "optimal_f1_new": best_f1_new,
    "beats_b_in_f1": best_f1_new > metrics_b["f1"],
    "old_setup_optimal_f1": 79.3,
    "raw_scores": raw_results,
}
OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  Resultats sauvegardes : {OUT_FILE.name}")
