"""
make_summary.py — Source de vérité UNIQUE des chiffres officiels (§3 du plan).

Agrège les sorties réelles en un seul fichier results/SUMMARY.json que le README,
l'article et les slides doivent citer (et eux seuls). Aucune valeur n'est saisie
à la main ici : tout est recalculé depuis les artefacts.

  - A/B/C : matrices de confusion + recall/precision/F1 (B = scores_raw argmax ;
            C = M1 argmax OR M2 conflict, repris du benchmark réel)
  - latence : scores_raw (médiane à chaud + IQR)
  - calibration (ECE, Platt) et conformal (couverture/cout) : repris tels quels
  - kappa : repris si annotation réelle disponible, sinon "en attente"

Lancer : venv\\Scripts\\python.exe scripts/make_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8")) if (R / p).exists() else None


def metrics(detected_by_id: dict, truth_hallu: dict, types: dict):
    tp = fp = tn = fn = 0
    for sid, is_h in truth_hallu.items():
        det = detected_by_id.get(sid, False)
        if is_h and det:       tp += 1
        elif is_h and not det: fn += 1
        elif not is_h and det: fp += 1
        else:                  tn += 1
    recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) else 0.0
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) else None
    f1 = round(2 * precision * recall / (precision + recall), 1) if precision and recall else 0.0
    fpr = round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0.0
    by_type = {}
    for t in ["T1", "T2", "T3", "T4", "T5"]:
        ids_t = [sid for sid, tt in types.items() if tt == t]
        d = sum(1 for sid in ids_t if detected_by_id.get(sid, False))
        by_type[t] = {"total": len(ids_t), "detected": d,
                      "recall_pct": round(d / len(ids_t) * 100, 1) if ids_t else 0.0}
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "recall_pct": recall,
            "precision_pct": precision, "f1_pct": f1, "fpr_pct": fpr, "by_type": by_type}


def main() -> None:
    scores = load("scores_raw.json")
    rows = scores["results"]
    truth = {r["id"]: r["is_hallucinated"] for r in rows}
    types = {r["id"]: r["hallucination_type"] for r in rows if r["is_hallucinated"]}

    # Variante A : référence sans détection (ne détecte rien par définition)
    det_a = {r["id"]: False for r in rows}
    # Variante B : M1 seul, règle argmax (réelle)
    det_b = {r["id"]: (r["label_argmax"] == "hallucinated") for r in rows}
    # Variante C : M1 OR M2 — m2_conflict repris du benchmark réel
    comp = load("resultats_comparatifs.json")
    m2 = {}
    if comp:
        for r in comp["variants"]["C"]["results"]:
            m2[r["id"]] = bool(r.get("m2_conflict", False))
    det_c = {r["id"]: (det_b[r["id"]] or m2.get(r["id"], False)) for r in rows}

    calib = load("calibration.json")
    conf = load("conformal_calibration.json")
    selfcons = load("self_consistency.json")
    judge = load("baseline_llm_judge.json")
    kappa = load("cohen_kappa_result.json")
    kappa_block = (
        {"status": "réel", "kappa_cohen": kappa.get("kappa_cohen"),
         "n": kappa.get("n_scenarios_annotes"), "interpretation": kappa.get("interpretation")}
        if kappa and kappa.get("n_annotateurs") == 2 and kappa.get("kappa_cohen") is not None
        else {"status": "EN ATTENTE — annoter data/taxonomy_annotation_sheet.csv puis lancer cohen_kappa.py"}
    )

    summary = {
        "_README": "CHIFFRES OFFICIELS. README, article et slides ne doivent citer que ce fichier.",
        "benchmark": "HaluEval-Agentic (90 scénarios : 60 hallucinés + 30 corrects)",
        "model": scores["model"],
        "variants": {
            "A_baseline": metrics(det_a, truth, types),
            "B_M1_argmax": metrics(det_b, truth, types),
            "C_M1_or_M2": metrics(det_c, truth, types),
        },
        "decision_rule": {
            "B": "label=hallucinated si P(contradiction) > P(entailment) (argmax)",
            "C": "B OR conflit BeliefState (M2)",
            "note_C": "M2 augmente le rappel mais dégrade fortement la précision (faux positifs).",
        },
        "latency_overhead_ms": scores["latency_overhead_ms"],
        "calibration": {
            "ece_raw": calib["raw_all"]["ece"], "brier_raw": calib["raw_all"]["brier"],
            "ece_test_raw": calib["platt_split"]["ece_test_raw"],
            "ece_test_platt": calib["platt_split"]["ece_test_platt"],
        } if calib else None,
        "conformal": conf["levels"] if conf else None,
        "self_consistency": {
            "auc": selfcons["auc"],
            "separation_fiable_vs_piegee": selfcons["separation"],
            "best_threshold_accuracy": selfcons["best_threshold"],
            "backend": selfcons["backend"],
        } if selfcons else None,
        "baseline_llm_judge": {
            "model": judge["model"], "recall_pct": judge["recall_pct"],
            "precision_pct": judge["precision_pct"], "f1_pct": judge["f1_pct"],
            "median_latency_ms": judge["median_latency_ms"],
        } if judge else None,
        "kappa_taxonomie": kappa_block,
    }
    (R / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    b, c = summary["variants"]["B_M1_argmax"], summary["variants"]["C_M1_or_M2"]
    print("=== CHIFFRES OFFICIELS (results/SUMMARY.json) ===")
    print(f"B (M1)  : recall {b['recall_pct']}% | precision {b['precision_pct']}% | "
          f"F1 {b['f1_pct']}% | FPR {b['fpr_pct']}%")
    print(f"C (M1+M2): recall {c['recall_pct']}% | precision {c['precision_pct']}% | "
          f"F1 {c['f1_pct']}% | FPR {c['fpr_pct']}%   <- M2 dégrade la précision")
    lo = summary["latency_overhead_ms"]
    print(f"Latence overhead : médiane {lo['median']} ms (IQR {lo['iqr']})")
    print(f"Calibration ECE brut : {summary['calibration']['ece_raw']}")
    print(f"Kappa taxonomie : {kappa_block['status']}")


if __name__ == "__main__":
    main()
