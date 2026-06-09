"""
self_consistency_eval.py — Évaluation de la self-consistency comme signal d'hallucination (C5).

On compare deux classes de questions :
  - FIABLE : faits connus et vérifiables -> le LLM répond de façon cohérente (score haut)
  - PIÉGÉE : fausse prémisse, fait obscur, anachronisme -> le LLM fabule (score bas)

Hypothèse : la self-consistency sépare les deux classes. On la valide en mesurant la
séparation des scores et la qualité de détection (AUC + accuracy au seuil retenu).

Lancer : venv\\Scripts\\python.exe src/experiments/self_consistency_eval.py [--backend mistral_api|ollama] [-k 5]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_FILE = ROOT / "results" / "self_consistency.json"

# label "prone" = question piégée (hallucination probable) -> on attend un score BAS
QUESTIONS = [
    ("Quelle est la capitale de la France ?", "fiable"),
    ("Combien de continents y a-t-il sur Terre ?", "fiable"),
    ("Qui a peint la Joconde ?", "fiable"),
    ("Quelle est la formule chimique de l'eau ?", "fiable"),
    ("En quelle annee a eu lieu la prise de la Bastille ?", "fiable"),
    ("Quel est le plus grand ocean du monde ?", "fiable"),
    ("Combien de cotes a un hexagone ?", "fiable"),
    ("Qui a ecrit la piece Romeo et Juliette ?", "fiable"),
    ("Quel roi de France a regne en l'an 3000 ?", "piegee"),
    ("Quel est le nom exact du chat de compagnie de Napoleon Bonaparte ?", "piegee"),
    ("Combien d'habitants compte exactement le village imaginaire de Trifouillis-les-Oies aujourd'hui ?", "piegee"),
    ("Pour quel prix Nobel de physique Albert Einstein a-t-il ete recompense en 1969 ?", "piegee"),
    ("Quelle marque de voiture personnelle conduisait Jules Cesar ?", "piegee"),
    ("Quelle est la 50e decimale exacte du nombre d'Euler e ?", "piegee"),
    ("Quel est le titre precis du 19e album studio solo de Mozart ?", "piegee"),
    ("Quel smartphone Cleopatre utilisait-elle au quotidien ?", "piegee"),
]


def best_threshold(rows):
    """Seuil maximisant l'accuracy (flag=incertain si score < seuil pour les piegees)."""
    cand = sorted({round(r["consistency"], 3) for r in rows})
    best_t, best_acc = 0.5, 0.0
    for t in cand:
        tp = sum(1 for r in rows if r["label"] == "piegee" and r["consistency"] < t)
        tn = sum(1 for r in rows if r["label"] == "fiable" and r["consistency"] >= t)
        acc = (tp + tn) / len(rows)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(best_t, 3), round(best_acc, 3)


def auc(scores, labels_prone):
    """AUC : capacite de (1 - score) a separer les questions piegees. labels_prone=1 si piegee."""
    pos = [1 - s for s, l in zip(scores, labels_prone) if l == 1]   # piegees : score bas -> (1-s) haut
    neg = [1 - s for s, l in zip(scores, labels_prone) if l == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mistral_api", choices=["mistral_api", "ollama"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    from src.halluguard.self_consistency import sample_answers, consistency_score

    rows = []
    print(f"Self-consistency : backend={args.backend}, k={args.k}, seuil={args.threshold}")
    for i, (q, label) in enumerate(QUESTIONS, 1):
        answers = sample_answers(q, k=args.k, temperature=0.7, backend=args.backend)
        score = consistency_score(answers)
        flagged = score < args.threshold     # True = signalee comme incertaine
        rows.append({"question": q, "label": label, "consistency": score,
                     "flagged_uncertain": flagged, "answers": answers})
        print(f"  [{i:02d}/{len(QUESTIONS)}] {label:7s} score={score:.3f} "
              f"{'-> FLAG incertain' if flagged else ''}")

    fiables = [r["consistency"] for r in rows if r["label"] == "fiable"]
    piegees = [r["consistency"] for r in rows if r["label"] == "piegee"]
    labels_prone = [1 if r["label"] == "piegee" else 0 for r in rows]
    scores = [r["consistency"] for r in rows]

    # Detection au seuil : on veut flag=True pour piegee, False pour fiable
    tp = sum(1 for r in rows if r["label"] == "piegee" and r["flagged_uncertain"])
    fn = sum(1 for r in rows if r["label"] == "piegee" and not r["flagged_uncertain"])
    fp = sum(1 for r in rows if r["label"] == "fiable" and r["flagged_uncertain"])
    tn = sum(1 for r in rows if r["label"] == "fiable" and not r["flagged_uncertain"])
    acc = round((tp + tn) / len(rows), 3)

    import statistics as st
    payload = {
        "description": "Self-consistency comme signal d'hallucination. "
                       "consistency = similarite cosinus moyenne des reponses echantillonnees.",
        "backend": args.backend, "k": args.k, "threshold": args.threshold,
        "n_questions": len(rows),
        "mean_consistency_fiable": round(st.mean(fiables), 3),
        "mean_consistency_piegee": round(st.mean(piegees), 3),
        "separation": round(st.mean(fiables) - st.mean(piegees), 3),
        "auc": auc(scores, labels_prone),
        "detection_at_threshold": {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "accuracy": acc},
        "best_threshold": dict(zip(("threshold", "accuracy"), best_threshold(rows))),
        "results": rows,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMoyenne self-consistency : fiable={payload['mean_consistency_fiable']} "
          f"vs piegee={payload['mean_consistency_piegee']} "
          f"(separation={payload['separation']})")
    bt = payload["best_threshold"]
    print(f"AUC={payload['auc']} | accuracy@{args.threshold}={acc} "
          f"(TP={tp} FP={fp} TN={tn} FN={fn}) | meilleur seuil={bt['threshold']} (acc={bt['accuracy']})")
    print(f"Ecrit : {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
