"""
conformal.py — Calibration du seuil de détection par conformal prediction (C6).

Axe du Sujet 15 : "conformal prediction".

Remplace les seuils ad hoc par un seuil tau avec GARANTIE de couverture sur la
détection des hallucinations. Split conformal (one-sided) :

  Score de conformité s = P(contradiction).  On détecte si s >= tau.
  Pour garantir un rappel >= 1 - alpha (rater au plus alpha des hallucinations) :
      tau = quantile d'ordre k des scores de contradiction sur les hallucinations
            de calibration, avec k = floor(alpha * (n_cal + 1)).
  Garantie marginale (échangeabilité) : P(s_test >= tau | halluciné) >= 1 - alpha.

On rapporte, sur un split test disjoint : couverture empirique (rappel) vs cible,
et le coût en faux positifs (FPR sur les scénarios corrects au même tau).

Entrée : results/scores_raw.json   Sortie : results/conformal_calibration.json
Lancer : venv\\Scripts\\python.exe src/experiments/conformal.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES_FILE = ROOT / "results" / "scores_raw.json"
OUT_FILE = ROOT / "results" / "conformal_calibration.json"
SEED = 42
ALPHAS = [0.05, 0.10, 0.20]
N_TRIALS = 200   # la garantie conformal est marginale -> on moyenne sur splits aléatoires


def split_conformal_threshold(scores_pos_cal: np.ndarray, alpha: float) -> float:
    """tau garantissant un rappel >= 1 - alpha (finite-sample, one-sided)."""
    n = len(scores_pos_cal)
    s = np.sort(scores_pos_cal)            # ascendant
    k = int(np.floor(alpha * (n + 1)))     # 1-based rank du quantile bas
    if k < 1:
        return float("-inf")               # détecter tout
    if k > n:
        k = n
    return float(s[k - 1])


def main() -> None:
    data = json.loads(SCORES_FILE.read_text(encoding="utf-8"))
    rows = data["results"]
    p = np.array([r["p_contradiction"] for r in rows], dtype=float)
    y = np.array([1 if r["is_hallucinated"] else 0 for r in rows], dtype=float)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    out = {
        "description": "Seuil de détection calibré par split conformal (garantie de rappel). "
                       f"Couverture/coût moyennés sur {N_TRIALS} splits aléatoires "
                       "(la garantie conformal est marginale).",
        "score": "P(contradiction), détection si score >= tau",
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_pos": int(len(idx_pos)),
        "n_neg": int(len(idx_neg)),
        "levels": [],
    }
    for alpha in ALPHAS:
        taus, recalls, fprs = [], [], []
        rng = np.random.default_rng(SEED + int(alpha * 1000))
        for _ in range(N_TRIALS):
            pp, nn = idx_pos.copy(), idx_neg.copy()
            rng.shuffle(pp)
            rng.shuffle(nn)
            pos_cal, pos_test = pp[: len(pp) // 2], pp[len(pp) // 2:]
            neg_test = nn[len(nn) // 2:]
            tau = split_conformal_threshold(p[pos_cal], alpha)
            recalls.append(float((p[pos_test] >= tau).mean()))
            fprs.append(float((p[neg_test] >= tau).mean()))
            if np.isfinite(tau):
                taus.append(tau)
        mean_recall = float(np.mean(recalls))
        mean_fpr = float(np.mean(fprs))
        out["levels"].append({
            "alpha": alpha,
            "target_recall": round(1 - alpha, 3),
            "tau_mean": round(float(np.mean(taus)), 4) if taus else None,
            "empirical_recall_mean": round(mean_recall, 3),
            "empirical_fpr_mean": round(mean_fpr, 3),
            "coverage_ok": mean_recall >= (1 - alpha) - 0.01,
        })
        print(f"alpha={alpha} | tau~{np.mean(taus):.4f} | rappel cible {1-alpha:.2f} "
              f"-> empirique {mean_recall:.3f} | FPR={mean_fpr:.3f}  (moy. {N_TRIALS} splits)")

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Écrit : {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
