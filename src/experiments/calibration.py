"""
calibration.py — Quantification d'incertitude et calibration du vérificateur (C7).

Axe du Sujet 15 : "calibration" + "uncertainty quantification".

Mesure si P(contradiction) est un estimateur calibré de la probabilité réelle
qu'un claim soit halluciné :
  - ECE (Expected Calibration Error) en M bins
  - Brier score
  - Reliability diagram (données par bin, pour la figure de l'article)
  - Platt scaling (régression logistique) : ECE avant/après, fit sur split
    calibration, évalué sur split test disjoint (seed fixe).

Entrée : results/scores_raw.json   Sortie : results/calibration.json
Lancer : venv\\Scripts\\python.exe src/experiments/calibration.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES_FILE = ROOT / "results" / "scores_raw.json"
OUT_FILE = ROOT / "results" / "calibration.json"
SEED = 42
N_BINS = 10


def ece_and_bins(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    """ECE (uniform binning) + données reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"bin": [round(lo, 2), round(hi, 2)], "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
        bins.append({"bin": [round(lo, 2), round(hi, 2)], "count": cnt,
                     "confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return round(ece, 4), bins


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return round(float(np.mean((p - y) ** 2)), 4)


def platt_scale(p_cal, y_cal, p_test):
    """Régression logistique 1D sur le score -> proba calibrée."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(p_cal.reshape(-1, 1), y_cal)
    return clf.predict_proba(p_test.reshape(-1, 1))[:, 1]


def main() -> None:
    data = json.loads(SCORES_FILE.read_text(encoding="utf-8"))
    rows = data["results"]
    p = np.array([r["p_contradiction"] for r in rows], dtype=float)
    y = np.array([1 if r["is_hallucinated"] else 0 for r in rows], dtype=float)

    # --- Calibration brute (tous les scénarios) ---
    ece_raw, bins_raw = ece_and_bins(p, y)
    brier_raw = brier(p, y)

    # --- Platt scaling : split stratifié calibration/test (seed fixe) ---
    rng = np.random.default_rng(SEED)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    cal_idx = np.concatenate([idx_pos[: len(idx_pos) // 2], idx_neg[: len(idx_neg) // 2]])
    test_idx = np.concatenate([idx_pos[len(idx_pos) // 2:], idx_neg[len(idx_neg) // 2:]])

    p_test_platt = platt_scale(p[cal_idx], y[cal_idx], p[test_idx])
    ece_test_raw, _ = ece_and_bins(p[test_idx], y[test_idx])
    ece_test_platt, _ = ece_and_bins(p_test_platt, y[test_idx])

    payload = {
        "description": "Calibration et UQ du vérificateur NLI. p = P(contradiction), y = halluciné.",
        "n_scenarios": len(rows),
        "n_bins": N_BINS,
        "seed": SEED,
        "raw_all": {
            "ece": ece_raw,
            "brier": brier_raw,
            "reliability_bins": bins_raw,
        },
        "platt_split": {
            "n_calibration": int(len(cal_idx)),
            "n_test": int(len(test_idx)),
            "ece_test_raw": ece_test_raw,
            "ece_test_platt": ece_test_platt,
            "improvement": round(ece_test_raw - ece_test_platt, 4),
        },
        "interpretation": (
            f"ECE brut={ece_raw} (0=parfaitement calibré). "
            f"Platt scaling: ECE test {ece_test_raw} -> {ece_test_platt} "
            f"({'amélioration' if ece_test_platt < ece_test_raw else 'pas d amélioration'})."
        ),
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ECE brut = {ece_raw} | Brier = {brier_raw}")
    print(f"Platt (test disjoint) : ECE {ece_test_raw} -> {ece_test_platt}")
    print(f"Écrit : {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
