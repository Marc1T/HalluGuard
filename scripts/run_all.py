"""
run_all.py — Régénère TOUS les résultats officiels en une commande (§3 du plan).

Ordre :
  1. regenerate_scores.py  -> results/scores_raw.json   (scores NLI bruts + latence)
  2. calibration.py        -> results/calibration.json  (ECE, Brier, Platt)
  3. conformal.py          -> results/conformal_calibration.json (couverture/cout)
  4. cohen_kappa.py        -> results/cohen_kappa_result.json (si feuille annotée)
  5. make_summary.py       -> results/SUMMARY.json       (source de vérité unique)

L'étape kappa est ignorée proprement si l'annotation réelle n'est pas encore faite.
Lancer : venv\\Scripts\\python.exe scripts/run_all.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("Scores NLI bruts + latence", "src/experiments/regenerate_scores.py", True),
    ("Calibration (ECE/Brier/Platt)", "src/experiments/calibration.py", True),
    ("Conformal prediction", "src/experiments/conformal.py", True),
    ("Tests statistiques (McNemar, bootstrap)", "src/experiments/statistical_analysis.py", False),
    ("Kappa inter-annotateurs (réel)", "src/experiments/cohen_kappa.py", False),
    ("Synthèse officielle SUMMARY.json", "scripts/make_summary.py", True),
]


def main() -> None:
    for title, script, required in STEPS:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
        rc = subprocess.run([PY, str(ROOT / script)], cwd=str(ROOT)).returncode
        if rc != 0:
            if required:
                sys.exit(f"[ÉCHEC] {script} (code {rc}). Arrêt.")
            print(f"[IGNORÉ] {script} non bloquant (code {rc}) — "
                  f"probablement annotation kappa non encore faite.")
    print(f"\nTerminé. Chiffres officiels : results/SUMMARY.json")


if __name__ == "__main__":
    main()
