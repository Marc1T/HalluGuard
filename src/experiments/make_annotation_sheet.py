"""
make_annotation_sheet.py — Génère la feuille d'annotation à l'aveugle pour l'étude
inter-annotateurs RÉELLE (C1). AUCUNE annotation n'est pré-remplie ni simulée.

Sortie : data/taxonomy_annotation_sheet.csv
Colonnes : id, node_type, query, claim, evidence, ann_marc, ann_souleymane
  -> Marc et Souleymane remplissent indépendamment ann_marc / ann_souleymane
     avec une étiquette parmi {T1, T2, T3, T4, T5}, SANS se concerter.
  -> Le type gold n'est PAS inclus (annotation à l'aveugle).

Définitions de la taxonomie (rappel pour les annotateurs) :
  T1 Perception    : claim basé sur des chunks récupérés inexacts (nœud retrieval)
  T2 Mémoire       : contradiction avec un fait établi / marqueur temporel (nœud reasoning)
  T3 Planification : enchaînement logique faux, formulé de façon neutre (nœud reasoning)
  T4 Causale       : claim contredisant directement les documents (nœud generation)
  T5 Délégation    : sortie d'outil divergente (nœud tool_call)

Lancer : venv\\Scripts\\python.exe src/experiments/make_annotation_sheet.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"
OUT_CSV = ROOT / "data" / "taxonomy_annotation_sheet.csv"          # version combinée (1 fichier, 2 colonnes)
SPLIT_DIR = ROOT / "data" / "annotation"                            # version séparée (1 fichier / annotateur)
SPLIT_FILES = {"marc": SPLIT_DIR / "taxonomy_marc.csv",
               "souleymane": SPLIT_DIR / "taxonomy_souleymane.csv"}


def main() -> None:
    rows = []
    for line in SCENARIOS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        if s["expected_label"] != "hallucinated":
            continue  # seuls les scénarios hallucinés portent un type T1-T5
        rows.append(s)

    # 1) Feuille combinée (pratique mais demande de la discipline pour rester aveugle)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "node_type", "query", "claim", "evidence",
                    "ann_marc", "ann_souleymane"])
        for s in rows:
            ev = s["evidences"][0] if s.get("evidences") else ""
            w.writerow([s["id"], s["node_type"], s["query"], s["claim"], ev, "", ""])

    # 2) Fichiers SÉPARÉS (aveugle garanti : chacun ne voit que son propre fichier)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for who, path in SPLIT_FILES.items():
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "node_type", "query", "claim", "evidence", "annotation"])
            for s in rows:
                ev = s["evidences"][0] if s.get("evidences") else ""
                w.writerow([s["id"], s["node_type"], s["query"], s["claim"], ev, ""])

    print(f"{len(rows)} scénarios à annoter (à l'aveugle, indépendamment).")
    print(f"  Combiné : {OUT_CSV.relative_to(ROOT)}")
    for who, path in SPLIT_FILES.items():
        print(f"  {who:<11}: {path.relative_to(ROOT)}  -> remplir la colonne 'annotation' (T1..T5)")
    print("Puis : python scripts/run_all.py  (le merge et le kappa sont automatiques)")


if __name__ == "__main__":
    main()
