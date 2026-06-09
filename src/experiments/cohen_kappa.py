"""
cohen_kappa.py — Accord inter-annotateurs RÉEL à 2 auteurs (C1).

Lit data/taxonomy_annotation_sheet.csv (rempli à la main par Marc et Souleymane,
à l'aveugle, sans concertation). Calcule le kappa de Cohen entre les deux
annotateurs, l'accord observé, la matrice de confusion et la liste des désaccords.

AUCUNE annotation n'est simulée ou générée. Si la feuille n'est pas remplie,
le script s'arrête et explique quoi faire — il n'invente rien.

Entrée  : data/taxonomy_annotation_sheet.csv  (généré par make_annotation_sheet.py)
Gold    : data/halueval/scenarios.jsonl        (pour l'accuracy indicative vs gold)
Sortie  : results/cohen_kappa_result.json
Lancer  : venv\\Scripts\\python.exe src/experiments/cohen_kappa.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHEET = ROOT / "data" / "taxonomy_annotation_sheet.csv"
SPLIT_MARC = ROOT / "data" / "annotation" / "taxonomy_marc.csv"
SPLIT_SOUL = ROOT / "data" / "annotation" / "taxonomy_souleymane.csv"
SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"
OUT_FILE = ROOT / "results" / "cohen_kappa_result.json"
TYPES = ["T1", "T2", "T3", "T4", "T5"]


def _read_col(path: Path, col: str) -> dict:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = (r.get(col) or "").strip().upper()
    return out


def load_annotations():
    """Fichiers séparés si présents (aveugle garanti), sinon feuille combinée."""
    if SPLIT_MARC.exists() and SPLIT_SOUL.exists():
        m_map, s_map = _read_col(SPLIT_MARC, "annotation"), _read_col(SPLIT_SOUL, "annotation")
        ids = [i for i in m_map if i in s_map]
        return ids, [m_map[i] for i in ids], [s_map[i] for i in ids], "fichiers séparés"
    if SHEET.exists():
        ids, marc, soul = [], [], []
        with open(SHEET, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ids.append(r["id"])
                marc.append((r.get("ann_marc") or "").strip().upper())
                soul.append((r.get("ann_souleymane") or "").strip().upper())
        return ids, marc, soul, "feuille combinée"
    sys.exit("[STOP] Aucune feuille d'annotation. Lancer d'abord : "
             "python src/experiments/make_annotation_sheet.py")


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977)."""
    if k < 0:    return "désaccord"
    if k < 0.20: return "accord faible"
    if k < 0.40: return "accord passable"
    if k < 0.60: return "accord modéré"
    if k < 0.80: return "accord substantiel"
    return "accord quasi-parfait"


def cohen_kappa(a1, a2, categories):
    n = len(a1)
    po = sum(1 for x, y in zip(a1, a2) if x == y) / n
    f1 = {c: a1.count(c) / n for c in categories}
    f2 = {c: a2.count(c) / n for c in categories}
    pe = sum(f1[c] * f2[c] for c in categories)
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0
    confusion = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for x, y in zip(a1, a2):
        if x in confusion and y in confusion[x]:
            confusion[x][y] += 1
    return po, pe, kappa, confusion


def main() -> None:
    ids, marc, soul, source = load_annotations()
    print(f"Source des annotations : {source}")

    # Garde-fou : refuser de calculer sur des annotations incomplètes/invalides
    pairs = [(i, m, s) for i, m, s in zip(ids, marc, soul)
             if m in TYPES and s in TYPES]
    n_filled = len(pairs)
    if n_filled < len(ids):
        missing = len(ids) - n_filled
        print(f"[ATTENTION] {missing}/{len(ids)} lignes non (ou mal) annotées "
              f"(valeurs attendues : {TYPES}).")
    if n_filled < 20:
        sys.exit(f"[STOP] Seulement {n_filled} lignes valides — annotez au moins 20 scénarios "
                 f"(idéalement les 60) dans {SHEET.name}, puis relancez.\n"
                 f"  Aucune valeur n'est inventée : le kappa ne sera calculé que sur de vraies annotations.")

    ids_f = [p[0] for p in pairs]
    a_marc = [p[1] for p in pairs]
    a_soul = [p[2] for p in pairs]

    po, pe, kappa, confusion = cohen_kappa(a_marc, a_soul, TYPES)
    disagreements = [{"id": i, "marc": m, "souleymane": s}
                     for i, m, s in pairs if m != s]

    # Accuracy indicative vs gold (n'entre PAS dans le kappa)
    gold = {}
    for line in SCENARIOS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["id"]] = d.get("hallucination_type")
    acc_marc = sum(1 for i, m in zip(ids_f, a_marc) if gold.get(i) == m) / n_filled
    acc_soul = sum(1 for i, s in zip(ids_f, a_soul) if gold.get(i) == s) / n_filled

    # Kappa sur le sous-ensemble réellement ambigu (T2/T3, nœud reasoning) :
    # c'est là que l'accord humain a une vraie valeur, le node_type déterminant
    # quasi mécaniquement T1/T4/T5. (Le gold ne sert qu'à stratifier, pas à noter.)
    def subset_kappa(gold_types):
        sel = [(m, s) for i, m, s in zip(ids_f, a_marc, a_soul) if gold.get(i) in gold_types]
        if len(sel) < 2:
            return {"n": len(sel), "kappa": None}
        po_s, pe_s, k_s, _ = cohen_kappa([m for m, _ in sel], [s for _, s in sel], TYPES)
        return {"n": len(sel), "accord_observe": round(po_s, 4), "kappa": round(k_s, 4),
                "interpretation": interpret_kappa(k_s)}

    out = {
        "n_annotateurs": 2,
        "annotateurs": ["Marc (auteur)", "Souleymane (auteur)"],
        "n_scenarios_annotes": n_filled,
        "accord_observe": round(po, 4),
        "accord_attendu_hasard": round(pe, 4),
        "kappa_cohen": round(kappa, 4),
        "interpretation": interpret_kappa(kappa),
        "kappa_sous_ensemble": {
            "reasoning_T2_T3": subset_kappa({"T2", "T3"}),
            "autres_T1_T4_T5": subset_kappa({"T1", "T4", "T5"}),
        },
        "n_desaccords": len(disagreements),
        "desaccords": disagreements,
        "confusion_matrix": confusion,
        "accuracy_vs_gold": {"marc": round(acc_marc, 3), "souleymane": round(acc_soul, 3)},
        "reference": "Cohen (1960) ; interprétation Landis & Koch (1977), seuil kappa > 0.60.",
        "source_annotations": source,
        "note": "Annotation réelle à 2 auteurs, à l'aveugle. Aucune valeur simulée.",
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nkappa de Cohen (Marc vs Souleymane) = {out['kappa_cohen']} "
          f"({out['interpretation']}) sur {n_filled} scénarios")
    print(f"Accord observé = {po*100:.1f}% | désaccords = {len(disagreements)}")
    print(f"Écrit : {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
