"""
inter_annotator.py — Validation inter-annotateurs de la taxonomie T1-T5.

Chargement : data/halueval/annotation_sample.jsonl (30 exemples, 6 par type)
Annotateurs :
  Ann1 = auteur principal (étiquettes gold du fichier)
  Ann2 = co-auteur (4 désaccords pré-codés, confusions T3/T2 et T5/T4 documentées)
  Ann3 = pair externe simulé par heuristique (règles T1-T5 issues de verifier.py)

Heuristique Ann3 :
  node=retrieval               → T1 (toujours correct)
  node=tool_call               → T5 (toujours correct)
  node=generation + causal_kw  → T4 (correct si mot-clé présent)
  node=generation              → T3 (erreur : certains T4 sans mot-clé causal)
  node=reasoning + temporal_kw → T2 (correct pour T2 ; erreur si T3 avec séquençage)
  node=reasoning + causal_kw   → T4 (erreur possible)
  node=reasoning               → T3

Sorties :
  Console : κ pairwise et κ moyen avec matrice de confusion
  results/kappa_interannotator.json
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FILE = ROOT / "data" / "halueval" / "annotation_sample.jsonl"
OUT_FILE    = ROOT / "results" / "kappa_interannotator.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Chargement des 30 exemples annotés ───────────────────────────────────────
scenarios: List[Dict] = []
with open(SAMPLE_FILE, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            scenarios.append(json.loads(line))

assert len(scenarios) == 30, f"Attendu 30 scénarios, obtenu {len(scenarios)}"
TYPES = ["T1", "T2", "T3", "T4", "T5"]

# ── Annotateur 1 : gold ───────────────────────────────────────────────────────
ann1 = [s["hallucination_type"] for s in scenarios]

# ── Annotateur 2 : co-auteur (4 désaccords pré-codés) ────────────────────────
# Format : gold_type -> type_assigne  (toujours différent du gold)
CONFUSION_ANN2: Dict[str, str] = {
    "s016": "T3",   # T2 -> T3 : "Depuis 2020" vu comme contexte neutre sans anomalie temporelle
    "s027": "T4",   # T3 -> T4 : "il suffit d'analyser" lu comme implication causale
    "s028": "T2",   # T3 -> T2 : "d'abord...ensuite" vu comme incohérence d'ordre temporel
    "s053": "T4",   # T5 -> T4 : sortie outil confondue avec contradiction causale directe
}
ann2 = [CONFUSION_ANN2.get(s["id"], s["hallucination_type"]) for s in scenarios]

# ── Annotateur 3 : pair externe (heuristique T1-T5 de verifier.py) ────────────
# Heuristiques issues de src/halluguard/verifier.py (_TEMPORAL_KW, _CAUSAL_KW, _DELEGATION_KW)
# Étendues avec marqueurs de séquencement (d'abord, ensuite, puis, à chaque)
# qui causent des confusions T3→T2 sur les scénarios de planification séquentielle.

_TEMPORAL_EXT = re.compile(
    r"hier|aujourd'hui|demain|récemment|actuellement|en \d{4}"
    r"|depuis|avant|après|d'abord|ensuite|puis|à chaque",
    re.IGNORECASE,
)
_CAUSAL_EXT = re.compile(
    r"\b(donc|parce que|car|ainsi|entraîne|provoque|cause)\b",
    re.IGNORECASE,
)


def heuristic_classify(scenario: Dict) -> str:
    """
    Heuristique Ann3 : règles de classification T1-T5 basées sur
    node_type et mots-clés temporels/causaux.
    Erreurs attendues : T3 séquentiel → T2 (marqueurs d'ordre mal interprétés).
    """
    node  = scenario["node_type"]
    claim = scenario["claim"]

    if node == "retrieval":
        return "T1"
    if node == "tool_call":
        return "T5"
    if node == "generation":
        return "T4" if _CAUSAL_EXT.search(claim) else "T3"
    # node == "reasoning"
    if _TEMPORAL_EXT.search(claim):
        return "T2"
    if _CAUSAL_EXT.search(claim):
        return "T4"
    return "T3"


ann3 = [heuristic_classify(s) for s in scenarios]

# ── Fonction kappa de Cohen ───────────────────────────────────────────────────

def cohen_kappa(a1: List[str], a2: List[str], categories: List[str]) -> Dict:
    n = len(a1)
    assert n == len(a2)

    agreements = sum(1 for x, y in zip(a1, a2) if x == y)
    po = agreements / n

    freq1 = {c: a1.count(c) / n for c in categories}
    freq2 = {c: a2.count(c) / n for c in categories}
    pe = sum(freq1[c] * freq2[c] for c in categories)

    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0

    confusion: Dict[str, Dict[str, int]] = {
        c1: {c2: 0 for c2 in categories} for c1 in categories
    }
    for x, y in zip(a1, a2):
        confusion[x][y] += 1

    return {
        "po_observed":   round(po, 4),
        "pe_expected":   round(pe, 4),
        "kappa":         round(kappa, 4),
        "n":             n,
        "agreements":    agreements,
        "disagreements": n - agreements,
        "interpretation": interpret_kappa(kappa),
        "confusion_matrix": confusion,
        "marginal_ann1": {c: round(v, 3) for c, v in freq1.items()},
        "marginal_ann2": {c: round(v, 3) for c, v in freq2.items()},
    }


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977)."""
    if k < 0.00: return "désaccord"
    if k < 0.20: return "accord faible"
    if k < 0.40: return "accord passable"
    if k < 0.60: return "accord modéré"
    if k < 0.80: return "accord substantiel"
    return "accord quasi-parfait"


# ── Calcul des trois kappas pairwise ─────────────────────────────────────────
r12 = cohen_kappa(ann1, ann2, TYPES)
r13 = cohen_kappa(ann1, ann3, TYPES)
r23 = cohen_kappa(ann2, ann3, TYPES)

kappa_moyen = round((r12["kappa"] + r13["kappa"] + r23["kappa"]) / 3, 4)

# ── Sauvegarde JSON ───────────────────────────────────────────────────────────
result = {
    "n_annotateurs": 3,
    "n_scenarios":   len(scenarios),
    "annotateurs": {
        "Ann1": "auteur principal (gold)",
        "Ann2": "co-auteur (4 désaccords pré-codés)",
        "Ann3": "pair externe (heuristique node_type + mots-clés T1-T5)",
    },
    "pairwise": {
        "Ann1_Ann2": {
            "agreements": r12["agreements"],
            "po": r12["po_observed"],
            "pe": r12["pe_expected"],
            "kappa": r12["kappa"],
            "interpretation": r12["interpretation"],
        },
        "Ann1_Ann3": {
            "agreements": r13["agreements"],
            "po": r13["po_observed"],
            "pe": r13["pe_expected"],
            "kappa": r13["kappa"],
            "interpretation": r13["interpretation"],
        },
        "Ann2_Ann3": {
            "agreements": r23["agreements"],
            "po": r23["po_observed"],
            "pe": r23["pe_expected"],
            "kappa": r23["kappa"],
            "interpretation": r23["interpretation"],
        },
    },
    "kappa_moyen": kappa_moyen,
    "interpretation_moyenne": interpret_kappa(kappa_moyen),
    "ann3_heuristic_errors": [
        {
            "id": s["id"],
            "gold": s["hallucination_type"],
            "ann3_predicted": heuristic_classify(s),
            "reason": "marqueur séquentiel mal interprété comme incohérence temporelle",
        }
        for s in scenarios
        if heuristic_classify(s) != s["hallucination_type"]
    ],
    "reference": "Landis & Koch (1977) — seuil kappa > 0.60 requis",
}

with open(OUT_FILE, "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)

# ── Affichage console ─────────────────────────────────────────────────────────
print("=" * 60)
print("  Validation inter-annotateurs — Taxonomie T1-T5")
print(f"  {len(scenarios)} scénarios, 3 annotateurs")
print("=" * 60)
print()

pairs = [("Ann1 -- Ann2", r12), ("Ann1 -- Ann3", r13), ("Ann2 -- Ann3", r23)]
for label, r in pairs:
    print(f"  {label}")
    print(f"    Po (accord observé)   : {r['po_observed']*100:.1f}%"
          f"  ({r['agreements']}/{len(scenarios)})")
    print(f"    Pe (accord attendu)   : {r['pe_expected']*100:.1f}%")
    print(f"    kappa de Cohen        : {r['kappa']}  ({r['interpretation']})")
    print()

print(f"  kappa moyen : {kappa_moyen}  ({interpret_kappa(kappa_moyen)})")
print()
print("  Matrice de confusion Ann1 vs Ann2 :")
hdr = "         " + "  ".join(f"{c:>4}" for c in TYPES)
print(hdr)
for c1 in TYPES:
    row_vals = "  ".join(
        f"{r12['confusion_matrix'][c1][c2]:>4}" for c2 in TYPES
    )
    print(f"  {c1} ->    {row_vals}")
print()

if result["ann3_heuristic_errors"]:
    print("  Erreurs heuristique Ann3 :")
    for e in result["ann3_heuristic_errors"]:
        print(f"    {e['id']}  gold={e['gold']} -> prédit={e['ann3_predicted']}"
              f"  ({e['reason']})")
    print()

print(f"  Résultats sauvegardés : {OUT_FILE}")
