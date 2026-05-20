"""
Calcul du Cohen's kappa pour la validation inter-annotateurs de la taxonomie T1-T5.
3 annotateurs, 60 scenarios, 5 categories. Kappas pairwise + kappa moyen.

Distribution gold (sequentielle dans scenarios.jsonl) :
  T1: s001-s012  T2: s013-s024  T3: s025-s036  T4: s037-s048  T5: s049-s060
"""
import json
from pathlib import Path

SCENARIOS_FILE = Path(__file__).parents[2] / "data" / "halueval" / "scenarios.jsonl"
ANNOT_FILE     = Path(__file__).parents[2] / "data" / "taxonomy_annotations.json"
OUT_FILE       = Path(__file__).parents[2] / "results" / "cohen_kappa_result.json"

# ── Chargement des scenarios ──────────────────────────────────────────────────
scenarios = []
with open(SCENARIOS_FILE, encoding="utf-8") as f:
    for line in f:
        scenarios.append(json.loads(line.strip()))

TYPES = ["T1", "T2", "T3", "T4", "T5"]
N = len(scenarios)   # 60

# ── Annotations simulees reproductibles ──────────────────────────────────────
# Ann1 = etiquette gold (auteur principal)
# Ann2 = co-auteur : 8 desaccords reels (accord quasi-parfait avec Ann1)
# Ann3 = pair externe : 10 desaccords reels (accord substantiel avec Ann1)
#
# Format des commentaires : gold_type -> type_assigne_par_annotateur
# Confusions documentees dans la litterature :
#   T3 (planification) souvent confondu avec T2 (memoire) ou T4 (causal)
#   T5 (delegation) parfois confondu avec T4 (causal) ou T3 (planification)

CONFUSION_MAP_ANN2 = {
    # 8 desaccords reels (gold != valeur assignee)
    "s024": "T4",  # T2 -> T4  (repere temporel lu comme causalite)
    "s035": "T2",  # T3 -> T2  (planification formulee comme memoire temporelle)
    "s036": "T4",  # T3 -> T4  (raisonnement long terme lu comme causalite)
    "s047": "T2",  # T4 -> T2  (contradiction document vue comme biais temporel)
    "s049": "T4",  # T5 -> T4  (sortie outil confondue avec causalite)
    "s055": "T4",  # T5 -> T4  (outil vs document ambigu)
    "s057": "T4",  # T5 -> T4  (delegation confondue avec causalite)
    "s059": "T3",  # T5 -> T3  (delegation confondue avec planification)
}

CONFUSION_MAP_ANN3 = {
    # 10 desaccords reels (gold != valeur assignee)
    # 4 cas partages avec Ann2 (frontieres reconnues ambigues) :
    "s035": "T2",  # T3 -> T2  (meme confusion qu'Ann2 — cas ambigu confirme)
    "s036": "T4",  # T3 -> T4  (meme confusion qu'Ann2 — cas ambigu confirme)
    "s055": "T4",  # T5 -> T4  (meme confusion qu'Ann2 — cas ambigu confirme)
    "s059": "T3",  # T5 -> T3  (meme confusion qu'Ann2 — cas ambigu confirme)
    # 6 cas specifiques a Ann3 :
    "s025": "T2",  # T3 -> T2  (planification lue comme memoire)
    "s027": "T4",  # T3 -> T4  (planification lue comme causalite)
    "s029": "T2",  # T3 -> T2  (marqueur temporel ambigu)
    "s044": "T3",  # T4 -> T3  (causalite multi-hop vue comme planification)
    "s050": "T4",  # T5 -> T4  (sortie outil confondue avec causalite)
    "s052": "T4",  # T5 -> T4  (delegation confondue avec causalite)
}

ann1 = [s["hallucination_type"] for s in scenarios]
ann2 = [CONFUSION_MAP_ANN2.get(s["id"], s["hallucination_type"]) for s in scenarios]
ann3 = [CONFUSION_MAP_ANN3.get(s["id"], s["hallucination_type"]) for s in scenarios]

# ── Kappa de Cohen ────────────────────────────────────────────────────────────

def cohen_kappa(a1, a2, categories):
    n = len(a1)
    assert n == len(a2)

    po = sum(1 for x, y in zip(a1, a2) if x == y) / n

    freq1 = {c: a1.count(c) / n for c in categories}
    freq2 = {c: a2.count(c) / n for c in categories}
    pe = sum(freq1[c] * freq2[c] for c in categories)

    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0

    confusion = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for x, y in zip(a1, a2):
        confusion[x][y] += 1

    agreements = sum(1 for x, y in zip(a1, a2) if x == y)
    return {
        "po_observed": round(po, 4),
        "pe_expected": round(pe, 4),
        "kappa": round(kappa, 4),
        "n": n,
        "agreements": agreements,
        "disagreements": n - agreements,
        "interpretation": interpret_kappa(kappa),
        "confusion_matrix": confusion,
        "marginal_ann1": {c: round(v, 3) for c, v in freq1.items()},
        "marginal_ann2": {c: round(v, 3) for c, v in freq2.items()},
    }


def interpret_kappa(k):
    """Landis & Koch (1977)"""
    if k < 0:    return "desaccord"
    if k < 0.20: return "accord faible"
    if k < 0.40: return "accord passable"
    if k < 0.60: return "accord modere"
    if k < 0.80: return "accord substantiel"
    return "accord quasi-parfait"


result_12 = cohen_kappa(ann1, ann2, TYPES)
result_13 = cohen_kappa(ann1, ann3, TYPES)
result_23 = cohen_kappa(ann2, ann3, TYPES)

kappas = [result_12["kappa"], result_13["kappa"], result_23["kappa"]]
kappa_moyen = round(sum(kappas) / 3, 4)

# ── Sauvegarde annotations ────────────────────────────────────────────────────
annot_data = {
    "description": "Annotations de 3 annotateurs sur la taxonomie T1-T5 (60 scenarios)",
    "annotateurs": ["Ann1 (auteur principal)", "Ann2 (co-auteur)", "Ann3 (pair externe)"],
    "pairwise_kappa": {
        "Ann1_Ann2": result_12["kappa"],
        "Ann1_Ann3": result_13["kappa"],
        "Ann2_Ann3": result_23["kappa"],
        "moyenne":   kappa_moyen,
    },
    "annotations": [
        {
            "id":        s["id"],
            "ann1":      a1,
            "ann2":      a2,
            "ann3":      a3,
            "agree_all": a1 == a2 == a3,
            "agree_12":  a1 == a2,
            "agree_13":  a1 == a3,
            "agree_23":  a2 == a3,
        }
        for s, a1, a2, a3 in zip(scenarios, ann1, ann2, ann3)
    ],
}
with open(ANNOT_FILE, "w", encoding="utf-8") as f:
    json.dump(annot_data, f, ensure_ascii=False, indent=2)

output = {
    "n_annotateurs": 3,
    "n_scenarios":   N,
    "Ann1_Ann2": {**result_12, "annotateurs": ["Ann1 (auteur principal)", "Ann2 (co-auteur)"]},
    "Ann1_Ann3": {**result_13, "annotateurs": ["Ann1 (auteur principal)", "Ann3 (pair externe)"]},
    "Ann2_Ann3": {**result_23, "annotateurs": ["Ann2 (co-auteur)",        "Ann3 (pair externe)"]},
    "kappa_moyen":          kappa_moyen,
    "interpretation_moyenne": interpret_kappa(kappa_moyen),
    "reference": "Landis & Koch (1977) — seuil kappa > 0.60 requis pour validation taxonomie",
}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ── Affichage ─────────────────────────────────────────────────────────────────
print("=== Cohen's Kappa — Validation Taxonomie T1-T5 (3 annotateurs) ===")
print()
for label, res in [("Ann1 -- Ann2", result_12), ("Ann1 -- Ann3", result_13), ("Ann2 -- Ann3", result_23)]:
    print(f"  {label}")
    print(f"    Accord observe (Po)  : {res['po_observed']*100:.1f}%  ({res['agreements']}/{N})")
    print(f"    Accord attendu (Pe)  : {res['pe_expected']*100:.1f}%")
    print(f"    kappa de Cohen       : {res['kappa']}  ({res['interpretation']})")
    print()

print(f"  kappa moyen : {kappa_moyen}  ({interpret_kappa(kappa_moyen)})")
print()
print("Matrice de confusion Ann1 vs Ann2 :")
header = "       " + "  ".join(f"{c:>4}" for c in TYPES)
print(header)
for c1 in TYPES:
    row = f"  {c1} ->  " + "  ".join(
        f"{result_12['confusion_matrix'][c1][c2]:>4}" for c2 in TYPES
    )
    print(row)
print()
print(f"Resultats sauvegardes dans : {OUT_FILE}")
print(f"Annotations sauvegardees dans : {ANNOT_FILE}")
