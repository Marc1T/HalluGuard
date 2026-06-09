"""
Tests statistiques pour l'article HalluGuard.
Calcule : McNemar (A/B, A/C, B/C), Bootstrap CI 95%, résumé JSON.
"""
import json, random, math
from pathlib import Path

RESULTS_FILE = Path(__file__).parents[2] / "results" / "resultats_comparatifs.json"
OUT_FILE     = Path(__file__).parents[2] / "results" / "statistical_analysis.json"

random.seed(42)

# ── helpers ──────────────────────────────────────────────────────────────────

def mcnemar(vec_x, vec_y):
    """McNemar's test avec correction de continuité de Yates.
    Retourne chi2, p-value (bilatéral), et la table 2×2."""
    b01 = sum(1 for x, y in zip(vec_x, vec_y) if not x and y)   # X=0, Y=1
    b10 = sum(1 for x, y in zip(vec_x, vec_y) if x and not y)   # X=1, Y=0
    discordant = b01 + b10
    if discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b01": 0, "b10": 0,
                "significant_005": False, "note": "no discordant pairs"}
    # Correction de continuité
    chi2 = (abs(b01 - b10) - 1) ** 2 / discordant
    # p-value depuis distribution chi2(1) — approximation Wilson-Hilferty
    p = chi2_sf(chi2, df=1)
    return {"chi2": round(chi2, 4), "p_value": round(p, 4),
            "b01": b01, "b10": b10,
            "significant_005": p < 0.05,
            "interpretation": interpret_mcnemar(p)}

def chi2_sf(x, df=1):
    """Survival function chi2(df=1) via approximation normale."""
    # Pour df=1 : chi2_sf(x) = 2 * Phi(-sqrt(x))
    z = math.sqrt(x)
    return 2 * (1 - normal_cdf(z))

def normal_cdf(x):
    """CDF normale standard — approximation Hart."""
    if x < 0:
        return 1 - normal_cdf(-x)
    t = 1 / (1 + 0.2316419 * x)
    poly = t * (0.319381530
               + t * (-0.356563782
               + t * (1.781477937
               + t * (-1.821255978
               + t * 1.330274429))))
    return 1 - (1/math.sqrt(2*math.pi)) * math.exp(-0.5*x*x) * poly

def interpret_mcnemar(p):
    if p < 0.001: return "hautement significatif (p<0.001)"
    if p < 0.01:  return "très significatif (p<0.01)"
    if p < 0.05:  return "significatif (p<0.05)"
    if p < 0.10:  return "tendance non significative (p<0.10)"
    return "non significatif (p>0.10)"

def bootstrap_ci(vec, n_boot=10000, alpha=0.05):
    """Bootstrap CI (percentile) pour un taux de détection binaire."""
    n = len(vec)
    obs_rate = sum(vec) / n
    boot_rates = []
    for _ in range(n_boot):
        sample = [random.choice(vec) for _ in range(n)]
        boot_rates.append(sum(sample) / n)
    boot_rates.sort()
    lo = boot_rates[int(alpha/2 * n_boot)]
    hi = boot_rates[int((1-alpha/2) * n_boot)]
    return {"rate": round(obs_rate*100, 1),
            "ci_lower": round(lo*100, 1),
            "ci_upper": round(hi*100, 1),
            "ci_95": f"[{round(lo*100,1)}%, {round(hi*100,1)}%]"}

# ── lecture données ───────────────────────────────────────────────────────────

with open(RESULTS_FILE) as f:
    data = json.load(f)

def get_detected_vec(variant_key):
    results = data["variants"][variant_key]["results"]
    id_to_detected = {r["id"]: r["detected"] for r in results}
    # ordre canonique s001-s060
    return [id_to_detected[f"s{str(i).zfill(3)}"] for i in range(1, 61)]

vecA = get_detected_vec("A")
vecB = get_detected_vec("B")
vecC = get_detected_vec("C")

# ── calculs ───────────────────────────────────────────────────────────────────

ci_A = bootstrap_ci(vecA)
ci_B = bootstrap_ci(vecB)
ci_C = bootstrap_ci(vecC)

mc_AB = mcnemar(vecA, vecB)
mc_AC = mcnemar(vecA, vecC)
mc_BC = mcnemar(vecB, vecC)

# Contingency tables complètes
def contingency(x, y, label_x, label_y):
    """Table 2x2 : (x=1,y=1), (x=1,y=0), (x=0,y=1), (x=0,y=0)"""
    return {
        f"{label_x}=1 {label_y}=1": sum(1 for a,b in zip(x,y) if a and b),
        f"{label_x}=1 {label_y}=0": sum(1 for a,b in zip(x,y) if a and not b),
        f"{label_x}=0 {label_y}=1": sum(1 for a,b in zip(x,y) if not a and b),
        f"{label_x}=0 {label_y}=0": sum(1 for a,b in zip(x,y) if not a and not b),
    }

# ── résultat final ────────────────────────────────────────────────────────────

output = {
    "meta": {
        "n_scenarios": 60,
        "n_bootstrap": 10000,
        "alpha": 0.05,
        "test": "McNemar avec correction de continuite de Yates",
        "ci_method": "Bootstrap percentile (seed=42)"
    },
    "detection_rates": {
        "A": ci_A,
        "B": ci_B,
        "C": ci_C
    },
    "mcnemar_tests": {
        "A_vs_B": {**mc_AB, "contingency": contingency(vecA, vecB, "A", "B")},
        "A_vs_C": {**mc_AC, "contingency": contingency(vecA, vecC, "A", "C")},
        "B_vs_C": {**mc_BC, "contingency": contingency(vecB, vecC, "B", "C")},
    },
    "summary_for_article": {
        "A_rate_ci":  ci_A["ci_95"],
        "B_rate_ci":  ci_B["ci_95"],
        "C_rate_ci":  ci_C["ci_95"],
        "A_vs_B_p":   mc_AB["p_value"],
        "A_vs_C_p":   mc_AC["p_value"],
        "B_vs_C_p":   mc_BC["p_value"],
        "B_vs_C_sig": mc_BC["significant_005"],
        "note_B_vs_C": (
            "Le test de McNemar ne porte que sur les 60 scenarios hallucines (rappel) : "
            "C detecte plus d'hallucinations que B. MAIS ce test ignore les faux positifs : "
            "sur les 30 scenarios corrects, M2 fait passer le FPR de 16.7% (B) a 80% (C), "
            "effondrant la precision (90.2% -> 68.4%) et le F1 (82.9% -> 76.5%). "
            "Le gain de rappel ne compense pas la perte de precision sur ce benchmark mono-tour."
        )
    }
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("=== Taux de détection (Bootstrap CI 95%) ===")
print(f"  Variante A : {ci_A['rate']}%  CI {ci_A['ci_95']}")
print(f"  Variante B : {ci_B['rate']}%  CI {ci_B['ci_95']}")
print(f"  Variante C : {ci_C['rate']}%  CI {ci_C['ci_95']}")
print()
print("=== Tests de McNemar (correction Yates) ===")
for pair, res in [("A vs B", mc_AB), ("A vs C", mc_AC), ("B vs C", mc_BC)]:
    print(f"  {pair} : chi2={res['chi2']}, p={res['p_value']}  -> {res['interpretation']}")
    print(f"          Table : b01={res['b01']} (discordant +), b10={res['b10']} (discordant -)")
print()
print(f"Résultats sauvegardés dans : {OUT_FILE}")
