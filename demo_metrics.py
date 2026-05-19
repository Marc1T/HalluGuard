"""demo_metrics.py — Affiche les metriques finales du benchmark HaluEval-Agentic."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    bench = ROOT / "results" / "resultats_comparatifs.json"
    d = json.loads(bench.read_text(encoding="utf-8"))
    v = d["variants"]
    a = v["A"]["metrics"]
    b = v["B"]["metrics"]
    c = v["C"]["metrics"]

    print(f"  Benchmark : {d['benchmark']} ({d['n_scenarios']} scenarios)")
    print()
    print(f"  Variante A (baseline)  : {a['detection_rate_pct']:.1f}% detection | {a['avg_latency_ms']:.0f} ms overhead")
    print(f"  Variante B (NLI M1)    : {b['detection_rate_pct']:.1f}% detection | {b['avg_latency_ms']:.1f} ms overhead")
    print(f"  Variante C (M1+M2+MCP) : {c['detection_rate_pct']:.1f}% detection | {c['avg_latency_ms']:.1f} ms overhead")
    print()
    print(f"  PBR@1  B={b['pbr1_pct']:.1f}%  C={c['pbr1_pct']:.1f}%")
    print(f"  Gain   A -> C : +{c['detection_rate_pct']:.1f} points de detection")
    print()
    print("  Detail par type (Variante C) :")
    for t, td in v["C"]["by_type"].items():
        print(f"    {t} : {td['detection_rate_pct']:.1f}% ({td['detected']}/{td['total']})")
except Exception as e:
    print(f"  Erreur lecture metriques : {e}")
