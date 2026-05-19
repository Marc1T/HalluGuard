"""demo_logs.py — Affiche les derniers echanges A2A et le compte des appels MCP."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Derniers echanges A2A
a2a_log = ROOT / "results" / "logs" / "a2a_exchanges.log"
if not a2a_log.exists():
    print("  (log A2A vide)")
else:
    lines = [l for l in a2a_log.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    recent = lines[-3:]
    if not recent:
        print("  (log A2A vide)")
    for line in recent:
        try:
            e = json.loads(line)
            r = e.get("verification_result", {})
            eid = e.get("exchange_id", "?")[:12]
            frm = e.get("from_agent", "?")
            to  = e.get("to_agent", "?")
            lbl = r.get("label", "?")
            score = r.get("score", "?")
            det = r.get("detected", "?")
            print(f"  [{eid}] {frm} -> {to} | {lbl} | score={score} | detected={det}")
        except Exception:
            pass

# Compte des appels MCP
mcp_log = ROOT / "results" / "logs" / "mcp_calls.log"
if mcp_log.exists():
    count = sum(1 for l in mcp_log.read_text(encoding="utf-8").strip().split("\n") if l.strip())
    print(f"  Appels MCP journalises (total) : {count}")
else:
    print("  (log MCP vide)")
