"""
Tests du serveur MCP HalluGuard - 4 tools via protocole stdio reel.
Lance le serveur en sous-process, appelle chaque tool, verifie la structure JSON.
Lancer avec : python src/tests/test_mcp.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

SEPARATOR = "-" * 60
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON  = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
MCP_SERVER   = PROJECT_ROOT / "src" / "mcp_servers" / "halluguard_mcp.py"
MCP_LOG      = PROJECT_ROOT / "results" / "logs" / "mcp_calls.log"


def _parse_tool_result(result) -> dict:
    """Extrait et parse le JSON retourne par un tool MCP."""
    if not result.content:
        raise ValueError("Reponse MCP vide")
    text = result.content[0].text
    return json.loads(text)


async def run_all_tests():
    """Lance le serveur MCP et execute les 4 tests."""

    env = dict(os.environ)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["HF_HUB_OFFLINE"] = "1"         # charge depuis cache local, pas d'appels HF API
    env["TRANSFORMERS_OFFLINE"] = "1"   # idem pour transformers

    params = StdioServerParameters(
        command=str(VENV_PYTHON),
        args=[str(MCP_SERVER)],
        env=env,
    )

    print("=" * 60)
    print("  HalluGuard - Tests serveur MCP (stdio)")
    print(f"  Serveur : {MCP_SERVER.name}")
    print(f"  Python  : {VENV_PYTHON.name}")
    print("=" * 60)
    print("\nDemarrage du serveur MCP en sous-process...")
    print("(premier appel NLI = chargement modele ~25s)\n")

    passed = 0

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Serveur MCP demarre et initialise.\n")

            # -------------------------------------------------------- #
            # Pre-warmup : charge le modele NLI dans le sous-process    #
            # (appel silencieux, non compte, timeout 600s)               #
            # -------------------------------------------------------- #
            print("Pre-chargement modele NLI (peut prendre plusieurs minutes sur CPU)...")
            try:
                await asyncio.wait_for(
                    session.call_tool("verify_claim", {
                        "claim": "warmup claim to preload NLI model",
                        "evidences": ["warmup evidence sentence"],
                    }),
                    timeout=600.0,
                )
                print("Modele NLI charge.\n")
            except Exception as _warmup_err:
                print(f"  Avertissement warmup : {_warmup_err}\n")

            # -------------------------------------------------------- #
            # Test 1 — add_fact                                          #
            # -------------------------------------------------------- #
            print(f"[Test 1] add_fact -> status, fact_id")
            print(SEPARATOR)
            try:
                raw = await asyncio.wait_for(
                    session.call_tool("add_fact", {
                        "fact": "The Eiffel Tower is located in Paris, France.",
                        "confidence": 0.99,
                        "step": 1,
                    }),
                    timeout=30.0,
                )
                result = _parse_tool_result(raw)
                print(f"  Output JSON :\n{json.dumps(result, indent=4, ensure_ascii=False)}")

                assert "status" in result, "champ 'status' manquant"
                assert "fact_id" in result, "champ 'fact_id' manquant"
                assert result["status"] in ("added", "evicted_oldest"), \
                    f"status inconnu : {result['status']}"
                print(f"  --> PASSED")
                passed += 1
            except Exception as e:
                print(f"  --> FAILED : {e}")

            # -------------------------------------------------------- #
            # Test 2 — get_belief_state                                  #
            # -------------------------------------------------------- #
            print(f"\n[Test 2] get_belief_state -> JSON complet")
            print(SEPARATOR)
            try:
                raw = await asyncio.wait_for(
                    session.call_tool("get_belief_state", {}),
                    timeout=30.0,
                )
                result = _parse_tool_result(raw)
                print(f"  Output JSON :\n{json.dumps(result, indent=4, ensure_ascii=False)}")

                assert "session_id" in result, "champ 'session_id' manquant"
                assert "active_facts" in result, "champ 'active_facts' manquant"
                assert "facts" in result, "champ 'facts' manquant"
                assert result["active_facts"] >= 1, "au moins 1 fait attendu"
                print(f"  --> PASSED")
                passed += 1
            except Exception as e:
                print(f"  --> FAILED : {e}")

            # -------------------------------------------------------- #
            # Test 3 — verify_claim  (charge le modele NLI)              #
            # -------------------------------------------------------- #
            print(f"\n[Test 3] verify_claim -> score, label, hallucination_type")
            print(SEPARATOR)
            try:
                raw = await asyncio.wait_for(
                    session.call_tool("verify_claim", {
                        "claim": "The Eiffel Tower is located in London.",
                        "evidences": ["The Eiffel Tower is in Paris, France."],
                        "node_type": "generation",
                    }),
                    timeout=60.0,
                )
                result = _parse_tool_result(raw)
                print(f"  Output JSON :\n{json.dumps(result, indent=4, ensure_ascii=False)}")

                assert "score" in result, "champ 'score' manquant"
                assert "label" in result, "champ 'label' manquant"
                assert "hallucination_type" in result, "champ 'hallucination_type' manquant"
                assert result["label"] == "hallucinated", \
                    f"attendu 'hallucinated', obtenu '{result['label']}'"
                assert result["score"] > 0.5, f"score trop faible : {result['score']}"
                print(f"  --> PASSED")
                passed += 1
            except Exception as e:
                print(f"  --> FAILED : {e}")

            # -------------------------------------------------------- #
            # Test 4 — check_temporal_coherence                          #
            # -------------------------------------------------------- #
            print(f"\n[Test 4] check_temporal_coherence -> conflict, fact_id, score")
            print(SEPARATOR)
            try:
                raw = await asyncio.wait_for(
                    session.call_tool("check_temporal_coherence", {
                        "claim": "The Eiffel Tower is located in London.",
                    }),
                    timeout=120.0,
                )
                result = _parse_tool_result(raw)
                print(f"  Output JSON :\n{json.dumps(result, indent=4, ensure_ascii=False)}")

                assert "conflict" in result, "champ 'conflict' manquant"
                assert "fact_id" in result, "champ 'fact_id' manquant"
                assert "score" in result, "champ 'score' manquant"
                assert result["conflict"] is True, \
                    f"attendu conflict=True (Paris vs London), obtenu {result['conflict']}"
                print(f"  --> PASSED")
                passed += 1
            except Exception as e:
                print(f"  --> FAILED : {e}")

    # ------------------------------------------------------------ #
    # Verification du fichier log MCP                               #
    # ------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  Fichier log MCP : {MCP_LOG}")
    print(f"{'='*60}")
    if MCP_LOG.exists():
        lines = [l for l in MCP_LOG.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        print(f"  Entrees totales : {len(lines)}")
        for line in lines[-4:]:   # afficher les 4 derniers appels
            entry = json.loads(line)
            print(f"  {entry['timestamp']} | {entry['tool']:<28} | result={str(entry['result'])[:60]}")
    else:
        print("  ATTENTION : fichier log absent")

    print(f"\n{'='*60}")
    print(f"  Resultat final : {passed}/4 tests passes")
    print(f"{'='*60}")
    return passed


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result == 4 else 1)
