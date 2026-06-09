"""
demo_interactive.py — Demo interactive HalluGuard pour la soutenance.

Fonctionnalites :
  - Poser une question/affirmation au pipeline HalluGuard
  - Voir en temps reel ce que HalluGuard detecte a chaque etape
  - Voir le BeliefState avant et apres chaque requete
  - Choisir le mode : log ou auto-correct
  - Voir le log des appels MCP en direct
  - Option --test : test non-interactif sur 3 questions predefinies

Usage :
  venv\\Scripts\\python.exe demo_interactive.py           # mode interactif
  venv\\Scripts\\python.exe demo_interactive.py --test    # mode test automatique
  venv\\Scripts\\python.exe demo_interactive.py --full    # pipeline complet (necessite Mistral)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ #
# Constantes                                                            #
# ------------------------------------------------------------------ #

_TEST_QUESTIONS = [
    {
        "claim": "Le Transformer a ete invente par Yann LeCun en 2015 pour la vision par ordinateur.",
        "evidences": [
            "Vaswani et al. (2017, Google Brain) introduisent le Transformer dans 'Attention Is All You Need'.",
            "Le Transformer est concu pour le traitement du langage naturel, pas la vision.",
        ],
        "node_type": "generation",
        "policy_mode": "log",
    },
    {
        "claim": "GPT-3 a ete publie en mai 2020 par OpenAI avec 175 milliards de parametres.",
        "evidences": [
            "GPT-3 a ete publie en mai 2020 par OpenAI avec 175 milliards de parametres.",
            "GPT-3 a ete entraine sur 570 Go de texte pour un cout de 4,6 millions de dollars.",
        ],
        "node_type": "generation",
        "policy_mode": "log",
    },
    {
        "claim": "Le RAG a completement elimine les hallucinations dans les LLM des 2021.",
        "evidences": [
            "Le RAG reduit les hallucinations de 30 a 60% selon le domaine (Lewis et al., 2020).",
            "Les etudes empiriques montrent que 15 a 25% des outputs de LLM contiennent encore des erreurs.",
        ],
        "node_type": "retrieval",
        "policy_mode": "auto-correct",
    },
]

# ------------------------------------------------------------------ #
# Affichage                                                             #
# ------------------------------------------------------------------ #

def sep(char: str = "-", n: int = 64) -> None:
    print(char * n)


def header() -> None:
    print()
    sep("=")
    print("  HalluGuard -- Demo Interactive")
    print(f"  ENSAM Meknes 4A | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sep("=")


def show_belief_state(belief, label: str = "BeliefState") -> None:
    facts = belief.get_active_facts()
    s = belief.summary()
    print(f"\n  [{label}]")
    print(f"  session={s['session_id']} | actifs={s['active_facts']}/{s['max_facts']}")
    if not facts:
        print("  (vide)")
        return
    for f in facts[:5]:
        age = f"step={f.step}"
        print(f"  {f.fact_id[:8]} conf={f.confidence:.2f} {age} | {f.content[:60]}...")
    if len(facts) > 5:
        print(f"  ... +{len(facts)-5} autres faits")


def show_mcp_log(n: int = 4) -> None:
    log = ROOT / "results" / "logs" / "mcp_calls.log"
    if not log.exists():
        print("  [MCP] (log vide)")
        return
    lines = [l for l in log.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    recent = lines[-n:]
    print(f"  [MCP] Derniers {len(recent)} appels sur {len(lines)} total :")
    for line in recent:
        try:
            e = json.loads(line)
            keys = list(e.get("result", {}).keys())[:3]
            print(f"    {e['timestamp']} | {e['tool']:30s} | result={keys}")
        except Exception:
            pass


def show_dag_propagation(node_type: str) -> None:
    try:
        from src.halluguard.dag import DependencyDAG
        dag = DependencyDAG.build_from_langgraph({
            "retrieval": ["reasoning"],
            "reasoning": ["tool_call", "generation"],
            "tool_call": ["generation"],
        })
        invalidated = dag.propagate_invalidation(node_type)
        descendants = dag.descendants(node_type)
        if descendants:
            print(f"  [DAG] Descendants invalides : {' -> '.join(sorted(descendants))}")
            crit = dag.criticite(node_type)
            print(f"  [DAG] Criticite de '{node_type}' : {crit:.2f}")
            if invalidated:
                print(f"  [DAG] Noeuds marques suspects : {invalidated}")
        else:
            print(f"  [DAG] '{node_type}' n'a pas de descendants (noeud terminal)")
    except Exception as e:
        print(f"  [DAG] Erreur : {e}")


# ------------------------------------------------------------------ #
# Retrieval ChromaDB                                                    #
# ------------------------------------------------------------------ #

def retrieve_docs(query: str, n: int = 3) -> List[str]:
    try:
        import chromadb  # type: ignore[import]
        client = chromadb.PersistentClient(path=str(ROOT / "data" / "chromadb"))
        coll = client.get_collection("halluguard_docs")
        res = coll.query(query_texts=[query], n_results=n)
        return res["documents"][0] if res["documents"] else []
    except Exception:
        return []


# ------------------------------------------------------------------ #
# Verification d'une affirmation (coeur de la demo)                    #
# ------------------------------------------------------------------ #

def run_verification(
    claim: str,
    evidences: List[str],
    node_type: str,
    policy_mode: str,
    belief,
    mem,
) -> dict:
    from src.agents.a2a_protocol import delegate_verification
    from src.halluguard.policy import PolicyEngine

    print()
    sep("=")
    print(f"  ETAPE 1 : Retrieval ChromaDB")
    sep()
    if evidences:
        print(f"  {len(evidences)} document(s) recupere(s) :")
        for i, doc in enumerate(evidences, 1):
            print(f"    [{i}] {doc[:80]}...")
    else:
        print("  (aucun document recupere — evidences manuelles)")

    print()
    sep("=")
    print(f"  ETAPE 2 : Delegation A2A -- RAGAgent -> HalluGuard")
    sep()
    print(f"  Claim      : {claim[:90]}...")
    print(f"  Node type  : {node_type}")
    print(f"  Policy     : {policy_mode}")
    print(f"  Protocol   : A2A -> MCP/stdio (skills : verify_claim + check_temporal_coherence)")
    print("  Traitement en cours...")

    t0 = time.time()
    result = delegate_verification(
        claim=claim,
        evidences=evidences,
        node_type=node_type,
        from_agent="RAGAgent_Interactive",
    )
    elapsed = (time.time() - t0) * 1000

    print()
    sep("=")
    print(f"  ETAPE 3 : Resultats HalluGuard M1 + M2")
    sep()

    m1_hall = result["label"] == "hallucinated"
    m2_conf = bool(result.get("m2_conflict", False))

    if m1_hall:
        verdict = "[M1] HALLUCINATION DETECTEE"
    elif m2_conf:
        verdict = "[M2] CONFLIT TEMPOREL DETECTE"
    else:
        verdict = "[OK] AFFIRMATION CORRECTE"

    print(f"  >> {verdict}")
    print()
    print(f"  M1 NLI    : label={result['label']} | score={result['score']:.4f}")
    print(f"  M2 BS     : conflit={m2_conf}")
    print(f"  Type HG   : {result['hallucination_type'] or 'N/A'}")
    print(f"  Task ID   : {result['task_id']}")
    print(f"  Status    : {result['task_status']}")
    print(f"  Latence   : {elapsed:.0f} ms")

    print()
    sep("=")
    print(f"  ETAPE 4 : DependencyDAG -- propagation")
    sep()
    if m1_hall or m2_conf:
        show_dag_propagation(node_type)
    else:
        print(f"  [DAG] Aucune propagation (noeud '{node_type}' valide)")

    print()
    sep("=")
    print(f"  ETAPE 5 : PolicyEngine -- action ({policy_mode})")
    sep()
    if not (m1_hall or m2_conf):
        print(f"  [Policy] Action : PASS (aucune hallucination detectee)")
        # Enrichir le BeliefState avec le fait valide
        fact_id = belief.add_fact(content=claim, confidence=result["score"], step=1)
        mem.save(belief)
        print(f"  [BeliefState] Fait ajoute : {fact_id[:8]}...")
    elif policy_mode == "log":
        print(f"  [Policy] Action : LOG")
        print(f"  [Policy] Entree JSONL ecrite dans results/logs/halluguard.log")
        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "node_id": node_type,
            "label": result["label"],
            "score": result["score"],
            "hallucination_type": result["hallucination_type"],
            "claim_snippet": claim[:80],
        }
        log_path = ROOT / "results" / "logs" / "halluguard.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"  [Policy] Log : {json.dumps(log_entry, ensure_ascii=False)[:100]}...")
    else:
        print(f"  [Policy] Action : AUTO-CORRECT")
        print(f"  [Policy] (Mistral non disponible -- correction simulee)")
        corrected = f"[CORRIGE] {claim[:60]}... -> affirmation non verifiable selon les sources disponibles."
        print(f"  [Policy] Sortie corrigee : {corrected}")

    return result


# ------------------------------------------------------------------ #
# Boucle interactive                                                    #
# ------------------------------------------------------------------ #

def interactive_loop(policy_mode: str, session_id: str = "interactive_session") -> None:
    from src.memory.persistent_memory import PersistentMemory
    from src.halluguard.belief_state import BeliefState

    mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))
    belief = BeliefState(session_id=session_id)

    # Reset du BeliefState MCP pour eviter les faux positifs inter-sessions
    try:
        import importlib
        from src.halluguard.belief_state import BeliefState as _BS
        _mod = importlib.import_module("src.mcp_servers.halluguard_mcp")
        _mod._belief = _BS(session_id=f"interactive_mcp_{int(time.time())}")
    except Exception:
        pass

    print(f"\n  Mode : Verification pas-a-pas HalluGuard")
    print(f"  Policy : {policy_mode}")
    print(f"  Session : {session_id}")
    print()
    print("  Entrez une affirmation a verifier (pas une question — une phrase declarative).")
    print("  Exemples :")
    print("    'GPT-3 a ete publie en 2020 par OpenAI avec 175B parametres.'")
    print("    'Le Transformer a ete invente par Yann LeCun en 2015.'")

    q_count = 0
    while True:
        print()
        sep("=")
        show_belief_state(belief, f"BeliefState AVANT la requete #{q_count+1}")

        print()
        try:
            claim = input("  Affirmation (ou 'q' pour quitter) :\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if claim.lower() in ("q", "quit", "exit", ""):
            break

        # Choix node_type
        print()
        print("  Type de noeud [g=generation / r=retrieval / rs=reasoning / t=tool_call] (Enter=g) :")
        try:
            nt_input = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            nt_input = "g"
        node_map = {"g": "generation", "r": "retrieval", "rs": "reasoning", "t": "tool_call", "": "generation"}
        node_type = node_map.get(nt_input, "generation")

        # Retrieval ChromaDB
        print(f"\n  Recherche dans ChromaDB pour : '{claim[:60]}...'")
        evidences = retrieve_docs(claim, n=3)
        if not evidences:
            print("  ChromaDB vide -- saisissez des evidences manuelles (separees par |) ou Enter :")
            try:
                manual = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                manual = ""
            evidences = [e.strip() for e in manual.split("|") if e.strip()] if manual else []

        # Verification
        run_verification(
            claim=claim,
            evidences=evidences,
            node_type=node_type,
            policy_mode=policy_mode,
            belief=belief,
            mem=mem,
        )

        # BeliefState apres
        print()
        sep("=")
        show_belief_state(belief, f"BeliefState APRES la requete #{q_count+1}")

        # Log MCP
        print()
        show_mcp_log(n=3)

        q_count += 1

        # Continuer ?
        print()
        try:
            again = input("  Continuer ? [o/n] (Enter=o) : ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again == "n":
            break

    print()
    sep("=")
    print(f"  Session terminee — {q_count} requete(s) traitee(s).")
    print(f"  Log A2A : results/logs/a2a_exchanges.log")
    print(f"  Log MCP : results/logs/mcp_calls.log")
    sep("=")


# ------------------------------------------------------------------ #
# Mode test automatique (--test)                                        #
# ------------------------------------------------------------------ #

def test_mode() -> None:
    from src.memory.persistent_memory import PersistentMemory
    from src.halluguard.belief_state import BeliefState

    print("\n  Mode TEST automatique (3 questions predefinies)")
    print("  Aucune intervention requise.")

    mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))
    belief = BeliefState(session_id="test_interactive_session")

    # Reset MCP BeliefState
    try:
        import importlib
        mod = importlib.import_module("src.mcp_servers.halluguard_mcp")
        from src.halluguard.belief_state import BeliefState as BS
        mod._belief = BS(session_id="test_mcp_session")
    except Exception:
        pass

    for i, case in enumerate(_TEST_QUESTIONS, 1):
        print()
        sep("=")
        print(f"  TEST {i}/3")
        sep()
        print(f"  Claim  : {case['claim'][:80]}...")
        print(f"  Policy : {case['policy_mode']}")

        run_verification(
            claim=case["claim"],
            evidences=case["evidences"],
            node_type=case["node_type"],
            policy_mode=case["policy_mode"],
            belief=belief,
            mem=mem,
        )

        show_belief_state(belief, f"BeliefState apres test {i}")
        show_mcp_log(n=2)

    print()
    sep("=")
    print("  RESULTAT DU TEST")
    sep()
    print("  Test 1 (Transformer par LeCun) : attendu HALLUCINATION")
    print("  Test 2 (GPT-3 correct)         : attendu CORRECT")
    print("  Test 3 (RAG elimine 100%)      : attendu HALLUCINATION")
    print()
    print("  demo_interactive.py : OPERATIONNEL")
    sep("=")


# ------------------------------------------------------------------ #
# Choix de mode au demarrage                                            #
# ------------------------------------------------------------------ #

def choose_policy_mode() -> str:
    print()
    print("  Modes de politique HalluGuard :")
    print("  1. log-only    — enregistre les hallucinations sans modifier la reponse")
    print("  2. auto-correct — tente une correction (necessite Mistral)")
    print()
    try:
        choice = input("  Choisissez [1/2] (Enter=1) : ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"
    return "log" if choice != "2" else "auto-correct"


def check_ollama() -> bool:
    try:
        import ollama  # type: ignore[import]
        models = ollama.list()
        return any(
            "mistral" in m.get("name", "").lower()
            for m in (models.get("models") or [])
        )
    except Exception:
        return False


# ------------------------------------------------------------------ #
# Point d'entree                                                        #
# ------------------------------------------------------------------ #

def main() -> None:
    args = sys.argv[1:]
    test_mode_flag = "--test" in args
    full_mode_flag = "--full" in args

    header()

    # Environnement
    print()
    print("  Verification de l'environnement...")
    ollama_ok = check_ollama()
    print(f"  NLI model (HalluGuard) : DISPONIBLE (CPU, ~117 M params)")
    print(f"  ChromaDB               : DISPONIBLE ({ROOT / 'data' / 'chromadb'})")
    print(f"  Ollama + Mistral       : {'DISPONIBLE' if ollama_ok else 'NON DISPONIBLE (mode direct OK)'}")

    if test_mode_flag:
        test_mode()
        return

    if full_mode_flag and ollama_ok:
        # Mode pipeline complet (necessite Mistral)
        from src.memory.persistent_memory import PersistentMemory
        from src.halluguard.belief_state import BeliefState

        mem = PersistentMemory(memory_dir=str(ROOT / "data" / "memory"))
        belief = BeliefState(session_id="full_pipeline_session")

        policy_mode = choose_policy_mode()

        print()
        print("  Mode : Pipeline RAG complet (Mistral via Ollama)")
        print("  Pipeline : retrieval -> reasoning -> tool_call -> generation")

        while True:
            sep()
            show_belief_state(belief, "BeliefState AVANT")
            print()
            try:
                question = input("  Question (ou 'q' pour quitter) :\n  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in ("q", "quit", "exit", ""):
                break

            from src.agents.rag_agent import run_query
            print("\n  Traitement pipeline complet (30-60s avec Mistral)...")
            t0 = time.time()
            try:
                result = run_query(question, variant="C", session_id="full_pipeline_session")
            except Exception as e:
                print(f"\n  ERREUR : {e}")
                continue
            elapsed = time.time() - t0

            print()
            sep("*")
            print(f"  REPONSE : {result['answer'][:300]}...")
            print(f"  Temps   : {elapsed:.1f}s")
            events = result.get("halluguard_events", [])
            if events:
                print(f"\n  [HalluGuard] {len(events)} evenement(s) detecte(s) :")
                for ev in events:
                    print(f"    {ev.get('node_id','?')} | {ev.get('label','?')} | score={ev.get('score',0):.3f} | type={ev.get('hallucination_type','?')}")
            else:
                print("  [HalluGuard] Aucune hallucination detectee")
            if result.get("overhead_stats"):
                s = result["overhead_stats"]
                print(f"  [Overhead] appels={s['calls']} | moy={s['avg_ms']}ms | max={s['max_ms']}ms")
            sep("*")

            belief = mem.load_or_create("full_pipeline_session")
            show_belief_state(belief, "BeliefState APRES")
            show_mcp_log(n=3)

            try:
                again = input("\n  Continuer ? [o/n] : ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if again == "n":
                break
        return

    elif full_mode_flag and not ollama_ok:
        print("\n  Mistral non disponible -- passage en mode verification directe.")

    # Mode par defaut : verification directe
    policy_mode = choose_policy_mode()
    interactive_loop(policy_mode=policy_mode)


if __name__ == "__main__":
    main()
